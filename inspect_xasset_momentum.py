# inspect_xasset_momentum.py
# Requirements: pip install pandas numpy yfinance matplotlib
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# =========================== CONFIG ===========================
ASSETS = ["BTC-USD", "ETH-USD", "SPY"]   # any mix of crypto / stocks / ETFs
START = "2019-01-01"
END = None                                # None = today
INTERVAL = "1d"                           # "1d" or "1h" (script assumes daily for annualization)
LOOKBACK = 126                            # ~6 months (trading days)
TOP_K = 2
REBALANCE = "M"                           # "M" monthly, "W" weekly, "D" daily
INIT_CAPITAL = 100_000.0
COST_BPS = 10                             # per trade notional (commissions + slippage), applied on each buy/sell
ALLOW_SHORT = False                       # strategy stays long-only by default
SAVE_CSV = True                           # write CSV artifacts to ./outputs
# =============================================================

def _ensure_dir(path: str):
    import os
    os.makedirs(path, exist_ok=True)

def download_prices(assets: List[str], start: str, end: Optional[str], interval: str) -> pd.DataFrame:
    df = yf.download(assets, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        px = df["Close"].copy()
    else:
        px = df[["Close"]].copy()
    if isinstance(px, pd.Series):
        px = px.to_frame()
    # Forward-fill to handle mixed calendars (crypto vs equities) for valuation continuity
    px = px.sort_index().ffill().dropna(how="all")
    return px

@dataclass
class MomentumDecision:
    rebalance_dt: pd.Timestamp
    exec_dt: Optional[pd.Timestamp]
    ranks: Dict[str, float]            # lookback returns per asset on decision date
    winners: List[str]
    planned_weights: Dict[str, float]  # equal weights for winners, 0 otherwise

def compute_rebalance_decisions(prices: pd.DataFrame, lookback: int, top_k: int, rebalance: str) -> List[MomentumDecision]:
    import pandas as pd
    # lookback total returns
    ret_lb = prices.pct_change(lookback)

    # Period-end “marks” (labels at month-end / week-end etc.)
    marks = ret_lb.resample(rebalance).last().dropna(how="all")

    decisions: List[MomentumDecision] = []
    idx = prices.index

    for t, row in marks.iterrows():
        row = row.dropna()
        if row.empty:
            continue

        k = min(top_k, len(row))
        winners = list(row.nlargest(k).index)
        weights = {c: (1.0 / k if c in winners else 0.0) for c in prices.columns}

        # Last actual bar in this period (<= t)
        t_eff = ret_lb.index.asof(t)   # maps 2025-09-30 -> 2025-09-10 in your case
        if pd.isna(t_eff):
            continue  # no valid bar in this period

        # Execute on the first bar strictly AFTER t_eff
        pos = idx.searchsorted(t_eff, side="right")
        exec_dt = idx[pos] if pos < len(idx) else None  # None for incomplete last period

        decisions.append(
            MomentumDecision(
                rebalance_dt=t,
                exec_dt=exec_dt,
                ranks=row.to_dict(),
                winners=winners,
                planned_weights=weights
            )
        )
    return decisions

def decisions_to_weight_schedule(decisions: List[MomentumDecision], index: pd.DatetimeIndex, columns: List[str]) -> pd.DataFrame:
    import numpy as np
    w = pd.DataFrame(np.nan, index=index, columns=columns)  # start with NaN rows
    for d in decisions:
        if d.exec_dt is None or d.exec_dt not in w.index:
            continue
        for c in columns:
            w.loc[d.exec_dt, c] = d.planned_weights.get(c, 0.0)  # zeros are meaningful
    # Hold last decided weights until next exec date; fill pre-start with 0
    w = w.ffill().fillna(0.0)
    return w

def format_rank_string(ranks: Dict[str, float], winners: List[str]) -> str:
    s = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{a}:{v:+.2%}{'*' if a in winners else ''}" for a, v in s]
    return " | ".join(parts)

def inspect_cross_asset_momentum(
    assets: List[str] = ASSETS,
    start: str = START,
    end: Optional[str] = END,
    interval: str = INTERVAL,
    lookback: int = LOOKBACK,
    top_k: int = TOP_K,
    rebalance: str = REBALANCE,
    init_capital: float = INIT_CAPITAL,
    cost_bps: float = COST_BPS,
    allow_short: bool = ALLOW_SHORT,
    save_csv: bool = SAVE_CSV
) -> Dict[str, pd.DataFrame]:
    """
    Runs a transparent simulation of Cross-Asset Momentum with full introspection:
    - Rebalance decisions (what, when, why)
    - Trade blotter (buys/sells, quantity, cost, cash)
    - Daily ledger (cash, positions value, total equity, daily return)
    """
    prices = download_prices(assets, start, end, interval)
    prices = prices.reindex(prices.index.unique()).sort_index()
    prices = prices[assets]  # keep column order

    # === Decisions ===
    decisions = compute_rebalance_decisions(prices, lookback, top_k, rebalance)
    # Build a human-friendly decisions table
    dec_rows = []
    for d in decisions:
        dec_rows.append({
            "rebalance_dt": d.rebalance_dt,
            "exec_dt": d.exec_dt,
            "winners": ", ".join(d.winners),
            "ranks_str": format_rank_string(d.ranks, d.winners)
        })
    decisions_df = pd.DataFrame(dec_rows).set_index("rebalance_dt").sort_index()

    # === Weight schedule effective after execution ===
    weight_sched = decisions_to_weight_schedule(decisions, prices.index, prices.columns)

    # === Simulate with explicit cash, positions, and trading costs ===
    cost_pct = cost_bps / 10_000.0

    # State
    cash = init_capital
    pos_shares = pd.Series(0.0, index=prices.columns)  # fractional shares allowed for clarity
    ledger_rows = []
    trades_rows = []

    prev_total = init_capital
    for t in prices.index:
        px = prices.loc[t]

        # Target weights effective from this date onward
        target_w = weight_sched.loc[t]

        # Current portfolio market value BEFORE trading (using today's prices)
        pos_val = float((pos_shares * px).sum())
        total_before = cash + pos_val

        # Compute current (implied) weights before trading
        curr_w = (pos_shares * px) / (total_before if total_before != 0 else 1.0)

        # If weights need to change today, execute trades NOW using today's price.
        # (These trades are the result of a decision made on an earlier 'rebalance_dt',
        # scheduled to execute on this 't' = exec_dt; no look-ahead on price changes.)
        if not np.allclose(curr_w.values, target_w.values, atol=1e-10):
            # Desired dollar exposure per asset
            target_dollar = target_w * total_before
            curr_dollar = pos_shares * px
            delta_dollar = target_dollar - curr_dollar

            for c in prices.columns:
                d_notional = float(delta_dollar[c])
                if abs(d_notional) < 1e-9:
                    continue
                side = "BUY" if d_notional > 0 else "SELL"
                trade_px = float(px[c])
                if np.isnan(trade_px) or trade_px <= 0:
                    continue  # skip impossible trade (shouldn't happen after ffill)
                trade_shares = d_notional / trade_px
                fee = abs(d_notional) * cost_pct

                # Cash impact: pay cash for buys; receive for sells; always pay fee
                cash_change = -d_notional - fee
                cash += cash_change
                pos_shares[c] += trade_shares

                # Revalue portfolio after this single trade (using current prices)
                pos_val_after = float((pos_shares * px).sum())
                total_after = cash + pos_val_after

                trades_rows.append({
                    "date": t,
                    "asset": c,
                    "side": side,
                    "shares": trade_shares,
                    "price": trade_px,
                    "trade_notional": d_notional,
                    "fee_cost": fee,
                    "cash_after": cash,
                    "positions_value_after": pos_val_after,
                    "portfolio_value_after": total_after,
                })

        # End-of-day valuation (after any trades today)
        pos_val = float((pos_shares * px).sum())
        total_after = cash + pos_val
        day_ret = (total_after / prev_total - 1.0) if prev_total != 0 else 0.0
        prev_total = total_after

        # Store per-asset diagnostics (positions & weights)
        weights_now = (pos_shares * px) / (total_after if total_after != 0 else 1.0)
        row = {
            "date": t,
            "cash": cash,
            "positions_value": pos_val,
            "total_value": total_after,
            "daily_return": day_ret,
        }
        # Optional: embed per-asset state
        for c in prices.columns:
            row[f"pos_shares_{c}"] = pos_shares[c]
            row[f"weight_{c}"] = weights_now[c]
        ledger_rows.append(row)

    ledger_df = pd.DataFrame(ledger_rows).set_index("date")
    trades_df = pd.DataFrame(trades_rows)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["date", "asset"]).reset_index(drop=True)

    # Simple performance summary
    if len(ledger_df) >= 2:
        equity = ledger_df["total_value"]
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252.0 / max(1, len(equity))) - 1.0
        dd = (equity / equity.cummax() - 1.0).min()
    else:
        cagr, dd = np.nan, np.nan
    summary = pd.DataFrame(
        {"metric": ["Final Equity", "CAGR (approx, daily)", "Max Drawdown"],
         "value": [ledger_df["total_value"].iloc[-1], cagr, dd]}
    )

    if save_csv:
        _ensure_dir("outputs")
        decisions_df.to_csv("outputs/momentum_decisions.csv", encoding="utf-8")
        weight_sched.to_csv("outputs/weight_schedule.csv", encoding="utf-8")
        ledger_df.to_csv("outputs/portfolio_ledger.csv", encoding="utf-8")
        trades_df.to_csv("outputs/trade_blotter.csv", encoding="utf-8", index=False)
        summary.to_csv("outputs/summary.csv", encoding="utf-8", index=False)
        print("Saved CSVs to ./outputs/: momentum_decisions.csv, weight_schedule.csv, portfolio_ledger.csv, trade_blotter.csv, summary.csv")

    return {
        "decisions": decisions_df,
        "weight_schedule": weight_sched,
        "ledger": ledger_df,
        "trades": trades_df,
        "summary": summary,
    }

# ------------------------ run if script ------------------------
if __name__ == "__main__":
    out = inspect_cross_asset_momentum()
    # Quick peek
    print("\n=== Summary ===")
    print(out["summary"].to_string(index=False))
    print("\n=== Last 5 trades ===")
    print(out["trades"].tail(5).to_string(index=False))
    print("\n=== Last 5 ledger rows ===")
    print(out["ledger"].tail(5)[["cash","positions_value","total_value","daily_return"]].to_string())
    print("\n=== Last 3 decisions ===")
    print(out["decisions"].tail(3).to_string())
