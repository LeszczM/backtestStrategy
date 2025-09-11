# inspect_pl_momentum.py
# Requirements: pip install pandas numpy yfinance matplotlib
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Optional

# =========================== CONFIG ===========================
# --- WIG20 (current blue-chips) + PLN-hedged S&P500 ETF (Yahoo: ETFBSPXPL.WA) ---
WIG20_TICKERS = [
    "SPL.WA",  # Santander Bank Polska
    "PKO.WA",  # PKO Bank Polski
    "MBK.WA",  # mBank
    "OPL.WA",  # Orange Polska
    "PEO.WA",  # Bank Pekao
    "KGH.WA",  # KGHM
    "PKN.WA",  # Orlen
    "PGE.WA",  # PGE
    "PZU.WA",  # PZU
    "ALR.WA",  # Alior Bank
    "BDX.WA",  # Budimex
    "CCC.WA",  # CCC
    "CDR.WA",  # CD Projekt
    "KRU.WA",  # Kruk
    "KTY.WA",  # Grupa Kęty
    "LPP.WA",  # LPP
    "DNP.WA",  # Dino Polska
    "ALE.WA",  # Allegro
    "PCO.WA",  # Pepco Group
    "ZAB.WA",  # Żabka Group
]
EXTRA_ETFS = ["ETFBSPXPL.WA"]  # Beta ETF S&P 500 PLN-Hedged

ASSETS = WIG20_TICKERS + EXTRA_ETFS

START = "2018-01-01"            # choose earlier/later as you wish
END = None                      # None = today
INTERVAL = "1d"
LOOKBACK = 126                  # ~6 months
TOP_K = 5                       # how many winners to hold each rebalance
REBALANCE = "W-FRI"                # month-end decisions, execute next bar
INIT_CAPITAL = 100_000.0
COST_BPS = 10                   # per trade notional
ALLOW_SHORT = False
SAVE_CSV = True
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
    px = px.sort_index().ffill().dropna(how="all")
    # drop columns with no data
    all_nan = [c for c in px.columns if px[c].dropna().empty]
    if all_nan:
        print("Warning: dropped symbols with no data:", all_nan)
        px = px.drop(columns=all_nan)
    return px

@dataclass
class MomentumDecision:
    rebalance_dt: pd.Timestamp
    exec_dt: Optional[pd.Timestamp]
    ranks: Dict[str, float]
    winners: List[str]
    planned_weights: Dict[str, float]

def compute_rebalance_decisions(prices: pd.DataFrame, lookback: int, top_k: int, rebalance: str) -> List[MomentumDecision]:
    """Decide at period-ends using lookback returns; execute on the next available bar."""
    ret_lb = prices.pct_change(lookback)

    # Period-end labels (may be month-end dates not in the daily index)
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

        # Map label t to the last actual bar <= t (e.g., 2025-09-30 -> 2025-09-10 if mid-month)
        t_eff = ret_lb.index.asof(t)
        if pd.isna(t_eff):
            continue
        # Execute on the first bar strictly AFTER t_eff (no look-ahead)
        pos = idx.searchsorted(t_eff, side="right")
        exec_dt = idx[pos] if pos < len(idx) else None

        decisions.append(MomentumDecision(t, exec_dt, row.to_dict(), winners, weights))
    return decisions

def decisions_to_weight_schedule(decisions: List[MomentumDecision], index: pd.DatetimeIndex, columns: List[str]) -> pd.DataFrame:
    """Target weights take effect FROM exec_dt forward (held until next exec)."""
    w = pd.DataFrame(np.nan, index=index, columns=columns)
    for d in decisions:
        if d.exec_dt is None or d.exec_dt not in w.index:
            continue
        for c in columns:
            w.loc[d.exec_dt, c] = d.planned_weights.get(c, 0.0)  # zeros are meaningful
    w = w.ffill().fillna(0.0)
    return w

def format_rank_string(ranks: Dict[str, float], winners: List[str]) -> str:
    s = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return " | ".join([f"{a}:{v:+.2%}{'*' if a in winners else ''}" for a, v in s])

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
    prices = download_prices(assets, start, end, interval).reindex(columns=assets, fill_value=np.nan).ffill()
    prices = prices.dropna(how="all", axis=1)  # if any ticker still empty, drop
    if prices.empty or len(prices.index) == 0:
        raise RuntimeError(
            f"No price data downloaded for {assets} in range {start}–{end}. "
            "Check tickers & internet, or try a later START date."
        )
    decisions = compute_rebalance_decisions(prices, lookback, top_k, rebalance)
    exec_dates = {d.exec_dt for d in decisions if d.exec_dt is not None}
    # Human-friendly decisions table
    dec_rows = []
    for d in decisions:
        dec_rows.append({
            "rebalance_dt": d.rebalance_dt,
            "exec_dt": d.exec_dt,
            "winners": ", ".join(d.winners),
            "ranks_str": format_rank_string(d.ranks, d.winners)
        })
    decisions_df = pd.DataFrame(dec_rows).set_index("rebalance_dt").sort_index()

    weight_sched = decisions_to_weight_schedule(decisions, prices.index, prices.columns)

    # --- Simulate with cash, positions (shares), fees ---
    cost_pct = cost_bps / 10_000.0
    cash = init_capital
    pos_shares = pd.Series(0.0, index=prices.columns)
    ledger_rows, trades_rows = [], []
    prev_total = init_capital

    for t in prices.index:
        px = prices.loc[t]

        target_w = weight_sched.loc[t]
        pos_val_before = float((pos_shares * px).sum())
    total_before = cash + pos_val_before
    curr_w = (pos_shares * px) / (total_before if total_before != 0 else 1.0)

    # ---- trade only on scheduled weekly exec dates ----
    if t in exec_dates:
        fee_rate = cost_bps / 10_000.0

        # Targets based on portfolio BEFORE trading (standard for rebalancing)
        target_dollar = target_w * total_before
        curr_dollar = pos_shares * px
        delta_dollar = target_dollar - curr_dollar

        # ------- 1) SELLS FIRST (long-only; no shorts) -------
        for c in prices.columns:
            d_notional = float(delta_dollar[c])
            if d_notional >= 0:
                continue  # not a sell
            trade_px = float(px[c])
            if not np.isfinite(trade_px) or trade_px <= 0:
                continue

            # can't sell more than we hold
            max_sell_shares = min(pos_shares[c], abs(d_notional) / trade_px)
            if max_sell_shares <= 0:
                continue

            sell_notional = max_sell_shares * trade_px
            fee = sell_notional * fee_rate

            # execute
            pos_shares[c] -= max_sell_shares
            cash += sell_notional - fee

            trades_rows.append({
                "date": t, "asset": c, "side": "SELL",
                "shares": -max_sell_shares, "price": trade_px,
                "trade_notional": -sell_notional, "fee_cost": fee,
                "cash_after": cash,
                "positions_value_after": float((pos_shares * px).sum()),
                "portfolio_value_after": cash + float((pos_shares * px).sum()),
            })

        # ------- 2) BUYS NEXT — SCALE TO AVAILABLE CASH (incl. fees) -------
        # Recompute current dollars after sells
        curr_dollar = pos_shares * px
        # Desired BUY notional from original plan
        buy_wants = {c: float(delta_dollar[c]) for c in prices.columns if float(delta_dollar[c]) > 0}
        buy_total = sum(buy_wants.values())

        if buy_total > 0 and cash > 0:
            # Scale so: sum(buys) * (1 + fee_rate) <= cash
            scale = min(1.0, cash / (buy_total * (1.0 + fee_rate)))

            for c, want_notional in buy_wants.items():
                trade_px = float(px[c])
                if not np.isfinite(trade_px) or trade_px <= 0:
                    continue

                buy_notional = want_notional * scale
                if buy_notional <= 0:
                    continue

                fee = buy_notional * fee_rate
                # guard for any numerical edge
                if buy_notional + fee > cash:
                    # final clamp
                    buy_notional = max(0.0, cash / (1.0 + fee_rate))
                    fee = buy_notional * fee_rate

                buy_shares = buy_notional / trade_px

                # execute
                cash -= (buy_notional + fee)
                pos_shares[c] += buy_shares

                trades_rows.append({
                    "date": t, "asset": c, "side": "BUY",
                    "shares": buy_shares, "price": trade_px,
                    "trade_notional": buy_notional, "fee_cost": fee,
                    "cash_after": cash,
                    "positions_value_after": float((pos_shares * px).sum()),
                    "portfolio_value_after": cash + float((pos_shares * px).sum()),
                })

        # Tiny float guard
        if -1e-9 < cash < 0:
            cash = 0.0

        # End-of-day valuation
        pos_val = float((pos_shares * px).sum())
        total_after = cash + pos_val
        day_ret = (total_after / prev_total - 1.0) if prev_total != 0 else 0.0
        prev_total = total_after

        weights_now = (pos_shares * px) / (total_after if total_after != 0 else 1.0)
        row = {
            "date": t,
            "cash": cash,
            "positions_value": pos_val,
            "total_value": total_after,
            "daily_return": day_ret,
        }
        for c in prices.columns:
            row[f"pos_shares_{c}"] = pos_shares[c]
            row[f"weight_{c}"] = weights_now[c]
        ledger_rows.append(row)

    ledger_df = pd.DataFrame(ledger_rows).set_index("date")
    trades_df = pd.DataFrame(trades_rows).sort_values(["date", "asset"]).reset_index(drop=True) if trades_rows else pd.DataFrame(
        columns=["date","asset","side","shares","price","trade_notional","fee_cost","cash_after","positions_value_after","portfolio_value_after"]
    )
    if len(ledger_rows) == 0:
    # create an empty ledger indexed by prices (so set_index won't fail)
        ledger_df = pd.DataFrame(
            {
                "cash": np.nan,
                "positions_value": np.nan,
                "total_value": np.nan,
                "daily_return": np.nan,
            },
            index=prices.index
        )
        ledger_df.index.name = "date"
    else:
        ledger_df = pd.DataFrame(ledger_rows).set_index("date").sort_index()

    # Tiny performance snapshot
    if len(ledger_df) >= 2:
        equity = ledger_df["total_value"]
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252.0 / max(1, len(equity))) - 1.0
        dd = (equity / equity.cummax() - 1.0).min()
    else:
        cagr, dd = np.nan, np.nan
    summary = pd.DataFrame({"metric": ["Final Equity", "CAGR (approx, daily)", "Max Drawdown"],
                            "value": [ledger_df["total_value"].iloc[-1], cagr, dd]})

    if save_csv:
        _ensure_dir("outputs")
        decisions_df.to_csv("outputs/pl_momentum_decisions.csv", encoding="utf-8")
        weight_sched.to_csv("outputs/pl_weight_schedule.csv", encoding="utf-8")
        ledger_df.to_csv("outputs/pl_portfolio_ledger.csv", encoding="utf-8")
        trades_df.to_csv("outputs/pl_trade_blotter.csv", encoding="utf-8", index=False)
        summary.to_csv("outputs/pl_summary.csv", encoding="utf-8", index=False)
        print("Saved CSVs to ./outputs/: pl_momentum_decisions.csv, pl_weight_schedule.csv, pl_portfolio_ledger.csv, pl_trade_blotter.csv, pl_summary.csv")

    return {"decisions": decisions_df, "weight_schedule": weight_sched, "ledger": ledger_df, "trades": trades_df, "summary": summary}

# ------------------------ run if script ------------------------
if __name__ == "__main__":
    out = inspect_cross_asset_momentum()
    print("\n=== Summary ===")
    print(out["summary"].to_string(index=False))
    print("\n=== Last 5 trades ===")
    print(out["trades"].tail(5).to_string(index=False))
    print("\n=== Last 5 ledger rows ===")
    print(out["ledger"].tail(5)[["cash","positions_value","total_value","daily_return"]].to_string())
    print("\n=== Last 3 decisions ===")
    print(out["decisions"].tail(3).to_string())
