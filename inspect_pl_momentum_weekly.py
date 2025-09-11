# inspect_pl_momentum_weekly.py
# Requirements: pip install pandas numpy yfinance
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Optional

# =========================== CONFIG ===========================
# WIG20 (adjust as needed; script drops any symbols with no data)
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
    "JSW.WA",  # JSW (if any symbol lacks data, it will be dropped)
]
EXTRA_ETFS = ["ETFBSPXPL.WA"]  # Beta ETF S&P 500 PLN-Hedged

ASSETS = WIG20_TICKERS + EXTRA_ETFS

START = "2018-01-01"
END = None
INTERVAL = "1d"

LOOKBACK = 126               # ~6 months
TOP_K = 5                    # hold top 5 each rebalance
REBALANCE = "W-FRI"          # weekly, period ends Friday; execute next bar
INIT_CAPITAL = 100_000.0
COST_BPS = 10                # commission+slippage per trade notional
ALLOW_SHORT = False          # long-only
CASH_BUFFER_PCT = 0.0        # e.g., 0.001 keeps 0.1% equity in cash
SAVE_CSV = True
OUT_DIR = "outputs"
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
    # drop columns with no actual data
    empty_cols = [c for c in px.columns if px[c].dropna().empty]
    if empty_cols:
        print("[WARN] Dropping symbols with no data:", empty_cols)
        px = px.drop(columns=empty_cols)
    return px

@dataclass
class MomentumDecision:
    rebalance_dt: pd.Timestamp
    exec_dt: Optional[pd.Timestamp]
    ranks: Dict[str, float]
    winners: List[str]
    planned_weights: Dict[str, float]

def compute_rebalance_decisions(prices: pd.DataFrame, lookback: int, top_k: int, rebalance: str) -> List[MomentumDecision]:
    """Decide at weekly period-ends using lookback returns; execute on the next bar (no look-ahead)."""
    ret_lb = prices.pct_change(lookback)
    marks = ret_lb.resample(rebalance).last().dropna(how="all")  # week-end labels (Fri)

    decisions: List[MomentumDecision] = []
    idx = prices.index

    for t, row in marks.iterrows():
        row = row.dropna()
        if row.empty:
            continue

        k = min(top_k, len(row))
        winners = list(row.nlargest(k).index)
        weights = {c: (1.0 / k if c in winners else 0.0) for c in prices.columns}

        # last actual bar <= period label
        t_eff = ret_lb.index.asof(t)
        if pd.isna(t_eff):
            continue

        # execute first bar strictly AFTER t_eff
        pos = idx.searchsorted(t_eff, side="right")
        exec_dt = idx[pos] if pos < len(idx) else None

        decisions.append(MomentumDecision(t, exec_dt, row.to_dict(), winners, weights))
    return decisions

def decisions_to_weight_schedule(decisions: List[MomentumDecision], index: pd.DatetimeIndex, columns: List[str]) -> pd.DataFrame:
    """Weights take effect FROM exec_dt forward (held until next exec)."""
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
    cash_buffer_pct: float = CASH_BUFFER_PCT,
    save_csv: bool = SAVE_CSV,
    out_dir: str = OUT_DIR
) -> Dict[str, pd.DataFrame]:

    prices = download_prices(assets, start, end, interval)
    prices = prices.reindex(columns=assets, fill_value=np.nan).ffill()
    prices = prices.dropna(how="all", axis=1)

    if prices.empty or len(prices.index) == 0:
        raise RuntimeError(
            f"No price data for {assets} in {start}–{end}. Check tickers/range/network."
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
            "ranks_str": format_rank_string(d.ranks, d.winners),
        })
    decisions_df = pd.DataFrame(dec_rows).set_index("rebalance_dt").sort_index()

    weight_sched = decisions_to_weight_schedule(decisions, prices.index, prices.columns)

    # --- Simulation state ---
    fee_rate = cost_bps / 10_000.0
    cash = init_capital
    pos_shares = pd.Series(0.0, index=prices.columns)
    ledger_rows, trades_rows = [], []
    prev_total = init_capital

    for t in prices.index:
        px = prices.loc[t]

        # portfolio valuation BEFORE any trading
        pos_val_before = float((pos_shares * px).sum())
        total_before = cash + pos_val_before

        # target weights in force from this date forward
        target_w = weight_sched.loc[t]

        # --- trade ONLY on scheduled exec dates ---
        if t in exec_dates:
            # keep optional cash buffer
            target_total_for_risk = total_before * (1.0 - cash_buffer_pct)
            target_dollar = target_w * target_total_for_risk

            curr_dollar = pos_shares * px
            delta_dollar = target_dollar - curr_dollar

            # 1) SELLS FIRST (long-only guard: don't sell below zero shares)
            for c in prices.columns:
                d_notional = float(delta_dollar[c])
                if d_notional >= 0:
                    continue  # not a sell
                trade_px = float(px[c])
                if not np.isfinite(trade_px) or trade_px <= 0:
                    continue

                max_sell_shares = min(pos_shares[c], abs(d_notional) / trade_px)
                if max_sell_shares <= 0:
                    continue

                sell_notional = max_sell_shares * trade_px
                fee = sell_notional * fee_rate

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

            # 2) BUYS — SCALE TO AVAILABLE CASH (incl. fees, buffer respected)
            curr_dollar = pos_shares * px  # after sells
            buy_wants = {c: float(delta_dollar[c]) for c in prices.columns if float(delta_dollar[c]) > 0}

            buy_total = sum(buy_wants.values())
            if buy_total > 0 and cash > 0:
                # ensure we don't use buffer cash and we pay fees
                cash_avail = cash
                scale = min(1.0, cash_avail / (buy_total * (1.0 + fee_rate))) if buy_total > 0 else 0.0

                for c, want_notional in buy_wants.items():
                    trade_px = float(px[c])
                    if not np.isfinite(trade_px) or trade_px <= 0:
                        continue

                    buy_notional = want_notional * scale
                    if buy_notional <= 0:
                        continue

                    fee = buy_notional * fee_rate
                    # final clamp against fp drift
                    if buy_notional + fee > cash:
                        buy_notional = max(0.0, cash / (1.0 + fee_rate))
                        fee = buy_notional * fee_rate

                    buy_shares = buy_notional / trade_px
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

            # tiny guard against -0.0
            if -1e-9 < cash < 0:
                cash = 0.0

        # --- End-of-day valuation (no trades on non-exec days) ---
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

    # ---- Build outputs (robust to empty trade list) ----
    if len(ledger_rows) == 0:
        ledger_df = pd.DataFrame(
            {"cash": np.nan, "positions_value": np.nan, "total_value": np.nan, "daily_return": np.nan},
            index=prices.index
        )
        ledger_df.index.name = "date"
    else:
        ledger_df = pd.DataFrame(ledger_rows).set_index("date").sort_index()

    trades_df = (
        pd.DataFrame(trades_rows).sort_values(["date", "asset"]).reset_index(drop=True)
        if trades_rows else pd.DataFrame(
            columns=["date","asset","side","shares","price","trade_notional","fee_cost","cash_after","positions_value_after","portfolio_value_after"]
        )
    )

    # Summary
    if len(ledger_df) >= 2 and ledger_df["total_value"].notna().any():
        equity = ledger_df["total_value"].astype(float)
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252.0 / max(1, len(equity))) - 1.0
        dd = (equity / equity.cummax() - 1.0).min()
    else:
        cagr, dd = np.nan, np.nan
    summary = pd.DataFrame(
        {"metric": ["Final Equity", "CAGR (approx, daily)", "Max Drawdown"],
         "value": [ledger_df["total_value"].iloc[-1], cagr, dd]}
    )

    if save_csv:
        _ensure_dir(out_dir)
        decisions_df.to_csv(f"{out_dir}/pl_momentum_decisions.csv", encoding="utf-8")
        weight_sched.to_csv(f"{out_dir}/pl_weight_schedule.csv", encoding="utf-8")
        ledger_df.to_csv(f"{out_dir}/pl_portfolio_ledger.csv", encoding="utf-8")
        trades_df.to_csv(f"{out_dir}/pl_trade_blotter.csv", encoding="utf-8", index=False)
        summary.to_csv(f"{out_dir}/pl_summary.csv", encoding="utf-8", index=False)
        print(f"Saved CSVs to ./{out_dir}/")

    return {"decisions": decisions_df, "weight_schedule": weight_sched, "ledger": ledger_df, "trades": trades_df, "summary": summary}

# ------------------------ Run ------------------------
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
