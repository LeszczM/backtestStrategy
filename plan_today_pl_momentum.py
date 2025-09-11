# plan_today_pl_momentum.py
# Requirements: pip install pandas numpy yfinance
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# =========================== CONFIG ===========================
WIG20_TICKERS = [
    "SPL.WA","PKO.WA","MBK.WA","OPL.WA","PEO.WA","KGH.WA","PKN.WA","PGE.WA","PZU.WA",
    "ALR.WA","BDX.WA","CCC.WA","CDR.WA","KRU.WA","KTY.WA","LPP.WA","DNP.WA","ALE.WA",
    "PCO.WA","JSW.WA"
]
EXTRA_ETFS = ["ETFBSPXPL.WA"]  # Beta ETF S&P 500 PLN-hedged
ASSETS = WIG20_TICKERS + EXTRA_ETFS

START = "2018-01-01"
END = None
INTERVAL = "1d"

LOOKBACK = 126           # ~6 months
TOP_K = 5                # pick top-5
REBALANCE = "W-FRI"      # weekly decision at Friday, execute next bar
FEE_BPS = 10             # commission + slippage per trade notional
INIT_CAPITAL = 100_000.0
CASH_BUFFER_PCT = 0.00   # e.g., 0.001 keeps ~0.1% in cash
ROUND_SHARES = True      # round down to integer shares (common on WSE)
ABS_MOMENTUM = False     # if True, only allocate to assets with lookback return > 0
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
    # drop symbols with no data
    empty_cols = [c for c in px.columns if px[c].dropna().empty]
    if empty_cols:
        print("[WARN] Dropping symbols with no data:", empty_cols)
        px = px.drop(columns=empty_cols)
    return px

@dataclass
class Decision:
    rebalance_dt: pd.Timestamp
    exec_dt: Optional[pd.Timestamp]
    ranks: Dict[str, float]
    winners: List[str]
    weights: Dict[str, float]

def latest_weekly_decision(prices: pd.DataFrame, lookback: int, top_k: int, rebalance: str,
                           abs_momentum: bool) -> Optional[Decision]:
    ret_lb = prices.pct_change(lookback)
    marks = ret_lb.resample(rebalance).last().dropna(how="all")
    if marks.empty:
        return None

    # pick the **latest** decision row
    t = marks.index[-1]
    row = marks.iloc[-1].dropna()

    if abs_momentum:
        row = row[row > 0]

    if row.empty:
        # nothing passes filter; all cash
        winners, weights = [], {c: 0.0 for c in prices.columns}
    else:
        k = min(top_k, len(row))
        winners = list(row.nlargest(k).index)
        w = 1.0 / max(1, len(winners))
        weights = {c: (w if c in winners else 0.0) for c in prices.columns}

    # map label to last real bar <= t, then exec is first bar AFTER that
    t_eff = ret_lb.index.asof(t)
    idx = prices.index
    pos = idx.searchsorted(t_eff, side="right")
    exec_dt = idx[pos] if (t_eff is not pd.NaT and pos < len(idx)) else None

    return Decision(t, exec_dt, row.to_dict(), winners, weights)

def plan_buys_today(prices: pd.DataFrame, decision: Decision, capital: float,
                    fee_bps: int, cash_buffer_pct: float,
                    round_shares: bool) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Returns a plan DataFrame (asset, price, weight, shares, notional, fee) and totals.
    Buys are scaled so cash never goes negative. No sells since we're starting from cash.
    """
    today = prices.index[-1]
    px = prices.loc[today]

    # capital available after buffer
    fee_rate = fee_bps / 10_000.0
    cash = float(capital * (1.0 - cash_buffer_pct))

    # desired notional per asset
    desired = pd.Series(decision.weights, index=prices.columns) * cash

    # scale to ensure sum(buys)*(1+fee) <= cash
    gross_desired = desired.sum()
    scale = min(1.0, cash / (gross_desired * (1.0 + fee_rate))) if gross_desired > 0 else 0.0
    buy_notional = desired * scale

    # compute shares (fractional), then optionally round down
    shares = buy_notional / px.replace(0, np.nan)

    if round_shares:
        shares = np.floor(shares.fillna(0.0))
        buy_notional = shares * px
        # fees based on rounded notional; if fees push over cash, shrink proportionally
        total_buy = float(buy_notional.sum())
        total_fee = total_buy * fee_rate
        if total_buy + total_fee > cash and total_buy > 0:
            adj = cash / (total_buy * (1.0 + fee_rate))
            shares = np.floor((shares * adj).fillna(0.0))
            buy_notional = shares * px

    fees = buy_notional * fee_rate
    spend = float(buy_notional.sum() + fees.sum())
    leftover = cash - spend

    # Build plan table
    plan = pd.DataFrame({
        "asset": prices.columns,
        "price": px.values,
        "weight": [decision.weights.get(c, 0.0) for c in prices.columns],
        "shares": shares.values,
        "notional": buy_notional.values,
        "fee": fees.values,
    })
    plan = plan[plan["weight"] > 0].sort_values("weight", ascending=False).reset_index(drop=True)

    totals = {
        "today": today,
        "rebalance_dt": decision.rebalance_dt,
        "exec_dt": decision.exec_dt,
        "capital": capital,
        "cash_after_buffer": cash,
        "spend_ex_fees": float(buy_notional.sum()),
        "fees_total": float(fees.sum()),
        "leftover_cash": float(leftover),
        "positions_value_now": float(buy_notional.sum()),
        "estimated_portfolio_value_now": float(buy_notional.sum() + leftover),
    }
    return plan, totals

def ranks_string(ranks: Dict[str, float], winners: List[str]) -> str:
    items = sorted(ranks.items(), key=lambda kv: kv[1], reverse=True)
    return " | ".join([f"{a}:{v:+.2%}{'*' if a in winners else ''}" for a,v in items])

def main():
    prices = download_prices(ASSETS, START, END, INTERVAL)
    prices = prices.reindex(columns=ASSETS, fill_value=np.nan).ffill()
    prices = prices.dropna(how="all", axis=1)

    if prices.empty:
        raise RuntimeError("No price data found. Check tickers or date range.")

    decision = latest_weekly_decision(prices, LOOKBACK, TOP_K, REBALANCE, ABS_MOMENTUM)
    if decision is None:
        raise RuntimeError("Could not compute a weekly decision (not enough data).")

    print("=== Cross-Asset Momentum: Start-Today Plan ===")
    print(f"Latest decision (weekly): label={decision.rebalance_dt.date()}  intended_exec={getattr(decision.exec_dt, 'date', lambda: None)()}")
    print(f"Data last bar (today):    {prices.index[-1].date()}")
    print(f"Winners (top {TOP_K}):    {', '.join(decision.winners) if decision.winners else '(none)'}")
    print(f"Ranks: {ranks_string(decision.ranks, decision.winners)}")
    if ABS_MOMENTUM:
        print("[Info] Absolute momentum ON: only assets with positive lookback return considered.")

    plan, totals = plan_buys_today(
        prices, decision, INIT_CAPITAL, FEE_BPS, CASH_BUFFER_PCT, ROUND_SHARES
    )

    print("\n--- Proposed Allocation (buy today at last price) ---")
    if plan.empty:
        print("No assets selected (all-cash per rules).")
    else:
        with pd.option_context("display.float_format", "{:,.2f}".format):
            print(plan.to_string(index=False))

    print("\n--- Totals ---")
    for k, v in totals.items():
        if isinstance(v, float):
            print(f"{k}: {v:,.2f}")
        else:
            print(f"{k}: {v}")

    if SAVE_CSV:
        _ensure_dir(OUT_DIR)
        plan.to_csv(f"{OUT_DIR}/today_plan.csv", index=False, encoding="utf-8")
        pd.DataFrame([totals]).to_csv(f"{OUT_DIR}/today_plan_totals.csv", index=False, encoding="utf-8")
        # also save the full price snapshot & ranks for transparency
        prices.to_csv(f"{OUT_DIR}/today_prices.csv", encoding="utf-8")
        pd.DataFrame([decision.ranks]).T.rename(columns={0: "lookback_return"}).to_csv(
            f"{OUT_DIR}/today_ranks.csv", encoding="utf-8"
        )
        print(f"\nSaved CSVs to ./{OUT_DIR}/: today_plan.csv, today_plan_totals.csv, today_prices.csv, today_ranks.csv")

if __name__ == "__main__":
    main()
