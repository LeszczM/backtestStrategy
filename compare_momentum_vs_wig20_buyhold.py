# compare_momentum_vs_wig20_buyhold.py
# Requirements: pip install pandas numpy yfinance matplotlib
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional

# =========================== CONFIG ===========================
# Universe: WIG20 stocks; momentum can include ETF if you want (toggle below)
WIG20_TICKERS = [
    "SPL.WA","PKO.WA","MBK.WA","OPL.WA","PEO.WA","KGH.WA","PKN.WA","PGE.WA","PZU.WA",
    "ALR.WA","BDX.WA","CCC.WA","CDR.WA","KRU.WA","KTY.WA","LPP.WA","DNP.WA","ALE.WA",
    "PCO.WA","JSW.WA"
]
PLN_HEDGED_SP500 = "ETFBSPXPL.WA"

# --- Data & backtest window ---
START = "2018-01-01"
END = None
INTERVAL = "1d"

# --- Momentum params ---
INCLUDE_ETF_IN_MOMENTUM = False        # Momentum universe = WIG20 + ETF if True, else WIG20 only
LOOKBACK = 126                        # ~6 months
TOP_K = 5
REBALANCE = "M"                       # "M" for monthly, "W-FRI" for weekly (Friday)
TURNOVER_EPS = 0.02                   # skip rebalance if sum(|Δw|) ≤ 2%

# --- Trading / costs / tax ---
INIT_CAPITAL = 100_000.0
COST_BPS = 10                         # commission + slippage per trade notional
CASH_BUFFER_PCT = 0.0                 # keep some cash aside (e.g., 0.001 = 0.1%)
BELKA_TAX = 0.19                      # 19%
TAX_MODE = "PER_TRADE"                # "PER_TRADE" or "ANNUAL"
ALLOW_LOSS_NETTING_ANNUAL = True      # only used in ANNUAL mode

# --- Outputs ---
SAVE_CSV = True
OUT_DIR = "outputs"
SAVE_PLOTS = True
SHOW_PLOTS = False                    # set True if you want windows to pop up
# =============================================================

def _ensure_dir(path: str):
    import os; os.makedirs(path, exist_ok=True)

def download_prices(assets: List[str], start: str, end: Optional[str], interval: str) -> pd.DataFrame:
    df = yf.download(assets, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        px = df["Close"].copy()
    else:
        px = df[["Close"]].copy()
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.sort_index().ffill().dropna(how="all")
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
    """Decide at period-ends using lookback total return; execute on next available bar."""
    ret_lb = prices.pct_change(lookback)
    marks = ret_lb.resample(rebalance).last().dropna(how="all")  # period-end labels

    decisions: List[MomentumDecision] = []
    idx = prices.index

    for t, row in marks.iterrows():
        row = row.dropna()
        if row.empty:
            continue
        k = min(top_k, len(row))
        winners = list(row.nlargest(k).index)
        weights = {c: (1.0 / k if c in winners else 0.0) for c in prices.columns}

        t_eff = ret_lb.index.asof(t)  # last real bar ≤ label
        if pd.isna(t_eff):
            continue
        pos = idx.searchsorted(t_eff, side="right")
        exec_dt = idx[pos] if pos < len(idx) else None

        decisions.append(MomentumDecision(t, exec_dt, row.to_dict(), winners, weights))
    return decisions

def decisions_to_weight_schedule(decisions: List[MomentumDecision], index: pd.DatetimeIndex, columns: List[str]) -> pd.DataFrame:
    w = pd.DataFrame(np.nan, index=index, columns=columns)
    for d in decisions:
        if d.exec_dt is None or d.exec_dt not in w.index:
            continue
        for c in columns:
            w.loc[d.exec_dt, c] = d.planned_weights.get(c, 0.0)
    return w.ffill().fillna(0.0)

# ---- FIFO lot helper for tax basis ----
class LotsFIFO:
    def __init__(self, columns: List[str]):
        self.lots: Dict[str, List[List[float]]] = {c: [] for c in columns}

    def buy(self, asset: str, shares: float, cost_per_share_incl_fee: float):
        if shares <= 0:
            return
        self.lots[asset].append([shares, cost_per_share_incl_fee])

    def sell(self, asset: str, shares_to_sell: float, sell_px: float, sell_fee_total: float) -> float:
        if shares_to_sell <= 0:
            return 0.0
        lots = self.lots[asset]
        remain = shares_to_sell
        realized = 0.0
        fee_per_share = (sell_fee_total / shares_to_sell) if shares_to_sell > 0 else 0.0

        new_lots = []
        for sh, cost_ps in lots:
            if remain <= 0:
                new_lots.append([sh, cost_ps]); continue
            use = min(sh, remain)
            proceeds_net = use * (sell_px - fee_per_share)
            cost = use * cost_ps
            realized += (proceeds_net - cost)
            sh_left = sh - use
            if sh_left > 1e-12:
                new_lots.append([sh_left, cost_ps])
            remain -= use
        self.lots[asset] = new_lots
        return realized

def run_momentum(prices: pd.DataFrame,
                 lookback: int, top_k: int, rebalance: str,
                 init_capital: float, cost_bps: int,
                 cash_buffer_pct: float,
                 belka_tax: float, tax_mode: str,
                 allow_loss_netting_annual: bool,
                 turnover_eps: float) -> pd.DataFrame:
    """Return a ledger with daily equity, taxes, etc. Strategy trades only at scheduled exec dates and respects turnover guard."""
    decisions = compute_rebalance_decisions(prices, lookback, top_k, rebalance)
    exec_dates = {d.exec_dt for d in decisions if d.exec_dt is not None}
    weight_sched = decisions_to_weight_schedule(decisions, prices.index, prices.columns)

    fee_rate = cost_bps / 10_000.0
    cash = init_capital
    lots = LotsFIFO(list(prices.columns))
    tax_liability = 0.0
    taxes_paid_cum = 0.0
    realized_pl_year: Dict[int, float] = {}

    pos_shares = pd.Series(0.0, index=prices.columns, dtype=float)
    ledger_rows = []
    prev_total = init_capital

    for i, t in enumerate(prices.index):
        px = prices.loc[t]
        year = t.year

        # Portfolio before trades
        pos_val_before = float((pos_shares * px).sum())
        total_before = cash + pos_val_before

        # Targets & current weights
        target_w = weight_sched.loc[t]
        curr_w = (pos_shares * px) / (total_before if total_before != 0 else 1.0)
        curr_w = curr_w.fillna(0.0)

        # Turnover guard
        delta_w = (target_w - curr_w).abs().sum()
        should_trade = (t in exec_dates) and (delta_w > turnover_eps + 1e-12)

        if should_trade:
            target_total_for_risk = total_before * (1.0 - cash_buffer_pct)
            target_dollar = target_w * target_total_for_risk
            curr_dollar = pos_shares * px
            delta_dollar = target_dollar - curr_dollar

            # 1) Sells first
            for c in prices.columns:
                d_notional = float(delta_dollar[c])
                if d_notional >= 0:  # not a sell
                    continue
                trade_px = float(px[c])
                if not np.isfinite(trade_px) or trade_px <= 0:
                    continue

                sell_shares = min(pos_shares[c], abs(d_notional) / trade_px)
                if sell_shares <= 0:
                    continue

                sell_notional = sell_shares * trade_px
                sell_fee = sell_notional * fee_rate

                realized = lots.sell(c, sell_shares, trade_px, sell_fee)
                pos_shares[c] -= sell_shares
                cash += (sell_notional - sell_fee)

                # tax on realized gains
                if tax_mode.upper() == "PER_TRADE":
                    gain = max(0.0, realized)
                    tax = gain * belka_tax
                    pay = min(cash, tax)
                    cash -= pay
                    tax_liability += (tax - pay)
                    taxes_paid_cum += pay
                else:  # ANNUAL
                    realized_pl_year[year] = realized_pl_year.get(year, 0.0) + realized

                # pay down any outstanding liability
                if tax_liability > 1e-12 and cash > 0:
                    pay = min(cash, tax_liability)
                    cash -= pay
                    tax_liability -= pay
                    taxes_paid_cum += pay

            # 2) Buys scaled to available cash (incl. fees)
            curr_dollar = pos_shares * px
            delta_dollar = target_dollar - curr_dollar
            buy_wants = {c: float(delta_dollar[c]) for c in prices.columns if float(delta_dollar[c]) > 0}
            buy_total = sum(buy_wants.values())
            free_cash = max(0.0, cash)

            if buy_total > 0 and free_cash > 0:
                scale = min(1.0, free_cash / (buy_total * (1.0 + fee_rate)))
                for c, want_notional in buy_wants.items():
                    trade_px = float(px[c])
                    if not np.isfinite(trade_px) or trade_px <= 0:
                        continue
                    buy_notional = want_notional * scale
                    if buy_notional <= 0:
                        continue
                    buy_fee = buy_notional * fee_rate
                    if buy_notional + buy_fee > cash:
                        buy_notional = max(0.0, cash / (1.0 + fee_rate))
                        buy_fee = buy_notional * fee_rate
                    buy_shares = buy_notional / trade_px if trade_px > 0 else 0.0

                    cash -= (buy_notional + buy_fee)
                    pos_shares[c] += buy_shares
                    if buy_shares > 0:
                        lot_cost_ps = (buy_notional + buy_fee) / buy_shares
                        lots.buy(c, buy_shares, lot_cost_ps)

            if -1e-9 < cash < 0:
                cash = 0.0

        # Year-end accrual for ANNUAL tax mode
        is_last_day = (i == len(prices.index) - 1)
        next_is_new_year = (not is_last_day) and (prices.index[i+1].year != year)
        if tax_mode.upper() == "ANNUAL" and (next_is_new_year or is_last_day):
            base = realized_pl_year.get(year, 0.0)
            taxable = base if ALLOW_LOSS_NETTING_ANNUAL else max(0.0, base)
            tax_due = max(0.0, taxable) * belka_tax
            if tax_due > 1e-12:
                pay = min(cash, tax_due)
                cash -= pay
                tax_liability += (tax_due - pay)
                taxes_paid_cum += pay
            realized_pl_year[year] = 0.0

        # End-of-day valuation
        pos_val = float((pos_shares * px).sum())
        total_after = cash + pos_val
        day_ret = (total_after / prev_total - 1.0) if prev_total != 0 else 0.0
        prev_total = total_after

        ledger_rows.append({
            "date": t, "cash": cash, "positions_value": pos_val, "total_value": total_after,
            "daily_return": day_ret, "tax_liability": tax_liability, "taxes_paid_cum": taxes_paid_cum
        })

    ledger = pd.DataFrame(ledger_rows).set_index("date").sort_index()
    return ledger

def run_wig20_buyhold(prices: pd.DataFrame,
                      init_capital: float, cost_bps: int) -> pd.DataFrame:
    """
    Equal-weight buy-and-hold WIG20:
    - Buy once on the first available date (weights = 1/N across available names).
    - No rebalancing, no sells until the end (so no realized gains → no Belka tax during backtest).
    - Fees only on the initial buys.
    """
    fee_rate = cost_bps / 10_000.0
    cols = list(prices.columns)
    first_day = prices.index[0]
    px0 = prices.loc[first_day]

    # Equal weights across available assets that have a valid price on first day
    valid = px0.dropna().index.tolist()
    if len(valid) == 0:
        raise RuntimeError("No valid WIG20 prices on the first day.")
    w = pd.Series(0.0, index=cols)
    w.loc[valid] = 1.0 / len(valid)

    cash = init_capital
    pos_shares = pd.Series(0.0, index=cols, dtype=float)

    # Initial buys (scale to ensure cash >= 0 including fees)
    target_dollar = w * cash
    buy_total = target_dollar.sum()
    scale = 1.0 / (1.0 + fee_rate) if buy_total > 0 else 0.0
    for c in valid:
        price = float(px0[c])
        if not np.isfinite(price) or price <= 0:
            continue
        buy_notional = target_dollar[c] * scale
        fee = buy_notional * fee_rate
        shares = buy_notional / price
        cash -= (buy_notional + fee)
        pos_shares[c] += shares

    # Daily valuation
    ledger_rows = []
    prev_total = init_capital
    for t in prices.index:
        px = prices.loc[t]
        pos_val = float((pos_shares * px).sum())
        total = cash + pos_val
        day_ret = (total / prev_total - 1.0) if prev_total != 0 else 0.0
        prev_total = total
        ledger_rows.append({
            "date": t, "cash": cash, "positions_value": pos_val, "total_value": total,
            "daily_return": day_ret, "tax_liability": 0.0, "taxes_paid_cum": 0.0
        })
    ledger = pd.DataFrame(ledger_rows).set_index("date").sort_index()
    return ledger

def summarize(ledger: pd.DataFrame, label: str) -> Dict[str, float]:
    eq = ledger["total_value"].astype(float)
    if len(eq) < 2:
        return {"Strategy": label, "Final": float(eq.iloc[-1]), "CAGR": np.nan,
                "MaxDD": np.nan, "TaxesPaid": float(ledger["taxes_paid_cum"].iloc[-1])}
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252.0 / len(eq)) - 1.0
    dd = (eq / eq.cummax() - 1.0).min()
    return {"Strategy": label, "Final": float(eq.iloc[-1]), "CAGR": cagr,
            "MaxDD": float(dd), "TaxesPaid": float(ledger["taxes_paid_cum"].iloc[-1])}

def plot_equity(mom_ledger: pd.DataFrame, bh_ledger: pd.DataFrame, out_dir: str):
    _ensure_dir(out_dir)
    plt.figure(figsize=(11,6))
    mom_ledger["total_value"].plot(label="Momentum")
    bh_ledger["total_value"].plot(label="Buy&Hold WIG20")
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (PLN)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/comparison_equity.png", dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # Optional: drawdown plot
    def drawdown(series: pd.Series) -> pd.Series:
        cum = series.astype(float)
        peak = cum.cummax()
        return cum / peak - 1.0

    plt.figure(figsize=(11,4))
    drawdown(mom_ledger["total_value"]).plot(label="Momentum")
    drawdown(bh_ledger["total_value"]).plot(label="Buy&Hold WIG20")
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/comparison_drawdown.png", dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def main():
    # Universe setup
    momentum_assets = WIG20_TICKERS + ([PLN_HEDGED_SP500] if INCLUDE_ETF_IN_MOMENTUM else [])
    wig20_assets = WIG20_TICKERS[:]  # baseline

    # Download
    px_mom = download_prices(momentum_assets, START, END, INTERVAL)
    px_bh = download_prices(wig20_assets, START, END, INTERVAL)

    # Align dates (intersection)
    common_idx = px_mom.index.intersection(px_bh.index)
    px_mom = px_mom.reindex(common_idx).ffill()
    px_bh = px_bh.reindex(common_idx).ffill()

    # Run strategies
    mom_ledger = run_momentum(px_mom, LOOKBACK, TOP_K, REBALANCE,
                              INIT_CAPITAL, COST_BPS, CASH_BUFFER_PCT,
                              BELKA_TAX, TAX_MODE, ALLOW_LOSS_NETTING_ANNUAL,
                              TURNOVER_EPS)

    bh_ledger = run_wig20_buyhold(px_bh, INIT_CAPITAL, COST_BPS)

    # Summaries
    s_mom = summarize(mom_ledger, f"Momentum({REBALANCE}, K={TOP_K}, lookback={LOOKBACK})")
    s_bh = summarize(bh_ledger, "Buy&Hold (WIG20 equal-weight)")

    comp = pd.DataFrame([s_mom, s_bh]).set_index("Strategy")
    print("\n=== Comparison Summary ===")
    with pd.option_context("display.float_format", "{:,.4f}".format):
        print(comp.to_string())

    # Save CSVs
    if SAVE_CSV:
        _ensure_dir(OUT_DIR)
        mom_ledger.to_csv(f"{OUT_DIR}/ledger_momentum.csv", encoding="utf-8")
        bh_ledger.to_csv(f"{OUT_DIR}/ledger_buyhold.csv", encoding="utf-8")
        comp.to_csv(f"{OUT_DIR}/comparison_summary.csv", encoding="utf-8")
        print(f"\nSaved CSVs to ./{OUT_DIR}/: comparison_summary.csv, ledger_momentum.csv, ledger_buyhold.csv")

    # Plots
    if SAVE_PLOTS:
        plot_equity(mom_ledger, bh_ledger, OUT_DIR)
        print(f"Saved plots to ./{OUT_DIR}/: comparison_equity.png, comparison_drawdown.png")

if __name__ == "__main__":
    main()
