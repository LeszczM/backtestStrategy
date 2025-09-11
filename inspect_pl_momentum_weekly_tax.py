# inspect_pl_momentum_weekly_tax.py
# Requirements: pip install pandas numpy yfinance
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Optional

# =========================== CONFIG ===========================
WIG20_TICKERS = [
    "SPL.WA","PKO.WA","MBK.WA","OPL.WA","PEO.WA","KGH.WA","PKN.WA","PGE.WA","PZU.WA",
    "ALR.WA","BDX.WA","CCC.WA","CDR.WA","KRU.WA","KTY.WA","LPP.WA","DNP.WA","ALE.WA",
    "PCO.WA","JSW.WA"
]
EXTRA_ETFS = ["ETFBSPXPL.WA"]
ASSETS = WIG20_TICKERS + EXTRA_ETFS

START = "2018-01-01"
END = None
INTERVAL = "1d"

LOOKBACK = 126
TOP_K = 5
REBALANCE = "M"
INIT_CAPITAL = 100_000.0
COST_BPS = 10
ALLOW_SHORT = False
CASH_BUFFER_PCT = 0.0
SAVE_CSV = True
OUT_DIR = "outputs"
TURNOVER_EPS = 0.02

# ---- Belka tax settings ----
BELKA_TAX = 0.19                  # 19%
TAX_MODE = "PER_TRADE"               # "ANNUAL" or "PER_TRADE"
ALLOW_LOSS_NETTING_ANNUAL = True  # net gains & losses within the calendar year
# =============================================================

def _ensure_dir(path: str):
    import os; os.makedirs(path, exist_ok=True)

def download_prices(assets: List[str], start: str, end: Optional[str], interval: str) -> pd.DataFrame:
    df = yf.download(assets, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        px = df["Close"].copy()
    else:
        px = df[["Close"]].copy()
    if isinstance(px, pd.Series): px = px.to_frame()
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
    ret_lb = prices.pct_change(lookback)
    marks = ret_lb.resample(rebalance).last().dropna(how="all")
    decisions: List[MomentumDecision] = []
    idx = prices.index
    for t, row in marks.iterrows():
        row = row.dropna()
        if row.empty: continue
        k = min(top_k, len(row))
        winners = list(row.nlargest(k).index)
        weights = {c: (1.0 / k if c in winners else 0.0) for c in prices.columns}
        t_eff = ret_lb.index.asof(t)
        if pd.isna(t_eff): continue
        pos = idx.searchsorted(t_eff, side="right")
        exec_dt = idx[pos] if pos < len(idx) else None
        decisions.append(MomentumDecision(t, exec_dt, row.to_dict(), winners, weights))
    return decisions

def decisions_to_weight_schedule(decisions: List[MomentumDecision], index: pd.DatetimeIndex, columns: List[str]) -> pd.DataFrame:
    w = pd.DataFrame(np.nan, index=index, columns=columns)
    for d in decisions:
        if d.exec_dt is None or d.exec_dt not in w.index: continue
        for c in columns:
            w.loc[d.exec_dt, c] = d.planned_weights.get(c, 0.0)
    return w.ffill().fillna(0.0)

def format_rank_string(ranks: Dict[str, float], winners: List[str]) -> str:
    s = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return " | ".join([f"{a}:{v:+.2%}{'*' if a in winners else ''}" for a, v in s])

# ---- FIFO lot helper ----
class LotsFIFO:
    def __init__(self, columns: List[str]):
        # per-asset list of (shares, cost_per_share_incl_buy_fee)
        self.lots: Dict[str, List[List[float]]] = {c: [] for c in columns}

    def buy(self, asset: str, shares: float, cost_per_share_incl_fee: float):
        if shares <= 0: return
        self.lots[asset].append([shares, cost_per_share_incl_fee])

    def sell(self, asset: str, shares_to_sell: float, sell_px: float, sell_fee_total: float) -> float:
        """
        FIFO realize P/L including fees:
        realized = (proceeds_net_of_fee) - (sum(cost_basis))
        Distribute sell_fee_total pro-rata across all shares sold in this trade.
        Returns realized P/L (can be negative).
        """
        if shares_to_sell <= 0: return 0.0
        lots = self.lots[asset]
        remain = shares_to_sell
        realized = 0.0
        # compute proceeds gross for the sold quantity
        proceeds_gross = shares_to_sell * sell_px
        # fee allocated per share sold
        fee_per_share = (sell_fee_total / shares_to_sell) if shares_to_sell > 0 else 0.0

        new_lots = []
        for sh, cost_ps in lots:
            if remain <= 0:
                new_lots.append([sh, cost_ps])
                continue
            use = min(sh, remain)
            proceeds_net = use * (sell_px - fee_per_share)
            cost = use * cost_ps
            realized += (proceeds_net - cost)
            sh_left = sh - use
            if sh_left > 1e-12:
                new_lots.append([sh_left, cost_ps])
            remain -= use
        self.lots[asset] = new_lots
        # if remain > 0 here, we tried to sell more than we had; ignore excess
        return realized

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
    out_dir: str = OUT_DIR,
    belka_tax: float = BELKA_TAX,
    tax_mode: str = TAX_MODE,
    allow_loss_netting_annual: bool = ALLOW_LOSS_NETTING_ANNUAL
) -> Dict[str, pd.DataFrame]:

    prices = download_prices(assets, start, end, interval)
    prices = prices.reindex(columns=assets, fill_value=np.nan).ffill().dropna(how="all", axis=1)
    if prices.empty: raise RuntimeError("No price data. Check tickers/range/network.")

    decisions = compute_rebalance_decisions(prices, lookback, top_k, rebalance)
    exec_dates = {d.exec_dt for d in decisions if d.exec_dt is not None}

    # decisions table
    dec_rows = []
    for d in decisions:
        dec_rows.append({"rebalance_dt": d.rebalance_dt, "exec_dt": d.exec_dt,
                         "winners": ", ".join(d.winners), "ranks_str": format_rank_string(d.ranks, d.winners)})
    decisions_df = pd.DataFrame(dec_rows).set_index("rebalance_dt").sort_index()

    weight_sched = decisions_to_weight_schedule(decisions, prices.index, prices.columns)

    fee_rate = cost_bps / 10_000.0
    cash = init_capital
    lots = LotsFIFO(list(prices.columns))
    tax_liability = 0.0
    taxes_paid_cum = 0.0
    realized_pl_year: Dict[int, float] = {}
    current_year = None

    pos_shares = pd.Series(0.0, index=prices.columns)
    ledger_rows, trades_rows = [], []
    prev_total = init_capital

    for i, t in enumerate(prices.index):
        px = prices.loc[t]

        # Year change: accrue annual tax at year end (after today's trades we'll also check again)
        year = t.year
        if current_year is None:
            current_year = year

        # portfolio valuation BEFORE trading
        pos_val_before = float((pos_shares * px).sum())
        total_before = cash + pos_val_before

        # target weights effective from this date
        target_w = weight_sched.loc[t]

        # ---- trade ONLY on scheduled exec dates ----
        if t in exec_dates:
            target_total_for_risk = total_before * (1.0 - cash_buffer_pct)
            target_dollar = target_w * target_total_for_risk
            curr_dollar = pos_shares * px
            delta_dollar = target_dollar - curr_dollar

            # 1) SELLS FIRST (long-only)
            for c in prices.columns:
                d_notional = float(delta_dollar[c])
                if d_notional >= 0:  # not a sell
                    continue
                trade_px = float(px[c])
                if not np.isfinite(trade_px) or trade_px <= 0:
                    continue
                # shares we intend to sell (cannot exceed position)
                shares_have = pos_shares[c]
                sell_shares = min(shares_have, abs(d_notional) / trade_px)
                if sell_shares <= 0: continue

                sell_notional = sell_shares * trade_px
                sell_fee = sell_notional * fee_rate

                # FIFO realized P/L including fees
                realized = lots.sell(c, sell_shares, trade_px, sell_fee)
                # Update holdings and cash (cash increases by proceeds - fee)
                pos_shares[c] -= sell_shares
                cash += (sell_notional - sell_fee)

                # --- tax processing on realized gain ---
                if tax_mode.upper() == "PER_TRADE":
                    gain = max(0.0, realized)  # only positive gains taxed
                    tax = gain * belka_tax
                    # pay from cash if available, otherwise accrue as liability (no negative cash)
                    pay = min(cash, tax)
                    cash -= pay
                    tax_liability += (tax - pay)
                    taxes_paid_cum += pay
                else:  # ANNUAL
                    realized_pl_year[year] = realized_pl_year.get(year, 0.0) + realized

                # Apply any outstanding tax liability using available cash immediately
                if tax_liability > 1e-12 and cash > 0:
                    pay = min(cash, tax_liability)
                    cash -= pay
                    tax_liability -= pay
                    taxes_paid_cum += pay

                trades_rows.append({
                    "date": t, "asset": c, "side": "SELL",
                    "shares": -sell_shares, "price": trade_px,
                    "trade_notional": -sell_notional, "fee_cost": sell_fee,
                    "realized_pl": realized, "tax_liability_after": tax_liability,
                    "cash_after": cash, "positions_value_after": float((pos_shares * px).sum()),
                    "portfolio_value_after": cash + float((pos_shares * px).sum()),
                })

            # 2) BUYS — scale to available cash (incl. fees, buffer respected)
            curr_dollar = pos_shares * px
            delta_dollar = target_dollar - curr_dollar
            buy_wants = {c: float(delta_dollar[c]) for c in prices.columns if float(delta_dollar[c]) > 0}
            buy_total = sum(buy_wants.values())

            # available cash cannot go negative; also tax_liability must be honored first
            free_cash = max(0.0, cash)  # already paid liability above with sells' proceeds
            if buy_total > 0 and free_cash > 0:
                scale = min(1.0, free_cash / (buy_total * (1.0 + fee_rate)))
                for c, want_notional in buy_wants.items():
                    trade_px = float(px[c])
                    if not np.isfinite(trade_px) or trade_px <= 0: continue
                    buy_notional = want_notional * scale
                    if buy_notional <= 0: continue
                    buy_fee = buy_notional * fee_rate
                    # clamp vs floating point
                    if buy_notional + buy_fee > cash:
                        buy_notional = max(0.0, cash / (1.0 + fee_rate))
                        buy_fee = buy_notional * fee_rate
                    buy_shares = 0.0
                    if trade_px > 0:
                        buy_shares = buy_notional / trade_px

                    # execute
                    cash -= (buy_notional + buy_fee)
                    pos_shares[c] += buy_shares
                    # store FIFO lot with cost incl. buy fee
                    if buy_shares > 0:
                        lot_cost_ps = (buy_notional + buy_fee) / buy_shares
                        lots.buy(c, buy_shares, lot_cost_ps)

                    trades_rows.append({
                        "date": t, "asset": c, "side": "BUY",
                        "shares": buy_shares, "price": trade_px,
                        "trade_notional": buy_notional, "fee_cost": buy_fee,
                        "cash_after": cash, "positions_value_after": float((pos_shares * px).sum()),
                        "portfolio_value_after": cash + float((pos_shares * px).sum()),
                    })

        # ----- YEAR-END tax accrual (ANNUAL mode) -----
        is_last_day = (i == len(prices.index) - 1)
        next_is_new_year = (not is_last_day) and (prices.index[i+1].year != year)
        if tax_mode.upper() == "ANNUAL" and (next_is_new_year or is_last_day):
            base = realized_pl_year.get(year, 0.0)
            taxable = (max(0.0, base) if allow_loss_netting_annual else max(0.0, base))
            tax_due = taxable * belka_tax
            if tax_due > 1e-12:
                # try to pay now from cash, otherwise accrue liability
                pay = min(cash, tax_due)
                cash -= pay
                tax_liability += (tax_due - pay)
                taxes_paid_cum += pay
            # reset yearly realized P/L
            realized_pl_year[year] = 0.0
            current_year = year

        # ----- end-of-day valuation -----
        pos_val = float((pos_shares * px).sum())
        total_after = cash + pos_val
        day_ret = (total_after / prev_total - 1.0) if prev_total != 0 else 0.0
        prev_total = total_after

        weights_now = (pos_shares * px) / (total_after if total_after != 0 else 1.0)
        row = {
            "date": t, "cash": cash, "positions_value": pos_val, "total_value": total_after,
            "daily_return": day_ret, "tax_liability": tax_liability, "taxes_paid_cum": taxes_paid_cum
        }
        for c in prices.columns:
            row[f"pos_shares_{c}"] = pos_shares[c]
            row[f"weight_{c}"] = weights_now[c]
        ledger_rows.append(row)

    # ---- Outputs ----
    ledger_df = pd.DataFrame(ledger_rows).set_index("date").sort_index()
    trades_df = (
        pd.DataFrame(trades_rows).sort_values(["date", "asset"]).reset_index(drop=True)
        if trades_rows else pd.DataFrame(
            columns=["date","asset","side","shares","price","trade_notional","fee_cost","realized_pl",
                     "tax_liability_after","cash_after","positions_value_after","portfolio_value_after"]
        )
    )

    # Simple summary
    if len(ledger_df) >= 2 and ledger_df["total_value"].notna().any():
        equity = ledger_df["total_value"].astype(float)
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252.0 / max(1, len(equity))) - 1.0
        dd = (equity / equity.cummax() - 1.0).min()
    else:
        cagr, dd = np.nan, np.nan
    summary = pd.DataFrame({
        "metric": ["Final Equity", "CAGR (approx, daily)", "Max Drawdown", "Tax Liability (end)", "Taxes Paid (cum)"],
        "value": [ledger_df["total_value"].iloc[-1], cagr, dd, ledger_df["tax_liability"].iloc[-1], ledger_df["taxes_paid_cum"].iloc[-1]]
    })

    if SAVE_CSV:
        _ensure_dir(out_dir)
        decisions_df.to_csv(f"{out_dir}/pl_momentum_decisions.csv", encoding="utf-8")
        weight_sched.to_csv(f"{out_dir}/pl_weight_schedule.csv", encoding="utf-8")
        ledger_df.to_csv(f"{out_dir}/pl_portfolio_ledger_tax.csv", encoding="utf-8")
        trades_df.to_csv(f"{out_dir}/pl_trade_blotter_tax.csv", encoding="utf-8", index=False)
        summary.to_csv(f"{out_dir}/pl_summary_tax.csv", encoding="utf-8", index=False)
        print(f"Saved CSVs to ./{out_dir}/ (tax-aware)")

    return {"decisions": decisions_df, "weight_schedule": weight_sched, "ledger": ledger_df, "trades": trades_df, "summary": summary}

# ------------------------ Run ------------------------
if __name__ == "__main__":
    out = inspect_cross_asset_momentum()
    print("\n=== Summary (tax-aware) ===")
    print(out["summary"].to_string(index=False))
    print("\n=== Last 5 trades ===")
    print(out["trades"].tail(5).to_string(index=False))
    print("\n=== Last 5 ledger rows ===")
    print(out["ledger"].tail(5)[["cash","positions_value","total_value","daily_return","tax_liability","taxes_paid_cum"]].to_string())
    print("\n=== Last 3 decisions ===")
    print(out["decisions"].tail(3).to_string())
