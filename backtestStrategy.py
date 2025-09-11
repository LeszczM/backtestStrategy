# backtest.py
# Requirements: pip install pandas numpy yfinance ta matplotlib scipy
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ----------------------- CONFIG (edit here) -----------------------
ASSETS = ["BTC-USD", "ETH-USD", "SPY"]
INTERVAL = "1d"            # "1d", "1h", etc.
START = "2018-01-01"
END = None                 # None = today
COST_BPS = 10              # per trade side, e.g., 10 bps = 0.1%
SLIPPAGE_BPS = 5           # simple model added to costs
ALLOW_SHORT = False
INIT_CAPITAL = 100_000
RISK_FREE_ANNUAL = 0.02    # for Sharpe/Sortino
REBAL_FREQ = "M"           # for momentum strategy: "M" monthly, "W" weekly, "D" daily
TOP_K = 2                  # momentum: allocate to top K assets
# ------------------------------------------------------------------

# ---------- Utilities ----------
def ann_factor(freq: str = "D") -> float:
    return {"D":252, "W":52, "M":12, "H":24*252}.get(freq.upper(), 252)

def pct_to_bps(x: float) -> float:
    return x * 1e4

def bps_to_pct(x_bps: float) -> float:
    return x_bps / 1e4

def to_period_str(interval: str) -> str:
    # crude map for annualization
    return "D" if interval.endswith("d") else ("H" if interval.endswith("h") else "D")

# ---------- Data ----------
def get_ohlc(assets: List[str], start: str, end: Optional[str], interval: str) -> pd.DataFrame:
    df = yf.download(assets, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        # Columns like ('Adj Close','BTC-USD'), we’ll keep 'Close' level only
        close = df["Close"].copy()
        volume = df.get("Volume")
    else:
        close = df[["Close"]].copy()
        volume = df.get("Volume")
    close = close.dropna(how="all")
    if isinstance(close, pd.Series):
        close = close.to_frame()
    return close.sort_index()

# ---------- Indicators ----------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # classic Wilder's RSI
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

# ---------- Strategy base ----------
class Strategy:
    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Returns target position per asset in [-1, 0, +1] or weights summing to 1.
        Convention here:
          * Signal strategies (SMA/RSI): return -1/0/+1 per asset (we’ll convert to weights).
          * Momentum: returns weights summing to 1 (long-only by default).
        Index aligns with prices.index, columns with assets.
        """
        raise NotImplementedError

# SMA crossover per asset (long-only by default)
@dataclass
class SMACross(Strategy):
    fast: int = 20
    slow: int = 100
    allow_short: bool = False
    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for c in prices.columns:
            f = sma(prices[c], self.fast)
            s = sma(prices[c], self.slow)
            raw = np.where(f > s, 1.0, (-1.0 if self.allow_short else 0.0))
            pos[c] = pd.Series(raw, index=prices.index)
        return pos

# RSI mean-reversion per asset
@dataclass
class RSIMeanReversion(Strategy):
    period: int = 14
    buy_th: float = 30.0
    sell_th: float = 70.0
    allow_short: bool = False
    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for c in prices.columns:
            r = rsi(prices[c], self.period)
            long = (r < self.buy_th).astype(float)
            short = (r > self.sell_th).astype(float) * (-1.0 if self.allow_short else 0.0)
            pos[c] = (long + short).replace(0, np.nan).ffill().fillna(0.0)
        return pos

# Cross-asset momentum: allocate to top K by past return; rebalance on REBAL_FREQ
@dataclass
class XAssetMomentum(Strategy):
    lookback: int = 126
    top_k: int = 2
    rebalance: str = "M"
    allow_short: bool = False  # long-only

    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        ret_lb = prices.pct_change(self.lookback)

        # resample to rebalance frequency; marks has period-end labels
        marks = ret_lb.resample(self.rebalance).last()
        marks = marks.dropna(how="all")

        weights = pd.DataFrame(0.0, index=marks.index, columns=prices.columns)
        for t, row in marks.iterrows():               # iterate over marks, not ret_lb
            row = row.dropna()
            if row.empty:
                continue
            k = min(self.top_k, len(row))             # guard if fewer valid assets
            top = row.nlargest(k).index
            weights.loc[t, top] = 1.0 / k

        # carry weights between rebalances and align to full price index
        weights = weights.reindex(prices.index).ffill().fillna(0.0)
        return weights
# ---------- Backtester ----------
@dataclass
class BTResult:
    equity: pd.Series
    weights: pd.DataFrame
    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    trades: pd.DataFrame  # boolean where weight change occurred
    metrics: Dict[str, float]

def compute_metrics(ret: pd.Series, interval: str, rf_annual: float = 0.0) -> Dict[str,float]:
    freq = to_period_str(interval)
    ann = ann_factor(freq)
    mu = ret.mean() * ann
    sig = ret.std(ddof=0) * math.sqrt(ann)
    rf = rf_annual
    sharpe = (mu - rf) / (sig + 1e-12)
    neg = ret[ret < 0]
    sortino = (mu - rf) / (neg.std(ddof=0) * math.sqrt(ann) + 1e-12)
    # drawdown
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    max_dd = dd.min()
    if max_dd != 0:
        calmar = mu / abs(max_dd)
    else:
        calmar = np.nan
    # hit rate & expectancy
    wins = (ret > 0).sum()
    losses = (ret < 0).sum()
    hit_rate = wins / max(1, (wins + losses))
    expectancy = ret.mean()
    # exposure (non-zero weight sum / num assets)
    # approximate: average of sum(|w|) / 1.0 (long-only)
    # computed later in run_backtest; placeholder here
    return {
        "CAGR": (cum.iloc[-1])**(ann/len(ret)) - 1 if len(ret) > 0 else np.nan,
        "AnnReturn": mu,
        "AnnVol": sig,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": float(max_dd),
        "Calmar": calmar,
        "HitRate": hit_rate,
        "Expectancy": expectancy,
    }

def run_backtest(prices: pd.DataFrame,
                 strategy: Strategy,
                 interval: str,
                 cost_bps: float = 0.0,
                 slippage_bps: float = 0.0,
                 allow_short: bool = False) -> BTResult:
    # positions_or_weights may be -1/0/1 per asset OR already weights summing to 1
    sig = strategy.generate_positions(prices).reindex(prices.index).fillna(0.0)
    # If signals look like -1..+1, convert to equal-weight across signaled assets
    row_sums = sig.abs().sum(axis=1)        # number of active legs if signals are ±1
    is_weight_like = row_sums.between(0.99, 1.01)  # sums ≈ 1 → already weights

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    # keep true weights as-is
    weights.loc[is_weight_like] = sig.loc[is_weight_like]

    # equal-weight across active signals when sum != 1
    to_norm = (row_sums > 0) & (~is_weight_like)
    weights.loc[to_norm] = sig.loc[to_norm].div(row_sums[to_norm], axis=0)

    # if no signals (row_sums == 0), the row stays 0.0
    if not allow_short:
        weights = weights.clip(lower=0.0)

    # daily returns
    rets = prices.pct_change().fillna(0.0)

    # turnover & costs (simple model): cost applied on weight changes
    w_prev = weights.shift(1).fillna(0.0)
    turnover = (weights - w_prev).abs().sum(axis=1)
    per_side_cost = bps_to_pct(cost_bps + slippage_bps)
    costs = turnover * per_side_cost

    gross = (weights.shift(1) * rets).sum(axis=1)  # use weights at t-1 applied to ret t
    net = gross - costs

    equity = (1 + net).cumprod()
    metrics = compute_metrics(net, interval)
    metrics["Turnover(Ann)"] = turnover.mean() * ann_factor(to_period_str(interval))
    metrics["Exposure(Avg)"] = weights.abs().sum(axis=1).mean()

    trades = (weights != w_prev)

    return BTResult(
        equity=equity,
        weights=weights,
        returns=net,
        gross_returns=gross,
        costs=costs,
        trades=trades,
        metrics=metrics
    )

# ---------- Reporting ----------
def summarize_results(results: Dict[str, BTResult]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        m = res.metrics
        rows.append({
            "Strategy": name,
            "CAGR": m["CAGR"],
            "AnnReturn": m["AnnReturn"],
            "AnnVol": m["AnnVol"],
            "Sharpe": m["Sharpe"],
            "Sortino": m["Sortino"],
            "MaxDD": m["MaxDD"],
            "Calmar": m["Calmar"],
            "HitRate": m["HitRate"],
            "Expectancy": m["Expectancy"],
            "Turnover(Ann)": m["Turnover(Ann)"],
            "Exposure(Avg)": m["Exposure(Avg)"],
            "FinalMultiple": res.equity.iloc[-1] if len(res.equity) else np.nan
        })
    df = pd.DataFrame(rows).set_index("Strategy").sort_values("Sharpe", ascending=False)
    return df

def plot_equity(results: Dict[str, BTResult], title: str = "Equity Curves"):
    plt.figure(figsize=(11,6))
    for name, res in results.items():
        res.equity.plot(label=name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_drawdown(res: BTResult, title: str):
    cum = (1 + res.returns).cumprod()
    peak = cum.cummax()
    dd = cum/peak - 1
    plt.figure(figsize=(11,3))
    dd.plot()
    plt.title(title + " – Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------- Main run ----------
def main():
    prices = get_ohlc(ASSETS, START, END, INTERVAL)

    strategies: Dict[str, Strategy] = {
        f"SMA({20},{100})": SMACross(fast=20, slow=100, allow_short=ALLOW_SHORT),
        f"RSI_MR(14,30/70)": RSIMeanReversion(period=14, buy_th=30, sell_th=70, allow_short=ALLOW_SHORT),
        f"XMomentum(lb=126,k={TOP_K},{REBAL_FREQ})": XAssetMomentum(lookback=126, top_k=TOP_K, rebalance=REBAL_FREQ)
    }

    results: Dict[str, BTResult] = {}
    for name, strat in strategies.items():
        res = run_backtest(prices, strat, INTERVAL, COST_BPS, SLIPPAGE_BPS, ALLOW_SHORT)
        results[name] = res

    table = summarize_results(results)
    print("\n=== Strategy Comparison ===")
    print(table.to_string(float_format=lambda x: f"{x:,.4f}"))

    plot_equity(results, title="Strategy Equity (Starting at 1.0)")
    # optional: individual drawdowns
    for name, res in results.items():
        plot_drawdown(res, name)

if __name__ == "__main__":
    main()
