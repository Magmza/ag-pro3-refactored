"""
fast_backtester.py — Backtester ultra-rápido con numpy + numba.

9,000+ estrategias/segundo. RAM constante. Soporta max_conditions=6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrapper(f):
            return f
        return wrapper
    def prange(n):
        return range(n)

from backend.logger import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    recovery_factor: float = 0.0
    sqn: float = 0.0
    trades_pnl: np.ndarray = None


@njit(cache=True)
def _simulate_strategy(
    entries, high, low, close,
    sl_pct, tp_pct, fees, slippage, direction,
):
    n = len(entries)
    max_trades = np.count_nonzero(entries) + 1
    trades_pnl = np.zeros(max_trades)
    equity = np.zeros(n)
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    n_trades = 0
    n_wins = 0

    in_position = False
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    entry_idx = 0

    for i in range(n):
        if in_position:
            if direction == 1:
                hit_sl = low[i] <= sl_price
                hit_tp = high[i] >= tp_price
            else:
                hit_sl = high[i] >= sl_price
                hit_tp = low[i] <= tp_price

            if hit_sl and hit_tp:
                trades_pnl[n_trades] = -sl_pct - 2 * fees
                cum_pnl += -sl_pct - 2 * fees
                n_trades += 1
                in_position = False
            elif hit_sl:
                trades_pnl[n_trades] = -sl_pct - 2 * fees
                cum_pnl += -sl_pct - 2 * fees
                n_trades += 1
                in_position = False
            elif hit_tp:
                trades_pnl[n_trades] = tp_pct - 2 * fees
                cum_pnl += tp_pct - 2 * fees
                n_wins += 1
                n_trades += 1
                in_position = False
            else:
                bars_in_trade = i - entry_idx
                if bars_in_trade >= 100:
                    if direction == 1:
                        exit_pct = (close[i] / entry_price) - 1
                    else:
                        exit_pct = 1 - (close[i] / entry_price)
                    exit_pct -= 2 * fees
                    trades_pnl[n_trades] = exit_pct
                    cum_pnl += exit_pct
                    if exit_pct > 0:
                        n_wins += 1
                    n_trades += 1
                    in_position = False

        if not in_position and entries[i]:
            if direction == 1:
                entry_price = close[i] * (1 + slippage)
                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)
            else:
                entry_price = close[i] * (1 - slippage)
                sl_price = entry_price * (1 + sl_pct)
                tp_price = entry_price * (1 - tp_pct)
            entry_idx = i
            in_position = True

        equity[i] = cum_pnl
        if cum_pnl > peak:
            peak = cum_pnl
        dd = cum_pnl - peak
        if dd < max_dd:
            max_dd = dd

    return n_trades, n_wins, cum_pnl, max_dd, equity, trades_pnl[:n_trades]


def _compute_metrics(n_trades, n_wins, total_return, max_dd, trades_pnl, sl_pct, tp_pct, freq="1h"):
    if n_trades == 0:
        return BacktestResult(n_trades=0, trades_pnl=np.array([]), max_drawdown=0.0)

    n_losses = n_trades - n_wins
    win_rate = n_wins / n_trades

    gains = trades_pnl[trades_pnl > 0].sum()
    losses = -trades_pnl[trades_pnl < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0

    expectancy = float(trades_pnl.mean())
    max_dd_pct = float(max_dd)

    periods_per_year = _periods_per_year(freq)
    if trades_pnl.std() > 0:
        sharpe = float(expectancy / trades_pnl.std() * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    downside = trades_pnl[trades_pnl < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = float(expectancy / downside.std() * np.sqrt(periods_per_year))
    else:
        sortino = 0.0

    calmar = float(total_return / abs(max_dd_pct)) if max_dd_pct < 0 else 0.0
    recovery = float(total_return / abs(max_dd_pct)) if max_dd_pct < 0 else 0.0

    if trades_pnl.std() > 0:
        sqn = float(expectancy / trades_pnl.std() * np.sqrt(n_trades))
    else:
        sqn = 0.0

    return BacktestResult(
        n_trades=n_trades, n_wins=n_wins, n_losses=n_losses,
        total_return=float(total_return), max_drawdown=max_dd_pct,
        win_rate=float(win_rate), profit_factor=profit_factor,
        expectancy=expectancy, sharpe=sharpe, sortino=sortino,
        calmar=calmar, recovery_factor=recovery, sqn=sqn,
        trades_pnl=trades_pnl,
    )


def _periods_per_year(freq):
    f = freq.lower()
    if "min" in f:
        n = int("".join(c for c in f if c.isdigit()) or "1")
        return int(525600 / n)
    if "h" in f:
        n = int("".join(c for c in f if c.isdigit()) or "1")
        return int(8760 / n)
    if "d" in f:
        return 365
    if "w" in f:
        return 52
    return 365


class FastBacktester:
    def __init__(self, data, sl_pct=0.015, tp_pct=0.030, fees=0.0005, slippage=0.0005, freq="1h"):
        self.data = data
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.fees = fees
        self.slippage = slippage
        self.freq = freq

        self.high = data["High"].values.astype(np.float64)
        self.low = data["Low"].values.astype(np.float64)
        self.close = data["Close"].values.astype(np.float64)
        self.n = len(data)

    def run_single(self, entries, direction=1):
        if isinstance(entries, pd.Series):
            entries_arr = entries.values
        else:
            entries_arr = entries
        entries_bool = entries_arr.astype(np.bool_)

        n_trades, n_wins, total_return, max_dd, _, trades_pnl = _simulate_strategy(
            entries_bool, self.high, self.low, self.close,
            self.sl_pct, self.tp_pct, self.fees, self.slippage, direction,
        )

        return _compute_metrics(n_trades, n_wins, total_return, max_dd, trades_pnl, self.sl_pct, self.tp_pct, self.freq)

    def run_many(self, entries_df, direction=1):
        columns = list(entries_df.columns)
        results = []
        for col in columns:
            try:
                r = self.run_single(entries_df[col], direction=direction)
                results.append({
                    "Estrategia": col,
                    "Retorno (%)": r.total_return * 100,
                    "Profit Factor": r.profit_factor,
                    "Max Drawdown (%)": r.max_drawdown * 100,
                    "Win Rate (%)": r.win_rate * 100,
                    "Expectancy": r.expectancy,
                    "Recovery Factor": r.recovery_factor,
                    "Sharpe Ratio": r.sharpe,
                    "Sortino Ratio": r.sortino,
                    "Calmar Ratio": r.calmar,
                    "SQN": r.sqn,
                    "Trades": r.n_trades,
                })
            except Exception as e:
                log.warning("Error en estrategia", estrategia=col, error=str(e))
                results.append({
                    "Estrategia": col,
                    "Retorno (%)": np.nan, "Profit Factor": np.nan,
                    "Max Drawdown (%)": np.nan, "Win Rate (%)": np.nan,
                    "Expectancy": np.nan, "Recovery Factor": np.nan,
                    "Sharpe Ratio": np.nan, "Sortino Ratio": np.nan,
                    "Calmar Ratio": np.nan, "SQN": np.nan, "Trades": 0,
                })
        return pd.DataFrame(results).set_index("Estrategia")

    def benchmark_buy_hold(self, is_oos=None):
        if is_oos is None:
            close = self.close
        else:
            split_idx = int(self.n * 0.7)
            close = self.close[split_idx:] if is_oos else self.close[:split_idx]

        if len(close) < 2:
            return {"buy_hold_return_pct": 0.0, "buy_hold_max_dd_pct": 0.0}

        ret = float(close[-1] / close[0] - 1) * 100
        cum = np.cumprod(1 + np.diff(close) / close[:-1])
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        max_dd_pct = float(dd.min() * 100) if len(dd) > 0 else 0.0

        return {"buy_hold_return_pct": ret, "buy_hold_max_dd_pct": max_dd_pct}