"""
backtester.py — Motor de backtest vectorizado con vectorbt.

Usado por walk_forward.py. Para el scanner principal usar fast_backtester.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import vectorbt as vbt

from backend.logger import get_logger

log = get_logger(__name__)

SlippageModel = Literal["fixed", "atr", "stochastic"]


@dataclass
class BacktestConfig:
    """Configuración del backtest."""
    fees: float = 0.0005
    slippage_pct: float = 0.0005
    slippage_model: SlippageModel = "fixed"
    slippage_atr_alpha: float = 0.3
    slippage_atr_window: int = 14
    slippage_stochastic_std: float = 0.0003
    sl_pct: float = 0.015
    tp_pct: float = 0.030
    freq: str = "1h"
    direction: Literal["long", "short", "both"] = "long"


class VectorizedBacktester:
    """Backtester con SL/TP fijos via vectorbt."""

    def __init__(self, data, config=None):
        self.data = data
        self.cfg = config or BacktestConfig()
        self._slippage_series = self._compute_slippage_series()

    def _compute_slippage_series(self):
        n = len(self.data)
        base = pd.Series(self.cfg.slippage_pct, index=self.data.index)

        if self.cfg.slippage_model == "fixed":
            return base

        high, low, close = self.data["High"], self.data["Low"], self.data["Close"]
        tr = pd.concat(
            [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=self.cfg.slippage_atr_window).mean()
        atr_pct = (atr / close).fillna(0)

        if self.cfg.slippage_model == "atr":
            return base + self.cfg.slippage_atr_alpha * atr_pct

        if self.cfg.slippage_model == "stochastic":
            noise = pd.Series(
                np.random.normal(0, self.cfg.slippage_stochastic_std, n),
                index=self.data.index,
            )
            return (base + self.cfg.slippage_atr_alpha * atr_pct + noise).clip(lower=0)

        return base

    def run(self, entries_long=None, entries_short=None, is_oos=None):
        if is_oos is not None:
            split_idx = int(len(self.data) * 0.7)
            if is_oos:
                data = self.data.iloc[split_idx:]
                el = entries_long.iloc[split_idx:] if entries_long is not None else None
                es = entries_short.iloc[split_idx:] if entries_short is not None else None
            else:
                data = self.data.iloc[:split_idx]
                el = entries_long.iloc[:split_idx] if entries_long is not None else None
                es = entries_short.iloc[:split_idx] if entries_short is not None else None
        else:
            data = self.data
            el, es = entries_long, entries_short

        if el is not None and es is not None:
            portfolio = vbt.Portfolio.from_signals(
                data["Close"],
                entries=el,
                short_entries=es,
                sl_stop=self.cfg.sl_pct,
                tp_stop=self.cfg.tp_pct,
                fees=self.cfg.fees,
                slippage=self._slippage_series.loc[data.index],
                freq=self.cfg.freq,
            )
        elif el is not None:
            portfolio = vbt.Portfolio.from_signals(
                data["Close"],
                entries=el,
                sl_stop=self.cfg.sl_pct,
                tp_stop=self.cfg.tp_pct,
                fees=self.cfg.fees,
                slippage=self._slippage_series.loc[data.index],
                freq=self.cfg.freq,
            )
        elif es is not None:
            portfolio = vbt.Portfolio.from_signals(
                data["Close"],
                short_entries=es,
                sl_stop=self.cfg.sl_pct,
                tp_stop=self.cfg.tp_pct,
                fees=self.cfg.fees,
                slippage=self._slippage_series.loc[data.index],
                freq=self.cfg.freq,
            )
        else:
            raise ValueError("Se requiere al menos entries_long o entries_short")

        return portfolio

    @staticmethod
    def _sanitize(arr):
        if isinstance(arr, pd.Series):
            return arr.replace([np.inf, -np.inf], np.nan)
        return np.where(np.isinf(arr), np.nan, arr)

    def calculate_professional_metrics(self, portfolio):
        trades = portfolio.trades

        def _to_series(val, name="value"):
            if isinstance(val, pd.Series):
                return val
            if isinstance(val, (int, float, np.integer, np.floating)):
                return pd.Series([val], name=name)
            arr = np.atleast_1d(val)
            return pd.Series(arr, name=name)

        total_return = _to_series(portfolio.total_return(), "total_return")
        max_dd = _to_series(portfolio.max_drawdown() * -1, "max_dd")
        win_rate = _to_series(trades.win_rate(), "win_rate")
        profit_factor = _to_series(trades.profit_factor(), "pf")
        n_trades = _to_series(trades.count(), "n_trades")
        sharpe = _to_series(portfolio.sharpe_ratio(), "sharpe")
        sortino = _to_series(portfolio.sortino_ratio(), "sortino")
        calmar = _to_series(portfolio.calmar_ratio(), "calmar")

        idx = total_return.index
        max_dd = max_dd.reindex(idx).fillna(0)
        win_rate = win_rate.reindex(idx).fillna(0)
        profit_factor = profit_factor.reindex(idx).fillna(0)
        n_trades = n_trades.reindex(idx).fillna(0).astype(int)
        sharpe = sharpe.reindex(idx).fillna(0)
        sortino = sortino.reindex(idx).fillna(0)
        calmar = calmar.reindex(idx).fillna(0)

        recovery_factor = total_return / np.where(max_dd > 0, max_dd, np.nan)

        wr = win_rate.fillna(0)
        expectancy = (wr * self.cfg.tp_pct) - ((1 - wr) * self.cfg.sl_pct)

        variance = (wr * self.cfg.tp_pct**2) + ((1 - wr) * self.cfg.sl_pct**2) - expectancy**2
        std_trade = np.sqrt(np.maximum(variance, 1e-12))
        sqn = (expectancy / std_trade) * np.sqrt(n_trades)

        metrics_df = pd.DataFrame({
            "Retorno (%)": self._sanitize(total_return * 100),
            "Profit Factor": self._sanitize(profit_factor),
            "Max Drawdown (%)": self._sanitize(max_dd * 100),
            "Win Rate (%)": self._sanitize(win_rate * 100),
            "Expectancy": self._sanitize(expectancy),
            "Recovery Factor": self._sanitize(recovery_factor),
            "Sharpe Ratio": self._sanitize(sharpe),
            "Sortino Ratio": self._sanitize(sortino),
            "Calmar Ratio": self._sanitize(calmar),
            "SQN": self._sanitize(sqn),
            "Trades": n_trades,
        })

        return metrics_df

    def benchmark_buy_hold(self, is_oos=None):
        if is_oos is not None:
            split_idx = int(len(self.data) * 0.7)
            data = self.data.iloc[split_idx:] if is_oos else self.data.iloc[:split_idx]
        else:
            data = self.data

        if len(data) < 2:
            return {"buy_hold_return_pct": 0.0, "buy_hold_max_dd_pct": 0.0}

        ret = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100

        cum = (1 + data["Close"].pct_change().fillna(0)).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        max_dd_pct = float(dd.min() * 100)

        return {
            "buy_hold_return_pct": float(ret),
            "buy_hold_max_dd_pct": max_dd_pct,
        }