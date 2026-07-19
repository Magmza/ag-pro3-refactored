"""
tests/test_backtester.py — Tests para el backtester.
"""
import numpy as np
import pandas as pd
import pytest

from backend.backtester import BacktestConfig, VectorizedBacktester


def test_backtester_initialization(sample_ohlcv):
    """Backtester se inicializa correctamente."""
    bt = VectorizedBacktester(sample_ohlcv)
    assert bt.cfg.sl_pct == 0.015
    assert bt.cfg.tp_pct == 0.030
    assert len(bt._slippage_series) == len(sample_ohlcv)


def test_slippage_models(sample_ohlcv):
    """Los 3 modelos de slippage deben producir series distintas."""
    cfg_fixed = BacktestConfig(slippage_model="fixed")
    cfg_atr = BacktestConfig(slippage_model="atr")
    cfg_stoch = BacktestConfig(slippage_model="stochastic")

    bt_fixed = VectorizedBacktester(sample_ohlcv, config=cfg_fixed)
    bt_atr = VectorizedBacktester(sample_ohlcv, config=cfg_atr)
    bt_stoch = VectorizedBacktester(sample_ohlcv, config=cfg_stoch)

    s_fixed = bt_fixed._slippage_series
    s_atr = bt_atr._slippage_series
    s_stoch = bt_stoch._slippage_series

    # ATR debe ser >= fixed en promedio
    assert s_atr.mean() >= s_fixed.mean() * 0.99
    # Stochastic no debe ser constante
    assert s_stoch.std() > 0


def test_run_with_entries(sample_ohlcv):
    """Backtest básico con señales long produce portfolio."""
    entries = pd.Series(False, index=sample_ohlcv.index)
    entries.iloc[10::50] = True  # entrada cada 50 barras

    bt = VectorizedBacktester(sample_ohlcv)
    portfolio = bt.run(entries_long=entries)

    assert portfolio is not None
    metrics = bt.calculate_professional_metrics(portfolio)
    assert "Sharpe Ratio" in metrics.columns
    assert "Profit Factor" in metrics.columns


def test_benchmark_buy_hold(sample_ohlcv):
    """Benchmark B&H devuelve métricas correctas."""
    bt = VectorizedBacktester(sample_ohlcv)
    bh = bt.benchmark_buy_hold()

    assert "buy_hold_return_pct" in bh
    assert "buy_hold_max_dd_pct" in bh
    assert isinstance(bh["buy_hold_return_pct"], float)


def test_benchmark_buy_hold_oos(sample_ohlcv):
    """Benchmark B&H en OOS usa último 30%."""
    bt = VectorizedBacktester(sample_ohlcv)
    bh_full = bt.benchmark_buy_hold()
    bh_oos = bt.benchmark_buy_hold(is_oos=True)

    # Deben ser distintos porque usan distintos periodos
    assert bh_full["buy_hold_return_pct"] != bh_oos["buy_hold_return_pct"]


def test_calculate_metrics_has_all_expected(sample_ohlcv):
    """calculate_professional_metrics devuelve todas las métricas esperadas."""
    entries = pd.Series(False, index=sample_ohlcv.index)
    entries.iloc[10::30] = True

    bt = VectorizedBacktester(sample_ohlcv)
    pf = bt.run(entries)
    metrics = bt.calculate_professional_metrics(pf)

    expected = [
        "Retorno (%)",
        "Profit Factor",
        "Max Drawdown (%)",
        "Win Rate (%)",
        "Expectancy",
        "Recovery Factor",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "SQN",
        "Trades",
    ]
    for col in expected:
        assert col in metrics.columns, f"Falta métrica {col}"


def test_short_signals(sample_ohlcv):
    """Backtester soporta señales short."""
    entries_short = pd.Series(False, index=sample_ohlcv.index)
    entries_short.iloc[20::40] = True

    bt = VectorizedBacktester(sample_ohlcv)
    portfolio = bt.run(entries_short=entries_short)

    assert portfolio is not None
