"""
tests/test_walk_forward.py — Tests para walk-forward, Monte Carlo y DSR.
"""
import numpy as np
import pandas as pd
import pytest

from backend.backtester import BacktestConfig, VectorizedBacktester
from backend.walk_forward import WalkForwardAnalyzer


@pytest.fixture
def backtester(sample_ohlcv):
    return VectorizedBacktester(sample_ohlcv, config=BacktestConfig())


def test_monte_carlo_basic(backtester):
    """Monte Carlo con 20 trades y 100 simulaciones."""
    analyzer = WalkForwardAnalyzer(sample_ohlcv := backtester.data, backtester)
    trades = [0.02, -0.01, 0.03, -0.015, 0.025, -0.01, 0.02, -0.02, 0.015, -0.005,
              0.03, -0.01, 0.02, -0.015, 0.025, -0.01, 0.02, -0.02, 0.015, -0.005]

    result = analyzer.run_monte_carlo(trades, n_simulations=100, seed=42)

    assert result.n_simulations == 100
    assert result.mc_max_dd_p5 <= result.mc_max_dd_p50 <= result.mc_max_dd_p95
    assert result.mc_return_p5 <= result.mc_return_p50 <= result.mc_return_p95
    assert 0 <= result.prob_ruin <= 1


def test_monte_carlo_too_few_trades(backtester):
    """Menos de 5 trades debe fallar."""
    analyzer = WalkForwardAnalyzer(backtester.data, backtester)
    with pytest.raises(ValueError, match="al menos 5 trades"):
        analyzer.run_monte_carlo([0.01, 0.02], n_simulations=10)


def test_monte_carlo_reproducible(backtester):
    """Misma semilla produce mismo resultado."""
    analyzer = WalkForwardAnalyzer(backtester.data, backtester)
    trades = [0.02, -0.01, 0.03, -0.015, 0.025, -0.01, 0.02, -0.02, 0.015, -0.005]

    r1 = analyzer.run_monte_carlo(trades, n_simulations=50, seed=42)
    r2 = analyzer.run_monte_carlo(trades, n_simulations=50, seed=42)
    assert r1.mc_return_p5 == r2.mc_return_p5


def test_deflated_sharpe_basic(backtester):
    """DSR con valores razonables devuelve número entre 0 y 1."""
    analyzer = WalkForwardAnalyzer(backtester.data, backtester)
    dsr = analyzer.deflated_sharpe_ratio(
        observed_sharpe=1.5,
        n_trials=1000,
        n_obs=252,
    )
    assert 0 <= dsr <= 1


def test_deflated_sharpe_more_trials_lower_dsr(backtester):
    """Más trials → DSR más bajo (más chance de falsos positivos)."""
    analyzer = WalkForwardAnalyzer(backtester.data, backtester)
    dsr_few = analyzer.deflated_sharpe_ratio(observed_sharpe=1.5, n_trials=10, n_obs=252)
    dsr_many = analyzer.deflated_sharpe_ratio(observed_sharpe=1.5, n_trials=100000, n_obs=252)
    assert dsr_many <= dsr_few


def test_deflated_sharpe_zero_sharpe(backtester):
    """Sharpe = 0 → DSR cercano a 0."""
    analyzer = WalkForwardAnalyzer(backtester.data, backtester)
    dsr = analyzer.deflated_sharpe_ratio(observed_sharpe=0.0, n_trials=100, n_obs=252)
    assert dsr < 0.5  # debería ser bajo


def test_walk_forward_returns_result(backtester, sample_ohlcv):
    """Walk-forward devuelve objeto resultado con métricas."""
    analyzer = WalkForwardAnalyzer(sample_ohlcv, backtester)
    entries = pd.Series(False, index=sample_ohlcv.index)
    entries.iloc[10::20] = True

    result = analyzer.run_walk_forward(entries, n_windows=5)

    assert result.n_windows == 5
    assert len(result.oos_returns) > 0
    assert isinstance(result.avg_oos_return, float)
    assert isinstance(result.consistency_ratio, float)
    summary = result.summary()
    assert "avg_oos_return_pct" in summary
