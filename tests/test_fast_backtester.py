"""tests/test_fast_backtester.py — Tests para FastBacktester."""
import numpy as np
import pandas as pd
import pytest

from backend.fast_backtester import FastBacktester


def test_fast_backtester_initialization(sample_ohlcv):
    bt = FastBacktester(sample_ohlcv)
    assert bt.n == len(sample_ohlcv)
    assert bt.sl_pct == 0.015


def test_run_single_returns_result(sample_ohlcv):
    entries = pd.Series(False, index=sample_ohlcv.index)
    entries.iloc[10::50] = True
    bt = FastBacktester(sample_ohlcv)
    result = bt.run_single(entries)
    assert result.n_trades > 0
    assert -1 <= result.total_return <= 10
    assert 0 <= result.win_rate <= 1


def test_run_single_no_entries(sample_ohlcv):
    entries = pd.Series(False, index=sample_ohlcv.index)
    bt = FastBacktester(sample_ohlcv)
    result = bt.run_single(entries)
    assert result.n_trades == 0
    assert result.total_return == 0.0


def test_run_many_returns_dataframe(sample_ohlcv):
    n = len(sample_ohlcv)
    entries_df = pd.DataFrame({
        "strat_1": np.random.RandomState(42).rand(n) > 0.95,
        "strat_2": np.random.RandomState(43).rand(n) > 0.97,
        "strat_3": np.random.RandomState(44).rand(n) > 0.90,
    }, index=sample_ohlcv.index)
    bt = FastBacktester(sample_ohlcv)
    df = bt.run_many(entries_df)
    assert len(df) == 3
    assert "Sharpe Ratio" in df.columns
    assert "Profit Factor" in df.columns
    assert "Trades" in df.columns


def test_run_many_handles_large_n(sample_ohlcv):
    n = len(sample_ohlcv)
    rng = np.random.RandomState(42)
    entries_df = pd.DataFrame(
        rng.rand(n, 500) > 0.95,
        index=sample_ohlcv.index,
        columns=[f"strat_{i}" for i in range(500)],
    )
    bt = FastBacktester(sample_ohlcv)
    df = bt.run_many(entries_df)
    assert len(df) == 500


def test_short_direction(sample_ohlcv):
    entries = pd.Series(False, index=sample_ohlcv.index)
    entries.iloc[20::40] = True
    bt = FastBacktester(sample_ohlcv)
    result_long = bt.run_single(entries, direction=1)
    result_short = bt.run_single(entries, direction=-1)
    assert result_long.n_trades == result_short.n_trades


def test_benchmark_buy_hold(sample_ohlcv):
    bt = FastBacktester(sample_ohlcv)
    bh = bt.benchmark_buy_hold()
    assert "buy_hold_return_pct" in bh
    assert "buy_hold_max_dd_pct" in bh


def test_benchmark_buy_hold_oos(sample_ohlcv):
    bt = FastBacktester(sample_ohlcv)
    bh_full = bt.benchmark_buy_hold()
    bh_oos = bt.benchmark_buy_hold(is_oos=True)
    assert bh_full["buy_hold_return_pct"] != bh_oos["buy_hold_return_pct"]


def test_metrics_calculation(sample_ohlcv):
    entries = pd.Series(False, index=sample_ohlcv.index)
    entries.iloc[10::30] = True
    bt = FastBacktester(sample_ohlcv)
    result = bt.run_single(entries)
    assert result.n_trades == result.n_wins + result.n_losses
    if result.n_trades > 0:
        assert 0 <= result.win_rate <= 1


def test_reproducibility(sample_ohlcv):
    entries = pd.Series(False, index=sample_ohlcv.index)
    entries.iloc[10::50] = True
    bt = FastBacktester(sample_ohlcv)
    r1 = bt.run_single(entries)
    r2 = bt.run_single(entries)
    assert r1.total_return == r2.total_return
    assert r1.n_trades == r2.n_trades


def test_fees_affect_returns(sample_ohlcv):
    entries = pd.Series(False, index=sample_ohlcv.index)
    entries.iloc[10::50] = True
    bt_low = FastBacktester(sample_ohlcv, fees=0.0001)
    bt_high = FastBacktester(sample_ohlcv, fees=0.005)
    r_low = bt_low.run_single(entries)
    r_high = bt_high.run_single(entries)
    assert r_high.total_return <= r_low.total_return + 1e-10