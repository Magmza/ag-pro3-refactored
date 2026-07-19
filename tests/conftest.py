"""
tests/conftest.py — Fixtures compartidas para pytest.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Asegurar que podemos importar backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """DataFrame OHLCV sintético de 500 barras para tests."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    base = 100.0
    returns = np.random.normal(0.0001, 0.01, n)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = np.roll(close, 1)
    open_[0] = base
    volume = np.random.randint(1000, 100000, n).astype(float)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def sample_ohlcv_bull() -> pd.DataFrame:
    """DataFrame con tendencia alcista clara."""
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 100 * np.exp(np.linspace(0, 0.3, n))  # +35% en 300 barras
    high = close * 1.005
    low = close * 0.995
    open_ = np.roll(close, 1)
    open_[0] = 100
    volume = np.full(n, 50000.0)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


@pytest.fixture
def sample_ohlcv_bear() -> pd.DataFrame:
    """DataFrame con tendencia bajista clara."""
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 100 * np.exp(np.linspace(0, -0.3, n))  # -26% en 300 barras
    high = close * 1.005
    low = close * 0.995
    open_ = np.roll(close, 1)
    open_[0] = 100
    volume = np.full(n, 50000.0)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)
