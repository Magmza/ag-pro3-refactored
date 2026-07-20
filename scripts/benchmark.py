"""Benchmark: FastBacktester vs VectorizedBacktester."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.fast_backtester import FastBacktester


def main():
    np.random.seed(42)
    n = 8757
    dates = pd.date_range("2022-01-01", periods=n, freq="1h")
    close = 3000.0 * np.cumprod(1 + np.random.normal(-0.0001, 0.015, n))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = np.roll(close, 1); open_[0] = 3000.0

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": 50000.0
    }, index=dates)
    df.index.name = "Date"

    for n_strats in [100, 500, 2000]:
        print(f"\n{'='*60}")
        print(f"TEST: {n_strats} estrategias x {n} velas")
        print(f"{'='*60}")

        rng = np.random.RandomState(42)
        entries_arr = rng.rand(n, n_strats) > 0.95
        entries_df = pd.DataFrame(
            entries_arr, index=df.index,
            columns=[f"strat_{i}" for i in range(n_strats)],
        )

        print("\n--- FastBacktester (numpy + numba) ---")
        fb = FastBacktester(df, sl_pct=0.015, tp_pct=0.030, fees=0.0005, slippage=0.0005)
        t0 = time.time()
        results = fb.run_many(entries_df)
        t1 = time.time()
        print(f"Tiempo: {t1-t0:.2f}s")
        print(f"Velocidad: {n_strats/(t1-t0):.0f} estrategias/segundo")
        print(f"Trades promedio: {results['Trades'].mean():.0f}")


if __name__ == "__main__":
    main()