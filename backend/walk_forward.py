"""
walk_forward.py — Análisis de robustez estadística para backtests.

Implementa tres técnicas críticas que el original NO tenía:

1. WALK-FORWARD ROLLING
   - Optimiza en ventana IS, valida en siguiente ventana OOS
   - Mueve la ventana y repite N veces
   - Promedia resultados OOS reales (no una sola prueba 70/30)

2. MONTE CARLO DE TRADES
   - Reordena la secuencia de trades OOS 10,000 veces
   - Calcula percentil 5 del max drawdown
   - Ese es tu "worst realistic DD" — no el mejor caso

3. DEFLATED SHARPE RATIO (Bailey & López de Prado 2014)
   - Ajusta el Sharpe por el número de estrategias probadas
   - Con muchas combinaciones, por puro chance algunas superan filtros
   - DSR te dice si tu mejor estrategia es estadísticamente real o suerte

Uso:
    from backend.walk_forward import WalkForwardAnalyzer
    analyzer = WalkForwardAnalyzer(df, generator, backtester)
    results = analyzer.run_walk_forward(entries_long, n_windows=10)
    mc_results = analyzer.run_monte_carlo(trades_pnl, n_simulations=10000)
    dsr = analyzer.deflated_sharpe_ratio(observed_sharpe, n_trials=230000)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from backend.logger import get_logger

log = get_logger(__name__)


@dataclass
class WalkForwardResult:
    """Resultado de un análisis walk-forward."""
    n_windows: int
    is_returns: list[float] = field(default_factory=list)
    oos_returns: list[float] = field(default_factory=list)
    is_sharpes: list[float] = field(default_factory=list)
    oos_sharpes: list[float] = field(default_factory=list)
    oos_max_dds: list[float] = field(default_factory=list)
    consistency_ratio: float = 0.0  # % de ventanas OOS positivas
    avg_oos_return: float = 0.0
    avg_oos_sharpe: float = 0.0

    def summary(self) -> dict[str, float]:
        return {
            "n_windows": self.n_windows,
            "avg_oos_return_pct": self.avg_oos_return * 100,
            "avg_oos_sharpe": self.avg_oos_sharpe,
            "consistency_ratio_pct": self.consistency_ratio * 100,
            "worst_oos_return_pct": min(self.oos_returns) * 100 if self.oos_returns else 0,
            "best_oos_return_pct": max(self.oos_returns) * 100 if self.oos_returns else 0,
            "worst_oos_max_dd_pct": min(self.oos_max_dds) * 100 if self.oos_max_dds else 0,
        }


@dataclass
class MonteCarloResult:
    """Resultado de Monte Carlo sobre secuencia de trades."""
    n_simulations: int
    original_max_dd: float
    original_final_return: float
    mc_max_dd_p5: float         # percentil 5 (peor caso razonable)
    mc_max_dd_p50: float        # mediana
    mc_max_dd_p95: float        # mejor caso
    mc_return_p5: float
    mc_return_p50: float
    mc_return_p95: float
    prob_ruin: float = 0.0      # P(de quebrar: DD > 50%)


class WalkForwardAnalyzer:
    """Análisis de robustez estadística."""

    def __init__(self, data: pd.DataFrame, backtester) -> None:
        """
        Args:
            data:       DataFrame OHLCV completo
            backtester: instancia de VectorizedBacktester
        """
        self.data = data
        self.bt = backtester

    # ──────────────────────────────────────────────────────────
    def run_walk_forward(
        self,
        entries: pd.Series | pd.DataFrame,
        n_windows: int = 10,
        is_ratio: float = 0.7,
    ) -> WalkForwardResult:
        """
        Walk-forward rolling: divide los datos en n_windows+1 segmentos.
        Para cada i: optimiza/evalúa en segmentos IS (is_ratio) y valida en el siguiente OOS.

        Implementación simplificada: asume que `entries` ya está calculada
        (no reoptimiza parámetros, solo mide consistencia temporal).

        Args:
            entries: Serie o DataFrame de señales booleanas
            n_windows: número de ventanas walk-forward
            is_ratio: porción IS dentro de cada ventana (0.7 = 70/30)
        """
        n = len(self.data)
        window_size = n // (n_windows + 1)
        result = WalkForwardResult(n_windows=n_windows)

        log.info("Iniciando walk-forward", n_windows=n_windows, window_size=window_size)

        for i in range(n_windows):
            start = i * window_size
            is_end = start + int(window_size * is_ratio)
            oos_end = start + window_size + window_size  # IS + OOS

            if oos_end > n:
                oos_end = n
            if is_end >= n:
                break

            is_data = self.data.iloc[start:is_end]
            oos_data = self.data.iloc[is_end:oos_end]

            if isinstance(entries, pd.DataFrame):
                is_entries = entries.iloc[start:is_end]
                oos_entries = entries.iloc[is_end:oos_end]
            else:
                is_entries = entries.iloc[start:is_end]
                oos_entries = entries.iloc[is_end:oos_end]

            if len(is_entries) < 10 or len(oos_entries) < 5:
                continue

            # Backtest IS
            try:
                bt_is = type(self.bt)(is_data, config=self.bt.cfg)
                pf_is = bt_is.run(is_entries)
                is_ret = float(pf_is.total_return().mean() if hasattr(pf_is.total_return(), "mean") else pf_is.total_return())
                is_sharpe = float(pf_is.sharpe_ratio().mean() if hasattr(pf_is.sharpe_ratio(), "mean") else pf_is.sharpe_ratio())

                # Backtest OOS
                bt_oos = type(self.bt)(oos_data, config=self.bt.cfg)
                pf_oos = bt_oos.run(oos_entries)
                oos_ret = float(pf_oos.total_return().mean() if hasattr(pf_oos.total_return(), "mean") else pf_oos.total_return())
                oos_sharpe = float(pf_oos.sharpe_ratio().mean() if hasattr(pf_oos.sharpe_ratio(), "mean") else pf_oos.sharpe_ratio())
                oos_dd = float(pf_oos.max_drawdown().mean() if hasattr(pf_oos.max_drawdown(), "mean") else pf_oos.max_drawdown())

                result.is_returns.append(is_ret)
                result.oos_returns.append(oos_ret)
                result.is_sharpes.append(is_sharpe)
                result.oos_sharpes.append(oos_sharpe)
                result.oos_max_dds.append(oos_dd)
            except Exception as e:
                log.warning("Error en ventana", window=i, error=str(e))
                continue

        if result.oos_returns:
            result.avg_oos_return = float(np.mean(result.oos_returns))
            result.avg_oos_sharpe = float(np.mean(result.oos_sharpes))
            result.consistency_ratio = float(np.mean([1 if r > 0 else 0 for r in result.oos_returns]))

        log.info(
            "Walk-forward completo",
            avg_oos_return_pct=result.avg_oos_return * 100,
            consistency_pct=result.consistency_ratio * 100,
        )
        return result

    # ──────────────────────────────────────────────────────────
    def run_monte_carlo(
        self,
        trade_returns: list[float] | np.ndarray | pd.Series,
        n_simulations: int = 10_000,
        initial_capital: float = 1.0,
        ruin_threshold: float = -0.50,
        seed: int | None = 42,
    ) -> MonteCarloResult:
        """
        Monte Carlo: reordena trades con reposición y mide distribución de outcomes.

        Args:
            trade_returns: lista de retornos por trade (decimales, ej. 0.03 = +3%)
            n_simulations: número de simulaciones
            initial_capital: capital inicial (1.0 = 100%)
            ruin_threshold: DD a partir del cual se considera ruin (-0.50 = -50%)
            seed: semilla aleatoria para reproducibilidad
        """
        if seed is not None:
            np.random.seed(seed)

        trades = np.array(trade_returns, dtype=float)
        n_trades = len(trades)
        if n_trades < 5:
            raise ValueError("Se requieren al menos 5 trades para Monte Carlo")

        # Outcome original
        original_final = float(np.prod(1 + trades) - 1)
        original_dd = self._max_dd_from_trades(trades)

        # Simulación: remuestreo con reposición
        sim_final_returns = np.zeros(n_simulations)
        sim_max_dds = np.zeros(n_simulations)
        n_ruin = 0

        for i in range(n_simulations):
            sampled = np.random.choice(trades, size=n_trades, replace=True)
            equity = np.cumprod(1 + sampled) * initial_capital
            sim_final_returns[i] = equity[-1] / initial_capital - 1
            sim_max_dds[i] = self._max_dd_from_equity(equity)
            if sim_max_dds[i] <= ruin_threshold:
                n_ruin += 1

        result = MonteCarloResult(
            n_simulations=n_simulations,
            original_max_dd=original_dd,
            original_final_return=original_final,
            mc_max_dd_p5=float(np.percentile(sim_max_dds, 5)),
            mc_max_dd_p50=float(np.percentile(sim_max_dds, 50)),
            mc_max_dd_p95=float(np.percentile(sim_max_dds, 95)),
            mc_return_p5=float(np.percentile(sim_final_returns, 5)),
            mc_return_p50=float(np.percentile(sim_final_returns, 50)),
            mc_return_p95=float(np.percentile(sim_final_returns, 95)),
            prob_ruin=n_ruin / n_simulations,
        )

        log.info(
            "Monte Carlo completo",
            n_simulations=n_simulations,
            mc_dd_p5_pct=result.mc_max_dd_p5 * 100,
            mc_return_p5_pct=result.mc_return_p5 * 100,
            prob_ruin_pct=result.prob_ruin * 100,
        )
        return result

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def deflated_sharpe_ratio(
        observed_sharpe: float,
        n_trials: int,
        n_obs: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

        Ajusta el Sharpe observado por:
        - Número de trials (más pruebas → más chance de falsos positivos)
        - Skewness y kurtosis de los retornos
        - Tamaño de muestra

        Un DSR > 0.95 sugiere que el Sharpe NO es producto del azar.

        Args:
            observed_sharpe: Sharpe ratio anualizado observado
            n_trials: número de estrategias probadas (ej. 230000)
            n_obs: número de observaciones (días/barras)
            skewness: skewness de los retornos
            kurtosis: kurtosis de los retornos (3 = normal)
        """
        if n_trials < 1 or n_obs < 2:
            return 0.0

        # Expected maximum Sharpe bajo N(0,1) considerando n_trials
        # E[max(Z_1..Z_n)] ≈ sqrt(2*ln(n)) para n grande
        # Para n_trials pequeño, usar aproximación más precisa
        if n_trials > 1:
            euler_mascheroni = 0.5772156649
            expected_max_sharpe = (
                np.sqrt(2 * np.log(n_trials)) -
                euler_mascheroni / np.sqrt(2 * np.log(n_trials))
            ) / np.sqrt(n_obs)
        else:
            expected_max_sharpe = 0.0

        # Ajuste por skewness y kurtosis (cornish-fisher)
        # SR_adj = SR * sqrt(1 - skew*SR + (kurt-1)/4 * SR^2)
        sr = observed_sharpe * np.sqrt(n_obs)  # de-anualizar a per-bar
        sr_adjusted = sr * np.sqrt(
            max(0.0001, 1 - skewness * sr + (kurtosis - 1) / 4 * sr**2)
        )

        # DSR = Φ((SR_obs - SR_max_esperado) * sqrt(N) / sqrt(1 - skew*SR + ...))
        # Versión simplificada:
        try:
            dsr = stats.norm.cdf((sr_adjusted - expected_max_sharpe * np.sqrt(n_obs)) * np.sqrt(n_obs))
        except Exception:
            dsr = 0.0

        return float(dsr)

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _max_dd_from_trades(trades: np.ndarray) -> float:
        """Calcula max drawdown a partir de secuencia de retornos por trade."""
        equity = np.cumprod(1 + trades)
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        return float(dd.min()) if len(dd) > 0 else 0.0

    @staticmethod
    def _max_dd_from_equity(equity: np.ndarray) -> float:
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        return float(dd.min()) if len(dd) > 0 else 0.0
