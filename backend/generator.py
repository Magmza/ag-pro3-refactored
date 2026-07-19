"""
generator.py — Generador de fuerza bruta de señales de trading.

Mejoras vs original:
- Pipeline LONG y SHORT separados (en el original las bear_features
  se calculaban pero NUNCA se usaban → bot 100% long-only).
- Dirección configurable: 'long', 'short', 'both'
- Logging estructurado
- Tipado, mejor manejo de NaN en volume
- Función para listar features disponibles (útil para tests)
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterator, Literal

import numpy as np
import pandas as pd
import ta

from backend.logger import get_logger

log = get_logger(__name__)

Direction = Literal["long", "short", "both"]


class StrategyGenerator:
    """
    Genera combinaciones booleanas de features técnicos.

    Uso:
        gen = StrategyGenerator(df)
        gen.calculate_all_features()
        for batch, total in gen.generate_combinations_in_batches(direction="both", max_conditions=3):
            ...
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.close = data["Close"]
        self.high = data["High"]
        self.low = data["Low"]
        self.open = data["Open"]
        # Volume puede ser 0 o faltante (ej. forex)
        self.volume = data.get("Volume", pd.Series(0, index=data.index, dtype=float))

        # Bull features = señales de COMPRA (long)
        self.bull_features: dict[str, pd.Series] = {}
        # Bear features = señales de VENTA (short)
        self.bear_features: dict[str, pd.Series] = {}

    # ──────────────────────────────────────────────────────────
    def calculate_all_features(self) -> None:
        """Calcula todos los indicadores. Idempotente."""
        self._calc_indicators()
        self._calc_price_action()
        self._calc_smc()
        log.info(
            "Features calculados",
            bull=len(self.bull_features),
            bear=len(self.bear_features),
        )

    # ──────────────────────────────────────────────────────────
    def list_features(self, direction: Direction = "both") -> list[str]:
        """Devuelve la lista de features disponibles para una dirección."""
        if direction == "long":
            return list(self.bull_features.keys())
        if direction == "short":
            return list(self.bear_features.keys())
        return list(self.bull_features.keys()) + list(self.bear_features.keys())

    # ──────────────────────────────────────────────────────────
    def generate_combinations_in_batches(
        self,
        direction: Direction = "both",
        max_conditions: int = 4,
        batch_size: int = 5000,
    ) -> Iterator[tuple[pd.DataFrame, int]]:
        """
        Genera combinaciones de features con AND lógico.

        Args:
            direction: 'long', 'short', o 'both'
            max_conditions: máximo número de condiciones a combinar (1-N)
            batch_size: cuántas estrategias por lote (yield)

        Yields:
            (DataFrame de entries, total_combinaciones_teoricas)
        """
        if not self.bull_features:
            self.calculate_all_features()

        if direction == "long":
            feature_dict = self.bull_features
        elif direction == "short":
            feature_dict = self.bear_features
        else:
            # Both: combinamos en un solo dict, prefijando L:/S: para identificar
            feature_dict = {}
            for k, v in self.bull_features.items():
                feature_dict[f"L:{k}"] = v
            for k, v in self.bear_features.items():
                feature_dict[f"S:{k}"] = v

        feature_names = list(feature_dict.keys())
        total_features = len(feature_names)
        total_combos = sum(
            _comb(total_features, k) for k in range(1, max_conditions + 1)
        )
        log.info(
            "Iniciando generación",
            direction=direction,
            features=total_features,
            max_conditions=max_conditions,
            total_combinations=total_combos,
        )

        all_entries: dict[str, pd.Series] = {}
        count = 0

        for k in range(1, max_conditions + 1):
            for combo in combinations(feature_names, k):
                strat_name = " + ".join(combo)
                combined_mask = feature_dict[combo[0]].copy()
                for i in range(1, len(combo)):
                    combined_mask = combined_mask & feature_dict[combo[i]]
                all_entries[strat_name] = combined_mask
                count += 1

                if count >= batch_size:
                    yield pd.DataFrame(all_entries), total_combos
                    all_entries = {}
                    count = 0

        if all_entries:
            yield pd.DataFrame(all_entries), total_combos

    # ──────────────────────────────────────────────────────────
    # ─── INDICADORES TÉCNICOS ─────────────────────────────────
    # ──────────────────────────────────────────────────────────
    def _calc_indicators(self) -> None:
        # RSI
        for w in [10, 14, 21, 30]:
            rsi = ta.momentum.RSIIndicator(self.close, window=w).rsi()
            self.bull_features[f"rsi_{w}_oversold"] = rsi < 30
            self.bear_features[f"rsi_{w}_overbought"] = rsi > 70

        # MACD
        for fast, slow, sign in [(12, 26, 9), (8, 21, 5)]:
            macd = ta.trend.MACD(self.close, window_slow=slow, window_fast=fast, window_sign=sign)
            macd_line = macd.macd()
            macd_signal = macd.macd_signal()
            self.bull_features[f"macd_{fast}_{slow}_bull_cross"] = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
            self.bear_features[f"macd_{fast}_{slow}_bear_cross"] = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))

        # EMA crosses
        for ema_f, ema_s in [(20, 50), (50, 200)]:
            ema_fast = ta.trend.EMAIndicator(self.close, window=ema_f).ema_indicator()
            ema_slow = ta.trend.EMAIndicator(self.close, window=ema_s).ema_indicator()
            self.bull_features[f"trend_bull_{ema_f}v{ema_s}"] = ema_fast > ema_slow
            self.bear_features[f"trend_bear_{ema_f}v{ema_s}"] = ema_fast < ema_slow

        # Bollinger
        for w in [20, 50]:
            bb = ta.volatility.BollingerBands(self.close, window=w)
            self.bull_features[f"bb_{w}_break_lower"] = self.close < bb.bollinger_lband()
            self.bear_features[f"bb_{w}_break_upper"] = self.close > bb.bollinger_hband()

        # Stochastic
        for w, smooth in [(14, 3), (21, 5)]:
            stoch = ta.momentum.StochasticOscillator(self.high, self.low, self.close, window=w, smooth_window=smooth)
            self.bull_features[f"stoch_{w}_oversold"] = stoch.stoch() < 20
            self.bear_features[f"stoch_{w}_overbought"] = stoch.stoch() > 80

        # Williams %R
        for w in [14, 21, 50]:
            wr = ta.momentum.WilliamsRIndicator(self.high, self.low, self.close, lbp=w)
            self.bull_features[f"wrm_{w}_oversold"] = wr.williams_r() < -80
            self.bear_features[f"wrm_{w}_overbought"] = wr.williams_r() > -20

        # CCI
        for w in [20, 50]:
            cci = ta.trend.CCIIndicator(self.high, self.low, self.close, window=w)
            self.bull_features[f"cci_{w}_oversold"] = cci.cci() < -100
            self.bear_features[f"cci_{w}_overbought"] = cci.cci() > 100

        # Awesome Oscillator
        ao = ta.momentum.AwesomeOscillatorIndicator(self.high, self.low)
        ao_val = ao.awesome_oscillator()
        self.bull_features["ao_bull_cross"] = (ao_val > 0) & (ao_val.shift(1) <= 0)
        self.bear_features["ao_bear_cross"] = (ao_val < 0) & (ao_val.shift(1) >= 0)

        # ROC
        for w in [10, 20]:
            roc = ta.momentum.ROCIndicator(self.close, window=w)
            self.bull_features[f"roc_{w}_bull"] = roc.roc() > 0
            self.bear_features[f"roc_{w}_bear"] = roc.roc() < 0

        # ADX
        for w in [14, 20]:
            adx = ta.trend.ADXIndicator(self.high, self.low, self.close, window=w)
            self.bull_features[f"adx_{w}_bull_trend"] = (adx.adx() > 25) & (adx.adx_pos() > adx.adx_neg())
            self.bear_features[f"adx_{w}_bear_trend"] = (adx.adx() > 25) & (adx.adx_pos() < adx.adx_neg())

        # Parabolic SAR
        # La librería `ta` puede devolver series con longitud distinta al input.
        # Reindexamos para alinear con self.close.
        psar = ta.trend.PSARIndicator(self.high, self.low, self.close)
        psar_series = psar.psar()
        if len(psar_series) != len(self.close):
            psar_series = psar_series.reindex(self.close.index).ffill()
        self.bull_features["psar_bull"] = self.close > psar_series
        self.bear_features["psar_bear"] = self.close < psar_series

        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(self.high, self.low)
        tenkan = ichimoku.ichimoku_conversion_line()
        kijun = ichimoku.ichimoku_base_line()
        self.bull_features["ichimoku_bull_cross"] = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
        self.bear_features["ichimoku_bear_cross"] = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))

        # Ichimoku cloud (proyectado 26 barras adelante como en TradingView)
        span_a = ichimoku.ichimoku_a().shift(26)
        span_b = ichimoku.ichimoku_b().shift(26)
        self.bull_features["ichimoku_price_above_cloud"] = (self.close > span_a) & (self.close > span_b)
        self.bear_features["ichimoku_price_below_cloud"] = (self.close < span_a) & (self.close < span_b)

        # Aroon
        aroon = ta.trend.AroonIndicator(self.high, self.low, window=25)
        self.bull_features["aroon_bull"] = aroon.aroon_up() > aroon.aroon_down()
        self.bear_features["aroon_bear"] = aroon.aroon_up() < aroon.aroon_down()

        # Keltner Channels
        kc = ta.volatility.KeltnerChannel(self.high, self.low, self.close, window=20)
        self.bull_features["kc_20_break_lower"] = self.close < kc.keltner_channel_lband()
        self.bear_features["kc_20_break_upper"] = self.close > kc.keltner_channel_hband()

        # Donchian Channels
        for w in [20, 50]:
            dc = ta.volatility.DonchianChannel(self.high, self.low, self.close, window=w)
            self.bull_features[f"dc_{w}_break_upper"] = self.close > dc.donchian_channel_hband()
            self.bear_features[f"dc_{w}_break_lower"] = self.close < dc.donchian_channel_lband()

        # Volume indicators (solo si hay volumen real)
        if self.volume.sum() > 0:
            vwap = ta.volume.VolumeWeightedAveragePrice(self.high, self.low, self.close, self.volume)
            vwap_line = vwap.volume_weighted_average_price()
            self.bull_features["vwap_cross_up"] = (self.close > vwap_line) & (self.close.shift(1) <= vwap_line.shift(1))
            self.bear_features["vwap_cross_down"] = (self.close < vwap_line) & (self.close.shift(1) >= vwap_line.shift(1))

            cmf = ta.volume.ChaikinMoneyFlowIndicator(self.high, self.low, self.close, self.volume, window=20)
            self.bull_features["cmf_bull"] = cmf.chaikin_money_flow() > 0.05
            self.bear_features["cmf_bear"] = cmf.chaikin_money_flow() < -0.05

            mfi = ta.volume.MFIIndicator(self.high, self.low, self.close, self.volume, window=14)
            self.bull_features["mfi_14_oversold"] = mfi.money_flow_index() < 20
            self.bear_features["mfi_14_overbought"] = mfi.money_flow_index() > 80

            obv = ta.volume.OnBalanceVolumeIndicator(self.close, self.volume)
            obv_sma = obv.on_balance_volume().rolling(window=20).mean()
            self.bull_features["obv_bull_trend"] = obv.on_balance_volume() > obv_sma
            self.bear_features["obv_bear_trend"] = obv.on_balance_volume() < obv_sma

    # ──────────────────────────────────────────────────────────
    def _calc_price_action(self) -> None:
        # Engulfing
        prev_open, prev_close = self.open.shift(1), self.close.shift(1)
        self.bull_features["pa_bull_engulfing"] = (
            (prev_close < prev_open) & (self.open < prev_close) & (self.close > prev_open)
        )
        self.bear_features["pa_bear_engulfing"] = (
            (prev_close > prev_open) & (self.open > prev_close) & (self.close < prev_open)
        )

        # Soportes / Resistencias
        for w in [50, 100]:
            support_level = self.low.rolling(window=w).min().shift(1)
            resistance_level = self.high.rolling(window=w).max().shift(1)
            self.bull_features[f"pa_support_bounce_{w}"] = (
                (self.low <= support_level * 1.002) & (self.close > support_level)
            )
            self.bear_features[f"pa_resistance_reject_{w}"] = (
                (self.high >= resistance_level * 0.998) & (self.close < resistance_level)
            )

        # Trendlines (proxy por pendiente de SMA)
        for period in [20, 50]:
            sma = self.close.rolling(window=period).mean()
            slope = sma.diff(3)
            self.bull_features[f"pa_trendline_bull_{period}"] = (
                (slope > 0) & (self.low <= sma * 1.001) & (self.close > sma)
            )
            self.bear_features[f"pa_trendline_bear_{period}"] = (
                (slope < 0) & (self.high >= sma * 0.999) & (self.close < sma)
            )

    # ──────────────────────────────────────────────────────────
    def _calc_smc(self) -> None:
        # Fair Value Gaps
        self.bull_features["smc_bull_fvg"] = self.low > self.high.shift(2)
        self.bear_features["smc_bear_fvg"] = self.high < self.low.shift(2)

        # Liquidity Sweeps
        for w in [10, 20, 50]:
            rolling_min = self.low.rolling(window=w).min().shift(1)
            rolling_max = self.high.rolling(window=w).max().shift(1)
            self.bull_features[f"smc_bull_sweep_{w}"] = (self.low < rolling_min) & (self.close > rolling_min)
            self.bear_features[f"smc_bear_sweep_{w}"] = (self.high > rolling_max) & (self.close < rolling_max)

        # Order Blocks (mitigation)
        strong_bull_move = (self.close > self.open * 1.005) & (self.close.shift(1) < self.open.shift(1))
        ob_bull_level = np.where(strong_bull_move, self.low.shift(1), np.nan)
        ob_bull_series = pd.Series(ob_bull_level, index=self.close.index).ffill()

        strong_bear_move = (self.close < self.open * 0.995) & (self.close.shift(1) > self.open.shift(1))
        ob_bear_level = np.where(strong_bear_move, self.high.shift(1), np.nan)
        ob_bear_series = pd.Series(ob_bear_level, index=self.close.index).ffill()

        self.bull_features["smc_mitigate_bull_ob"] = (self.low <= ob_bull_series * 1.002) & (self.close > ob_bull_series)
        self.bear_features["smc_mitigate_bear_ob"] = (self.high >= ob_bear_series * 0.998) & (self.close < ob_bear_series)


# ──────────────────────────────────────────────────────────────
def _comb(n: int, k: int) -> int:
    """math.comb con safe-guard."""
    import math
    if k > n or n < 0 or k < 0:
        return 0
    return math.comb(n, k)
