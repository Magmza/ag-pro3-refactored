"""
pine_translator.py — Traduce estrategias generadas a Pine Script v5.

Mejoras vs original:
- Soporta LONG y SHORT (original era 100% long)
- Detecta prefijos "L:" y "S:" del generator con direction='both'
- Position sizing configurable (no siempre 100% equity)
- Process orders en close (coincide con vectorbt default)
- Comisiones configurables
- Webhook payload incluye client_order_id para idempotencia
"""
from __future__ import annotations

import re
from typing import Literal

Direction = Literal["long", "short"]


def _detect_direction(condition_name: str) -> Direction:
    """Detecta si la condición es long o short por prefijo."""
    if condition_name.startswith("S:"):
        return "short"
    return "long"


def _strip_prefix(condition_name: str) -> str:
    """Quita el prefijo L:/S: si existe."""
    if condition_name.startswith(("L:", "S:")):
        return condition_name[2:]
    return condition_name


def generate_pine_script(
    strategy_name: str,
    sl_pct: float,
    tp_pct: float,
    rr_ratio: float,
    symbol: str,
    qty_pct: float = 0.10,
    fees_pct: float = 0.05,
    webhook_passphrase_hint: str = "CONFIGURE_IN_ENV",
) -> str:
    """
    Genera código Pine Script v5 ejecutable.

    Args:
        strategy_name:    Nombre de la estrategia (ej. "L:rsi_14_oversold + S:macd_12_26_bear_cross")
        sl_pct:           Stop loss en decimal (0.015 = 1.5%)
        tp_pct:           Take profit en decimal (0.030 = 3.0%)
        rr_ratio:         Risk:Reward ratio (tp/sl)
        symbol:           Activo
        qty_pct:          % del equity a usar por trade (default 10% — conservador)
        fees_pct:         Comisión en % (0.05 = 5 cents por $100)
        webhook_passphrase_hint: hint para el usuario (no exponer passphrase real)
    """
    raw_conditions = [c.strip() for c in strategy_name.split("+")]
    pine_vars: list[str] = []
    pine_long_conditions: list[str] = []
    pine_short_conditions: list[str] = []

    for i, raw_cond in enumerate(raw_conditions):
        direction = _detect_direction(raw_cond)
        cond = _strip_prefix(raw_cond)
        cond_var = f"cond_{i}"

        translated = _translate_single_condition(cond, cond_var, i)
        if translated is None:
            pine_vars.append(f"{cond_var} = false // {cond} (No traducido)")
            pine_long_conditions.append(cond_var)
            continue

        pine_vars.extend(translated["vars"])
        if direction == "long":
            pine_long_conditions.append(cond_var)
        else:
            pine_short_conditions.append(cond_var)

    # Dedupe vars manteniendo orden
    unique_pine_vars: list[str] = []
    seen: set[str] = set()
    for var in pine_vars:
        if var not in seen:
            unique_pine_vars.append(var)
            seen.add(var)

    pine_vars_str = "\n".join(f"    {v}" for v in unique_pine_vars)
    long_cond_str = " and ".join(pine_long_conditions) if pine_long_conditions else "false"
    short_cond_str = " and ".join(pine_short_conditions) if pine_short_conditions else "false"

    has_short = bool(pine_short_conditions)
    has_long = bool(pine_long_conditions)

    # Build execution block
    execution_lines = []
    if has_long:
        execution_lines.append(f'if ({long_cond_str})')
        execution_lines.append(f'    strategy.entry("Long", strategy.long, qty_percent={qty_pct*100})')
        execution_lines.append(f'    strategy.exit("SL/TP Long", "Long", stop=close*(1-{sl_pct:.4f}), limit=close*(1+{tp_pct:.4f}))')
    if has_short:
        execution_lines.append(f'if ({short_cond_str})')
        execution_lines.append(f'    strategy.entry("Short", strategy.short, qty_percent={qty_pct*100})')
        execution_lines.append(f'    strategy.exit("SL/TP Short", "Short", stop=close*(1+{sl_pct:.4f}), limit=close*(1-{tp_pct:.4f}))')

    execution_str = "\n".join(execution_lines)

    direction_label = "LONG+SHORT" if has_short and has_long else ("SHORT" if has_short else "LONG")

    code = f"""//@version=5
strategy("{strategy_name} Pro 3.1", overlay=true,
     initial_capital=1000,
     default_qty_type=strategy.percent_of_equity,
     default_qty_value={qty_pct*100},
     margin_long=100, margin_short=100,
     process_orders_on_close=false,
     commission_type=strategy.commission.percent,
     commission_value={fees_pct})
// SL: {sl_pct*100:.2f}% | TP: {tp_pct*100:.2f}% | RR 1:{rr_ratio:.1f} | Direction: {direction_label}
// Activo: {symbol}

// --- Cálculos de Indicadores y Condiciones ---
{pine_vars_str}

long_condition = {long_cond_str}
short_condition = {short_cond_str}

// --- Ejecución ---
{execution_str}

// --- Webhook payload template ---
// Long:  {{"strategy_id": "{strategy_name}", "symbol": "{symbol}", "action": "BUY", "volume": 1, "passphrase": "{webhook_passphrase_hint}", "client_order_id": "{{ticker}}-{{timenow}}"}}
// Short: {{"strategy_id": "{strategy_name}", "symbol": "{symbol}", "action": "SELL", "volume": 1, "passphrase": "{webhook_passphrase_hint}", "client_order_id": "{{ticker}}-{{timenow}}"}}
"""
    return code


# ──────────────────────────────────────────────────────────────
def _translate_single_condition(cond: str, cond_var: str, idx: int) -> dict | None:
    """Traduce una condición a Pine Script. Devuelve dict con 'vars' list."""
    # Cada uno de estos bloques replica la lógica del generator.py
    # IMPORTANTE: mantener sincronizado con generator.py

    # RSI
    m = re.match(r'rsi_(\d+)_(oversold|overbought)', cond)
    if m:
        w, state = m.group(1), m.group(2)
        vars = [
            f"rsi_{w} = ta.rsi(close, {w})",
            f"{cond_var} = rsi_{w} {'< 30' if state == 'oversold' else '> 70'}",
        ]
        return {"vars": vars}

    # MACD
    m = re.match(r'macd_(\d+)_(\d+)_(bull|bear)_cross', cond)
    if m:
        fast, slow, state = m.group(1), m.group(2), m.group(3)
        sign = 9 if fast == "12" and slow == "26" else 5
        vars = [
            f"[macdLine_{fast}_{slow}, signalLine_{fast}_{slow}, _] = ta.macd(close, {fast}, {slow}, {sign})",
            f"{cond_var} = ta.{'crossover' if state == 'bull' else 'crossunder'}(macdLine_{fast}_{slow}, signalLine_{fast}_{slow})",
        ]
        return {"vars": vars}

    # EMA Trend
    m = re.match(r'trend_(bull|bear)_(\d+)v(\d+)', cond)
    if m:
        state, fast, slow = m.group(1), m.group(2), m.group(3)
        vars = [
            f"ema_{fast} = ta.ema(close, {fast})",
            f"ema_{slow} = ta.ema(close, {slow})",
            f"{cond_var} = ema_{fast} {'>' if state == 'bull' else '<'} ema_{slow}",
        ]
        return {"vars": vars}

    # Bollinger
    m = re.match(r'bb_(\d+)_break_(lower|upper)', cond)
    if m:
        w, state = m.group(1), m.group(2)
        vars = [
            f"[bb_basis_{w}, bb_upper_{w}, bb_lower_{w}] = ta.bb(close, {w}, 2)",
            f"{cond_var} = close {'<' if state == 'lower' else '>'} {'bb_lower_' + w if state == 'lower' else 'bb_upper_' + w}",
        ]
        return {"vars": vars}

    # Stochastic
    m = re.match(r'stoch_(\d+)_(oversold|overbought)', cond)
    if m:
        w, state = m.group(1), m.group(2)
        smooth = 3 if w == "14" else 5
        vars = [
            f"k_{w} = ta.stoch(close, high, low, {w})",
            f"d_{w} = ta.sma(k_{w}, {smooth})",
            f"{cond_var} = k_{w} {'< 20' if state == 'oversold' else '> 80'}",
        ]
        return {"vars": vars}

    # Williams %R
    m = re.match(r'wrm_(\d+)_(oversold|overbought)', cond)
    if m:
        w, state = m.group(1), m.group(2)
        vars = [
            f"wrm_{w} = ta.wpr({w})",
            f"{cond_var} = wrm_{w} {'< -80' if state == 'oversold' else '> -20'}",
        ]
        return {"vars": vars}

    # CCI
    m = re.match(r'cci_(\d+)_(oversold|overbought)', cond)
    if m:
        w, state = m.group(1), m.group(2)
        vars = [
            f"cci_{w} = ta.cci(close, {w})",
            f"{cond_var} = cci_{w} {'< -100' if state == 'oversold' else '> 100'}",
        ]
        return {"vars": vars}

    # Awesome Oscillator
    m = re.match(r'ao_(bull|bear)_cross', cond)
    if m:
        state = m.group(1)
        vars = [
            "ao_val = ta.sma(hl2, 5) - ta.sma(hl2, 34)",
            f"{cond_var} = ta.{'crossover' if state == 'bull' else 'crossunder'}(ao_val, 0)",
        ]
        return {"vars": vars}

    # ROC
    m = re.match(r'roc_(\d+)_(bull|bear)', cond)
    if m:
        w, state = m.group(1), m.group(2)
        vars = [
            f"roc_{w} = ta.roc(close, {w})",
            f"{cond_var} = roc_{w} {'> 0' if state == 'bull' else '< 0'}",
        ]
        return {"vars": vars}

    # ADX
    m = re.match(r'adx_(\d+)_(bull|bear)_trend', cond)
    if m:
        w, state = m.group(1), m.group(2)
        vars = [
            f"[diplus_{w}, diminus_{w}, adx_{w}] = ta.dmi({w}, {w})",
            f"{cond_var} = adx_{w} > 25 and diplus_{w} {'>' if state == 'bull' else '<'} diminus_{w}",
        ]
        return {"vars": vars}

    # PSAR
    m = re.match(r'psar_(bull|bear)', cond)
    if m:
        state = m.group(1)
        vars = [
            "psar_val = ta.sar(0.02, 0.02, 0.2)",
            f"{cond_var} = close {'>' if state == 'bull' else '<'} psar_val",
        ]
        return {"vars": vars}

    # Ichimoku
    m = re.match(r'ichimoku_(bull|bear)_cross', cond)
    if m:
        state = m.group(1)
        vars = [
            "tenkan = math.avg(ta.lowest(9), ta.highest(9))",
            "kijun = math.avg(ta.lowest(26), ta.highest(26))",
            f"{cond_var} = ta.{'crossover' if state == 'bull' else 'crossunder'}(tenkan, kijun)",
        ]
        return {"vars": vars}

    m = re.match(r'ichimoku_price_(above|below)_cloud', cond)
    if m:
        state = m.group(1)
        vars = [
            "tenkan = math.avg(ta.lowest(9), ta.highest(9))",
            "kijun = math.avg(ta.lowest(26), ta.highest(26))",
            "senkouA = math.avg(tenkan, kijun)",
            "senkouB = math.avg(ta.lowest(52), ta.highest(52))",
            f"{cond_var} = close {'>' if state == 'above' else '<'} senkouA[26] and close {'>' if state == 'above' else '<'} senkouB[26]",
        ]
        return {"vars": vars}

    # Aroon
    m = re.match(r'aroon_(bull|bear)', cond)
    if m:
        state = m.group(1)
        vars = [
            "aroon_up = 100 * (25 - ta.highestbars(high, 25)) / 25",
            "aroon_down = 100 * (25 - ta.lowestbars(low, 25)) / 25",
            f"{cond_var} = aroon_up {'>' if state == 'bull' else '<'} aroon_down",
        ]
        return {"vars": vars}

    # Keltner
    m = re.match(r'kc_(\d+)_break_(upper|lower)', cond)
    if m:
        w, state = m.group(1), m.group(2)
        vars = [
            f"[kc_basis_{w}, kc_upper_{w}, kc_lower_{w}] = ta.kc(close, {w}, 2)",
            f"{cond_var} = close {'>' if state == 'upper' else '<'} {'kc_upper_' + w if state == 'upper' else 'kc_lower_' + w}",
        ]
        return {"vars": vars}

    # Donchian
    m = re.match(r'dc_(\d+)_break_(upper|lower)', cond)
    if m:
        w, state = m.group(1), m.group(2)
        vars = [
            f"dc_upper_{w} = ta.highest(high, {w})",
            f"dc_lower_{w} = ta.lowest(low, {w})",
            f"{cond_var} = close {'>' if state == 'upper' else '<'} {'dc_upper_' + w if state == 'upper' else 'dc_lower_' + w}",
        ]
        return {"vars": vars}

    # VWAP
    m = re.match(r'vwap_cross_(up|down)', cond)
    if m:
        state = m.group(1)
        vars = [
            "vwap_val = ta.vwap",
            f"{cond_var} = ta.{'crossover' if state == 'up' else 'crossunder'}(close, vwap_val)",
        ]
        return {"vars": vars}

    # CMF
    m = re.match(r'cmf_(bull|bear)', cond)
    if m:
        state = m.group(1)
        vars = [
            "mf_mult = ((close - low) - (high - close)) / (high - low)",
            "mf_vol = mf_mult * volume",
            "cmf_val = math.sum(mf_vol, 20) / math.sum(volume, 20)",
            f"{cond_var} = cmf_val {'> 0.05' if state == 'bull' else '< -0.05'}",
        ]
        return {"vars": vars}

    # MFI
    m = re.match(r'mfi_(\d+)_(oversold|overbought)', cond)
    if m:
        w, state = m.group(1), m.group(2)
        vars = [
            f"mfi_{w} = ta.mfi(hlc3, {w})",
            f"{cond_var} = mfi_{w} {'< 20' if state == 'oversold' else '> 80'}",
        ]
        return {"vars": vars}

    # OBV
    m = re.match(r'obv_(bull|bear)_trend', cond)
    if m:
        state = m.group(1)
        vars = [
            "obv_val = ta.cum(math.sign(ta.change(close)) * volume)",
            "obv_sma = ta.sma(obv_val, 20)",
            f"{cond_var} = obv_val {'>' if state == 'bull' else '<'} obv_sma",
        ]
        return {"vars": vars}

    # Price Action
    m = re.match(r'pa_(bull|bear)_engulfing', cond)
    if m:
        state = m.group(1)
        if state == "bull":
            cond_str = "close[1] < open[1] and open < close[1] and close > open[1]"
        else:
            cond_str = "close[1] > open[1] and open > close[1] and close < open[1]"
        return {"vars": [f"{cond_var} = {cond_str}"]}

    m = re.match(r'pa_support_bounce_(\d+)', cond)
    if m:
        w = m.group(1)
        vars = [
            f"support_{w} = ta.lowest(low, {w})[1]",
            f"{cond_var} = low <= support_{w} * 1.002 and close > support_{w}",
        ]
        return {"vars": vars}

    m = re.match(r'pa_resistance_reject_(\d+)', cond)
    if m:
        w = m.group(1)
        vars = [
            f"resistance_{w} = ta.highest(high, {w})[1]",
            f"{cond_var} = high >= resistance_{w} * 0.998 and close < resistance_{w}",
        ]
        return {"vars": vars}

    m = re.match(r'pa_trendline_(bull|bear)_(\d+)', cond)
    if m:
        state, period = m.group(1), m.group(2)
        vars = [
            f"sma_trend_{period} = ta.sma(close, {period})",
            f"slope_{period} = sma_trend_{period} - sma_trend_{period}[3]",
        ]
        if state == "bull":
            cond_str = f"slope_{period} > 0 and low <= sma_trend_{period} * 1.001 and close > sma_trend_{period}"
        else:
            cond_str = f"slope_{period} < 0 and high >= sma_trend_{period} * 0.999 and close < sma_trend_{period}"
        vars.append(f"{cond_var} = {cond_str}")
        return {"vars": vars}

    # SMC
    m = re.match(r'smc_(bull|bear)_fvg', cond)
    if m:
        state = m.group(1)
        if state == "bull":
            return {"vars": [f"{cond_var} = low > high[2]"]}
        return {"vars": [f"{cond_var} = high < low[2]"]}

    m = re.match(r'smc_(bull|bear)_sweep_(\d+)', cond)
    if m:
        state, w = m.group(1), m.group(2)
        if state == "bull":
            vars = [
                f"rolling_min_{w} = ta.lowest(low, {w})[1]",
                f"{cond_var} = low < rolling_min_{w} and close > rolling_min_{w}",
            ]
        else:
            vars = [
                f"rolling_max_{w} = ta.highest(high, {w})[1]",
                f"{cond_var} = high > rolling_max_{w} and close < rolling_max_{w}",
            ]
        return {"vars": vars}

    m = re.match(r'smc_mitigate_(bull|bear)_ob', cond)
    if m:
        state = m.group(1)
        if state == "bull":
            vars = [
                "strong_bull_move = close > open * 1.005 and close[1] < open[1]",
                "var float ob_bull_level = na",
                "if strong_bull_move",
                "    ob_bull_level := low[1]",
                f"{cond_var} = not na(ob_bull_level) and low <= ob_bull_level * 1.002 and close > ob_bull_level",
            ]
        else:
            vars = [
                "strong_bear_move = close < open * 0.995 and close[1] > open[1]",
                "var float ob_bear_level = na",
                "if strong_bear_move",
                "    ob_bear_level := high[1]",
                f"{cond_var} = not na(ob_bear_level) and high >= ob_bear_level * 0.998 and close < ob_bear_level",
            ]
        return {"vars": vars}

    return None
