"""
tests/test_pine_translator.py — Tests para el traductor a Pine Script.
"""
import pytest

from backend.pine_translator import generate_pine_script


def test_generates_valid_pine_v5():
    """El output debe ser Pine Script v5 válido."""
    code = generate_pine_script(
        strategy_name="rsi_14_oversold",
        sl_pct=0.015,
        tp_pct=0.030,
        rr_ratio=2.0,
        symbol="ETH/USDT",
    )
    assert "@version=5" in code
    assert 'strategy("rsi_14_oversold Pro 3.1"' in code
    assert "ETH/USDT" in code
    assert "SL: 1.50%" in code
    assert "TP: 3.00%" in code


def test_long_only_strategy():
    """Estrategia solo long genera entry LONG."""
    code = generate_pine_script(
        strategy_name="rsi_14_oversold",
        sl_pct=0.015,
        tp_pct=0.030,
        rr_ratio=2.0,
        symbol="ETH/USDT",
    )
    assert 'strategy.entry("Long", strategy.long' in code
    assert 'strategy.entry("Short"' not in code


def test_short_only_strategy():
    """Estrategia solo short genera entry SHORT."""
    code = generate_pine_script(
        strategy_name="S:rsi_14_overbought",
        sl_pct=0.015,
        tp_pct=0.030,
        rr_ratio=2.0,
        symbol="ETH/USDT",
    )
    assert 'strategy.entry("Short", strategy.short' in code
    assert 'strategy.entry("Long"' not in code


def test_mixed_long_short_strategy():
    """Combinación L: + S: genera entries LONG y SHORT."""
    code = generate_pine_script(
        strategy_name="L:rsi_14_oversold + S:macd_12_26_bear_cross",
        sl_pct=0.015,
        tp_pct=0.030,
        rr_ratio=2.0,
        symbol="ETH/USDT",
    )
    assert 'strategy.entry("Long", strategy.long' in code
    assert 'strategy.entry("Short", strategy.short' in code
    assert "Direction: LONG+SHORT" in code


def test_position_sizing_configurable():
    """qty_pct controla el tamaño de posición."""
    code = generate_pine_script(
        strategy_name="rsi_14_oversold",
        sl_pct=0.015,
        tp_pct=0.030,
        rr_ratio=2.0,
        symbol="ETH/USDT",
        qty_pct=0.05,  # 5% equity por trade
    )
    assert "default_qty_value=5.0" in code
    assert "qty_percent=5.0" in code


def test_multiple_indicators_combined():
    """Combinación de 3+ indicadores se traduce correctamente."""
    code = generate_pine_script(
        strategy_name="rsi_14_oversold + adx_20_bull_trend + macd_12_26_bull_cross",
        sl_pct=0.015,
        tp_pct=0.030,
        rr_ratio=2.0,
        symbol="ETH/USDT",
    )
    assert "ta.rsi(close, 14)" in code
    assert "ta.dmi(20, 20)" in code
    assert "ta.macd(close, 12, 26, 9)" in code
    assert "cond_0" in code
    assert "cond_1" in code
    assert "cond_2" in code


def test_webhook_payload_included():
    """El código incluye template de webhook."""
    code = generate_pine_script(
        strategy_name="rsi_14_oversold",
        sl_pct=0.015,
        tp_pct=0.030,
        rr_ratio=2.0,
        symbol="ETH/USDT",
    )
    assert "Webhook payload" in code
    assert "strategy_id" in code
    assert "client_order_id" in code


def test_unknown_condition_fallback():
    """Condiciones no reconocidas se marcan como false."""
    code = generate_pine_script(
        strategy_name="unknown_indicator_xyz",
        sl_pct=0.015,
        tp_pct=0.030,
        rr_ratio=2.0,
        symbol="ETH/USDT",
    )
    assert "No traducido" in code
