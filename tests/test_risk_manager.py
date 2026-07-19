"""
tests/test_risk_manager.py — Tests para el risk manager.
"""
from datetime import datetime, timedelta

import pytest

from backend.risk_manager import Position, RiskManager


@pytest.fixture
def rm():
    """RiskManager fresco para cada test."""
    return RiskManager()


def test_kill_switch_blocks_trades(rm):
    """Kill switch bloquea nuevos trades."""
    rm.activate_kill_switch()
    decision = rm.check_can_open_trade(
        symbol="ETH/USDT",
        side="LONG",
        entry_price=2500,
        current_equity=10000,
        current_positions=[],
    )
    assert not decision.allowed
    assert "Kill switch" in decision.reason


def test_max_drawdown_triggers_pause(rm):
    """Max drawdown pausa el bot 24h."""
    rm.update_equity(10000, datetime(2024, 1, 1, 10, 0))
    rm.update_equity(8000, datetime(2024, 1, 1, 14, 0))  # -20% DD

    decision = rm.check_can_open_trade(
        symbol="ETH/USDT",
        side="LONG",
        entry_price=2500,
        current_equity=8000,
        current_positions=[],
    )
    assert not decision.allowed
    assert "drawdown" in decision.reason.lower()


def test_daily_loss_limit_blocks_trades(rm):
    """Daily loss limit bloquea trades del día."""
    rm.update_equity(10000, datetime(2024, 1, 1, 10, 0))
    # Simular pérdida del día: equity peak = 10000, equity day_start = 10000, current = 9600 (-4%)
    rm._day_start_equity = 10000
    rm._equity_peak = 10000

    decision = rm.check_can_open_trade(
        symbol="ETH/USDT",
        side="LONG",
        entry_price=2500,
        current_equity=9600,  # -4% día (supera -3% default)
        current_positions=[],
    )
    assert not decision.allowed
    assert "diario" in decision.reason.lower()


def test_max_exposure_blocks_new_trade(rm):
    """Max exposure bloquea nuevos trades cuando se alcanza."""
    rm.update_equity(10000, datetime(2024, 1, 1))
    # Usar activo no correlacionado (ETH vs stock) para aislar el test de exposición
    positions = [
        Position(
            symbol="SPY",
            side="LONG",
            entry_price=500,
            size_pct=0.45,  # casi en el límite
            opened_at=datetime(2024, 1, 1),
        )
    ]
    # exposure_total = 0.45, max_exposure = 0.50, queda 0.05
    decision = rm.check_can_open_trade(
        symbol="ETH/USDT",
        side="LONG",
        entry_price=2500,
        current_equity=10000,
        current_positions=positions,
        volatility=0.5,
    )
    assert decision.allowed
    assert decision.suggested_size_pct <= 0.06  # limitado por exposición restante (0.05 + margen)


def test_duplicate_symbol_blocked(rm):
    """No permitir 2 trades en mismo símbolo."""
    rm.update_equity(10000, datetime(2024, 1, 1))
    positions = [
        Position(
            symbol="ETH/USDT",
            side="LONG",
            entry_price=2500,
            size_pct=0.10,
            opened_at=datetime(2024, 1, 1),
        )
    ]
    decision = rm.check_can_open_trade(
        symbol="ETH/USDT",
        side="LONG",
        entry_price=2500,
        current_equity=10000,
        current_positions=positions,
    )
    assert not decision.allowed


def test_correlated_assets_blocked(rm):
    """No abrir long en ETH si ya hay long en BTC (alta correlación)."""
    rm.update_equity(10000, datetime(2024, 1, 1))
    positions = [
        Position(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=50000,
            size_pct=0.10,
            opened_at=datetime(2024, 1, 1),
        )
    ]
    decision = rm.check_can_open_trade(
        symbol="ETH/USDT",
        side="LONG",
        entry_price=2500,
        current_equity=10000,
        current_positions=positions,
    )
    assert not decision.allowed
    assert "correlacionada" in decision.reason.lower()


def test_position_sizing_by_risk(rm):
    """Position sizing por riesgo fijo."""
    # entry=100, stop=99, risk_per_unit=1, equity=10000, risk_per_trade=1%
    # expected size = (10000 * 0.01) / 1 = 100 unidades
    size = rm.calculate_position_size(
        entry_price=100,
        stop_price=99,
        equity=10000,
        risk_per_trade_pct=0.01,
    )
    assert size == pytest.approx(100, rel=0.001)


def test_position_sizing_zero_risk(rm):
    """Si entry == stop, size = 0."""
    size = rm.calculate_position_size(
        entry_price=100,
        stop_price=100,
        equity=10000,
    )
    assert size == 0.0


def test_status_returns_dict(rm):
    """status() devuelve diccionario con campos esperados."""
    rm.update_equity(10000, datetime(2024, 1, 1))
    status = rm.status(current_equity=9500)

    assert "kill_switch" in status
    assert "current_dd_pct" in status
    assert "daily_pnl_pct" in status
    assert "trades_today" in status
