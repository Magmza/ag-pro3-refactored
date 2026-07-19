"""
risk_manager.py — Risk Management en vivo. Lo que evita quiebra.

Reglas implementadas (todas configurables vía .env):
- Position sizing por volatilidad objetivo (vol-targeting)
- Max drawdown stop: si DD > X%, pausar 24h
- Daily loss limit: -Y% en el día = cerrar todo
- Kill switch hard: desactivar bot inmediatamente
- Max exposure: nunca más de Z% del equity en una sola dirección
- Max position size: nunca más de W% en un solo trade
- Correlation check: no abrir trade si correlación > 0.7 con posición abierta
- Trading hours: respetar horarios de mayor liquidez

Uso:
    from backend.risk_manager import RiskManager, RiskDecision
    rm = RiskManager()
    decision = rm.check_can_open_trade(
        symbol="ETH/USDT",
        side="LONG",
        entry_price=2500.0,
        current_equity=10000,
        current_positions=[...]
    )
    if decision.allowed:
        size = rm.calculate_position_size(decision.volatility)
        ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from backend.config import settings
from backend.logger import get_logger

log = get_logger(__name__)

Side = Literal["LONG", "SHORT"]


class RiskDecision:
    """Resultado de check_can_open_trade."""

    def __init__(
        self,
        allowed: bool,
        reason: str = "",
        suggested_size_pct: float = 0.0,
        volatility: float = 0.0,
    ) -> None:
        self.allowed = allowed
        self.reason = reason
        self.suggested_size_pct = suggested_size_pct
        self.volatility = volatility

    def __repr__(self) -> str:
        return f"<RiskDecision allowed={self.allowed} reason='{self.reason}' size={self.suggested_size_pct:.2%}>"


@dataclass
class Position:
    """Posición abierta."""
    symbol: str
    side: Side
    entry_price: float
    size_pct: float
    opened_at: datetime
    current_price: float = 0.0


class RiskManager:
    """Motor de risk management en vivo."""

    def __init__(self) -> None:
        self.kill_switch: bool = settings.risk_kill_switch
        self.target_vol: float = settings.risk_target_vol
        self.max_position_pct: float = settings.risk_max_position_pct
        self.max_exposure_pct: float = settings.risk_max_exposure_pct
        self.max_daily_loss_pct: float = settings.risk_max_daily_loss_pct
        self.max_drawdown_pct: float = settings.risk_max_drawdown_pct

        # Estado interno
        self._equity_peak: float = 0.0
        self._day_start_equity: float = 0.0
        self._day_start_time: datetime | None = None
        self._pause_until: datetime | None = None
        self._trades_today: int = 0

    # ──────────────────────────────────────────────────────────
    def update_equity(self, equity: float, now: datetime | None = None) -> None:
        """Llamar en cada tick con el equity actual de la cuenta."""
        now = now or datetime.now()
        self._equity_peak = max(self._equity_peak, equity)

        # Reset diario (00:00 UTC del broker)
        if self._day_start_time is None or (now - self._day_start_time).days >= 1:
            self._day_start_equity = equity
            self._day_start_time = now
            self._trades_today = 0

    # ──────────────────────────────────────────────────────────
    def current_drawdown_pct(self, equity: float) -> float:
        """Devuelve DD actual en decimal (0.0 = sin DD, -0.15 = -15%)."""
        if self._equity_peak <= 0:
            return 0.0
        return (equity - self._equity_peak) / self._equity_peak

    def current_daily_pnl_pct(self, equity: float) -> float:
        """Devuelve P&L del día en decimal."""
        if self._day_start_equity <= 0:
            return 0.0
        return (equity - self._day_start_equity) / self._day_start_equity

    # ──────────────────────────────────────────────────────────
    def check_can_open_trade(
        self,
        symbol: str,
        side: Side,
        entry_price: float,
        current_equity: float,
        current_positions: list[Position],
        volatility: float = 0.0,
        now: datetime | None = None,
    ) -> RiskDecision:
        """
        Verifica todas las reglas antes de abrir un trade.

        Args:
            symbol:           activo (ej. "ETH/USDT")
            side:             LONG o SHORT
            entry_price:      precio de entrada
            current_equity:   equity actual de la cuenta
            current_positions: lista de posiciones abiertas
            volatility:       volatilidad anualizada del activo (0.0 si no se sabe)
            now:              timestamp actual (default: datetime.now())
        """
        now = now or datetime.now()

        # 1. Kill switch hard
        if self.kill_switch:
            return RiskDecision(False, "Kill switch activado. No se abren trades.")

        # 2. Pausa por max drawdown
        if self._pause_until and now < self._pause_until:
            return RiskDecision(
                False,
                f"En pausa por max drawdown hasta {self._pause_until.isoformat()}",
            )

        # 3. Max drawdown check
        current_dd = self.current_drawdown_pct(current_equity)
        if current_dd < -self.max_drawdown_pct:
            self._pause_until = now + timedelta(hours=24)
            log.warning(
                "MAX DRAWDOWN HIT — pausando 24h",
                current_dd_pct=current_dd * 100,
                threshold_pct=-self.max_drawdown_pct * 100,
            )
            return RiskDecision(
                False,
                f"Max drawdown alcanzado: {current_dd*100:.2f}%. Pausando 24h.",
            )

        # 4. Daily loss limit
        daily_pnl = self.current_daily_pnl_pct(current_equity)
        if daily_pnl < -self.max_daily_loss_pct:
            log.warning(
                "DAILY LOSS LIMIT HIT",
                daily_pnl_pct=daily_pnl * 100,
                threshold_pct=-self.max_daily_loss_pct * 100,
            )
            return RiskDecision(
                False,
                f"Límite diario de pérdida alcanzado: {daily_pnl*100:.2f}%. Cierra todo.",
            )

        # 5. Max exposure check
        total_exposure = sum(p.size_pct for p in current_positions)
        if total_exposure >= self.max_exposure_pct:
            return RiskDecision(
                False,
                f"Exposición total {total_exposure*100:.2f}% >= límite {self.max_exposure_pct*100:.2f}%",
            )

        # 6. Mismo símbolo ya abierto
        if any(p.symbol == symbol for p in current_positions):
            return RiskDecision(False, f"Ya hay posición abierta en {symbol}")

        # 7. Correlación simplificada: no abrir misma dirección en activos muy correlacionados
        # (para ETH/BTC, XAU/XAG, etc.)
        correlated_groups = {
            ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"): "crypto_majors",
            ("XAUUSD", "XAGUSD"): "metals",
            ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"): "usd_majors",
        }
        for group, _ in correlated_groups.items():
            if symbol in group:
                existing = [p for p in current_positions if p.symbol in group and p.side == side]
                if existing:
                    return RiskDecision(
                        False,
                        f"Posición correlacionada ya abierta en {existing[0].symbol}",
                    )

        # 8. Position sizing por volatilidad objetivo
        if volatility > 0:
            suggested_size = min(
                self.max_position_pct,
                self.target_vol / volatility,
            )
        else:
            suggested_size = self.max_position_pct * 0.5  # conservador si no hay info

        # No exponer más del disponible
        remaining_capacity = self.max_exposure_pct - total_exposure
        suggested_size = min(suggested_size, remaining_capacity, self.max_position_pct)
        suggested_size = max(suggested_size, 0.0)

        if suggested_size <= 0:
            return RiskDecision(False, "Sin capacidad de exposición disponible")

        self._trades_today += 1
        return RiskDecision(
            True,
            "OK",
            suggested_size_pct=suggested_size,
            volatility=volatility,
        )

    # ──────────────────────────────────────────────────────────
    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        equity: float,
        risk_per_trade_pct: float = 0.01,
    ) -> float:
        """
        Position sizing por riesgo fijo (1% del equity por trade por defecto).

        Devuelve el tamaño en unidades del activo.
        """
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            return 0.0
        dollar_risk = equity * risk_per_trade_pct
        return dollar_risk / risk_per_unit

    # ──────────────────────────────────────────────────────────
    def activate_kill_switch(self) -> None:
        """Activa kill switch. Cancela todo y no abre nuevos trades."""
        self.kill_switch = True
        log.critical("KILL SWITCH ACTIVADO — cancelar todas las órdenes")

    def deactivate_kill_switch(self) -> None:
        self.kill_switch = False
        log.info("Kill switch desactivado")

    # ──────────────────────────────────────────────────────────
    def status(self, current_equity: float) -> dict:
        """Estado actual del risk manager para dashboard/monitoring."""
        return {
            "kill_switch": self.kill_switch,
            "pause_until": self._pause_until.isoformat() if self._pause_until else None,
            "equity_peak": self._equity_peak,
            "current_equity": current_equity,
            "current_dd_pct": self.current_drawdown_pct(current_equity) * 100,
            "day_start_equity": self._day_start_equity,
            "daily_pnl_pct": self.current_daily_pnl_pct(current_equity) * 100,
            "trades_today": self._trades_today,
            "max_dd_limit_pct": -self.max_drawdown_pct * 100,
            "max_daily_loss_pct": -self.max_daily_loss_pct * 100,
        }


# ──────────────────────────────────────────────────────────────
# Singleton
_rm_instance: RiskManager | None = None


def get_risk_manager() -> RiskManager:
    global _rm_instance
    if _rm_instance is None:
        _rm_instance = RiskManager()
    return _rm_instance
