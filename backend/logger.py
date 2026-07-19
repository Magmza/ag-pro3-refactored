"""
logger.py — Logging estructurado con loguru + structlog.

Uso:
    from backend.logger import get_logger
    log = get_logger(__name__)
    log.info("mensaje", extra={"key": "value"})
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from backend.config import settings

_LOGS_DIR = settings.project_root / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Remover handler default
logger.remove()

# ─── Consola (color, formato breve) ───────────────────────────
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
    backtrace=True,
    diagnose=settings.app_env != "production",
)

# ─── Archivo rotativo diario (JSON para machine-parsing) ─────
logger.add(
    _LOGS_DIR / "app_{time:YYYY-MM-DD}.jsonl",
    level="DEBUG",
    rotation="00:00",          # rota a medianoche
    retention="30 days",       # mantiene 30 días
    compression="zip",
    serialize=True,            # JSON estructurado
    backtrace=True,
    diagnose=False,            # nunca mostrar variables en prod (security)
)

# ─── Archivo de errores (separate) ───────────────────────────
logger.add(
    _LOGS_DIR / "errors_{time:YYYY-MM-DD}.log",
    level="ERROR",
    rotation="00:00",
    retention="90 days",
    compression="zip",
    backtrace=True,
    diagnose=settings.app_env != "production",
)


def get_logger(name: str = __name__):
    """Retorna un logger enlazado al módulo que lo invoca."""
    return logger.bind(module=name)


# Exportar también la instancia raíz por si se quiere usar directo
log = get_logger("ag_pro3")
