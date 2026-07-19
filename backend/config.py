"""
config.py — Carga centralizada de configuración desde .env

NUNCA usar os.getenv() disperso por el código. Siempre importar desde acá.
Esto garantiza que un solo lugar valide y exponga las variables.
"""
from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Cargar .env al importar este módulo (idempotente)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _get_bool(key: str, default: bool = False) -> bool:
    return _get(key, str(default)).lower() in ("1", "true", "yes", "on")


def _get_float(key: str, default: float) -> float:
    try:
        return float(_get(key, str(default)))
    except ValueError:
        return default


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


class Settings:
    """Configuración global de la aplicación. Singleton vía get_settings()."""

    def __init__(self) -> None:
        # ─── App ────────────────────────────────────────────────
        self.app_env: str = _get("APP_ENV", "development")
        self.app_host: str = _get("APP_HOST", "0.0.0.0")
        self.app_port: int = _get_int("APP_PORT", 8000)
        self.log_level: str = _get("LOG_LEVEL", "INFO")

        # ─── Webhook ────────────────────────────────────────────
        self.webhook_passphrase: str = _get("WEBHOOK_PASSPHRASE", "")
        if self.is_production and len(self.webhook_passphrase) < 32:
            raise RuntimeError(
                "WEBHOOK_PASSPHRASE debe tener >=32 caracteres en production. "
                "Generá uno con: python -c \"import secrets;print(secrets.token_urlsafe(32))\""
            )
        if not self.webhook_passphrase:
            # Dev mode: autogeneramos y avisamos
            self.webhook_passphrase = "DEV_ONLY_" + secrets.token_urlsafe(24)
            print(
                "[config] WEBHOOK_PASSPHRASE no configurado. "
                f"Usando passphrase efímera para dev: {self.webhook_passphrase[:16]}..."
            )

        # ─── Binance ────────────────────────────────────────────
        self.binance_api_key: str = _get("BINANCE_API_KEY", "")
        self.binance_api_secret: str = _get("BINANCE_API_SECRET", "")
        self.binance_testnet: bool = _get_bool("BINANCE_TESTNET", True)

        # ─── OANDA ──────────────────────────────────────────────
        self.oanda_api_key: str = _get("OANDA_API_KEY", "")
        self.oanda_account_id: str = _get("OANDA_ACCOUNT_ID", "")
        self.oanda_env: str = _get("OANDA_ENV", "practice")

        # ─── Telegram ───────────────────────────────────────────
        self.telegram_bot_token: str = _get("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id: str = _get("TELEGRAM_CHAT_ID", "")

        # ─── Risk Management ────────────────────────────────────
        self.risk_target_vol: float = _get_float("RISK_TARGET_VOL", 0.15)
        self.risk_max_position_pct: float = _get_float("RISK_MAX_POSITION_PCT", 0.25)
        self.risk_max_exposure_pct: float = _get_float("RISK_MAX_EXPOSURE_PCT", 0.50)
        self.risk_max_daily_loss_pct: float = _get_float("RISK_MAX_DAILY_LOSS_PCT", 0.03)
        self.risk_max_drawdown_pct: float = _get_float("RISK_MAX_DRAWDOWN_PCT", 0.15)
        self.risk_kill_switch: bool = _get_bool("RISK_KILL_SWITCH", False)

        # ─── Data ───────────────────────────────────────────────
        self.duka_path: str = _get("DUKA_PATH", "")
        self.data_dir: str = _get("DATA_DIR", "./data")

    # ─── Helpers ────────────────────────────────────────────────
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_path(self) -> Path:
        p = PROJECT_ROOT / self.data_dir if not Path(self.data_dir).is_absolute() else Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def has_binance_credentials(self) -> bool:
        return bool(self.binance_api_key and self.binance_api_secret)

    @property
    def has_oanda_credentials(self) -> bool:
        return bool(self.oanda_api_key and self.oanda_account_id)

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton de configuración. Usar SIEMPRE get_settings() en vez de Settings()."""
    return Settings()


# Instancia lista para importar
settings = get_settings()
