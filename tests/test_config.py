"""
tests/test_config.py — Tests para la configuración.
"""
import os
import pytest
from pathlib import Path


def test_settings_singleton():
    """get_settings devuelve siempre la misma instancia."""
    from backend.config import get_settings
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_settings_has_all_fields():
    """Settings debe tener todos los campos esperados."""
    from backend.config import settings
    assert hasattr(settings, "app_env")
    assert hasattr(settings, "webhook_passphrase")
    assert hasattr(settings, "binance_api_key")
    assert hasattr(settings, "risk_target_vol")
    assert hasattr(settings, "risk_max_drawdown_pct")
    assert hasattr(settings, "data_dir")


def test_settings_defaults():
    """Defaults son razonables."""
    from backend.config import settings
    assert 0 < settings.risk_target_vol < 1
    assert 0 < settings.risk_max_position_pct <= 1
    assert 0 < settings.risk_max_exposure_pct <= 1
    assert settings.risk_max_daily_loss_pct > 0


def test_data_path_creates_directory(tmp_path):
    """data_path crea el directorio si no existe."""
    from backend.config import settings
    p = settings.data_path
    assert p.exists()
    assert p.is_dir()


def test_helpers_exist():
    """Helper properties funcionan."""
    from backend.config import settings
    # is_production debe ser bool
    assert isinstance(settings.is_production, bool)
    assert isinstance(settings.has_binance_credentials, bool)
    assert isinstance(settings.has_telegram, bool)
