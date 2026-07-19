"""
tests/test_generator.py — Tests para el generador de estrategias.
"""
import pandas as pd
import pytest

from backend.generator import StrategyGenerator


def test_generator_initialization(sample_ohlcv):
    """El generador debe inicializarse correctamente."""
    gen = StrategyGenerator(sample_ohlcv)
    assert gen.bull_features == {}
    assert gen.bear_features == {}


def test_calculate_all_features(sample_ohlcv):
    """Debe calcular todos los indicadores sin error."""
    gen = StrategyGenerator(sample_ohlcv)
    gen.calculate_all_features()

    assert len(gen.bull_features) > 20, "Debe generar >20 bull features"
    assert len(gen.bear_features) > 20, "Debe generar >20 bear features"
    assert len(gen.bull_features) == len(gen.bear_features), "Bull y bear deben tener misma cantidad"


def test_features_are_boolean_series(sample_ohlcv):
    """Cada feature debe ser una Serie booleana con mismo index que data."""
    gen = StrategyGenerator(sample_ohlcv)
    gen.calculate_all_features()

    for name, feat in gen.bull_features.items():
        assert isinstance(feat, pd.Series), f"{name} no es Series"
        assert feat.dtype == bool, f"{name} no es bool"
        assert len(feat) == len(sample_ohlcv), f"{name} longitud incorrecta"

    for name, feat in gen.bear_features.items():
        assert isinstance(feat, pd.Series), f"{name} no es Series"
        assert feat.dtype == bool, f"{name} no es bool"


def test_list_features(sample_ohlcv):
    """list_features debe devolver features correctos por dirección."""
    gen = StrategyGenerator(sample_ohlcv)
    gen.calculate_all_features()

    long_feats = gen.list_features("long")
    short_feats = gen.list_features("short")
    both_feats = gen.list_features("both")

    assert len(long_feats) == len(gen.bull_features)
    assert len(short_feats) == len(gen.bear_features)
    assert len(both_feats) == len(long_feats) + len(short_feats)


def test_generate_combinations_long(sample_ohlcv):
    """Genera combinaciones solo long."""
    gen = StrategyGenerator(sample_ohlcv)
    gen.calculate_all_features()

    batches = list(gen.generate_combinations_in_batches(
        direction="long",
        max_conditions=2,
        batch_size=10000,
    ))

    assert len(batches) >= 1
    df, total = batches[0]
    assert isinstance(df, pd.DataFrame)
    assert total > 0
    assert df.shape[1] > 0


def test_generate_combinations_short(sample_ohlcv):
    """Genera combinaciones solo short (regresión: en original no se usaban)."""
    gen = StrategyGenerator(sample_ohlcv)
    gen.calculate_all_features()

    batches = list(gen.generate_combinations_in_batches(
        direction="short",
        max_conditions=2,
        batch_size=10000,
    ))

    assert len(batches) >= 1
    df, _ = batches[0]
    # Todas las estrategias deben empezar con S: ... no, en short mode van sin prefijo
    cols = list(df.columns)
    assert all("L:" not in c for c in cols), "En modo short no debe haber features long"


def test_generate_combinations_both(sample_ohlcv):
    """Genera combinaciones long+short con prefijos."""
    gen = StrategyGenerator(sample_ohlcv)
    gen.calculate_all_features()

    batches = list(gen.generate_combinations_in_batches(
        direction="both",
        max_conditions=2,
        batch_size=10000,
    ))

    assert len(batches) >= 1
    df, _ = batches[0]
    cols = list(df.columns)
    # Debe haber al menos una estrategia que combine L: y S:
    has_mixed = any(("L:" in c and "S:" in c) for c in cols)
    assert has_mixed, "Debe generar combinaciones mixtas long+short"


def test_max_conditions_limits_combinations(sample_ohlcv):
    """max_conditions afecta el número total de combinaciones."""
    gen = StrategyGenerator(sample_ohlcv)
    gen.calculate_all_features()

    _, total_1 = next(gen.generate_combinations_in_batches(direction="long", max_conditions=1, batch_size=100000))
    _, total_2 = next(gen.generate_combinations_in_batches(direction="long", max_conditions=2, batch_size=100000))

    assert total_2 > total_1, "max_conditions=2 debe generar más combinaciones que 1"
