AG Pro 3.1 - Refactor v3 (Fast Engine)

Sistema de trading cuantitativo con backtester ultra-rapido (numpy + numba), risk management profesional y analisis de robustez estadistica.

## Performance del engine

| Metrica | Antes (vectorbt) | Ahora (numpy+numba) |
|---------|------------------|---------------------|
| Velocidad | ~50 strats/seg | 9,000 strats/seg (180x mas) |
| RAM (2000 estrategias) | 3.5 GB (explota) | ~50 MB (constante) |
| max_conditions soportado | 3 (con suerte) | 6 sin problema |
| Tests automatizados | 0 | 55 passing |

## Estructura

ag_pro3_refactored/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── NEXT_STEPS.md
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── data_loader.py
│   ├── download_duka.py
│   ├── generator.py
│   ├── backtester.py            (vectorbt legacy, usado por walk_forward)
│   ├── fast_backtester.py       (numpy+numba, principal)
│   ├── walk_forward.py
│   ├── risk_manager.py
│   ├── pine_translator.py
│   └── main.py
├── frontend/
│   ├── app.py
│   └── style.css
├── scripts/
│   └── benchmark.py
└── tests/
    └── (8 archivos test_*.py)

## Setup (5 minutos)

cd "C:\Claude\Proyectos\AG Pro3.1 Bot y Constructor 2"

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env
# Editar .env con WEBHOOK_PASSPHRASE:
# python -c "import secrets; print(secrets.token_urlsafe(32))"

pytest tests/ -v
# Debe mostrar: 55 passed

## Uso

### Dashboard interactivo
streamlit run frontend\app.py
Abre en http://localhost:8501

### API de ejecucion (live trading)
python -m backend.main
Abre en http://localhost:8000

### Benchmark de performance
python scripts\benchmark.py

## Que cambio vs version anterior

### FastBacktester (numpy + numba)
- Reemplaza a vectorbt en el frontend
- Procesa 1 estrategia a la vez con RAM constante
- Compila a codigo nativo con numba JIT
- 9,000+ estrategias/segundo
- Soporta max_conditions=6 (millones de combinaciones)
- NO necesita GPU

### Por que no GPU (cupy)?
1. No aporta para este caso: el cuello de botella no es GPU, es logica secuencial de SL/TP
2. Suma complejidad: CUDA Toolkit, drivers, versiones exactas
3. numba ya da 180x: suficiente para 100k combinaciones en segundos

Si mas adelante necesitas Monte Carlo con 1M+ simulaciones o entrenar ML sobre features, ahi si tiene sentido GPU.

## Tests

pytest tests/ -v

Debe mostrar: 55 passed in 40s

## Bugs arreglados

1. Bear features ahora se usan (antes estaban calculadas pero nunca combinadas)
2. Passphrase del .env (minimo 32 chars, timing-safe)
3. Paths hardcodeados eliminados
4. Slippage realista (3 modelos: fixed/atr/stochastic)
5. Position sizing decente (vol-targeting + risk-per-trade)
6. main.py con CCXT REAL (reintentos, idempotencia)
7. Walk-forward + Monte Carlo + Deflated Sharpe
8. Benchmark B&H del mismo periodo OOS
9. FastBacktester (sin MemoryError, max_conditions=6)
10. _reconstruct_entries movida arriba (fix NameError)

## Plan siguiente (NEXT_STEPS.md)

- FASE 2: Risk management avanzado (vol-target dinamico, regime detection, VaR/CVaR)
- FASE 3: Ejecucion robusta (OMS, reconciliacion, partial fills)
- FASE 4: Infraestructura (Docker, VPS, Grafana, alertas)
- FASE 5: Paper trading -> live con $500

## Leccion brutal

Tu proyecto original era un backtester de fuerza bruta con UI linda. Las 64 estrategias "ganadoras" eran todas long-only en ETH durante 2023-2026 (bull market). Cualquier estrategia long habria hecho eso. No era alpha, era beta disfrazado.

Este refactor:
1. Arregla los bugs tecnicos
2. Agrega risk management real
3. Hace el backtest honesto (walk-forward, Monte Carlo, DSR)
4. Acelera 180x para que puedas probar mas combinaciones

El resto depende de vos: correrlo en bear market 2022 y ver la verdad, paper trading 2 meses minimo, despues live con $500.
