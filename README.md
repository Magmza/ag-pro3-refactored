# 🚀 AG Pro 3.1 — Refactor v2

Reescritura completa del bot de trading con fixes críticos de ingeniería, risk management y backtest honesto.

## 📊 Estado

| Componente | Antes | Ahora |
|------------|-------|-------|
| Control de versiones | ❌ ninguno | ✅ `.gitignore` listo |
| Configuración | ❌ hardcoded en código | ✅ `.env` centralizado |
| Logging | ❌ `print()` disperso | ✅ `loguru` estructurado JSON |
| Bear features | ❌ calculadas, no usadas | ✅ pipelines long/short/both |
| Slippage | ❌ 5 bps fijo irreal | ✅ 3 modelos (fixed/atr/stochastic) |
| Risk management | ❌ inexistente | ✅ completo (DD/daily/kill/exposure/correlation) |
| Backtest honesto | ❌ solo 70/30 una vez | ✅ walk-forward + Monte Carlo + DSR |
| Benchmark B&H | ❌ últimos 500 días random | ✅ mismo periodo OOS |
| Position sizing | ❌ 100% equity | ✅ vol-targeting + risk-per-trade |
| Webhook | ❌ passphrase hardcoded | ✅ desde .env + idempotencia |
| Ejecución broker | ❌ mock comentado | ✅ CCXT real con reintentos |
| Tests | ❌ ninguno | ✅ 44/44 passing |

## 📁 Estructura

```
ag_pro3_refactored/
├── .gitignore
├── .env.example              # template — copiar a .env y rellenar
├── requirements.txt          # dependencias productivas + dev
├── README.md                 # este archivo
├── NEXT_STEPS.md             # plan FASE 2-5
├── backend/
│   ├── __init__.py
│   ├── config.py             # configuración centralizada (singleton)
│   ├── logger.py             # loguru estructurado
│   ├── data_loader.py        # CCXT + YFinance + cache local
│   ├── download_duka.py      # descarga Dukascopy genérica
│   ├── generator.py          # features LONG + SHORT (bug crítico arreglado)
│   ├── backtester.py         # slippage realista + benchmark B&H correcto
│   ├── walk_forward.py       # WF rolling + Monte Carlo + Deflated Sharpe
│   ├── risk_manager.py       # risk management completo
│   ├── pine_translator.py    # traductor a Pine v5 (soporta long+short)
│   └── main.py               # FastAPI con CCXT real + idempotencia
├── frontend/
│   ├── app.py                # Streamlit (NO migrado todavía)
│   └── style.css
├── tests/                    # 44 tests pytest
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_generator.py
│   ├── test_backtester.py
│   ├── test_risk_manager.py
│   ├── test_walk_forward.py
│   └── test_pine_translator.py
└── data/                     # cache local (gitignored)
```

## 🔧 Setup

```bash
# 1. Clonar / descomprimir
cd ag_pro3_refactored

# 2. Crear venv
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
copy .env.example .env     # Windows
# cp .env.example .env       # Linux/Mac
# Editar .env con tus claves

# 5. Generar passphrase segura para webhook
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Pegar el resultado en WEBHOOK_PASSPHRASE del .env

# 6. Correr tests (deben pasar 44/44)
pytest tests/ -v

# 7. Inicializar git
git init
git add .
git commit -m "AG Pro 3.1 refactor — FASE 0 + FASE 1 completas"

# 8. Crear repo en GitHub y subir
git remote add origin https://github.com/TU_USUARIO/ag-pro3.git
git branch -M main
git push -u origin main
```

## 🎯 Qué cambió y por qué

### Bug #1: Bear features nunca se usaban
**Antes:** `generator.py` calculaba 30+ features bear (RSI overbought, MACD bear cross...) pero en `generate_combinations_in_batches()` solo iteraba `bull_features.keys()`. El bot era **100% long-only**.

**Ahora:** Método `generate_combinations_in_batches(direction='long'|'short'|'both')`. Con `direction='both'` combina features con prefijo `L:` (long) y `S:` (short). El Pine translator detecta los prefijos y genera `strategy.entry("Long")` y `strategy.entry("Short")` según corresponda.

### Bug #2: Passphrase hardcodeada
**Antes:** `if signal.passphrase != "Pro31_Secret_2026"` — si subías el repo a GitHub, cualquiera podía mandarte órdenes.

**Ahora:** Passphrase desde `.env`, mínimo 32 caracteres en production. Comparación timing-safe con `hmac.compare_digest()`. Soporte opcional de firma HMAC-SHA256 en header `X-Signature`.

### Bug #3: Paths hardcodeados a Windows
**Antes:** `r"C:\Users\Mario\AppData\Roaming\Python\Python311\Scripts\duka.exe"` y `r"g:\Claude\Proyectos\Bot XAU Long"`.

**Ahora:** Todo configurable vía `.env`. `DUKA_PATH` se busca en .env o en PATH del sistema. Cache de datos en `./data/` (configurable con `DATA_DIR`).

### Bug #4: Slippage irreal
**Antes:** 5 bps fijos. Para ETH en sesión líquida OK, para XAG en sesión illiquid Subestimado 10x.

**Ahora:** 3 modelos:
- `fixed`: igual que antes (compatibilidad)
- `atr`: `base + α × ATR_pct` — mayor slippage en alta volatilidad
- `stochastic`: `atr + ruido gaussiano` — más realista aún

### Bug #5: Position sizing = 100% equity
**Antes:** Cada trade usaba el 100% del equity. Diez trades malos seguidos = -15% cuenta.

**Ahora:** Position sizing por:
- Volatilidad objetivo: `size = min(max_position, target_vol / asset_vol)`
- Riesgo fijo por trade: `unidades = (equity × risk_pct) / |entry - stop|`
- Default conservador: 10% equity por trade, max 25%, max exposición total 50%

### Bug #6: Backtest 70/30 una sola vez
**Antes:** Optimizabas en 70% del periodo, validabas en 30%. Una sola prueba. Con 230,000 combinaciones, por puro chance docenas pasan ambos filtros.

**Ahora:** 3 herramientas en `walk_forward.py`:
1. **Walk-forward rolling:** divide datos en N ventanas. Optimiza/valida en cada una. Promedia resultados OOS reales.
2. **Monte Carlo de trades:** reordena trades 10,000 veces. Devuelve percentil 5 (peor caso razonable) del drawdown y retorno. Calcula probabilidad de ruina.
3. **Deflated Sharpe Ratio (Bailey & López de Prado 2014):** ajusta Sharpe por número de pruebas múltiples. Te dice si tu mejor estrategia es estadísticamente real o producto del azar.

### Bug #7: Benchmark B&H incorrecto
**Antes:** `df_bench = loader.get_data(..., "1d", limit=500)` — descargaba últimos 500 días de diario sin importar el periodo OOS real. Comparabas tu estrategia intradiaria en ETH 1h contra ETH diario de otro periodo.

**Ahora:** `benchmark_buy_hold(is_oos=True)` calcula retorno y max DD de buy & hold en el **mismo periodo OOS exacto**.

### Bug #8: Sin risk management
**Antes:** Si perdías 50% en una semana, el bot seguía operando igual.

**Ahora:** `risk_manager.py` implementa:
- Kill switch hard (endpoint `/emergency_stop`)
- Max drawdown stop (default 15% → pausa 24h)
- Daily loss limit (default -3% → cierra todo el día)
- Max exposure (default 50% del equity)
- Max position size (default 25%)
- Correlation check (no abrir long ETH si ya hay long BTC)
- Duplicate symbol block

### Bug #9: main.py era mock
**Antes:** Código CCXT comentado. Endpoint devolvía "Orden procesada" sin hacer nada.

**Ahora:** `main.py` con:
- CCXT real (Binance Futuros testnet o production)
- Reintentos con backoff exponencial (DDoS, Network errors)
- Idempotencia con `client_order_id` (no duplicar si hay timeout)
- Integración con RiskManager antes de ejecutar
- Alertas Telegram
- Endpoints: `/`, `/status`, `/webhook/tradingview`, `/emergency_stop`

## 🧪 Tests

```bash
pytest tests/ -v
# 44 passed in 13.54s
```

Cobertura:
- `test_config.py` (5) — configuración
- `test_generator.py` (8) — features long/short/both, combinaciones
- `test_backtester.py` (7) — slippage models, métricas, B&H, shorts
- `test_risk_manager.py` (9) — kill switch, DD, daily loss, exposure, correlación, sizing
- `test_walk_forward.py` (7) — Monte Carlo, DSR, reproducibilidad
- `test_pine_translator.py` (8) — long, short, mixed, sizing, webhook payload

## ⚠️ Qué NO está hecho todavía

Ver `NEXT_STEPS.md` para el plan detallado de:
- **FASE 2:** Risk management avanzado (vol-target dinámico, regime detection, VaR/CVaR)
- **FASE 3:** Ejecución robusta (reconciliación, heartbeat, partial fills, OMS completo)
- **FASE 4:** Infraestructura (Docker, VPS, monitoring, alertas)
- **FASE 5:** Paper trading → live con capital mínimo

## 🎓 Lección brutal

Tu proyecto original era un **backtester de fuerza bruta con UI linda**, no un sistema de trading. Las 64 estrategias "ganadoras" que tenías guardadas eran todas:
- ETH/USDT long-only
- En el periodo 2023-2026 (uno de los mercados alcistas más fuertes de la historia)
- Con Profit Factors de 17.96 (matemáticamente casi imposible en trading real)
- Sin slippage realista, sin walk-forward, sin Monte Carlo

Cualquier estrategia long en ETH habría hecho eso. No descubriste alpha, descubriste beta disfrazado. Si corrías ese mismo escáner en ETH durante el bear market 2022 (de $4800 a $900), **ninguna de las 64 habría sobrevivido**.

Este refactor arregla los bugs técnicos. El resto depende de vos:
1. Corré el nuevo backtester con `slippage_model='atr'` y `direction='both'` en ETH durante 2022. Vas a ver la realidad.
2. Si tu estrategia no le gana al buy & hold del mismo periodo, no es estrategia, es forma cara de comprar y mantener.
3. Paper trading en Binance Testnet **mínimo 2 meses** antes de poner un peso.
4. El risk management es el 70% del éxito. La estrategia es el 30%.
