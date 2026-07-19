# 📋 NEXT_STEPS — Plan FASE 2 a FASE 5

La FASE 0 (higiene) y FASE 1 (backtest honesto) están hechas. Esto es lo que falta para llegar a dinero real.

---

## FASE 2 — Risk Management Avanzado (2-3 semanas)

### 2.1 Vol-targeting dinámico
- [ ] Calcular volatilidad realizada (20D, 60D) del portafolio completo
- [ ] Ajustar `target_vol` según régimen (alta vol → reducir size, baja vol → aumentar)
- [ ] Implementar en `risk_manager.py` método `update_target_vol()`

### 2.2 Regime Detection
- [ ] Clasificar mercado en bull / bear / range usando:
  - EMA 50 vs EMA 200
  - ADX > 25
  - Volatilidad realizada vs histórica
- [ ] Cada estrategia debe declarar en qué régimen funciona
- [ ] Apagar estrategias cuyo régimen no coincide con el actual

### 2.3 VaR y CVaR
- [ ] Value at Risk 95% y 99% (1 día, 7 días)
- [ ] Conditional VaR (Expected Shortfall)
- [ ] Stress test: qué pasa si el mercado cae 10% en un día

### 2.4 Correlación dinámica real
- [ ] Calcular matriz de correlación rolling 30D entre activos operados
- [ ] Bloquear trades si correlación realizada > 0.7 con posición abierta
- [ ] Reemplazar el hardcoded `correlated_groups` actual

---

## FASE 3 — Ejecución Robusta (3-4 semanas)

### 3.1 Order Management System (OMS)
Crear `backend/oms.py` con:
- [ ] Clase `Order` con estados: PENDING → SUBMITTED → PARTIAL → FILLED / CANCELED / REJECTED
- [ ] Clase `Position` con PnL no realizado, exposición, margen usado
- [ ] Clase `Fill` con timestamp, slippage real, comisión
- [ ] Event log: cada cambio de estado se loguea con timestamp

### 3.2 Reconciliación
- [ ] Cada 60s comparar estado interno vs broker
- [ ] Si discrepancia → alerta Telegram + reconciliar desde broker
- [ ] Log de discrepancias para auditoría

### 3.3 Reconexión automática
- [ ] WebSocket con reconexión exponencial (1s, 2s, 4s, 8s, max 60s)
- [ ] En reconexión: re-suscribir a todos los canales
- [ ] Heartbeat cada 30s. Si no hay heartbeat en 2 min → alerta crítica

### 3.4 Manejo de edge cases
- [ ] Partial fills: dividir orden en N sub-ordenes
- [ ] Rejected orders: log + reintentar con parámetros ajustados
- [ ] Margin call: cerrar posición más perdedora automáticamente
- [ ] Funding rate (Binance Futuros): contabilizar en PnL

### 3.5 Order types avanzados
- [ ] Limit orders con price improvement
- [ ] TWAP para entradas grandes
- [ ] Iceberg orders (visibilidad parcial)
- [ ] Trailing stop loss dinámico

---

## FASE 4 — Infraestructura Production (2-3 semanas)

### 4.1 Containerización
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] `Dockerfile` multi-stage (build + runtime)
- [ ] `docker-compose.yml` con: app, redis (cache), postgres (trades), grafana, prometheus
- [ ] Volumes para datos y logs
- [ ] Healthcheck en cada servicio

### 4.2 VPS Deployment
- [ ] VPS cercano al broker:
  - Binance: AWS Tokyo (ap-northeast-1) o Frankfurt (eu-central-1)
  - OANDA: AWS us-east-1 (Virginia)
- [ ] Mínimo 4 vCPU, 8GB RAM, 80GB SSD
- [ ] firewall: solo puertos 22 (SSH), 8000 (API)
- [ ] SSH key only, no password

### 4.3 Systemd / Supervisor
```ini
# /etc/systemd/system/agpro3.service
[Unit]
Description=AG Pro 3.1 Trading Bot
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/opt/agpro3
ExecStart=/opt/agpro3/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

- [ ] Reinicio automático si crashea
- [ ] Log rotation
- [ ] Notificación si servicio cae

### 4.4 Monitoring
- [ ] **Prometheus** metrics:
  - `trades_total`, `trades_filled`, `trades_rejected`
  - `pnl_daily`, `pnl_realized`, `pnl_unrealized`
  - `latency_webhook_ms`, `latency_order_ms`
  - `equity`, `exposure_pct`, `drawdown_pct`
- [ ] **Grafana** dashboards:
  - Dashboard principal: equity curve, daily PnL, exposure, # trades
  - Dashboard de salud: latencia, errores, rate limits, reconexiones
- [ ] **Alertas**:
  - PnL diario < -3% → Telegram crítico
  - Latency > 500ms → Telegram warning
  - Reconexiones > 5/hora → Telegram warning
  - Equity < 80% del peak → Telegram crítico

### 4.5 Logging Centralizado
- [ ] Loguru → JSON a archivo local (ya hecho)
- [ ] Shipper a Loki o Better Stack
- [ ] Retención 90 días mínimo
- [ ] Búsqueda por `strategy_id`, `symbol`, `trace_id`

### 4.6 Backup
- [ ] DB backup diario automático (cron + pg_dump)
- [ ] Backup off-site (S3, Backblaze B2)
- [ ] Test de restore mensual
- [ ] Backup de `.env` en gestor de contraseñas (1Password, Bitwarden)

---

## FASE 5 — Paper Trading → Live (3-6 meses)

### 5.1 Paper Trading (mínimo 2 meses)
- [ ] Binance Testnet configurado (`BINANCE_TESTNET=true`)
- [ ] Mismo código que producción (solo cambia el endpoint)
- [ ] Mismos parámetros: SL, TP, position sizing, risk limits
- [ ] Daily journal: comparar live vs backtest esperado
- [ ] Semanalmente: revisar brecha backtest vs paper

**Criterios de aprobación para pasar a live:**
- ✅ Sharpe paper > 0.5 (mínimo)
- ✅ Max DD paper < 1.5x backtest DD
- ✅ Slippage real < 2x backtest slippage
- ✅ 0 bugs críticos en 60 días
- ✅ Reconexiones funcionan bajo test de corte de red

### 5.2 Live con capital mínimo (3 meses)
- [ ] Empezar con **$500** (cantidad que estés 100% dispuesto a perder)
- [ ] 1 sola estrategia, 1 solo activo, 1 solo timeframe
- [ ] `APP_ENV=production` en `.env`
- [ ] `BINANCE_TESTNET=false` en `.env`
- [ ] Position size máximo: 5% equity por trade (conservador)
- [ ] Risk limits estrictos:
  - Max DD: 10% (más conservador que paper)
  - Daily loss: 2%
  - Kill switch siempre armado

### 5.3 Tracking y análisis
- [ ] Diario de trading (Notion, Obsidian o spreadsheet):
  - Cada trade: razón, contexto de mercado, resultado, aprendizaje
  - Cada error: qué pasó, por qué, cómo evitarlo
  - Semanal: métricas vs paper trading
- [ ] Comparar live vs backtest semanalmente:
  - Si Sharpe live < 0.5x backtest → investigar slippage/latencia/bugs
  - Si DD live > 1.5x backtest → reducir size
  - Si correlación live vs backtest < 0.7 → algo roto, pausar

### 5.4 Scaling
- [ ] Solo escalar capital después de 3 meses verdes con Sharpe > 0.7
- [ ] Duplicar capital gradualmente (no multiplicar por 10)
- [ ] Agregar 1 activo nuevo a la vez (no 5)
- [ ] Agregar 1 estrategia nueva a la vez
- [ ] Cada 3 meses: review completo de estrategia y risk

---

## 🎯 Cronograma realista

| Fase | Duración | Costo | Resultado |
|------|----------|-------|-----------|
| FASE 0 + 1 | ✅ Hecho | $0 | Código limpio + backtest honesto |
| FASE 2 | 2-3 sem | $0 | Risk management profesional |
| FASE 3 | 3-4 sem | $0 | Ejecución robusta |
| FASE 4 | 2-3 sem | $20-50/mes VPS | Infraestructura production |
| FASE 5 paper | 2 meses | $0 | Validación real |
| FASE 5 live | 3+ meses | $500 capital | Live trading |
| **Total** | **8-12 meses** | **~$500-1000** | **Sistema productivo** |

## 💰 Expectativas

**Si hacés todo bien:**
- Sharpe 0.7-1.2 en live
- 15-25% retorno anual con 10-15% max DD
- Para sacar $1000/mes necesitás $50-80k de capital operando al 15-25% anual

**Si apurás o saltas fases:**
- 80% probabilidad de perder el capital
- Lo vas a racionalizar como "lección pagada" pero va a ser falta de disciplina

**El cuento de "bot que te hace rico con $500" es mentira.** Los bots profesionales manejan $100k+ y rinden 15-25% anual. Esa es la realidad.

---

## 🚨 Reglas de oro (pegarlas en la pared)

1. **Risk management es el 70% del éxito.** La estrategia es el 30%.
2. **Sharpe > 3 en backtest = overfit.** Dudá de cualquier número demasiado bueno.
3. **Backtest ≠ live.** Slippage + latency + partial fills cortan 30-50% del retorno backtesteado.
4. **Si el bot corre en tu PC, NO es producción.**
5. **Si no podés explicar por qué funciona la estrategia, no la operes.**
6. **Una estrategia mediocre con buen RM sobrevive. Una brillante sin RM revienta.**
7. **Si necesitás sacar el dinero del bot para pagar gastos, NO es momento de operar bots.**
8. **Paper trading mínimo 2 meses. Sin excepciones.**
9. **Si DD > 15% en live, pausar 1 semana. Investigar antes de seguir.**
10. **El kill switch es tu mejor amigo. Tenelo siempre a mano.**
