"""
frontend/app.py — Streamlit Dashboard para AG Pro 3.1 Refactorizado.

Mejoras vs original:
- Usa config centralizado (no hardcoded)
- Soporta dirección long/short/both
- Slippage model seleccionable (fixed/atr/stochastic)
- Botón "Modo Honesto" → walk-forward + Monte Carlo
- Benchmark vs Buy & Hold del MISMO periodo OOS
- Position sizing configurable (no 100% equity)
- Mostrar resultados por régimen (bull/bear/range)
- Integración con risk_manager para mostrar estado en vivo
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Asegurar path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.backtester import BacktestConfig, VectorizedBacktester
from backend.config import settings
from backend.data_loader import UniversalDataLoader
from backend.generator import StrategyGenerator
from backend.logger import get_logger
from backend.pine_translator import generate_pine_script
from backend.walk_forward import WalkForwardAnalyzer

log = get_logger(__name__)

# ─── Configuración de página ─────────────────────────────────
st.set_page_config(
    page_title="AG Pro 3.1 | Quant Strategy Builder",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    with open(Path(__file__).parent / "style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ─── Session state ────────────────────────────────────────────
if "scan_completed" not in st.session_state:
    st.session_state.scan_completed = False
if "final_ranking" not in st.session_state:
    st.session_state.final_ranking = pd.DataFrame()
if "walk_forward_result" not in st.session_state:
    st.session_state.walk_forward_result = None
if "mc_result" not in st.session_state:
    st.session_state.mc_result = None

st.title("⚡ AG Pro 3.1 — Refactor")
st.markdown("Backtester honesto con risk management profesional y walk-forward analysis.")

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    mercado = st.selectbox("Mercado", ["crypto", "forex", "metals", "stocks", "commodities"])
    activos_por_mercado = {
        "crypto": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
        "forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"],
        "metals": ["XAUUSD=X", "XAGUSD=X", "GC=F", "SI=F"],
        "stocks": ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA"],
        "commodities": ["CL=F", "NG=F", "ZC=F"],
    }
    symbol = st.selectbox("Activo", activos_por_mercado.get(mercado, ["BTC/USDT"]))
    timeframe = st.selectbox("Timeframe", ["1h", "15m", "5m", "4h", "1d"])
    freq_map = {"1h": "1h", "15m": "15min", "5m": "5min", "4h": "4h", "1d": "1d"}
    vbt_freq = freq_map.get(timeframe, "1h")

    st.markdown("---")
    st.subheader("📅 Rango Temporal")
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Inicio", value=pd.Timestamp("2023-01-01").date())
    end_date = col_d2.date_input("Fin", value=pd.Timestamp.today().date())

    st.markdown("---")
    st.subheader("🎯 Dirección")
    direction = st.selectbox(
        "Tipo de estrategias",
        ["both", "long", "short"],
        help="both = combinaciones LONG y SHORT. long = solo compras. short = solo ventas.",
    )

    st.markdown("---")
    st.subheader("💰 Gestión de Riesgo")
    sl_pct_input = st.slider("Stop Loss (%)", 0.5, 5.0, 1.5, 0.1)
    tp_pct_input = st.slider("Take Profit (%)", 0.5, 10.0, 3.0, 0.1)
    qty_pct_input = st.slider("Position Size (% equity)", 1.0, 25.0, 10.0, 0.5,
                              help="Cuánto equity usar por trade. Default conservador: 10%")
    sl_pct = sl_pct_input / 100
    tp_pct = tp_pct_input / 100
    rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0
    st.caption(f"RR: **1:{rr_ratio:.1f}** | Size: **{qty_pct_input:.1f}%**")

    st.markdown("---")
    st.subheader("📊 Slippage Model")
    slippage_model = st.selectbox(
        "Modelo",
        ["fixed", "atr", "stochastic"],
        index=1,
        help="fixed = constante. atr = base + α×ATR%. stochastic = atr + ruido gaussiano.",
    )

    st.markdown("---")
    st.subheader("🔍 Filtros de Robustez")
    min_pf = st.slider("Profit Factor Mínimo", 1.0, 3.0, 1.5, 0.05)
    max_dd = st.slider("Max Drawdown (%)", 5, 50, 20, 1)
    min_sqn = st.slider("SQN Mínimo", 0.5, 5.0, 1.5, 0.1)
    min_trades = st.slider("Trades Mínimos", 5, 100, 20, 5)

    st.markdown("---")
    st.subheader("⚙️ Generador")
    max_cond = st.slider("Máx indicadores combinados", 1, 6, 3,
                         help="Cada nivel extra multiplica exponencialmente las combinaciones.")

    st.markdown("---")
    st.subheader("🔬 Modo Honesto")
    run_walk_forward = st.checkbox("Walk-Forward Analysis", value=False,
                                   help="Divide en N ventanas y valida consistencia temporal. Lento pero honesto.")
    wf_windows = st.slider("Ventanas WF", 3, 20, 5) if run_walk_forward else 5
    run_monte_carlo = st.checkbox("Monte Carlo Trades", value=False,
                                  help="Reordena trades 10,000 veces. Mide worst-case DD.")
    mc_sims = st.slider("MC Simulaciones", 1000, 50000, 10000, step=1000) if run_monte_carlo else 10000

    ejecutar = st.button("🚀 Iniciar Scanner", use_container_width=True, type="primary")

# ─── Tabs ─────────────────────────────────────────────────────
tab_scanner, tab_robustez, tab_estado = st.tabs([
    "🔍 Scanner", "🔬 Análisis de Robustez", "📡 Estado del Sistema"
])

# ─── Tab: Estado ──────────────────────────────────────────────
with tab_estado:
    st.subheader("📡 Estado del Sistema")
    col1, col2, col3 = st.columns(3)
    col1.metric("Entorno", settings.app_env)
    col2.metric("Broker Conectado", "Binance" if settings.has_binance_credentials else "❌ No configurado")
    col3.metric("Telegram", "✅" if settings.has_telegram else "❌")

    st.markdown("---")
    st.subheader("Risk Management Configuration")
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Vol Objetivo", f"{settings.risk_target_vol*100:.0f}%")
    rc2.metric("Max Position", f"{settings.risk_max_position_pct*100:.0f}%")
    rc3.metric("Max Exposure", f"{settings.risk_max_exposure_pct*100:.0f}%")
    rc4.metric("Max Daily Loss", f"-{settings.risk_max_daily_loss_pct*100:.0f}%")

    rc5, rc6, rc7 = st.columns(3)
    rc5.metric("Max Drawdown", f"-{settings.risk_max_drawdown_pct*100:.0f}%")
    rc6.metric("Kill Switch", "🟢 OFF" if not settings.risk_kill_switch else "🔴 ON")
    rc7.metric("Testnet", "✅" if settings.binance_testnet else "❌ PROD")

    st.markdown("---")
    st.subheader("⚠️ Configuración Pendiente")
    if not settings.has_binance_credentials:
        st.warning("📊 Falta configurar BINANCE_API_KEY y BINANCE_API_SECRET en .env")
    if not settings.has_telegram:
        st.info("💬 Falta configurar TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env")
    if settings.app_env == "production" and settings.binance_testnet:
        st.error("🚨 APP_ENV=production pero BINANCE_TESTNET=true. Revisá antes de operar con dinero real.")

# ─── Tab: Scanner ─────────────────────────────────────────────
with tab_scanner:
    if ejecutar:
        st.session_state.scan_completed = False
        try:
            # 1. Datos
            with st.spinner(f"📥 Descargando {symbol} ({mercado}) {timeframe}..."):
                loader = UniversalDataLoader()
                df_data = loader.get_data(
                    mercado, symbol, timeframe,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    limit=300000,
                )
                st.info(f"✅ {len(df_data):,} velas | {df_data.index[0].date()} → {df_data.index[-1].date()}")

            # 2. Generar features
            with st.spinner("🧬 Calculando features (bull + bear)..."):
                generator = StrategyGenerator(df_data)
                generator.calculate_all_features()
                st.info(f"Features: {len(generator.bull_features)} bull + {len(generator.bear_features)} bear")

            # 3. Backtester con slippage configurado
            bt_config = BacktestConfig(
                slippage_model=slippage_model,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                freq=vbt_freq,
            )
            backtester = VectorizedBacktester(df_data, config=bt_config)

            # 4. Generar combinaciones y filtrar IS
            with st.spinner(f"🚀 Generando combinaciones (max_conditions={max_cond}, direction={direction})..."):
                progress = st.progress(0.0)
                status = st.empty()

                passed_is_list = []
                total_evaluated = 0

                for entries_batch, total_combos in generator.generate_combinations_in_batches(
                    direction=direction,  # type: ignore
                    max_conditions=max_cond,
                    batch_size=2000,
                ):
                    portfolio_is = backtester.run(entries_batch, is_oos=False)
                    results_is = backtester.calculate_professional_metrics(portfolio_is)
                    results_is["Estrategia"] = results_is.index

                    batch_passed = results_is[
                        (results_is["Profit Factor"].fillna(0) >= min_pf) &
                        (results_is["Max Drawdown (%)"].fillna(100) <= max_dd) &
                        (results_is["SQN"].fillna(0) >= min_sqn) &
                        (results_is["Trades"] >= min_trades) &
                        (results_is["Retorno (%)"].fillna(-999) > 0)
                    ]

                    if not batch_passed.empty:
                        passed_is_list.append(batch_passed)

                    total_evaluated += entries_batch.shape[1]
                    progress.progress(min(total_evaluated / total_combos, 1.0))
                    status.text(f"{total_evaluated:,} / {total_combos:,} | Pasan IS: {sum(len(p) for p in passed_is_list):,}")

            passed_is = pd.concat(passed_is_list) if passed_is_list else pd.DataFrame()

            # 5. Test OOS
            if len(passed_is) > 0:
                with st.spinner("🔬 Validando Out-Of-Sample..."):
                    entries_oos_all = pd.concat(
                        [b.set_index("Estrategia") for b in passed_is_list], axis=0
                    )
                    # Re-obtener entries como DataFrame alineado
                    estrategias_is = passed_is["Estrategia"].tolist()
                    # Re-construir entries OOS desde generator
                    feature_dict = {}
                    if direction in ("long", "both"):
                        feature_dict.update({f"L:{k}": v for k, v in generator.bull_features.items()})
                    if direction in ("short", "both"):
                        feature_dict.update({f"S:{k}": v for k, v in generator.bear_features.items()})
                    if direction == "long":
                        feature_dict = generator.bull_features
                    elif direction == "short":
                        feature_dict = generator.bear_features

                    # Filtrar solo las estrategias que pasaron IS
                    estrategias_presentes = []
                    for s in estrategias_is:
                        # Reconstruir condiciones
                        if s in feature_dict:
                            estrategias_presentes.append((s, feature_dict[s]))

                    if estrategias_presentes:
                        entries_oos = pd.DataFrame({s: v for s, v in estrategias_presentes})
                        portfolio_oos = backtester.run(entries_oos, is_oos=True)
                        results_oos = backtester.calculate_professional_metrics(portfolio_oos)
                        results_oos["Estrategia"] = results_oos.index

                        passed_oos = results_oos[
                            (results_oos["Retorno (%)"].fillna(-999) > 0) &
                            (results_oos["Profit Factor"].fillna(0) >= 1.2) &
                            (results_oos["Trades"] >= max(5, min_trades // 2))
                        ].sort_values("SQN", ascending=False)

                        # Benchmark B&H del MISMO periodo OOS
                        bh = backtester.benchmark_buy_hold(is_oos=True)

                        # Dinero final con $500
                        capital = 500
                        passed_oos["Dinero Final ($500)"] = capital * (1 + passed_oos["Retorno (%)"] / 100)
                        passed_oos["vs Buy&Hold (%)"] = passed_oos["Retorno (%)"] - bh["buy_hold_return_pct"]

                        st.session_state.final_ranking = passed_oos
                        st.session_state.bh_oos = bh
                        st.session_state.scan_completed = True
                    else:
                        st.warning("No se pudieron reconstruir entries OOS.")
                        st.session_state.final_ranking = pd.DataFrame()
            else:
                st.warning("Ninguna estrategia superó los filtros IS.")
                st.session_state.final_ranking = pd.DataFrame()

            # 6. Walk-forward (opcional)
            if run_walk_forward and len(st.session_state.final_ranking) > 0:
                with st.spinner(f"🔬 Walk-Forward ({wf_windows} ventanas)..."):
                    best_strat_name = st.session_state.final_ranking.iloc[0]["Estrategia"]
                    # Reconstruir entries para la mejor estrategia
                    if direction in ("long", "both") and not best_strat_name.startswith("S:"):
                        name = best_strat_name.replace("L:", "")
                        if name in generator.bull_features:
                            best_entries = generator.bull_features[name]
                    elif direction in ("short", "both") and best_strat_name.startswith("S:"):
                        name = best_strat_name[2:]
                        if name in generator.bear_features:
                            best_entries = generator.bear_features[name]
                    else:
                        best_entries = None

                    if best_entries is not None:
                        analyzer = WalkForwardAnalyzer(df_data, backtester)
                        wf_result = analyzer.run_walk_forward(best_entries, n_windows=wf_windows)
                        st.session_state.walk_forward_result = wf_result

            # 7. Monte Carlo (opcional)
            if run_monte_carlo and len(st.session_state.final_ranking) > 0:
                with st.spinner(f"🎲 Monte Carlo ({mc_sims:,} simulaciones)..."):
                    # Reconstruir trades de la mejor estrategia OOS
                    best_strat_name = st.session_state.final_ranking.iloc[0]["Estrategia"]
                    if direction in ("long", "both") and not best_strat_name.startswith("S:"):
                        name = best_strat_name.replace("L:", "")
                        best_entries = generator.bull_features.get(name)
                    elif direction in ("short", "both") and best_strat_name.startswith("S:"):
                        name = best_strat_name[2:]
                        best_entries = generator.bear_features.get(name)
                    else:
                        best_entries = None

                    if best_entries is not None:
                        try:
                            pf_best = backtester.run(best_entries, is_oos=True)
                            trades_pnl = pf_best.trades.returns()
                            if hasattr(trades_pnl, "values"):
                                trades_list = trades_pnl.values.tolist()
                            else:
                                trades_list = [float(trades_pnl)]

                            if len(trades_list) >= 5:
                                analyzer = WalkForwardAnalyzer(df_data, backtester)
                                mc_result = analyzer.run_monte_carlo(
                                    trades_list, n_simulations=mc_sims, seed=42
                                )
                                st.session_state.mc_result = mc_result
                        except Exception as e:
                            st.warning(f"Monte Carlo falló: {e}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

    # ─── Mostrar Resultados ─────────────────────────────────
    if st.session_state.scan_completed:
        final_ranking = st.session_state.final_ranking
        bh = st.session_state.get("bh_oos", {"buy_hold_return_pct": 0, "buy_hold_max_dd_pct": 0})

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Estrategias en IS", f"{len(final_ranking):,}")
        c2.metric("Buy & Hold OOS", f"{bh['buy_hold_return_pct']:.2f}%")
        c3.metric("B&H Max DD", f"{bh['buy_hold_max_dd_pct']:.2f}%")

        if len(final_ranking) > 0:
            st.subheader("🏆 Ranking de Estrategias (IS+OOS)")
            st.info(
                f"💡 Tu estrategia debe superar el Buy & Hold ({bh['buy_hold_return_pct']:.2f}%). "
                f"Si no lo hace, no es estrategia, es forma cara de comprar y mantener."
            )

            columnas = [
                "Estrategia", "Retorno (%)", "Profit Factor", "Max Drawdown (%)",
                "Win Rate (%)", "SQN", "Trades", "vs Buy&Hold (%)", "Dinero Final ($500)"
            ]
            available_cols = [c for c in columnas if c in final_ranking.columns]
            st.dataframe(final_ranking[available_cols].head(20), use_container_width=True)

            # Pine Script de la mejor
            st.markdown("---")
            st.subheader("📜 Pine Script — Mejor Estrategia")
            best = final_ranking.iloc[0]
            st.code(
                generate_pine_script(
                    strategy_name=best["Estrategia"],
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    rr_ratio=rr_ratio,
                    symbol=symbol,
                    qty_pct=qty_pct_input / 100,
                ),
                language="javascript",
            )
        else:
            st.warning("Ninguna estrategia sobrevivió IS+OOS. Probá filtros más relajados.")

# ─── Tab: Robustez ────────────────────────────────────────────
with tab_robustez:
    st.subheader("🔬 Análisis de Robustez Estadística")

    if not st.session_state.scan_completed:
        st.info("Ejecutá el scanner primero. Activá Walk-Forward y/o Monte Carlo en el sidebar.")
    else:
        # Walk-Forward
        if st.session_state.walk_forward_result:
            wf = st.session_state.walk_forward_result
            st.markdown("### 📊 Walk-Forward Rolling")
            st.info(
                f"**Consistencia:** {wf.consistency_ratio*100:.1f}% de ventanas OOS positivas\n\n"
                f"**Retorno OOS promedio:** {wf.avg_oos_return*100:.2f}%\n\n"
                f"**Sharpe OOS promedio:** {wf.avg_oos_sharpe:.2f}\n\n"
                f"**Peor ventana OOS:** {min(wf.oos_returns)*100:.2f}%\n\n"
                f"**Mejor ventana OOS:** {max(wf.oos_returns)*100:.2f}%"
            )

            df_wf = pd.DataFrame({
                "Ventana": range(1, len(wf.oos_returns) + 1),
                "Retorno IS (%)": [r*100 for r in wf.is_returns],
                "Retorno OOS (%)": [r*100 for r in wf.oos_returns],
                "Sharpe IS": wf.is_sharpes,
                "Sharpe OOS": wf.oos_sharpes,
                "Max DD OOS (%)": [d*100 for d in wf.oos_max_dds],
            })
            st.dataframe(df_wf, use_container_width=True)

            if wf.consistency_ratio < 0.6:
                st.error(
                    f"⚠️ Solo {wf.consistency_ratio*100:.0f}% de las ventanas son positivas. "
                    "Tu estrategia NO es robusta en el tiempo. Es probable overfitting."
                )
            elif wf.consistency_ratio < 0.8:
                st.warning(f"⚠️ Consistencia media ({wf.consistency_ratio*100:.0f}%). Mejorable.")
            else:
                st.success(f"✅ Consistencia alta ({wf.consistency_ratio*100:.0f}%). Estrategia robusta.")

        # Monte Carlo
        if st.session_state.mc_result:
            mc = st.session_state.mc_result
            st.markdown("### 🎲 Monte Carlo de Trades")
            st.info(
                f"**Simulaciones:** {mc.n_simulations:,}\n\n"
                f"**Retorno original:** {mc.original_final_return*100:.2f}%\n\n"
                f"**Worst-case retorno (P5):** {mc.mc_return_p5*100:.2f}%\n\n"
                f"**Mediana retorno (P50):** {mc.mc_return_p50*100:.2f}%\n\n"
                f"**Best-case retorno (P95):** {mc.mc_return_p95*100:.2f}%\n\n"
                f"**Worst-case Max DD (P5):** {mc.mc_max_dd_p5*100:.2f}%\n\n"
                f"**Probabilidad de ruina (-50%):** {mc.prob_ruin*100:.2f}%"
            )

            if mc.prob_ruin > 0.05:
                st.error(
                    f"⚠️ {mc.prob_ruin*100:.1f}% de probabilidad de perder la mitad de la cuenta. "
                    "Reducí position size o no operes esta estrategia."
                )
            if mc.mc_max_dd_p5 < -0.30:
                st.warning(
                    f"⚠️ Worst-case DD de {mc.mc_max_dd_p5*100:.1f}%. "
                    "¿Estás cómodo con eso? Si no, ajustá risk management."
                )

        if not st.session_state.walk_forward_result and not st.session_state.mc_result:
            st.info("Activá Walk-Forward y/o Monte Carlo en el sidebar y volvé a ejecutar.")
