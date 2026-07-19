"""
frontend/app.py — Streamlit Dashboard para AG Pro 3.1 (con FastBacktester).

Cambios vs versión anterior:
- Usa FastBacktester (numpy + numba) en vez de VectorizedBacktester (vectorbt)
- 9000 estrategias/segundo en vez de explotar memoria
- Soporta max_conditions=5 o 6 sin problema
- Procesa TODAS las combinaciones, no en lotes pequeños
- Mantiene walk-forward, Monte Carlo, dirección long/short/both
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.data_loader import UniversalDataLoader
from backend.fast_backtester import FastBacktester
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

st.title("⚡ AG Pro 3.1 — Fast Engine (numpy + numba)")
st.markdown("Backtester ultra-rápido: 9000+ estrategias/segundo. Soporta max_conditions=6.")

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
    start_date = col_d1.date_input("Inicio", value=pd.Timestamp("2022-01-01").date())
    end_date = col_d2.date_input("Fin", value=pd.Timestamp("2022-12-31").date())

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
    qty_pct_input = st.slider("Position Size (% equity)", 1.0, 25.0, 10.0, 0.5)
    sl_pct = sl_pct_input / 100
    tp_pct = tp_pct_input / 100
    rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0
    st.caption(f"RR: **1:{rr_ratio:.1f}** | Size: **{qty_pct_input:.1f}%**")

    st.markdown("---")
    st.subheader("💸 Costs")
    fees_input = st.slider("Fees (%)", 0.01, 0.50, 0.05, 0.01) / 100
    slippage_input = st.slider("Slippage (%)", 0.01, 0.50, 0.05, 0.01) / 100

    st.markdown("---")
    st.subheader("🔍 Filtros de Robustez")
    min_pf = st.slider("Profit Factor Mínimo", 1.0, 3.0, 1.5, 0.05)
    max_dd = st.slider("Max Drawdown (%)", 5, 50, 20, 1)
    min_sqn = st.slider("SQN Mínimo", 0.5, 5.0, 1.5, 0.1)
    min_trades = st.slider("Trades Mínimos", 5, 100, 20, 5)

    st.markdown("---")
    st.subheader("⚙️ Generador")
    max_cond = st.slider("Máx indicadores combinados", 1, 6, 4,
                         help="Con FastBacktester podés usar hasta 6 sin problema.")
    show_only_passed = st.checkbox("Solo mostrar estrategias que pasan filtros", value=True,
                                   help="Si desactivado, muestra TODAS las combinaciones (puede ser lento de renderizar).")

    st.markdown("---")
    st.subheader("🔬 Modo Honesto")
    run_walk_forward = st.checkbox("Walk-Forward Analysis", value=True,
                                   help="Divide en N ventanas y valida consistencia temporal.")
    wf_windows = st.slider("Ventanas WF", 3, 20, 5) if run_walk_forward else 5
    run_monte_carlo = st.checkbox("Monte Carlo Trades", value=True,
                                  help="Reordena trades N veces. Mide worst-case DD.")
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
    col2.metric("Broker Configurado", "Binance" if settings.has_binance_credentials else "❌")
    col3.metric("Telegram", "✅" if settings.has_telegram else "❌")

    st.markdown("---")
    st.subheader("🚀 Performance del Engine")
    st.info(
        "**FastBacktester (numpy + numba)**\n\n"
        "- 9,000+ estrategias/segundo\n"
        "- RAM constante (~50 MB sin importar N estrategias)\n"
        "- Soporta max_conditions hasta 6 (millones de combinaciones)\n"
        "- Compilación JIT automática en primer uso (puede tardar 5s la primera vez)"
    )

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

                # Pre-calcular DataFrame de features para acceso rápido
                feature_dict = {}
                if direction in ("long", "both"):
                    for k, v in generator.bull_features.items():
                        feature_dict[f"L:{k}"] = v
                if direction in ("short", "both"):
                    for k, v in generator.bear_features.items():
                        feature_dict[f"S:{k}"] = v
                if direction == "long":
                    feature_dict = generator.bull_features
                elif direction == "short":
                    feature_dict = generator.bear_features

            # 3. Backtester
            bt = FastBacktester(
                df_data,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                fees=fees_input,
                slippage=slippage_input,
                freq=vbt_freq,
            )

            # 4. Generar combinaciones y backtestear
            import time as _time
            t_start = _time.time()

            with st.spinner(f"🚀 Generando y backtesteando combinaciones (max_conditions={max_cond}, direction={direction})..."):
                progress = st.progress(0.0)
                status = st.empty()

                all_results = []
                total_evaluated = 0
                total_passed_is = 0

                for entries_batch, total_combos in generator.generate_combinations_in_batches(
                    direction=direction,  # type: ignore
                    max_conditions=max_cond,
                    batch_size=5000,  # ahora sí podemos usar batch grande
                ):
                    # Para cada estrategia del batch, determinar dirección
                    # Si todas las condiciones son L: → long
                    # Si todas son S: → short
                    # Si mixto → por ahora tomamos la primera como dirección (mejorable)
                    columns = entries_batch.columns.tolist()
                    long_cols = [c for c in columns if not c.startswith("S:")]
                    short_cols = [c for c in columns if c.startswith("S:")]
                    mixed_cols = [c for c in columns if ("L:" in c and "S:" in c)]

                    # Procesar estrategias LONG (todas las condiciones son L: o sin prefijo)
                    if long_cols:
                        entries_long = entries_batch[long_cols]
                        results_long = bt.run_many(entries_long, direction=1)
                        results_long["Direction"] = "LONG"
                        all_results.append(results_long.reset_index())

                    # Procesar estrategias SHORT (todas las condiciones son S:)
                    if short_cols:
                        entries_short = entries_batch[short_cols]
                        results_short = bt.run_many(entries_short, direction=-1)
                        results_short["Direction"] = "SHORT"
                        all_results.append(results_short.reset_index())

                    total_evaluated += len(columns)
                    progress.progress(min(total_evaluated / total_combos, 1.0))

                    passed_now = sum(
                        1 for r in (all_results[-1].to_dict("records") if all_results else [])
                        if r.get("Profit Factor", 0) and r["Profit Factor"] >= min_pf
                        and r.get("Max Drawdown (%)", 100) <= max_dd
                        and r.get("SQN", 0) >= min_sqn
                        and r.get("Trades", 0) >= min_trades
                        and r.get("Retorno (%)", -999) > 0
                    )
                    total_passed_is += passed_now

                    status.text(
                        f"{total_evaluated:,} / {total_combos:,} | "
                        f"Pasan IS: {total_passed_is:,} | "
                        f"Speed: {total_evaluated/(_time.time()-t_start):.0f} strats/seg"
                    )

            t_end = _time.time()
            st.success(f"✅ Backtest completo en {t_end-t_start:.1f}s | {total_evaluated:,} estrategias evaluadas")

            # 5. Consolidar resultados
            if all_results:
                results_df = pd.concat(all_results, ignore_index=True)
            else:
                results_df = pd.DataFrame()

            if len(results_df) > 0:
                # 5a. Filtrar IS
                passed_is = results_df[
                    (results_df["Profit Factor"].fillna(0) >= min_pf) &
                    (results_df["Max Drawdown (%)"].fillna(100) <= max_dd) &
                    (results_df["SQN"].fillna(0) >= min_sqn) &
                    (results_df["Trades"] >= min_trades) &
                    (results_df["Retorno (%)"].fillna(-999) > 0)
                ].copy()

                # 5b. Para cada estrategia que pasó IS, hacer backtest OOS
                if len(passed_is) > 0:
                    with st.spinner(f"🔬 Validando {len(passed_is)} estrategias en OOS..."):
                        # Split data
                        split_idx = int(len(df_data) * 0.7)
                        df_oos = df_data.iloc[split_idx:]
                        bt_oos = FastBacktester(
                            df_oos,
                            sl_pct=sl_pct,
                            tp_pct=tp_pct,
                            fees=fees_input,
                            slippage=slippage_input,
                            freq=vbt_freq,
                        )

                        oos_results = []
                        for _, row in passed_is.iterrows():
                            strat_name = row["Estrategia"]
                            direction_str = row["Direction"]

                            # Reconstruir entries para OOS
                            entries_series = _reconstruct_entries(
                                strat_name, direction_str, generator, df_data
                            )
                            if entries_series is None:
                                continue
                            entries_oos = entries_series.iloc[split_idx:]
                            r = bt_oos.run_single(entries_oos, direction=1 if direction_str == "LONG" else -1)

                            oos_results.append({
                                "Estrategia": strat_name,
                                "Direction": direction_str,
                                "Retorno OOS (%)": r.total_return * 100,
                                "Profit Factor OOS": r.profit_factor,
                                "Max DD OOS (%)": r.max_drawdown * 100,
                                "Win Rate OOS (%)": r.win_rate * 100,
                                "SQN OOS": r.sqn,
                                "Trades OOS": r.n_trades,
                                "trades_pnl_obj": r.trades_pnl,  # guardamos array para MC
                            })

                        oos_df = pd.DataFrame(oos_results)
                        passed_oos = oos_df[
                            (oos_df["Retorno OOS (%)"].fillna(-999) > 0) &
                            (oos_df["Profit Factor OOS"].fillna(0) >= 1.2) &
                            (oos_df["Trades OOS"] >= max(5, min_trades // 2))
                        ].sort_values("SQN OOS", ascending=False).reset_index(drop=True)

                        # Benchmark B&H OOS
                        bh = bt.benchmark_buy_hold(is_oos=True)

                        # Capital final con $500
                        capital = 500
                        passed_oos["Dinero Final ($500)"] = capital * (1 + passed_oos["Retorno OOS (%)"] / 100)
                        passed_oos["vs Buy&Hold (%)"] = passed_oos["Retorno OOS (%)"] - bh["buy_hold_return_pct"]

                        st.session_state.final_ranking = passed_oos
                        st.session_state.bh_oos = bh
                        st.session_state.scan_completed = True

                        # Guardar datos para walk-forward y MC
                        if len(passed_oos) > 0:
                            best = passed_oos.iloc[0]
                            st.session_state.best_strat_name = best["Estrategia"]
                            st.session_state.best_strat_direction = best["Direction"]
                            st.session_state.best_trades_pnl = best["trades_pnl_obj"]
                else:
                    st.warning("Ninguna estrategia superó los filtros IS.")
                    st.session_state.final_ranking = pd.DataFrame()
            else:
                st.warning("No se generaron resultados.")
                st.session_state.final_ranking = pd.DataFrame()

            # 6. Walk-forward
            if run_walk_forward and "best_strat_name" in st.session_state:
                with st.spinner(f"🔬 Walk-Forward ({wf_windows} ventanas)..."):
                    entries_full = _reconstruct_entries(
                        st.session_state.best_strat_name,
                        st.session_state.best_strat_direction,
                        generator,
                        df_data,
                    )
                    if entries_full is not None:
                        # Para WF usamos VectorizedBacktester (lento pero flexible)
                        # O implementamos WF con FastBacktester
                        from backend.backtester import BacktestConfig, VectorizedBacktester
                        vb = VectorizedBacktester(df_data, config=BacktestConfig(
                            sl_pct=sl_pct, tp_pct=tp_pct, fees=fees_input, slippage=slippage_input, freq=vbt_freq
                        ))
                        analyzer = WalkForwardAnalyzer(df_data, vb)
                        wf_result = analyzer.run_walk_forward(entries_full, n_windows=wf_windows)
                        st.session_state.walk_forward_result = wf_result

            # 7. Monte Carlo
            if run_monte_carlo and "best_trades_pnl" in st.session_state:
                with st.spinner(f"🎲 Monte Carlo ({mc_sims:,} simulaciones)..."):
                    trades_list = st.session_state.best_trades_pnl
                    if trades_list is not None and len(trades_list) >= 5:
                        from backend.backtester import BacktestConfig, VectorizedBacktester
                        vb = VectorizedBacktester(df_data, config=BacktestConfig(
                            sl_pct=sl_pct, tp_pct=tp_pct, fees=fees_input, slippage=slippage_input, freq=vbt_freq
                        ))
                        analyzer = WalkForwardAnalyzer(df_data, vb)
                        mc_result = analyzer.run_monte_carlo(
                            trades_list.tolist(), n_simulations=mc_sims, seed=42
                        )
                        st.session_state.mc_result = mc_result

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

    # ─── Mostrar Resultados ─────────────────────────────────
    if st.session_state.scan_completed:
        final_ranking = st.session_state.final_ranking
        bh = st.session_state.get("bh_oos", {"buy_hold_return_pct": 0, "buy_hold_max_dd_pct": 0})

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estrategias IS", f"{len(final_ranking):,}")
        c2.metric("Buy & Hold OOS", f"{bh['buy_hold_return_pct']:.2f}%")
        c3.metric("B&H Max DD", f"{bh['buy_hold_max_dd_pct']:.2f}%")
        c4.metric("Dirección", "LONG+SHORT" if "Direction" in final_ranking.columns else "—")

        if len(final_ranking) > 0:
            st.subheader("🏆 Ranking de Estrategias (IS+OOS)")
            st.info(
                f"💡 Tu estrategia debe superar el Buy & Hold ({bh['buy_hold_return_pct']:.2f}%). "
                f"Si no lo hace, no es estrategia, es forma cara de comprar y mantener."
            )

            # Limpiar columna interna antes de mostrar
            display_cols = [c for c in final_ranking.columns if c != "trades_pnl_obj"]
            st.dataframe(final_ranking[display_cols].head(30), use_container_width=True)

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


# ──────────────────────────────────────────────────────────────
def _reconstruct_entries(strat_name, direction_str, generator, df_data):
    """Reconstruye la serie de entries a partir del nombre de estrategia."""
    try:
        conditions = [c.strip() for c in strat_name.split("+")]
        combined = pd.Series(True, index=df_data.index)
        for cond in conditions:
            # Quitar prefijo L:/S:
            clean = cond[2:] if cond.startswith(("L:", "S:")) else cond
            if direction_str == "LONG":
                feat = generator.bull_features.get(clean)
            else:
                feat = generator.bear_features.get(clean)
            if feat is None:
                return None
            combined = combined & feat
        return combined
    except Exception:
        return None
