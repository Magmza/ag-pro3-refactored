"""
data_loader.py — Motor unificado para descargar datos históricos.

Mejoras vs original:
- Sin paths hardcodeados a Windows (G:\Claude\Proyectos\Bot XAU Long ...)
- Cache local usando settings.data_path
- Logging estructurado en vez de print
- Tipado completo
- Timeframe 4h ahora hace resample correcto (no finge ser 1h)
- Validación de columnas antes de devolver DataFrame
- Manejo de errores consistente
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import ccxt
import pandas as pd
import yfinance as yf

from backend.config import settings
from backend.logger import get_logger

log = get_logger(__name__)

AssetClass = Literal["crypto", "forex", "metals", "stocks", "commodities"]


class UniversalDataLoader:
    """
    Descarga datos históricos OHLCV desde múltiples fuentes:
      - crypto       → Binance (o Bybit fallback) vía CCXT
      - forex/metals → Dukascopy CSV local (si existe) o YFinance
      - stocks       → YFinance
      - commodities  → YFinance (futuros)
    """

    # Mapeo de timeframes YFinance (lo que realmente soporta)
    _YF_TF_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "60m",
        "4h": "60m",   # YFinance no tiene 4h, se resamplea después
        "1d": "1d",
        "1w": "1wk",
    }

    # Límites intradía de YFinance
    _YF_INTRADAY_LIMITS = {
        "1m": 6,
        "5m": 59,
        "15m": 59,
        "30m": 59,
        "60m": 729,
    }

    def __init__(self) -> None:
        # Clientes CCXT sin credenciales (datos públicos)
        try:
            self.binance = ccxt.binance({"enableRateLimit": True})
            self.bybit = ccxt.bybit({"enableRateLimit": True})
        except Exception as e:
            log.warning("No se pudo inicializar CCXT", error=str(e))
            self.binance = None
            self.bybit = None

    # ──────────────────────────────────────────────────────────
    def get_data(
        self,
        asset_class: AssetClass,
        symbol: str,
        timeframe: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """
        Devuelve DataFrame con index=Date (tz-naive) y columnas OHLCV.
        """
        asset_class = asset_class.lower()  # type: ignore

        if asset_class == "crypto":
            df = self._fetch_crypto(symbol, timeframe, start_date, end_date, limit)
        else:
            df = self._fetch_traditional(asset_class, symbol, timeframe, start_date, end_date, limit)

        # Validación final
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame incompleto. Faltan columnas: {missing}")

        if df.isnull().any().any():
            log.warning(
                "DataFrame tiene NaNs, haciendo forward fill conservativo",
                symbol=symbol,
                timeframe=timeframe,
                nans=int(df.isnull().sum().sum()),
            )
            df = df.ffill().dropna()

        return df

    # ──────────────────────────────────────────────────────────
    def _fetch_traditional(
        self,
        asset_class: str,
        symbol: str,
        timeframe: str,
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> pd.DataFrame:
        """Para forex/metales/stocks/commodities. Cache local → YFinance."""

        # 1. Buscar CSV local (descargado por download_duka.py)
        clean_sym = symbol.replace("=X", "").replace("=F", "").lower()
        local_filename = f"hist_{clean_sym}_{timeframe.lower()}.csv"
        local_path = settings.data_path / local_filename

        if local_path.exists():
            log.info("Cargando datos locales Dukascopy", path=str(local_path))
            df = pd.read_csv(local_path)
            date_col = "Date" if "Date" in df.columns else "datetime"
            df["Date"] = pd.to_datetime(df[date_col])
            df = df.set_index("Date").sort_index()
            # Filtrar por rango
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            return self._normalize_columns(df)

        # 2. Fallback YFinance
        log.info("No hay CSV local, usando YFinance", symbol=symbol)
        return self._fetch_yfinance(symbol, timeframe, start_date, end_date, limit)

    # ──────────────────────────────────────────────────────────
    def _fetch_crypto(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> pd.DataFrame:
        """Descarga OHLCV de Binance o Bybit con paginación."""
        if not self.binance:
            raise RuntimeError("CCXT no está inicializado")

        exchange = self.binance
        try:
            exchange.load_markets()
            if symbol not in exchange.markets and self.bybit:
                log.info("Symbol no en Binance, probando Bybit", symbol=symbol)
                exchange = self.bybit
                exchange.load_markets()
        except Exception as e:
            log.warning("Error cargando markets", error=str(e))

        since = None
        if start_date:
            dt = datetime.strptime(start_date, "%Y-%m-%d")
            since = int(dt.timestamp() * 1000)

        all_ohlcv: list[list] = []

        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000, since=since)
                if not ohlcv:
                    break

                last_ts = ohlcv[-1][0]
                if since is not None and last_ts <= since and len(ohlcv) == 1:
                    break  # evitar loop infinito si la API repite

                all_ohlcv.extend(ohlcv)
                since = last_ts + 1

                if len(all_ohlcv) >= limit:
                    all_ohlcv = all_ohlcv[:limit]
                    break

                time.sleep(0.15)  # rate limit amable
            except Exception as e:
                log.error("Error descargando crypto paginado", error=str(e))
                break

        if not all_ohlcv:
            raise ValueError(f"No se pudieron descargar datos crypto para {symbol}")

        df = pd.DataFrame(all_ohlcv, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"], unit="ms")
        df = df.set_index("Date").sort_index()

        if end_date:
            end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            df = df[df.index < end_dt]

        log.info("Crypto descargado", symbol=symbol, rows=len(df))
        return df

    # ──────────────────────────────────────────────────────────
    def _fetch_yfinance(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> pd.DataFrame:
        """Descarga vía YFinance con manejo de límites intradía."""
        yf_tf = self._YF_TF_MAP.get(timeframe, "1d")
        needs_resample_4h = timeframe == "4h"

        now = datetime.now()

        # Recortar start_date según límites intradía de YFinance
        if yf_tf in self._YF_INTRADAY_LIMITS:
            min_start = now - timedelta(days=self._YF_INTRADAY_LIMITS[yf_tf])
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                if start_dt < min_start:
                    log.warning(
                        "YFinance no tiene datos tan antiguos para este TF, recortando",
                        timeframe=timeframe,
                        original=start_date,
                        recortado=min_start.strftime("%Y-%m-%d"),
                    )
                    start_date = min_start.strftime("%Y-%m-%d")
            else:
                start_date = min_start.strftime("%Y-%m-%d")

        if not start_date:
            days_back = 1800
            start_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = now.strftime("%Y-%m-%d")

        log.info("Descargando YFinance", symbol=symbol, tf=yf_tf, start=start_date, end=end_date)

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=yf_tf)

        if df.empty:
            raise ValueError(f"YFinance devolvió vacío para {symbol}")

        df = df.reset_index()
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(columns={date_col: "Date"})

        # Quitar timezone para compatibilidad con vectorbt
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.set_index("Date").sort_index()

        # Resample a 4h si se pidió 4h
        if needs_resample_4h:
            df = self._resample_4h(df)

        df = self._normalize_columns(df)
        log.info("YFinance descargado", symbol=symbol, rows=len(df))
        return df.tail(limit)

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Asegura columnas: Open, High, Low, Close, Volume (todas float)."""
        col_map = {c.lower(): c.capitalize() for c in df.columns}
        df = df.rename(columns=col_map)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].astype(float)
        return df

    @staticmethod
    def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
        """Resamplea datos 1h a 4h OHLCV."""
        return df.resample("4h").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna()
