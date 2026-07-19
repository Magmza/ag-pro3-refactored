r"""
download_duka.py — Descarga datos históricos desde Dukascopy.

Mejoras vs original:
- Sin paths hardcodeados a Windows (ej. C:\Users\Mario\AppData...)
- Path de duka.exe viene de .env (DUKA_PATH) o se busca en PATH
- Símbolo, timeframe y rango configurables vía CLI o argumentos
- Logging estructurado en vez de print()
- Tipado, manejo de errores robusto
- Cache en data/ en vez de directorio suelto
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from dateutil.relativedelta import relativedelta

# Permitir ejecutar como script directo o como módulo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.logger import get_logger

log = get_logger(__name__)


def get_months_list(start: str, end: str) -> list[tuple[str, str]]:
    """Devuelve lista de (first_day, last_day) por mes entre start y end."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    months: list[tuple[str, str]] = []

    current = start_dt
    while current <= end_dt:
        first_day = current.replace(day=1)
        next_month = first_day + relativedelta(months=1)
        last_day = next_month - relativedelta(days=1)
        if last_day > end_dt:
            last_day = end_dt
        months.append(
            (first_day.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d"))
        )
        current = next_month
    return months


def find_duka_binary() -> str | None:
    """Localiza el binario `duka` desde .env o PATH."""
    if settings.duka_path and Path(settings.duka_path).exists():
        return settings.duka_path

    # Buscar en PATH
    found = shutil.which("duka")
    return found


def download_month(
    symbol: str, timeframe: str, start: str, end: str, temp_dir: Path, duka_bin: str
) -> Path | None:
    """Descarga un mes. Devuelve path al CSV o None si falla."""
    file_start = start.replace("-", "_")
    file_end = end.replace("-", "_")
    expected_filename = f"{symbol}-{file_start}-{file_end}.csv"
    temp_filepath = temp_dir / expected_filename
    local_filepath = Path.cwd() / expected_filename

    # Cache hit
    if temp_filepath.exists() and temp_filepath.stat().st_size > 1000:
        log.info("Mes ya descargado (cache)", start=start, end=end, path=str(temp_filepath))
        return temp_filepath

    if temp_filepath.exists():
        temp_filepath.unlink()

    cmd = [duka_bin, symbol, "-c", timeframe, "-t", "2", "-s", start, "-e", end]
    max_retries = 5

    for attempt in range(1, max_retries + 1):
        log.info("Descargando", symbol=symbol, start=start, end=end, attempt=attempt)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)

            if local_filepath.exists() and local_filepath.stat().st_size > 1000:
                local_filepath.rename(temp_filepath)
                return temp_filepath

            if local_filepath.exists():
                local_filepath.unlink()
            log.warning("Descarga vacía o inválida", attempt=attempt)
        except subprocess.CalledProcessError as e:
            log.error("duka falló", stderr=e.stderr[:200] if e.stderr else "")
        except subprocess.TimeoutExpired:
            log.warning("Timeout en descarga", attempt=attempt)
        except Exception as e:
            log.exception("Error inesperado descargando", error=str(e))

        if attempt < max_retries:
            time.sleep(30 * attempt)  # backoff lineal suave

    log.error("No se pudo descargar el mes tras reintentos", start=start, end=end)
    return None


def merge_csvs(csv_paths: Iterable[Path], output_file: Path) -> pd.DataFrame | None:
    """Concatena CSVs mensuales en un único DataFrame ordenado y limpio."""
    df_list = []
    for csv_file in csv_paths:
        try:
            df_list.append(pd.read_csv(csv_file))
        except Exception as e:
            log.warning("No se pudo leer CSV", path=str(csv_file), error=str(e))

    if not df_list:
        return None

    final_df = pd.concat(df_list, ignore_index=True)
    final_df["time"] = pd.to_datetime(final_df["time"])
    final_df = (
        final_df.sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )
    final_df = final_df.rename(
        columns={
            "time": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    final_df.to_csv(output_file, index=False)
    log.info("CSV final guardado", path=str(output_file), rows=len(final_df))
    return final_df


def run_downloader(
    symbol: str = "XAGUSD",
    timeframe: str = "H1",
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    output_filename: str | None = None,
) -> Path | None:
    """
    Descarga y consolida datos históricos.

    Args:
        symbol:           Símbolo Dukascopy (ej. XAGUSD, XAUUSD, EURUSD)
        timeframe:        H1, M15, M5, D1...
        start_date:       YYYY-MM-DD
        end_date:         YYYY-MM-DD (default: hoy)
        output_filename:  Nombre del CSV final (default: hist_{symbol}_{tf}.csv)

    Returns:
        Path al CSV final o None si falla.
    """
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    output_filename = output_filename or f"hist_{symbol.lower()}_{timeframe.lower()}.csv"

    temp_dir = settings.data_path / f"duka_temp_{symbol.lower()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_file = settings.data_path / output_filename

    duka_bin = find_duka_binary()
    if not duka_bin:
        log.error(
            "No se encontró el binario `duka`. Instalalo con `pip install dukascopy` "
            "y configurá DUKA_PATH en .env si no está en PATH."
        )
        return None

    log.info(
        "Iniciando descarga Dukascopy",
        symbol=symbol,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
    )

    months = get_months_list(start_date, end_date)
    all_csvs: list[Path] = []

    for start, end in months:
        csv_path = download_month(symbol, timeframe, start, end, temp_dir, duka_bin)
        if csv_path:
            all_csvs.append(csv_path)
        time.sleep(2)  # ser amables con el servidor

    if not all_csvs:
        log.error("No se descargó ningún mes. Abortando.")
        return None

    df = merge_csvs(all_csvs, output_file)
    if df is None:
        return None

    log.info(
        "Descarga completa",
        rows=len(df),
        start=str(df["Date"].iloc[0]),
        end=str(df["Date"].iloc[-1]),
    )
    return output_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga datos Dukascopy")
    parser.add_argument("--symbol", default="XAGUSD", help="Símbolo (ej. XAGUSD, XAUUSD)")
    parser.add_argument("--timeframe", default="H1", help="H1, M15, M5, D1")
    parser.add_argument("--start", default="2020-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: hoy)")
    args = parser.parse_args()

    result = run_downloader(args.symbol, args.timeframe, args.start, args.end)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
