import argparse
import io
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from kiteconnect import KiteConnect
from kiteconnect.exceptions import PermissionException

from env_utils import load_env_file
from kite_equity import build_nse_eq_token_map


NIFTY_SMALLCAP_250_CSV_URL = "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv"
NSE_ARCHIVES_SMALLCAP_250_CSV_URL = (
    "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"
)
from kite_intervals import CHUNK_DAYS_BY_INTERVAL, chunk_step, default_chunk_days

DEFAULT_INTERVAL = "minute"
DEFAULT_YEARS_BACK = 5
DEFAULT_OUTPUT_DIR = "data/1m"
HISTORICAL_PERMISSION_HELP = """
Kite rejected historical_data: "Insufficient permission for that call."

This script needs Kite Connect historical candle access. Common fixes:

1. App type must be "Kite Connect" (paid), not "Personal" (free).
   https://developers.kite.trade/

2. Subscribe to the Historical data add-on for that app (if not already).

3. Re-login AFTER enabling historical access (old tokens lack the scope):
   python kite_login.py

4. If you just changed subscription or app type, revoke the app at
   https://kite.zerodha.com/apps and log in again.
"""
DEFAULT_CONSTITUENTS_FILE = "data/nifty_smallcap250_constituents.csv"
CONSTITUENT_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
CONSTITUENT_CSV_SOURCES = (
    NSE_ARCHIVES_SMALLCAP_250_CSV_URL,
    NIFTY_SMALLCAP_250_CSV_URL,
)


def load_symbols_from_csv(csv_path: Path, symbol_column: str = "symbol") -> List[str]:
    from download_config import load_symbols_from_csv as _load

    return _load(csv_path, symbol_column)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OHLCV for NSE stocks via Kite Connect (CLI flags or JSON config)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config (symbols_csv, from_date, to_date, interval/intervals, ...)",
    )
    parser.add_argument("--api-key", default=os.getenv("KITE_API_KEY"), help="Kite API key")
    parser.add_argument("--access-token", default=os.getenv("KITE_ACCESS_TOKEN"), help="Kite access token")
    parser.add_argument(
        "--symbols-csv",
        type=Path,
        help="CSV of symbols (column 'symbol' by default). Skips index download when set.",
    )
    parser.add_argument(
        "--symbol-column",
        default="symbol",
        help="Column name in --symbols-csv (default: symbol)",
    )
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        choices=sorted(CHUNK_DAYS_BY_INTERVAL),
        help=f"Kite candle interval (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--from-date",
        help="Start date YYYY-MM-DD. Overrides --years-back when used with --to-date.",
    )
    parser.add_argument(
        "--to-date",
        help="End date YYYY-MM-DD. Overrides --years-back when used with --from-date.",
    )
    parser.add_argument(
        "--years-back",
        type=int,
        default=DEFAULT_YEARS_BACK,
        help=f"Trailing years of history (default: {DEFAULT_YEARS_BACK})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Directory for per-symbol OHLCV CSV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--constituents-file",
        default=DEFAULT_CONSTITUENTS_FILE,
        help="Path to save/load latest constituent symbol list",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=None,
        help="Max days per historical API request (default: Kite limit for --interval)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.4,
        help="Delay between API requests",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR
    if args.chunk_days is None:
        args.chunk_days = default_chunk_days(args.interval)
    return args


def validate_auth(api_key: str, access_token: str) -> None:
    if not api_key or not access_token:
        raise ValueError(
            "Missing API credentials. Pass --api-key and --access-token or set "
            "KITE_API_KEY and KITE_ACCESS_TOKEN."
        )


def resolve_date_range(args: argparse.Namespace) -> tuple[datetime, datetime]:
    if args.from_date and args.to_date:
        from_dt = datetime.strptime(args.from_date, "%Y-%m-%d")
        to_dt = datetime.strptime(args.to_date, "%Y-%m-%d")
        return from_dt, to_dt

    if args.from_date or args.to_date:
        raise ValueError("Pass both --from-date and --to-date, or use --years-back only.")

    if args.years_back <= 0:
        raise ValueError("--years-back must be a positive integer")

    to_dt = datetime.now()
    from_dt = to_dt - relativedelta(years=args.years_back)
    return from_dt, to_dt


def _normalize_symbols(df: pd.DataFrame) -> List[str]:
    symbol_col = next(
        (col for col in df.columns if col.strip().lower() == "symbol"),
        None,
    )
    if symbol_col is None:
        raise RuntimeError("Could not find 'Symbol' column in constituents CSV.")

    return sorted(
        df[symbol_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )


def download_constituents(
    constituents_file: Path,
    *,
    csv_urls: tuple[str, ...] | None = None,
) -> List[str]:
    sources = csv_urls or CONSTITUENT_CSV_SOURCES
    session = requests.Session()
    session.headers.update(CONSTITUENT_REQUEST_HEADERS)
    response = None
    last_error: Exception | None = None

    for url in sources:
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc

    if response is None or not response.ok:
        raise RuntimeError(
            f"Failed to download index constituents CSV from: {list(sources)}"
        ) from last_error
    df = pd.read_csv(io.StringIO(response.text))
    symbols = sorted(_normalize_symbols(df))
    constituents_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": symbols}).to_csv(constituents_file, index=False)
    return symbols


def verify_historical_access(
    kite: KiteConnect, instrument_token: int, interval: str
) -> None:
    """Fail fast if this session cannot call historical_data."""
    probe_end = datetime.now()
    probe_start = probe_end - timedelta(days=1)
    try:
        kite.historical_data(
            instrument_token=instrument_token,
            from_date=probe_start,
            to_date=probe_end,
            interval=interval,
            oi=False,
        )
    except PermissionException as exc:
        raise PermissionException(
            f"{exc}. {HISTORICAL_PERMISSION_HELP.strip()}"
        ) from exc


def fetch_symbol_ohlcv(
    kite: KiteConnect,
    instrument_token: int,
    from_dt: datetime,
    to_dt: datetime,
    interval: str,
    chunk_days: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    current_start = from_dt
    all_rows: List[dict] = []
    step = chunk_step(interval)

    while current_start <= to_dt:
        current_end = min(current_start + timedelta(days=chunk_days), to_dt)
        rows = kite.historical_data(
            instrument_token=instrument_token,
            from_date=current_start,
            to_date=current_end,
            interval=interval,
            oi=False,
        )
        if rows:
            all_rows.extend(rows)

        current_start = current_end + step
        time.sleep(sleep_seconds)

    if not all_rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows)
    return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def run_download(
    *,
    kite: KiteConnect,
    symbols: List[str],
    from_dt: datetime,
    to_dt: datetime,
    intervals: List[str],
    output_dir: Path,
    chunk_days: int | None,
    sleep_seconds: float,
    skip_existing: bool = False,
) -> None:
    token_map = build_nse_eq_token_map(kite)

    missing = [sym for sym in symbols if sym not in token_map]
    if missing:
        print(f"Warning: {len(missing)} symbols not found in NSE EQ instruments and will be skipped.")

    probe_symbol = next((sym for sym in symbols if sym in token_map), None)
    if probe_symbol is None:
        raise RuntimeError("No symbols matched NSE EQ instruments; cannot download OHLCV.")

    print("Checking Kite historical_data permission...")
    try:
        verify_historical_access(kite, token_map[probe_symbol], intervals[0])
    except PermissionException as exc:
        print(exc)
        raise SystemExit(1) from exc

    if "minute" in intervals and (to_dt - from_dt).days > 365 * 3:
        print(
            "Note: Kite typically retains ~3 years of 1-minute candles; "
            "older dates may return no rows."
        )

    total_done = 0
    total_skipped = 0

    for interval in intervals:
        interval_chunk = chunk_days if chunk_days is not None else default_chunk_days(interval)
        interval_out = output_dir if len(intervals) == 1 else output_dir / interval
        interval_out.mkdir(parents=True, exist_ok=True)

        print(
            f"\n[{interval}] Downloading OHLCV {from_dt.date()} -> {to_dt.date()} "
            f"(chunk={interval_chunk}d) -> {interval_out}"
        )

        done = 0
        skipped = 0
        for symbol in symbols:
            token = token_map.get(symbol)
            if token is None:
                skipped += 1
                continue

            out_file = interval_out / f"{symbol}.csv"
            if skip_existing and out_file.is_file() and out_file.stat().st_size > 0:
                skipped += 1
                continue

            try:
                df = fetch_symbol_ohlcv(
                    kite=kite,
                    instrument_token=token,
                    from_dt=from_dt,
                    to_dt=to_dt,
                    interval=interval,
                    chunk_days=interval_chunk,
                    sleep_seconds=sleep_seconds,
                )
                df.to_csv(out_file, index=False)
                done += 1
                print(f"[{interval}] [{done}/{len(symbols)}] {symbol} -> {out_file} ({len(df)} rows)")
            except PermissionException as exc:
                print(exc)
                raise SystemExit(1) from exc
            except Exception as exc:
                print(f"[{interval}] Error for {symbol}: {exc}")
                skipped += 1
                time.sleep(max(1.0, sleep_seconds))

        print(f"[{interval}] Done. Downloaded: {done}, Skipped/Failed: {skipped}")
        total_done += done
        total_skipped += skipped

    if len(intervals) > 1:
        print(f"\nAll intervals complete. Downloaded: {total_done}, Skipped/Failed: {total_skipped}")


def main() -> None:
    load_env_file()
    args = parse_args()
    validate_auth(args.api_key, args.access_token)

    if args.config:
        from download_config import load_config

        cfg = load_config(args.config.resolve())
        symbols = cfg.symbols
        from_dt, to_dt = cfg.from_dt, cfg.to_dt
        intervals = cfg.intervals
        output_dir = cfg.output_dir
        chunk_days = cfg.chunk_days
        sleep_seconds = cfg.sleep_seconds
        print(f"Config: {args.config}")
        print(f"Loaded {len(symbols)} symbols from config")
    else:
        if args.from_date and args.to_date:
            from_dt = datetime.strptime(args.from_date, "%Y-%m-%d")
            to_dt = datetime.strptime(args.to_date, "%Y-%m-%d")
        elif not (args.from_date or args.to_date):
            from_dt, to_dt = resolve_date_range(args)
        else:
            raise ValueError("Pass both --from-date and --to-date, or use --years-back only.")

        intervals = [args.interval]
        output_dir = Path(args.output_dir)
        chunk_days = args.chunk_days
        sleep_seconds = args.sleep_seconds

        if args.symbols_csv:
            symbols = load_symbols_from_csv(args.symbols_csv.resolve(), args.symbol_column)
            print(f"Loaded {len(symbols)} symbols from {args.symbols_csv}")
        else:
            constituents_file = Path(args.constituents_file)
            print("Downloading latest NIFTY Smallcap 250 constituents...")
            symbols = download_constituents(constituents_file)
            print(f"Saved {len(symbols)} symbols -> {constituents_file}")

    if to_dt < from_dt:
        raise ValueError("End date must be >= start date.")

    output_dir.mkdir(parents=True, exist_ok=True)

    kite = KiteConnect(api_key=args.api_key)
    kite.set_access_token(args.access_token)

    print("Loading NSE instrument tokens from Kite...")
    run_download(
        kite=kite,
        symbols=symbols,
        from_dt=from_dt,
        to_dt=to_dt,
        intervals=intervals,
        output_dir=output_dir,
        chunk_days=chunk_days,
        sleep_seconds=sleep_seconds,
    )


if __name__ == "__main__":
    main()
