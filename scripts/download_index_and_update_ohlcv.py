#!/usr/bin/env python3
"""Download Nifty Smallcap 250 index OHLCV and update stock data to latest date.

Usage:
    # Set env vars first (from .env)
    export KITE_API_KEY=... KITE_ACCESS_TOKEN=...

    # Download just the index (fast, ~50 rows)
    python scripts/download_index_and_update_ohlcv.py --index-only

    # Download index + update all 250 stocks to latest
    python scripts/download_index_and_update_ohlcv.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Kite instrument tokens ──────────────────────────────────────────────────
# NSE Indices segment — these are the actual index tokens, not ETFs
INDEX_TOKENS = {
    "NIFTY_SMALLCAP_250": 288009,   # NSE:NIFTY SMALLCAP 250
    "NIFTY_50":            256265,   # NSE:NIFTY 50
    "NIFTY_MIDCAP_150":   288265,   # NSE:NIFTY MIDCAP 150
}

DATASET_ROOT = Path(__file__).parent.parent / "dataset_smallcap250"
INDEX_OUTPUT_DIR = DATASET_ROOT / "ohlcv" / "indices"


def get_kite() -> "KiteConnect":  # type: ignore[name-defined]
    from kiteconnect import KiteConnect  # type: ignore[import]

    api_key = os.environ.get("KITE_API_KEY", "").strip()
    access_token = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
    if not api_key or not access_token:
        raise EnvironmentError("Set KITE_API_KEY and KITE_ACCESS_TOKEN environment variables.")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    try:
        profile = kite.profile()
        logger.info("Authenticated as %s (%s)", profile.get("user_name"), profile.get("user_id"))
    except Exception as exc:
        raise EnvironmentError(f"Kite auth failed: {exc}") from exc
    return kite


def fetch_daily(kite, token: int, from_dt: date, to_dt: date) -> pd.DataFrame:
    """Fetch day-interval OHLCV for one instrument."""
    records = kite.historical_data(
        instrument_token=token,
        from_date=from_dt.strftime("%Y-%m-%d 00:00:00"),
        to_date=to_dt.strftime("%Y-%m-%d 00:00:00"),
        interval="day",
    )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")


def download_index(kite, name: str, token: int, out_dir: Path, *, from_dt: date, to_dt: date) -> Path:
    """Download one index to a CSV file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{name}.csv"

    df = fetch_daily(kite, token, from_dt, to_dt)
    if df.empty:
        logger.warning("No data returned for %s (token %d)", name, token)
        return out_file

    df.to_csv(out_file, index=False)
    logger.info("Saved %s: %d rows → %s", name, len(df), out_file)
    return out_file


def update_stock_ohlcv(kite, dataset_root: Path, *, to_dt: date, sleep: float = 0.4) -> None:
    """Extend each stock's day OHLCV CSV to *to_dt*."""
    day_dir = dataset_root / "ohlcv" / "day"
    csv_files = sorted(day_dir.glob("*.csv"))
    logger.info("Updating %d stock OHLCV files to %s ...", len(csv_files), to_dt)

    # Load instruments to get token → symbol mapping
    instr_dir = dataset_root / "instruments"
    instr_files = sorted(instr_dir.glob("nse_eq_*.csv"))
    if not instr_files:
        logger.error("No instruments CSV found in %s", instr_dir)
        return
    instr_df = pd.read_csv(instr_files[-1])
    token_by_symbol = {str(r["tradingsymbol"]): int(r["instrument_token"]) for _, r in instr_df.iterrows()}

    ok, skipped, failed = 0, 0, 0
    for f in csv_files:
        symbol = f.stem
        token = token_by_symbol.get(symbol)
        if token is None:
            skipped += 1
            continue

        existing = pd.read_csv(f, parse_dates=["date"])
        if existing.empty:
            last_date = date(2021, 1, 1)
        else:
            last_ts = pd.to_datetime(existing["date"]).max()
            last_date = last_ts.date() if hasattr(last_ts, "date") else last_ts

        from_dt = last_date + timedelta(days=1)
        if from_dt > to_dt:
            skipped += 1
            continue

        try:
            new_rows = fetch_daily(kite, token, from_dt, to_dt)
            if new_rows.empty:
                skipped += 1
                continue
            # Normalize timezones: strip tz info from both sides before concat
            existing["date"] = pd.to_datetime(existing["date"]).dt.tz_localize(None)
            new_rows["date"] = pd.to_datetime(new_rows["date"]).dt.tz_localize(None)
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.drop_duplicates(subset=["date"]).sort_values("date")
            combined.to_csv(f, index=False)
            ok += 1
            logger.info("Updated %s: added %d rows (now through %s)", symbol, len(new_rows), to_dt)
        except Exception as exc:
            logger.warning("Failed %s: %s", symbol, exc)
            failed += 1

        time.sleep(sleep)

    logger.info("Stock update done: ok=%d skipped=%d failed=%d", ok, skipped, failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-only", action="store_true", help="Only download index data, skip stock update")
    parser.add_argument("--from-date", default="2024-01-01", help="History start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", default=date.today().isoformat(), help="End date (default: today)")
    parser.add_argument("--sleep", type=float, default=0.4, help="Sleep seconds between API calls")
    args = parser.parse_args()

    # Load .env if present
    env_file = Path(__file__).parent.parent.parent / "nse_smallcap250_historical_data" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    kite = get_kite()
    from_dt = date.fromisoformat(args.from_date)
    to_dt = date.fromisoformat(args.to_date)

    # ── Download indices ──────────────────────────────────────────────────────
    for name, token in INDEX_TOKENS.items():
        try:
            download_index(kite, name, token, INDEX_OUTPUT_DIR, from_dt=from_dt, to_dt=to_dt)
        except Exception as exc:
            logger.warning("Index %s failed: %s", name, exc)
        time.sleep(args.sleep)

    if args.index_only:
        logger.info("Done (index only).")
        return

    # ── Update stock OHLCV ───────────────────────────────────────────────────
    update_stock_ohlcv(kite, DATASET_ROOT, to_dt=to_dt, sleep=args.sleep)
    logger.info("All done.")


if __name__ == "__main__":
    main()
