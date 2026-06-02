"""Intraday helpers: Kite chunk windows and candle normalization."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

# Max calendar days per Kite historical request (conservative for 5minute bars).
INTERVAL_CHUNK_DAYS: dict[str, int] = {
    "minute": 60,
    "3minute": 90,
    "5minute": 90,
    "10minute": 90,
    "15minute": 120,
    "30minute": 120,
    "60minute": 200,
}

INTERVAL_DIR: dict[str, str] = {
    "minute": "1m",
    "3minute": "3m",
    "5minute": "5m",
    "10minute": "10m",
    "15minute": "15m",
    "30minute": "30m",
    "60minute": "60m",
}

OHLCV_COLS = ["datetime", "open", "high", "low", "close", "volume"]


def _normalize_candles(candles: list[dict]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=OHLCV_COLS)

    rows: list[dict] = []
    for c in candles:
        dt = c.get("date") or c.get("timestamp")
        rows.append(
            {
                "datetime": pd.to_datetime(dt),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c.get("volume", 0)),
            }
        )
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    return df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)


def iter_chunks(
    start: date,
    end: date,
    interval: str,
) -> list[tuple[datetime, datetime]]:
    """Split [start, end] into Kite-safe windows with NSE session bounds."""
    chunk_days = INTERVAL_CHUNK_DAYS.get(interval, 90)
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        chunks.append(
            (
                datetime.combine(cursor, datetime.min.time().replace(hour=9, minute=15)),
                datetime.combine(chunk_end, datetime.min.time().replace(hour=15, minute=30)),
            )
        )
        cursor = chunk_end + timedelta(days=1)
    return chunks
