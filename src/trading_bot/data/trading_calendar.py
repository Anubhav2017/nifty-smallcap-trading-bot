"""NSE session calendar helpers (weekends + optional index-derived holidays)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def trading_days_between(
    start: date,
    end: date,
    known_trading_days: set[date] | None = None,
) -> list[date]:
    """Return trading sessions in [start, end] inclusive."""
    if known_trading_days is not None:
        return sorted(d for d in known_trading_days if start <= d <= end)
    return [ts.date() for ts in pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end))]


def resolve_trading_day(
    d: date,
    known_trading_days: set[date] | None = None,
    *,
    max_scan_days: int = 366,
) -> tuple[date, str | None]:
    """Map a calendar day to the next NSE session on or after *d*.

    Returns (*session_date*, *note*) where *note* is set when *d* was adjusted.
    """
    if known_trading_days is not None:
        if d in known_trading_days:
            return d, None
        cursor = d
        for _ in range(max_scan_days):
            if cursor in known_trading_days:
                return cursor, (
                    f"{d.isoformat()} is not a trading day; "
                    f"using next trading day {cursor.isoformat()}"
                )
            cursor += timedelta(days=1)
        raise ValueError(
            f"No trading day on or after {d.isoformat()} within {max_scan_days} days"
        )

    if not is_weekend(d):
        return d, None

    cursor = d
    while is_weekend(cursor):
        cursor += timedelta(days=1)
    return cursor, (
        f"{d.isoformat()} is not a trading day (weekend); "
        f"using next trading day {cursor.isoformat()}"
    )


def resolve_period(
    start: date,
    end: date,
    known_trading_days: set[date] | None = None,
) -> tuple[date, date, list[str]]:
    """Adjust period boundaries to the next trading session when needed."""
    notes: list[str] = []
    resolved_start, note = resolve_trading_day(start, known_trading_days)
    if note:
        notes.append(f"Start date adjusted: {note}")
    resolved_end, note = resolve_trading_day(end, known_trading_days)
    if note:
        notes.append(f"End date adjusted: {note}")
    if resolved_start > resolved_end:
        raise ValueError(
            f"Invalid period after trading-day adjustment: "
            f"{resolved_start.isoformat()} > {resolved_end.isoformat()}"
        )
    return resolved_start, resolved_end, notes
