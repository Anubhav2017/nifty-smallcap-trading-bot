"""Kite Connect historical candle intervals and API chunk limits."""

from datetime import timedelta
from typing import Dict

CHUNK_DAYS_BY_INTERVAL: Dict[str, int] = {
    "minute": 60,
    "2minute": 60,
    "3minute": 100,
    "4minute": 100,
    "5minute": 100,
    "10minute": 100,
    "15minute": 200,
    "30minute": 200,
    "60minute": 400,
    "day": 2000,
}

CHUNK_STEP_BY_INTERVAL: Dict[str, timedelta] = {
    "minute": timedelta(minutes=1),
    "2minute": timedelta(minutes=2),
    "3minute": timedelta(minutes=3),
    "4minute": timedelta(minutes=4),
    "5minute": timedelta(minutes=5),
    "10minute": timedelta(minutes=10),
    "15minute": timedelta(minutes=15),
    "30minute": timedelta(minutes=30),
    "60minute": timedelta(minutes=60),
    "day": timedelta(days=1),
}


def default_chunk_days(interval: str) -> int:
    if interval not in CHUNK_DAYS_BY_INTERVAL:
        supported = ", ".join(sorted(CHUNK_DAYS_BY_INTERVAL))
        raise ValueError(f"Unsupported interval {interval!r}. Use one of: {supported}")
    return CHUNK_DAYS_BY_INTERVAL[interval]


def chunk_step(interval: str) -> timedelta:
    return CHUNK_STEP_BY_INTERVAL.get(interval, timedelta(minutes=1))
