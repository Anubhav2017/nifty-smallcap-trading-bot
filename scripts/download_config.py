"""Load download settings from a JSON config file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dateutil.relativedelta import relativedelta

import pandas as pd

from env_utils import resolve_repo_path
from kite_intervals import CHUNK_DAYS_BY_INTERVAL, default_chunk_days


@dataclass(frozen=True)
class DownloadConfig:
    symbols: List[str]
    from_date: str
    to_date: str
    intervals: List[str]
    output_dir: Path
    chunk_days: Optional[int]
    sleep_seconds: float
    symbol_column: str

    @property
    def from_dt(self) -> datetime:
        return datetime.strptime(self.from_date, "%Y-%m-%d")

    @property
    def to_dt(self) -> datetime:
        return datetime.strptime(self.to_date, "%Y-%m-%d")

    def chunk_days_for(self, interval: str) -> int:
        if self.chunk_days is not None:
            return self.chunk_days
        return default_chunk_days(interval)

    def output_dir_for(self, interval: str) -> Path:
        if len(self.intervals) == 1:
            return self.output_dir
        return self.output_dir / interval


def _parse_date_field(raw: Any, field: str) -> str:
    if raw is None or raw == "":
        raise ValueError(f"Config field '{field}' is required (YYYY-MM-DD).")
    text = str(raw).strip()
    datetime.strptime(text, "%Y-%m-%d")
    return text


def resolve_date_range(
    raw: dict[str, Any],
    *,
    default_from: str,
    default_to: str,
) -> Tuple[str, str]:
    """Resolve from_date/to_date from explicit fields or years_back (rolling window)."""
    if raw.get("from_date") is not None and raw.get("to_date") is not None:
        return (
            _parse_date_field(raw["from_date"], "from_date"),
            _parse_date_field(raw["to_date"], "to_date"),
        )
    years_back = raw.get("years_back")
    if years_back is not None:
        years = int(years_back)
        if years <= 0:
            raise ValueError("years_back must be a positive integer.")
        to_dt = datetime.now()
        from_dt = to_dt - relativedelta(years=years)
        return from_dt.strftime("%Y-%m-%d"), to_dt.strftime("%Y-%m-%d")
    from_date = raw.get("from_date", default_from)
    to_date = raw.get("to_date", default_to)
    return _parse_date_field(from_date, "from_date"), _parse_date_field(to_date, "to_date")


def _parse_intervals(raw: dict[str, Any]) -> List[str]:
    if "intervals" in raw and "interval" in raw:
        raise ValueError("Use either 'interval' or 'intervals' in config, not both.")

    if "intervals" in raw:
        intervals = raw["intervals"]
        if not isinstance(intervals, list) or not intervals:
            raise ValueError("'intervals' must be a non-empty list of strings.")
        intervals = [str(i).strip() for i in intervals]
    elif "interval" in raw:
        intervals = [str(raw["interval"]).strip()]
    else:
        raise ValueError("Config must include 'interval' or 'intervals'.")

    supported = set(CHUNK_DAYS_BY_INTERVAL)
    for interval in intervals:
        if interval not in supported:
            supported_str = ", ".join(sorted(supported))
            raise ValueError(f"Unsupported interval {interval!r}. Use one of: {supported_str}")
    return intervals


def load_symbols_from_csv(csv_path: Path, symbol_column: str) -> List[str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"symbols_csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"symbols_csv is empty: {csv_path}")

    col = symbol_column.strip()
    if col not in df.columns:
        match = next(
            (c for c in df.columns if str(c).strip().lower() == col.lower()),
            None,
        )
        if match is None:
            raise ValueError(
                f"Column {symbol_column!r} not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )
        col = match

    symbols = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )
    if not symbols:
        raise ValueError(f"No symbols found in column {col!r} of {csv_path}")
    return sorted(symbols)


def load_config(path: Path, base_dir: Optional[Path] = None) -> DownloadConfig:
    """Load config from JSON. Relative paths resolve against the project root."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    config_path = path.resolve()
    if base_dir is not None:
        resolve = lambda rel: (Path(base_dir) / rel).resolve()
    else:
        resolve = lambda rel: resolve_repo_path(config_path, rel)

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object.")

    symbols_csv = raw.get("symbols_csv")
    if not symbols_csv:
        raise ValueError("Config must include 'symbols_csv' (path to a CSV of NSE symbols).")

    symbol_column = str(raw.get("symbol_column", "symbol"))
    symbols = load_symbols_from_csv(resolve(str(symbols_csv)), symbol_column)

    from_date = _parse_date_field(raw.get("from_date"), "from_date")
    to_date = _parse_date_field(raw.get("to_date"), "to_date")
    if to_date < from_date:
        raise ValueError(f"to_date ({to_date}) must be on or after from_date ({from_date}).")

    intervals = _parse_intervals(raw)
    output_dir = resolve(str(raw.get("output_dir", "data")))

    chunk_days = raw.get("chunk_days")
    if chunk_days is not None:
        chunk_days = int(chunk_days)
        if chunk_days <= 0:
            raise ValueError("chunk_days must be a positive integer.")

    sleep_seconds = float(raw.get("sleep_seconds", 0.4))
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be >= 0.")

    return DownloadConfig(
        symbols=symbols,
        from_date=from_date,
        to_date=to_date,
        intervals=intervals,
        output_dir=output_dir,
        chunk_days=chunk_days,
        sleep_seconds=sleep_seconds,
        symbol_column=symbol_column,
    )
