"""Data access helpers for the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from trading_bot.data.dataset_store import (
    list_symbols,
    load_manifest,
    load_ohlcv,
    load_universe,
)
from trading_bot.data.screener_excel import SCREENER_SUFFIX, list_screener_symbols
from trading_bot.features.chart_indicators import add_chart_indicators


def default_dataset_root() -> Path:
    return Path("dataset_smallcap250")


def available_symbols(root: Path, interval: str = "day") -> List[str]:
    ohlcv = set(list_symbols(interval, root=root))
    screener = set(list_screener_symbols(root / "screener_excel"))
    return sorted(ohlcv | screener)


def symbol_choices(root: Path) -> List[Tuple[str, str]]:
    try:
        uni = load_universe(root)
        name_map = dict(zip(uni["symbol"].str.upper(), uni["name"].fillna("")))
    except Exception:
        name_map = {}
    return [
        (f"{sym} — {name_map[sym]}" if name_map.get(sym) else sym, sym)
        for sym in available_symbols(root)
    ]


def load_bars(
    symbol: str,
    interval: str,
    root: Path,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    df = load_ohlcv(symbol, interval, root=root)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end) + pd.Timedelta(days=1)]
    return df.sort_values("date").reset_index(drop=True)


def load_with_indicators(
    symbol: str,
    interval: str,
    root: Path,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    df = load_bars(symbol, interval, root, start, end)
    return add_chart_indicators(df) if not df.empty else df


def dataset_summary(root: Path) -> dict:
    try:
        manifest = load_manifest(root)
    except FileNotFoundError:
        manifest = {}
    return {
        "root": str(root),
        "day_symbols": len(list_symbols("day", root=root)),
        "minute_symbols": len(list_symbols("minute", root=root)),
        "screener_symbols": len(list_screener_symbols(root / "screener_excel")),
        "date_range": manifest.get("date_range", {}),
    }


def screener_path(root: Path, symbol: str) -> Path:
    return root / "screener_excel" / f"{symbol.upper()}{SCREENER_SUFFIX}"


def bar_date_bounds(symbol: str, interval: str, root: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    df = load_ohlcv(symbol, interval, root=root)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df["date"].min(), df["date"].max()
