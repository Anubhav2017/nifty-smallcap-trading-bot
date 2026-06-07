"""Load a built NSE equity OHLCV dataset from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

DEFAULT_DATASET_ROOT = Path("dataset")


def dataset_root(path: Optional[Path] = None) -> Path:
    return (path or DEFAULT_DATASET_ROOT).resolve()


def manifest_path(root: Optional[Path] = None) -> Path:
    return dataset_root(root) / "manifest.json"


def load_manifest(root: Optional[Path] = None) -> dict:
    path = manifest_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"No manifest at {path}. Run build_equity_dataset.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def list_symbols(interval: str, root: Optional[Path] = None) -> List[str]:
    folder = dataset_root(root) / "ohlcv" / interval
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.csv"))


def load_universe(root: Optional[Path] = None) -> pd.DataFrame:
    """Universe file with symbol, instrument_token, name."""
    root = dataset_root(root)
    enriched = root / "universe" / "universe_enriched.csv"
    if enriched.is_file():
        return pd.read_csv(enriched)
    manifest = load_manifest(root)
    path = root / manifest["universe"]["symbols_csv"]
    return pd.read_csv(path)


def load_ohlcv(
    symbol: str,
    interval: str,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load one symbol's OHLCV CSV."""
    path = dataset_root(root) / "ohlcv" / interval / f"{symbol.upper()}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_panel(
    symbols: List[str],
    interval: str,
    root: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV for multiple symbols."""
    return {sym: load_ohlcv(sym, interval, root) for sym in symbols}
