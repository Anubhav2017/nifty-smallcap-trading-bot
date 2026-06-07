"""Load built NSE equity OHLCV datasets (dataset_smallcap250/, dataset_nifty50/, etc.)."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from trading_bot.config import Config

logger = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = Path("dataset_smallcap250")
_DAILY_COLS = ["date", "open", "high", "low", "close", "volume"]


def dataset_root(path: Optional[Path] = None) -> Path:
    return (path or DEFAULT_DATASET_ROOT).resolve()


def dataset_root_from_config(cfg: Config | None = None) -> Path:
    if cfg is None:
        return dataset_root()
    data = cfg.get("data", {}) or {}
    return dataset_root(Path(str(data.get("dataset_root", DEFAULT_DATASET_ROOT))))


def manifest_path(root: Optional[Path] = None) -> Path:
    return dataset_root(root) / "manifest.json"


def load_manifest(root: Optional[Path] = None) -> dict:
    path = manifest_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"No manifest at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def list_symbols(interval: str, root: Optional[Path] = None) -> List[str]:
    if interval != "day":
        raise ValueError(f"Only daily OHLCV is supported (got {interval!r}).")
    folder = dataset_root(root) / "ohlcv" / "day"
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.csv"))


def load_universe(root: Optional[Path] = None) -> pd.DataFrame:
    root = dataset_root(root)
    enriched = root / "universe" / "universe_enriched.csv"
    if enriched.is_file():
        return pd.read_csv(enriched)
    manifest = load_manifest(root)
    rel = manifest["universe"].get("enriched_csv") or manifest["universe"]["symbols_csv"]
    path = root / rel if not Path(rel).is_absolute() else Path(rel)
    return pd.read_csv(path)


def load_instruments(root: Optional[Path] = None) -> pd.DataFrame:
    """Load NSE EQ instrument dump from the dataset (``instruments/nse_eq_latest.csv``)."""
    root = dataset_root(root)
    manifest = load_manifest(root)
    rel = manifest.get("instruments", {}).get("latest", "instruments/nse_eq_latest.csv")
    path = root / rel if not Path(str(rel)).is_absolute() else Path(rel)
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"tradingsymbol": str})


def load_ohlcv(
    symbol: str,
    interval: str,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    if interval != "day":
        raise ValueError(f"Only daily OHLCV is supported (got {interval!r}).")
    path = dataset_root(root) / "ohlcv" / "day" / f"{symbol.upper()}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.sort_values("date").reset_index(drop=True)


def load_daily_bars(
    symbol: str,
    start: date,
    end: date,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    """Daily OHLCV for *symbol* in [start, end] from ``ohlcv/day/{SYMBOL}.csv``."""
    root = dataset_root(root)
    sym = symbol.upper()
    day_path = root / "ohlcv" / "day" / f"{sym}.csv"
    try:
        if not day_path.is_file():
            raise FileNotFoundError(day_path)
        df = _normalize_daily(load_ohlcv(sym, "day", root))
    except FileNotFoundError:
        return pd.DataFrame(columns=_DAILY_COLS)

    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].reset_index(drop=True)
