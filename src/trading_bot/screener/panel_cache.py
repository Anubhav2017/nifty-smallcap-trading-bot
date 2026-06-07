"""Pre-built historical screener panel cache (parquet + manifest)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from trading_bot.data.dataset_store import dataset_root
from trading_bot.screener.historical import HistoricalScreener

DEFAULT_CACHE_DIR = "screener_panel"
PANEL_FILE = "panel.parquet"
MANIFEST_FILE = "manifest.json"


def cache_dir(root: Path | str) -> Path:
    return dataset_root(Path(root)) / DEFAULT_CACHE_DIR


def manifest_path(root: Path | str) -> Path:
    return cache_dir(root) / MANIFEST_FILE


def panel_path(root: Path | str) -> Path:
    return cache_dir(root) / PANEL_FILE


def load_manifest(root: Path | str) -> dict:
    path = manifest_path(root)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_panel(
    root: Path | str,
    as_of: str | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Load cached panel; optionally filter to one date and/or symbols."""
    path = panel_path(root)
    if not path.is_file():
        raise FileNotFoundError(
            f"No panel cache at {path}. Run: scripts/historical_screener.py build-cache"
        )
    df = pd.read_parquet(path)
    if as_of is not None:
        df = df[df["as_of_date"] == pd.Timestamp(as_of).strftime("%Y-%m-%d")]
    if symbols:
        sym_set = {s.upper() for s in symbols}
        df = df[df["symbol"].isin(sym_set)]
    return df.reset_index(drop=True)


def build_panel_cache(
    root: Path | str,
    start: str,
    end: str,
    *,
    freq: str = "W-FRI",
    symbols: Optional[list[str]] = None,
    incremental: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Build or extend the screener panel cache.

    Returns ``(panel_df, manifest)``.
    """
    root = dataset_root(Path(root))
    out_dir = cache_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()
    manifest = load_manifest(root)
    panel_file = panel_path(root)

    if incremental and panel_file.is_file() and manifest:
        existing = pd.read_parquet(panel_file)
        cache_start = manifest.get("date_range", {}).get("from")
        cache_end = manifest.get("date_range", {}).get("to")
        if cache_start and cache_end:
            start_ts = min(pd.Timestamp(start), pd.Timestamp(cache_start))
            end_ts = max(pd.Timestamp(end), pd.Timestamp(cache_end))
            start, end = start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")

    screener = HistoricalScreener(root)
    new_panel = screener.build_panel(start, end, symbols, freq=freq, use_adjusted=True)

    if not existing.empty:
        combined = pd.concat([existing, new_panel], ignore_index=True)
        combined = combined.drop_duplicates(subset=["symbol", "as_of_date"], keep="last")
        combined = combined.sort_values(["as_of_date", "symbol"]).reset_index(drop=True)
    else:
        combined = new_panel

    combined.to_parquet(panel_file, index=False)

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "date_range": {"from": start, "to": end},
        "freq": freq,
        "symbols": int(combined["symbol"].nunique()),
        "rows": int(len(combined)),
        "columns": list(combined.columns),
        "incremental": incremental,
    }
    manifest_path(root).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return combined, manifest
