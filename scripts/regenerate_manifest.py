#!/usr/bin/env python3
"""Regenerate dataset_smallcap250/manifest.json from existing on-disk artifacts.

The dataset was built without the final manifest-writing step (or the manifest
was removed).  ``trading_bot.data.universe.Universe`` needs the manifest for the
universe ``date_range.from`` (the point-in-time "add" date for all symbols) and
the instruments / universe CSV paths.

This script does NOT re-download anything; it only inspects the files already
present under the dataset root and writes a manifest matching the schema in
``scripts/build_equity_dataset.py::_write_manifest``.

Usage:
    python scripts/regenerate_manifest.py
    python scripts/regenerate_manifest.py --dataset dataset_smallcap250
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _scan_ohlcv_date_range(day_dir: Path) -> tuple[str, str]:
    """Return (min_date, max_date) ISO strings across all OHLCV CSVs.

    Reads only the first/last date of each file (cheap) by parsing the full
    date column per file but keeping just min/max.
    """
    min_date: str | None = None
    max_date: str | None = None
    for csv in sorted(day_dir.glob("*.csv")):
        try:
            dates = pd.read_csv(csv, usecols=["date"])["date"]
        except Exception:
            continue
        if dates.empty:
            continue
        d = pd.to_datetime(dates, errors="coerce").dropna()
        if d.empty:
            continue
        lo = d.min().date().isoformat()
        hi = d.max().date().isoformat()
        if min_date is None or lo < min_date:
            min_date = lo
        if max_date is None or hi > max_date:
            max_date = hi
    if min_date is None or max_date is None:
        raise SystemExit(f"No parseable OHLCV dates found under {day_dir}")
    return min_date, max_date


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset_smallcap250", help="Dataset root folder")
    ap.add_argument("--universe-name", default="nifty_smallcap_250",
                    help="Universe name to record in the manifest")
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    day_dir = root / "ohlcv" / "day"
    enriched_path = root / "universe" / "universe_enriched.csv"

    if not day_dir.is_dir():
        raise SystemExit(f"Missing OHLCV directory: {day_dir}")
    if not enriched_path.is_file():
        raise SystemExit(f"Missing universe enriched CSV: {enriched_path}")

    enriched = pd.read_csv(enriched_path)
    symbols = [str(s).strip().upper() for s in enriched["symbol"].tolist()]

    # "found" column: which symbols mapped to a Kite token.  Default all True
    # if the column is absent so they are treated as tradable.
    if "found" in enriched.columns:
        found_mask = enriched["found"].astype(bool)
    else:
        found_mask = pd.Series([True] * len(enriched))
    missing = enriched.loc[~found_mask, "symbol"].tolist()

    from_date, to_date = _scan_ohlcv_date_range(day_dir)

    instruments_latest = root / "instruments" / "nse_eq_latest.csv"
    bse_dir = root / "bse_announcements"

    manifest: dict = {
        "version": 1,
        "asset_class": "nse_equity",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "regenerated_from_disk": True,
        "universe": {
            "name": args.universe_name,
            "symbol_count": len(symbols),
            "symbols_csv": "universe/universe_enriched.csv",
            "enriched_csv": "universe/universe_enriched.csv",
        },
        "date_range": {"from": from_date, "to": to_date},
        "intervals": ["day"],
        "ohlcv_path_template": "ohlcv/{interval}/{symbol}.csv",
        "columns": ["date", "open", "high", "low", "close", "volume"],
        "instruments": {
            "latest": "instruments/nse_eq_latest.csv"
            if instruments_latest.is_file()
            else "",
            "note": "NSE EQ only; regenerated from existing files (no re-download).",
        },
        "missing_from_kite": missing,
    }

    if bse_dir.is_dir():
        manifest["bse_announcements"] = {
            "path": "bse_announcements",
            "layout": "{symbol}/announcements.csv",
        }

    out = root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"  symbols       : {len(symbols)} ({len(missing)} missing from Kite)")
    print(f"  date_range    : {from_date} -> {to_date}")
    print(f"  instruments   : {manifest['instruments']['latest'] or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
