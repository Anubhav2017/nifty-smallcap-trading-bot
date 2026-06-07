#!/usr/bin/env python3
"""Remove minute OHLCV from a built dataset and update manifest.json."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def remove_minute_data(root: Path, *, dry_run: bool = False) -> None:
    root = root.resolve()
    minute_dir = root / "ohlcv" / "minute"
    manifest_path = root / "manifest.json"

    if minute_dir.is_dir():
        n = len(list(minute_dir.glob("*.csv")))
        print(f"{'Would remove' if dry_run else 'Removing'} {n} minute CSV(s) under {minute_dir}")
        if not dry_run:
            shutil.rmtree(minute_dir)
    else:
        print(f"No minute directory at {minute_dir}")

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        intervals = manifest.get("intervals", [])
        if "minute" in intervals:
            manifest["intervals"] = [i for i in intervals if i != "minute"]
            if not manifest["intervals"]:
                manifest["intervals"] = ["day"]
            print(f"{'Would update' if dry_run else 'Updating'} manifest intervals → {manifest['intervals']}")
            if not dry_run:
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        else:
            print("manifest.json already has no minute interval")
    else:
        print(f"No manifest at {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="dataset_smallcap250",
        help="Dataset root (default: dataset_smallcap250)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing files")
    args = parser.parse_args()
    remove_minute_data(Path(args.dataset_root), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
