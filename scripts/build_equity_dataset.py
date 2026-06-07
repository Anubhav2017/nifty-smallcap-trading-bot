#!/usr/bin/env python3
"""Build a reusable NSE cash-equity OHLCV dataset (no F&O)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from kiteconnect import KiteConnect

from dataset_config import DatasetBuildConfig, load_dataset_config, load_universe_symbols
from download_kite_ohlcv import download_constituents, run_download, validate_auth
from env_utils import load_env_file
from kite_equity import enrich_universe, nse_eq_instruments_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build structured NSE equity dataset under dataset/ (OHLCV + universe + instruments)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.dataset.equity.json"),
        help="Dataset build config JSON (default: config.dataset.equity.json)",
    )
    parser.add_argument("--api-key", default=os.getenv("KITE_API_KEY"))
    parser.add_argument("--access-token", default=os.getenv("KITE_ACCESS_TOKEN"))
    return parser.parse_args()


def _write_manifest(cfg: DatasetBuildConfig, symbols: list[str], enriched: pd.DataFrame) -> None:
    cfg.meta_dir.mkdir(parents=True, exist_ok=True)
    try:
        symbols_rel = str(cfg.symbols_csv.relative_to(cfg.dataset_root))
    except ValueError:
        symbols_rel = str(cfg.symbols_csv)

    manifest = {
        "version": 1,
        "asset_class": "nse_equity",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe": {
            "name": cfg.universe_name,
            "symbol_count": len(symbols),
            "symbols_csv": symbols_rel,
            "enriched_csv": "universe/universe_enriched.csv",
        },
        "date_range": {"from": cfg.from_date, "to": cfg.to_date},
        "intervals": cfg.intervals,
        "ohlcv_path_template": "ohlcv/{interval}/{symbol}.csv",
        "columns": ["date", "open", "high", "low", "close", "volume"],
        "instruments": {
            "latest": "instruments/nse_eq_latest.csv",
            "note": "NSE EQ only; refreshed on each build if save_instruments=true",
        },
        "missing_from_kite": enriched.loc[~enriched["found"], "symbol"].tolist(),
    }
    out = cfg.dataset_root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def build(cfg: DatasetBuildConfig, api_key: str, access_token: str) -> None:
    cfg.dataset_root.mkdir(parents=True, exist_ok=True)
    cfg.universe_dir.mkdir(parents=True, exist_ok=True)

    if cfg.refresh_universe_from_nse:
        urls = (cfg.nse_constituents_url,) if cfg.nse_constituents_url else None
        print(f"Refreshing {cfg.universe_name} universe from NSE -> {cfg.symbols_csv}")
        download_constituents(cfg.symbols_csv, csv_urls=urls)

    symbols = load_universe_symbols(cfg)
    print(f"Universe: {len(symbols)} symbols from {cfg.symbols_csv}")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    if cfg.save_instruments:
        cfg.instruments_dir.mkdir(parents=True, exist_ok=True)
        inst = nse_eq_instruments_df(kite)
        stamp = datetime.now().strftime("%Y%m%d")
        dated = cfg.instruments_dir / f"nse_eq_{stamp}.csv"
        latest = cfg.instruments_dir / "nse_eq_latest.csv"
        inst.to_csv(dated, index=False)
        inst.to_csv(latest, index=False)
        print(f"Saved {len(inst)} NSE EQ instruments -> {latest}")

    enriched = enrich_universe(symbols, kite)
    enriched_path = cfg.universe_dir / "universe_enriched.csv"
    enriched.to_csv(enriched_path, index=False)
    found = enriched["found"].sum()
    print(f"Universe enriched: {found}/{len(symbols)} symbols mapped to Kite tokens -> {enriched_path}")

    tradable = enriched.loc[enriched["found"], "symbol"].tolist()
    run_download(
        kite=kite,
        symbols=tradable,
        from_dt=cfg.from_dt,
        to_dt=cfg.to_dt,
        intervals=cfg.intervals,
        output_dir=cfg.ohlcv_dir,
        chunk_days=cfg.chunk_days,
        sleep_seconds=cfg.sleep_seconds,
        skip_existing=cfg.skip_existing,
    )

    _write_manifest(cfg, symbols, enriched)
    print(f"\nDataset ready under: {cfg.dataset_root}")
    print("Load later: from dataset_loader import load_ohlcv, load_manifest, list_symbols")


def main() -> None:
    load_env_file()
    args = parse_args()
    validate_auth(args.api_key, args.access_token)
    cfg = load_dataset_config(args.config.resolve())
    build(cfg, args.api_key, args.access_token)


if __name__ == "__main__":
    main()
