#!/usr/bin/env python3
"""Build a reusable NSE cash-equity dataset (OHLCV + optional BSE + Screener)."""

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


def _needs_kite(cfg: DatasetBuildConfig, *, skip_ohlcv: bool, skip_screener: bool) -> bool:
    """Kite is required for OHLCV; also checked when Screener is enabled (full dataset build)."""
    if not skip_ohlcv:
        return True
    return not skip_screener and cfg.screener_config is not None


def _needs_screener(cfg: DatasetBuildConfig, *, skip_screener: bool) -> bool:
    return not skip_screener and cfg.screener_config is not None


def validate_kite_session(api_key: str, access_token: str) -> None:
    """Ensure Kite credentials exist and the access token is still valid."""
    validate_auth(api_key, access_token)
    from kite_login import _format_user, check_existing_session

    ok, profile, err = check_existing_session(api_key, access_token)
    if not ok or profile is None:
        raise SystemExit(
            f"Kite login invalid or expired: {err or 'unknown error'}\n"
            "Run: python scripts/kite_login.py"
        )
    print(f"Kite session OK ({_format_user(profile)})")


def validate_screener_session(cfg: DatasetBuildConfig) -> None:
    """Ensure Screener cookies exist and the session is logged in."""
    if cfg.screener_config is None:
        return
    if not cfg.screener_config.is_file():
        raise SystemExit(f"Screener config not found: {cfg.screener_config}")

    from download_screener_excel import check_screener_session, load_screener_config

    screener_cfg = load_screener_config(cfg.screener_config)
    check_screener_session(screener_cfg, probe_export=False)


def run_preflight_checks(
    cfg: DatasetBuildConfig,
    api_key: str,
    access_token: str,
    *,
    skip_ohlcv: bool,
    skip_bse: bool,
    skip_screener: bool,
) -> None:
    """Validate auth before starting long-running downloads."""
    checks: list[str] = []
    if _needs_kite(cfg, skip_ohlcv=skip_ohlcv, skip_screener=skip_screener):
        checks.append("kite")
    if _needs_screener(cfg, skip_screener=skip_screener):
        checks.append("screener")

    if not checks:
        return

    print("=== Preflight checks ===")
    if "kite" in checks:
        validate_kite_session(api_key, access_token)
    if "screener" in checks:
        validate_screener_session(cfg)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build structured NSE equity dataset: OHLCV, universe, instruments, "
            "and optionally BSE announcements + Screener.in Excel exports."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/dataset.smallcap250.json"),
        help="Dataset build config JSON (default: config/dataset.smallcap250.json)",
    )
    parser.add_argument("--api-key", default=os.getenv("KITE_API_KEY"))
    parser.add_argument("--access-token", default=os.getenv("KITE_ACCESS_TOKEN"))
    parser.add_argument(
        "--skip-ohlcv",
        action="store_true",
        help="Skip Kite OHLCV download (universe refresh still runs if configured)",
    )
    parser.add_argument(
        "--skip-minute",
        action="store_true",
        help="Skip minute-level OHLCV intervals (only daily/higher intervals are downloaded)",
    )
    parser.add_argument(
        "--skip-bse",
        action="store_true",
        help="Skip BSE announcements download even if bse_config is set",
    )
    parser.add_argument(
        "--skip-screener",
        action="store_true",
        help="Skip Screener.in Excel download even if screener_config is set",
    )
    return parser.parse_args()


def _filter_intervals(intervals: list[str], *, skip_minute: bool) -> list[str]:
    """Return the configured intervals, optionally dropping minute-level ones."""
    if not skip_minute:
        return list(intervals)
    return [iv for iv in intervals if "minute" not in iv.lower()]


def _write_manifest(
    cfg: DatasetBuildConfig,
    symbols: list[str],
    enriched: pd.DataFrame,
    *,
    intervals: list[str] | None = None,
    bse_dir: Path | None = None,
    screener_dir: Path | None = None,
) -> None:
    cfg.meta_dir.mkdir(parents=True, exist_ok=True)
    try:
        symbols_rel = str(cfg.symbols_csv.relative_to(cfg.dataset_root))
    except ValueError:
        symbols_rel = str(cfg.symbols_csv)

    manifest: dict = {
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
        "intervals": list(intervals) if intervals is not None else cfg.intervals,
        "ohlcv_path_template": "ohlcv/{interval}/{symbol}.csv",
        "columns": ["date", "open", "high", "low", "close", "volume"],
        "instruments": {
            "latest": "instruments/nse_eq_latest.csv",
            "note": "NSE EQ only; refreshed on each build if save_instruments=true",
        },
        "missing_from_kite": enriched.loc[~enriched["found"], "symbol"].tolist(),
    }
    if bse_dir is not None:
        try:
            bse_rel = str(bse_dir.relative_to(cfg.dataset_root))
        except ValueError:
            bse_rel = str(bse_dir)
        manifest["bse_announcements"] = {
            "path": bse_rel,
            "layout": "{symbol}/announcements.csv",
        }
    if screener_dir is not None:
        try:
            screener_rel = str(screener_dir.relative_to(cfg.dataset_root))
        except ValueError:
            screener_rel = str(screener_dir)
        manifest["screener_excel"] = {
            "path": screener_rel,
            "layout": "{symbol}[_consolidated|_standalone].xlsx",
        }
    out = cfg.dataset_root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def _refresh_universe(cfg: DatasetBuildConfig) -> list[str]:
    cfg.dataset_root.mkdir(parents=True, exist_ok=True)
    cfg.universe_dir.mkdir(parents=True, exist_ok=True)

    if cfg.refresh_universe_from_nse:
        urls = (cfg.nse_constituents_url,) if cfg.nse_constituents_url else None
        print(f"Refreshing {cfg.universe_name} universe from NSE -> {cfg.symbols_csv}")
        download_constituents(cfg.symbols_csv, csv_urls=urls)

    symbols = load_universe_symbols(cfg)
    print(f"Universe: {len(symbols)} symbols from {cfg.symbols_csv}")
    return symbols


def _download_ohlcv(
    cfg: DatasetBuildConfig,
    symbols: list[str],
    api_key: str,
    access_token: str,
    *,
    intervals: list[str] | None = None,
) -> pd.DataFrame:
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
    effective_intervals = list(intervals) if intervals is not None else list(cfg.intervals)
    run_download(
        kite=kite,
        symbols=tradable,
        from_dt=cfg.from_dt,
        to_dt=cfg.to_dt,
        intervals=effective_intervals,
        output_dir=cfg.ohlcv_dir,
        chunk_days=cfg.chunk_days,
        sleep_seconds=cfg.sleep_seconds,
        skip_existing=cfg.skip_existing,
    )
    return enriched


def _download_bse(cfg: DatasetBuildConfig) -> Path | None:
    if cfg.bse_config is None:
        return None
    if not cfg.bse_config.is_file():
        raise FileNotFoundError(f"BSE config not found: {cfg.bse_config}")

    from download_bse_announcements import load_bse_config, run_download as run_bse_download

    print(f"\n=== BSE announcements ({cfg.bse_config.name}) ===")
    bse_cfg = load_bse_config(cfg.bse_config)
    run_bse_download(bse_cfg)
    return bse_cfg.output_dir


def _download_screener(cfg: DatasetBuildConfig) -> Path | None:
    if cfg.screener_config is None:
        return None
    if not cfg.screener_config.is_file():
        raise FileNotFoundError(f"Screener config not found: {cfg.screener_config}")

    from download_screener_excel import load_screener_config, run_download as run_screener_download

    print(f"\n=== Screener.in Excel ({cfg.screener_config.name}) ===")
    screener_cfg = load_screener_config(cfg.screener_config)
    run_screener_download(screener_cfg)
    return screener_cfg.output_dir


def build(
    cfg: DatasetBuildConfig,
    api_key: str,
    access_token: str,
    *,
    skip_ohlcv: bool = False,
    skip_minute: bool = False,
    skip_bse: bool = False,
    skip_screener: bool = False,
) -> None:
    symbols = _refresh_universe(cfg)

    effective_intervals = _filter_intervals(cfg.intervals, skip_minute=skip_minute)
    if skip_minute:
        dropped = [iv for iv in cfg.intervals if iv not in effective_intervals]
        if dropped:
            print(f"--skip-minute: dropping interval(s) {dropped}; will download {effective_intervals}")
        else:
            print("--skip-minute set, but config has no minute-level intervals to drop.")

    if skip_ohlcv:
        enriched_path = cfg.universe_dir / "universe_enriched.csv"
        if enriched_path.is_file():
            enriched = pd.read_csv(enriched_path)
        else:
            enriched = pd.DataFrame({"symbol": symbols, "found": [False] * len(symbols)})
            print("Skipping OHLCV — universe_enriched.csv not found; manifest will list all symbols as missing from Kite.")
    elif not effective_intervals:
        enriched_path = cfg.universe_dir / "universe_enriched.csv"
        if enriched_path.is_file():
            enriched = pd.read_csv(enriched_path)
        else:
            enriched = pd.DataFrame({"symbol": symbols, "found": [False] * len(symbols)})
        print("All configured intervals were skipped — no OHLCV download will run.")
    else:
        enriched = _download_ohlcv(
            cfg, symbols, api_key, access_token, intervals=effective_intervals
        )

    bse_dir: Path | None = None
    if not skip_bse and cfg.bse_config is not None:
        bse_dir = _download_bse(cfg)
    elif cfg.bse_config is None and not skip_bse:
        print("\nBSE announcements: skipped (no bse_config in dataset config)")

    screener_dir: Path | None = None
    if not skip_screener and cfg.screener_config is not None:
        screener_dir = _download_screener(cfg)
    elif cfg.screener_config is None and not skip_screener:
        print("\nScreener Excel: skipped (no screener_config in dataset config)")

    _write_manifest(
        cfg,
        symbols,
        enriched,
        intervals=effective_intervals,
        bse_dir=bse_dir,
        screener_dir=screener_dir,
    )
    print(f"\nDataset ready under: {cfg.dataset_root}")
    print("Load later: from dataset_loader import load_ohlcv, load_manifest, list_symbols")


def main() -> None:
    load_env_file()
    args = parse_args()
    cfg = load_dataset_config(args.config.resolve())

    run_preflight_checks(
        cfg,
        args.api_key,
        args.access_token,
        skip_ohlcv=args.skip_ohlcv,
        skip_bse=args.skip_bse,
        skip_screener=args.skip_screener,
    )

    build(
        cfg,
        args.api_key,
        args.access_token,
        skip_ohlcv=args.skip_ohlcv,
        skip_minute=args.skip_minute,
        skip_bse=args.skip_bse,
        skip_screener=args.skip_screener,
    )


if __name__ == "__main__":
    main()
