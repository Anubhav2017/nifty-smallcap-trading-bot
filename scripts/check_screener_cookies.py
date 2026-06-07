#!/usr/bin/env python3
"""Verify Screener.in cookies are valid for Excel downloads."""

from __future__ import annotations

import argparse
from pathlib import Path

from download_screener_excel import check_screener_session, load_screener_config
from env_utils import load_env_file, repo_root

DEFAULT_CONFIG = repo_root() / "config/screener.smallcap250.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that Screener.in cookies are configured and working."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Screener config JSON (default: config/screener.smallcap250.json)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only check login (skip Excel export probe)",
    )
    parser.add_argument(
        "--probe-symbol",
        default="RELIANCE",
        help="Symbol for export probe (default: RELIANCE)",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    cfg = load_screener_config(args.config.resolve())
    print("Checking Screener.in cookies...")
    check_screener_session(
        cfg,
        probe_export=not args.quick,
        probe_symbol=args.probe_symbol.upper(),
    )


if __name__ == "__main__":
    main()
