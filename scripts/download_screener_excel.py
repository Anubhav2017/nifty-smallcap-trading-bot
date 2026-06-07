#!/usr/bin/env python3
"""Bulk-download Screener.in 'Export to Excel' files for a symbol universe."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from dataset_config import load_dataset_config, load_universe_symbols
from download_config import load_symbols_from_csv
from env_utils import load_env_file, repo_root, resolve_repo_path
from screener_client import (
    ScreenerAuthError,
    ScreenerError,
    ScreenerExportError,
    ScreenerNotFoundError,
    ScreenerRateLimitError,
    ScreenerSession,
)


@dataclass(frozen=True)
class ScreenerDownloadConfig:
    symbols: List[str]
    output_dir: Path
    cookies_file: Optional[Path]
    sleep_seconds: float
    skip_existing: bool
    consolidated: bool
    rate_limit_max_retries: int
    rate_limit_base_seconds: float
    request_pause_seconds: float


def _load_json_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object.")
    return raw


def load_screener_config(path: Path) -> ScreenerDownloadConfig:
    config_path = path.resolve()
    raw = _load_json_config(path)

    symbols: List[str] = []
    if raw.get("symbols_from"):
        ds_path = resolve_repo_path(config_path, str(raw["symbols_from"]))
        ds_cfg = load_dataset_config(ds_path)
        symbols = load_universe_symbols(ds_cfg)
    elif raw.get("symbols_csv"):
        symbols = load_symbols_from_csv(
            resolve_repo_path(config_path, str(raw["symbols_csv"])),
            str(raw.get("symbol_column", "symbol")),
        )
    else:
        raise ValueError("Config must include 'symbols_from' or 'symbols_csv'.")

    cookies_file = raw.get("cookies_file")
    cookies_path = resolve_repo_path(config_path, str(cookies_file)) if cookies_file else None

    return ScreenerDownloadConfig(
        symbols=symbols,
        output_dir=resolve_repo_path(config_path, str(raw.get("output_dir", "screener_excel"))),
        cookies_file=cookies_path,
        sleep_seconds=float(raw.get("sleep_seconds", 4.0)),
        skip_existing=bool(raw.get("skip_existing", True)),
        consolidated=bool(raw.get("consolidated", True)),
        rate_limit_max_retries=int(raw.get("rate_limit_max_retries", 6)),
        rate_limit_base_seconds=float(raw.get("rate_limit_base_seconds", 30)),
        request_pause_seconds=float(raw.get("request_pause_seconds", 0.75)),
    )


def _cookie_source_label(cfg: ScreenerDownloadConfig) -> str:
    if cfg.cookies_file is not None:
        return str(cfg.cookies_file)
    return "SCREENER_SESSIONID (.env)"


def _missing_required_cookies(path: Path) -> list[str]:
    """Return required cookie names absent from a Netscape cookies.txt file."""
    found: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) == 7:
            found.add(parts[5].strip().lower())
    return sorted({"sessionid", "csrftoken"} - found)


def check_screener_session(
    cfg: ScreenerDownloadConfig,
    *,
    probe_export: bool = False,
    probe_symbol: str = "RELIANCE",
) -> None:
    """Validate Screener cookies (login page + optional Excel export probe).

    Raises SystemExit with actionable guidance when cookies are missing or invalid.
    """
    import os

    source = _cookie_source_label(cfg)
    if cfg.cookies_file is not None:
        if not cfg.cookies_file.is_file():
            raise SystemExit(
                f"Screener cookies file not found: {cfg.cookies_file}\n"
                "Log in at https://www.screener.in, export Netscape-format cookies, "
                "and save them to screener_cookies.txt in the project root."
            )
        missing = _missing_required_cookies(cfg.cookies_file)
        if missing:
            raise SystemExit(
                f"Screener cookies file is missing required entries: {', '.join(missing)}\n"
                f"File: {cfg.cookies_file}\n"
                "Re-export cookies from your browser after logging in at https://www.screener.in"
            )
    elif not os.getenv("SCREENER_SESSIONID", "").strip():
        raise SystemExit(
            "No Screener cookies configured.\n"
            "Save screener_cookies.txt in the project root or set SCREENER_SESSIONID in .env"
        )

    client = ScreenerSession(
        cookies_file=cfg.cookies_file,
        rate_limit_max_retries=cfg.rate_limit_max_retries,
        rate_limit_base_seconds=cfg.rate_limit_base_seconds,
        request_pause_seconds=cfg.request_pause_seconds,
    )
    try:
        client.verify_logged_in()
    except ScreenerAuthError as exc:
        raise SystemExit(
            f"Screener login check failed: {exc}\n"
            f"Update cookies at {source} and try again."
        ) from exc

    print(f"  Login: OK (watchlist accessible)")

    if probe_export:
        try:
            ref = client.resolve_company(probe_symbol, consolidated=cfg.consolidated)
            content = client.export_excel(ref)
        except ScreenerAuthError as exc:
            raise SystemExit(
                f"Screener login OK but Excel export failed (auth): {exc}\n"
                f"Refresh cookies at {source}."
            ) from exc
        except ScreenerNotFoundError as exc:
            raise SystemExit(f"Export probe symbol not found: {exc}") from exc
        except ScreenerExportError as exc:
            raise SystemExit(
                f"Screener login OK but Excel export failed: {exc}\n"
                "Ensure your Screener.in account can export Excel files."
            ) from exc
        except ScreenerError as exc:
            raise SystemExit(f"Screener export probe failed: {exc}") from exc

        label = "consolidated" if ref.consolidated else "standalone"
        print(
            f"  Export probe ({probe_symbol}): OK "
            f"({len(content) // 1024} KB, {label})"
        )

    print(f"Screener cookies OK (source: {source})")


def _output_path(output_dir: Path, symbol: str, consolidated: bool) -> Path:
    suffix = "_consolidated" if consolidated else "_standalone"
    return output_dir / f"{symbol}{suffix}.xlsx"


def _existing_output_path(output_dir: Path, symbol: str) -> Optional[Path]:
    for consolidated in (True, False):
        path = _output_path(output_dir, symbol, consolidated)
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _write_manifest(cfg: ScreenerDownloadConfig, results: list[dict[str, Any]]) -> None:
    ok = sum(1 for r in results if r.get("status") == "ok")
    manifest = {
        "version": 1,
        "source": "screener.in",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(cfg.output_dir),
        "symbol_count": len(cfg.symbols),
        "downloaded_ok": ok,
        "consolidated": cfg.consolidated,
        "per_symbol_layout": "{symbol}[_consolidated|_standalone].xlsx",
        "results": results,
    }
    (cfg.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def run_download(cfg: ScreenerDownloadConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    client = ScreenerSession(
        cookies_file=cfg.cookies_file,
        rate_limit_max_retries=cfg.rate_limit_max_retries,
        rate_limit_base_seconds=cfg.rate_limit_base_seconds,
        request_pause_seconds=cfg.request_pause_seconds,
    )

    try:
        email = client.verify_logged_in()
        print(f"Screener session OK ({email})")
    except ScreenerAuthError as exc:
        raise SystemExit(str(exc)) from exc

    results: list[dict[str, Any]] = []
    done = 0
    skipped = 0

    print(
        f"Downloading Excel exports for {len(cfg.symbols)} symbols -> {cfg.output_dir}"
    )

    for idx, symbol in enumerate(cfg.symbols, start=1):
        existing = _existing_output_path(cfg.output_dir, symbol)
        if cfg.skip_existing and existing is not None:
            skipped += 1
            print(f"[{idx}/{len(cfg.symbols)}] {symbol} skip (exists)")
            results.append(
                {
                    "nse_symbol": symbol,
                    "status": "skipped",
                    "path": str(existing),
                }
            )
            continue

        print(f"[{idx}/{len(cfg.symbols)}] {symbol} ...", flush=True)
        try:
            ref = client.resolve_company(symbol, consolidated=cfg.consolidated)
            content = client.export_excel(ref)
            out_path = _output_path(cfg.output_dir, symbol, ref.consolidated)
            out_path.write_bytes(content)
            done += 1
            label = "consolidated" if ref.consolidated else "standalone (no consolidated export)"
            print(f"    saved {out_path.name} ({len(content) // 1024} KB, {label})")
            results.append(
                {
                    "nse_symbol": symbol,
                    "status": "ok",
                    "screener_slug": ref.screener_slug,
                    "warehouse_id": ref.warehouse_id,
                    "company_url": ref.company_url,
                    "consolidated": ref.consolidated,
                    "path": str(out_path),
                    "bytes": len(content),
                }
            )
        except ScreenerNotFoundError as exc:
            print(f"    not found: {exc}")
            results.append({"nse_symbol": symbol, "status": "not_found", "error": str(exc)})
        except ScreenerRateLimitError as exc:
            print(f"    rate limit: {exc}")
            results.append({"nse_symbol": symbol, "status": "rate_limited", "error": str(exc)})
            time.sleep(cfg.rate_limit_base_seconds)
        except (ScreenerAuthError, ScreenerExportError) as exc:
            print(f"    failed: {exc}")
            results.append({"nse_symbol": symbol, "status": "error", "error": str(exc)})
            if isinstance(exc, ScreenerAuthError):
                break
        except ScreenerError as exc:
            print(f"    error: {exc}")
            results.append({"nse_symbol": symbol, "status": "error", "error": str(exc)})

        time.sleep(cfg.sleep_seconds)

    _write_manifest(cfg, results)
    print(f"\nDone. Downloaded: {done}, skipped (existing): {skipped}, output: {cfg.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Screener.in Excel exports for symbols in a config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "config/screener.smallcap250.json",
        help="Screener download config JSON",
    )
    parser.add_argument(
        "--check-session",
        action="store_true",
        help="Verify Screener login (and Excel export) and exit",
    )
    parser.add_argument(
        "--probe-export",
        action="store_true",
        help="With --check-session, also download a sample Excel export",
    )
    parser.add_argument(
        "--probe-symbol",
        default="RELIANCE",
        help="Symbol used for --probe-export (default: RELIANCE)",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    cfg = load_screener_config(args.config.resolve())

    if args.check_session:
        check_screener_session(
            cfg,
            probe_export=args.probe_export,
            probe_symbol=args.probe_symbol.upper(),
        )
        return

    run_download(cfg)


if __name__ == "__main__":
    main()
