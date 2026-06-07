#!/usr/bin/env python3
"""Bulk-download BSE corporate announcements for a dataset symbol universe."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

from bse_announcements import (
    AnnouncementFetchResult,
    BseAnnouncementError,
    BseScripNotFoundError,
    PROGRESS_STYLE_COMPACT,
    PROGRESS_STYLE_VERBOSE,
    fetch_symbol_announcements,
    is_transient_bse_error,
    load_scrip_cache,
    save_scrip_cache,
)
from bse_category_filter import CategoryFilter, filters_match, parse_category_filter
from dataset_config import load_dataset_config, load_universe_symbols
from download_config import load_symbols_from_csv, resolve_date_range
from env_utils import resolve_repo_path
from tqdm import tqdm


@dataclass(frozen=True)
class BseDownloadConfig:
    symbols: List[str]
    output_dir: Path
    from_date: str
    to_date: str
    download_pdfs: bool
    sleep_seconds: float
    page_sleep_seconds: float
    pdf_sleep_seconds: float
    request_timeout: float
    max_retries: int
    max_workers: int
    pdf_workers: int
    skip_existing: bool
    show_progress: bool
    progress_style: str
    category_filter: CategoryFilter
    retry_until_complete: bool
    retry_wait_seconds: float
    scrip_cache_path: Path
    bse_download_folder: Path


def _load_json_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object.")
    return raw


def load_bse_config(path: Path) -> BseDownloadConfig:
    config_path = path.resolve()
    raw = _load_json_config(path)

    symbols: List[str] = []
    date_raw: dict[str, Any] = dict(raw)
    if raw.get("symbols_from"):
        ds_path = resolve_repo_path(config_path, str(raw["symbols_from"]))
        ds_cfg = load_dataset_config(ds_path)
        symbols = load_universe_symbols(ds_cfg)
        if "years_back" not in raw and "from_date" not in raw and "to_date" not in raw:
            date_raw = {
                "from_date": ds_cfg.from_date,
                "to_date": ds_cfg.to_date,
            }
    elif raw.get("symbols_csv"):
        symbols = load_symbols_from_csv(
            resolve_repo_path(config_path, str(raw["symbols_csv"])),
            str(raw.get("symbol_column", "symbol")),
        )
    else:
        raise ValueError("Config must include 'symbols_from' or 'symbols_csv'.")

    output_dir = resolve_repo_path(config_path, str(raw.get("output_dir", "bse_announcements")))
    from_date, to_date = resolve_date_range(
        date_raw,
        default_from="2021-01-01",
        default_to=datetime.now().strftime("%Y-%m-%d"),
    )

    return BseDownloadConfig(
        symbols=symbols,
        output_dir=output_dir,
        from_date=from_date,
        to_date=to_date,
        download_pdfs=bool(raw.get("download_pdfs", True)),
        sleep_seconds=float(raw.get("sleep_seconds", 1.0)),
        page_sleep_seconds=float(raw.get("page_sleep_seconds", 0.3)),
        pdf_sleep_seconds=float(raw.get("pdf_sleep_seconds", 0.2)),
        request_timeout=float(raw.get("request_timeout", 60)),
        max_retries=int(raw.get("max_retries", 4)),
        max_workers=max(1, int(raw.get("max_workers", 4))),
        pdf_workers=max(1, int(raw.get("pdf_workers", 8))),
        skip_existing=bool(raw.get("skip_existing", True)),
        show_progress=bool(raw.get("show_progress", True)),
        progress_style=_validate_progress_style(
            str(raw.get("progress_style", PROGRESS_STYLE_COMPACT)).strip().lower()
        ),
        category_filter=parse_category_filter(raw),
        retry_until_complete=bool(raw.get("retry_until_complete", True)),
        retry_wait_seconds=float(raw.get("retry_wait_seconds", 600)),
        scrip_cache_path=(output_dir / str(raw.get("scrip_cache", "meta/bse_scrip_codes.json"))).resolve(),
        bse_download_folder=(output_dir / str(raw.get("bse_download_folder", ".cache/bse"))).resolve(),
    )


def _validate_progress_style(style: str) -> str:
    if style in (PROGRESS_STYLE_COMPACT, PROGRESS_STYLE_VERBOSE):
        return style
    raise ValueError(
        f"progress_style must be {PROGRESS_STYLE_COMPACT!r} or {PROGRESS_STYLE_VERBOSE!r}, got {style!r}"
    )


def _worker_bar_position(cfg: BseDownloadConfig, idx: int) -> Optional[int]:
    if not cfg.show_progress:
        return None
    slot = (idx - 1) % max(1, cfg.max_workers)
    if cfg.progress_style == PROGRESS_STYLE_COMPACT:
        return slot + 1  # position 0 = overall symbols bar
    return slot


def _meta_path(symbol_dir: Path) -> Path:
    return symbol_dir / "_meta.json"


def _cleanup_empty_symbol_dirs(output_dir: Path) -> int:
    """Remove symbol folders left behind by failed runs (no announcements.json)."""
    removed = 0
    for path in output_dir.iterdir():
        if not path.is_dir() or path.name in ("meta", ".cache"):
            continue
        if (path / "announcements.json").is_file():
            continue
        # Drop empty tree (e.g. failed fetch that never wrote data)
        for child in path.rglob("*"):
            if child.is_file():
                break
        else:
            shutil.rmtree(path)
            removed += 1
    return removed


def _symbol_done(
    symbol_dir: Path,
    from_date: str,
    to_date: str,
    category_filter: CategoryFilter,
) -> bool:
    meta_file = _meta_path(symbol_dir)
    if not meta_file.is_file():
        return False
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    return (
        meta.get("status") == "ok"
        and meta.get("from_date") == from_date
        and meta.get("to_date") == to_date
        and filters_match(meta, category_filter)
    )


def _write_symbol_meta(
    symbol_dir: Path,
    result: AnnouncementFetchResult,
    status: str,
    category_filter: CategoryFilter,
) -> None:
    meta = {
        "nse_symbol": result.nse_symbol,
        "bse_scrip_code": result.bse_scrip_code,
        "status": status,
        "from_date": result.from_date,
        "to_date": result.to_date,
        "announcement_count": len(result.announcements),
        "pdfs_downloaded": result.pdfs_downloaded,
        "pdfs_skipped": result.pdfs_skipped,
        "pdfs_failed": result.pdfs_failed,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        **category_filter.meta_dict(),
    }
    _meta_path(symbol_dir).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _write_announcements_csv(symbol_dir: Path, announcements: list[dict[str, Any]]) -> None:
    if not announcements:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in announcements:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    path = symbol_dir / "announcements.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in announcements:
            writer.writerow({k: row.get(k) for k in keys})


def _write_manifest(cfg: BseDownloadConfig, results: list[dict[str, Any]]) -> None:
    ok = sum(1 for r in results if r.get("status") == "ok")
    manifest = {
        "version": 1,
        "source": "bseindia.com",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(cfg.output_dir),
        "from_date": cfg.from_date,
        "to_date": cfg.to_date,
        "download_pdfs": cfg.download_pdfs,
        **cfg.category_filter.meta_dict(),
        "symbol_count": len(cfg.symbols),
        "downloaded_ok": ok,
        "per_symbol_layout": "{symbol}/announcements.json, attachments/*.pdf",
        "results": results,
    }
    (cfg.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def _process_symbol(
    cfg: BseDownloadConfig,
    symbol: str,
    idx: int,
    total: int,
    scrip_cache: dict[str, str],
    cache_lock: threading.Lock,
    print_lock: threading.Lock,
) -> dict[str, Any]:
    sym_dir = cfg.output_dir / symbol
    worker_cache = cfg.bse_download_folder / f"worker_{threading.get_ident()}"

    def log(msg: str) -> None:
        with print_lock:
            if cfg.show_progress:
                tqdm.write(msg)
            else:
                print(msg, flush=True)

    if cfg.skip_existing and _symbol_done(sym_dir, cfg.from_date, cfg.to_date, cfg.category_filter):
        log(f"[{idx}/{total}] {symbol} skip (exists)")
        return {"nse_symbol": symbol, "status": "skipped", "path": str(sym_dir)}

    bar_position = _worker_bar_position(cfg, idx)
    try:
        result = fetch_symbol_announcements(
            nse_symbol=symbol,
            output_dir=cfg.output_dir,
            from_date=cfg.from_date,
            to_date=cfg.to_date,
            download_pdfs=cfg.download_pdfs,
            scrip_cache=scrip_cache,
            download_folder=cfg.bse_download_folder,
            page_sleep_seconds=cfg.page_sleep_seconds,
            pdf_sleep_seconds=cfg.pdf_sleep_seconds,
            request_timeout=cfg.request_timeout,
            max_retries=cfg.max_retries,
            pdf_workers=cfg.pdf_workers,
            cache_lock=cache_lock,
            bse_cache_dir=worker_cache,
            show_progress=cfg.show_progress,
            progress_position=bar_position,
            category_filter=cfg.category_filter,
            progress_style=cfg.progress_style,
        )
        _write_announcements_csv(sym_dir, result.announcements)
        _write_symbol_meta(sym_dir, result, "ok", cfg.category_filter)
        log(
            f"[{idx}/{total}] {symbol} done — "
            f"{len(result.announcements)} announcements, scrip {result.bse_scrip_code}, "
            f"PDFs: {result.pdfs_downloaded} new, "
            f"{result.pdfs_skipped} skipped, {result.pdfs_failed} failed"
        )
        return {
            "nse_symbol": symbol,
            "status": "ok",
            "bse_scrip_code": result.bse_scrip_code,
            "announcement_count": len(result.announcements),
            "path": str(sym_dir),
        }
    except BseScripNotFoundError as exc:
        log(f"[{idx}/{total}] {symbol} not on BSE: {exc}")
        return {"nse_symbol": symbol, "status": "not_found", "error": str(exc)}
    except (BseAnnouncementError, ValueError, TimeoutError, ConnectionError) as exc:
        transient = is_transient_bse_error(exc)
        tag = "transient" if transient else "failed"
        log(f"[{idx}/{total}] {symbol} {tag}: {exc}")
        if sym_dir.is_dir() and not (sym_dir / "announcements.json").is_file():
            shutil.rmtree(sym_dir, ignore_errors=True)
        return {
            "nse_symbol": symbol,
            "status": "error",
            "error": str(exc),
            "transient": transient,
        }
    except Exception as exc:
        if not is_transient_bse_error(exc):
            raise
        log(f"[{idx}/{total}] {symbol} transient: {exc}")
        if sym_dir.is_dir() and not (sym_dir / "announcements.json").is_file():
            shutil.rmtree(sym_dir, ignore_errors=True)
        return {
            "nse_symbol": symbol,
            "status": "error",
            "error": str(exc),
            "transient": True,
        }


def _pending_work(cfg: BseDownloadConfig) -> List[Tuple[int, str]]:
    """Symbols that still need a successful download."""
    pending: List[Tuple[int, str]] = []
    for idx, symbol in enumerate(cfg.symbols, start=1):
        sym_dir = cfg.output_dir / symbol
        if _symbol_done(sym_dir, cfg.from_date, cfg.to_date, cfg.category_filter):
            continue
        pending.append((idx, symbol))
    return pending


def _format_wait(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins and secs:
        return f"{mins}m {secs}s"
    if mins:
        return f"{mins}m"
    return f"{secs}s"


def _run_round(
    cfg: BseDownloadConfig,
    work: List[Tuple[int, str]],
    scrip_cache: dict[str, str],
    scrip_cache_path: Path,
    cache_lock: threading.Lock,
    print_lock: threading.Lock,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(cfg.symbols)

    if cfg.show_progress:
        tqdm.set_lock(threading.RLock())

    overall: Optional[tqdm] = None
    if cfg.show_progress and cfg.progress_style == PROGRESS_STYLE_COMPACT:
        overall = tqdm(
            total=len(work),
            desc="This round",
            unit="sym",
            position=0,
            leave=True,
            mininterval=0.5,
        )

    def _record_result(res: dict[str, Any]) -> None:
        results.append(res)
        if overall is not None:
            overall.update(1)

    if cfg.max_workers <= 1:
        for idx, symbol in work:
            _record_result(
                _process_symbol(
                    cfg, symbol, idx, total, scrip_cache, cache_lock, print_lock
                )
            )
            save_scrip_cache(scrip_cache_path, scrip_cache)
            if cfg.sleep_seconds > 0:
                time.sleep(cfg.sleep_seconds)
    else:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            futures = {
                pool.submit(
                    _process_symbol,
                    cfg,
                    symbol,
                    idx,
                    total,
                    scrip_cache,
                    cache_lock,
                    print_lock,
                ): symbol
                for idx, symbol in work
            }
            for fut in as_completed(futures):
                _record_result(fut.result())
        save_scrip_cache(scrip_cache_path, scrip_cache)

    if overall is not None:
        overall.close()
    return results


def run_download(cfg: BseDownloadConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.bse_download_folder.mkdir(parents=True, exist_ok=True)
    scrip_cache_path = cfg.scrip_cache_path
    scrip_cache_path.parent.mkdir(parents=True, exist_ok=True)
    scrip_cache = load_scrip_cache(scrip_cache_path)

    removed = _cleanup_empty_symbol_dirs(cfg.output_dir)
    if removed:
        print(f"Removed {removed} empty symbol folder(s) from earlier failed runs.")

    print(
        f"Downloading BSE announcements for {len(cfg.symbols)} symbols "
        f"({cfg.from_date} -> {cfg.to_date}) -> {cfg.output_dir}"
    )
    print(
        f"Parallel: {cfg.max_workers} symbol worker(s), "
        f"{cfg.pdf_workers} PDF worker(s) per symbol"
    )
    filt = cfg.category_filter
    if filt.is_all():
        print("Categories: all (no filter)")
    else:
        print(
            f"Categories: {filt.mode} — {', '.join(filt.categories or ())}"
        )
        if filt.exclude_subcategories:
            print(
                "Exclude subcategories: " + ", ".join(filt.exclude_subcategories)
            )
    if cfg.retry_until_complete:
        print(
            f"Retry until complete: on transient BSE errors, wait "
            f"{_format_wait(cfg.retry_wait_seconds)} and retry pending symbols."
        )
    if cfg.show_progress and cfg.progress_style == PROGRESS_STYLE_COMPACT:
        slots = max(1, cfg.max_workers)
        print(f"Progress: 1 round line + up to {slots} active symbol line(s).")

    cache_lock = threading.Lock()
    print_lock = threading.Lock()
    results_by_symbol: dict[str, dict[str, Any]] = {}
    round_no = 0

    while True:
        pending = _pending_work(cfg)
        if not pending:
            break

        round_no += 1
        print(f"\n=== Round {round_no}: {len(pending)} symbol(s) pending ===")

        round_results = _run_round(
            cfg, pending, scrip_cache, scrip_cache_path, cache_lock, print_lock
        )
        for res in round_results:
            results_by_symbol[res["nse_symbol"]] = res

        still_pending = _pending_work(cfg)
        if not still_pending:
            break
        if not cfg.retry_until_complete:
            print(f"{len(still_pending)} symbol(s) still pending; retry disabled in config.")
            break

        retryable = False
        for sym, _ in still_pending:
            res = results_by_symbol.get(sym, {})
            if res.get("status") != "error":
                retryable = True
                break
            if res.get("transient") or is_transient_bse_error(
                Exception(res.get("error", ""))
            ):
                retryable = True
                break
        if not retryable:
            print(
                f"{len(still_pending)} symbol(s) still pending with non-transient errors; stopping."
            )
            break

        print(
            f"{len(still_pending)} symbol(s) still pending. "
            f"Waiting {_format_wait(cfg.retry_wait_seconds)} before retry..."
        )
        time.sleep(cfg.retry_wait_seconds)

    # Include skipped symbols in final manifest
    for idx, symbol in enumerate(cfg.symbols, start=1):
        if symbol in results_by_symbol:
            continue
        sym_dir = cfg.output_dir / symbol
        if _symbol_done(sym_dir, cfg.from_date, cfg.to_date, cfg.category_filter):
            results_by_symbol[symbol] = {
                "nse_symbol": symbol,
                "status": "skipped",
                "path": str(sym_dir),
            }

    results = [results_by_symbol.get(s, {"nse_symbol": s, "status": "missing"}) for s in cfg.symbols]
    done = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    errors = [r for r in results if r.get("status") == "error"]
    not_found = sum(1 for r in results if r.get("status") == "not_found")
    _write_manifest(cfg, results)
    print(
        f"\nFinished. ok: {done}, skipped: {skipped}, not_found: {not_found}, "
        f"errors: {len(errors)}, output: {cfg.output_dir}"
    )
    if errors:
        print("Still failing:", ", ".join(r["nse_symbol"] for r in errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BSE corporate announcements for symbols in a config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/bse.smallcap250.json"),
        help="BSE announcements download config JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_bse_config(args.config.resolve())
    run_download(cfg)


if __name__ == "__main__":
    main()
