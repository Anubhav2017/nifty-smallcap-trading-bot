"""Fetch BSE corporate announcements and optional PDF attachments."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import requests
from bse import BSE
from bse_category_filter import CategoryFilter, filter_announcement_rows, filters_match
from tqdm import tqdm

BASE_URL = "https://www.bseindia.com"
API_URL = "https://api.bseindia.com/BseIndiaAPI/api"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)
DATE_FMT = "%Y%m%d"
ANNOUNCEMENTS_URL = f"{API_URL}/AnnSubCategoryGetData/w"
DEFAULT_REQUEST_TIMEOUT = 60.0
PROGRESS_STYLE_COMPACT = "compact"
PROGRESS_STYLE_VERBOSE = "verbose"


class SymbolProgress:
    """One throttled status line per symbol (compact terminal output)."""

    def __init__(self, symbol: str, position: Optional[int] = None) -> None:
        label = f"{symbol[:12]:<12}"
        self._bar = tqdm(
            total=1,
            desc=label,
            position=position,
            leave=False,
            mininterval=1.0,
            bar_format="{desc} | {postfix}",
            dynamic_ncols=True,
        )
        self.set_status("starting")

    def set_status(self, text: str) -> None:
        self._bar.set_postfix_str(text, refresh=True)

    def set_ann(self, category: str, total: int, cat_index: int, cat_count: int) -> None:
        short = category if len(category) <= 22 else category[:19] + "..."
        self.set_status(f"ann {cat_index}/{cat_count} {short} ({total} rows)")

    def set_pdf(self, done: int, total: int, failed: int) -> None:
        self.set_status(f"PDFs {done}/{total} failed={failed}")

    def close(self) -> None:
        self._bar.close()


class BseAnnouncementError(Exception):
    """BSE announcements fetch failed."""


class BseScripNotFoundError(BseAnnouncementError):
    """NSE symbol could not be mapped to a BSE scrip code."""


_TRANSIENT_MARKERS = (
    "403",
    "429",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "forbidden",
    "gateway",
    "connection",
)


def is_transient_bse_error(exc: BaseException) -> bool:
    """True for rate limits, gateway errors, and timeouts worth retrying later."""
    if isinstance(exc, BseScripNotFoundError):
        return False
    if isinstance(exc, (TimeoutError, requests.Timeout, requests.ConnectionError, ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in (403, 429, 502, 503, 504)
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


@dataclass(frozen=True)
class AnnouncementFetchResult:
    nse_symbol: str
    bse_scrip_code: str
    announcements: list[dict[str, Any]]
    from_date: str
    to_date: str
    pdfs_downloaded: int
    pdfs_skipped: int
    pdfs_failed: int


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def load_scrip_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k.upper(): str(v) for k, v in data.items()}


def save_scrip_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(cache.items())), indent=2) + "\n", encoding="utf-8")


def resolve_bse_scrip_code(
    bse: BSE,
    nse_symbol: str,
    cache: dict[str, str],
    *,
    max_retries: int = 4,
    cache_lock: Optional[threading.Lock] = None,
) -> str:
    key = nse_symbol.upper()
    if cache_lock:
        with cache_lock:
            if key in cache:
                return cache[key]
    elif key in cache:
        return cache[key]
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            code = bse.getScripCode(nse_symbol)
            if cache_lock:
                with cache_lock:
                    cache[key] = code
            else:
                cache[key] = code
            return code
        except ValueError as exc:
            raise BseScripNotFoundError(f"No BSE scrip code for {nse_symbol!r}") from exc
        except (TimeoutError, requests.Timeout, requests.ConnectionError, ConnectionError) as exc:
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise BseAnnouncementError(
        f"BSE scrip lookup failed for {nse_symbol!r} after {max_retries} tries: {last_err}"
    ) from last_err


def _bse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.5",
            "Origin": BASE_URL,
            "Referer": f"{BASE_URL}/",
            "Connection": "keep-alive",
        }
    )
    return session


def _announcement_row_key(row: dict[str, Any]) -> str:
    for field in ("NEWSID", "BSENewsid", "NEWS_DT", "HEADLINE"):
        val = row.get(field)
        if val is not None and str(val).strip():
            return f"{field}:{val}"
    return json.dumps(row, sort_keys=True, default=str)


def _fetch_announcements_page(
    session: requests.Session,
    *,
    scrip_code: str,
    from_date: datetime,
    to_date: datetime,
    page_no: int,
    timeout: float,
    max_retries: int,
    category: str = "-1",
    subcategory: str = "-1",
) -> dict[str, Any]:
    params = {
        "pageno": page_no,
        "strCat": category,
        "subcategory": subcategory,
        "strPrevDate": from_date.strftime(DATE_FMT),
        "strToDate": to_date.strftime(DATE_FMT),
        "strSearch": "P",
        "strscrip": scrip_code,
        "strType": "C",
    }
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = session.get(ANNOUNCEMENTS_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return data
            raise BseAnnouncementError(f"Unexpected announcements JSON type: {type(data)}")
        except (requests.Timeout, requests.ConnectionError, TimeoutError) as exc:
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
        except requests.HTTPError as exc:
            raise BseAnnouncementError(
                f"BSE announcements HTTP error (page {page_no}, scrip {scrip_code}): {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise BseAnnouncementError(
                f"BSE announcements invalid JSON (page {page_no}, scrip {scrip_code})"
            ) from exc
    raise BseAnnouncementError(
        f"BSE announcements request timed out (page {page_no}, scrip {scrip_code}) "
        f"after {max_retries} tries"
    ) from last_err


def _paginate_announcements(
    http: requests.Session,
    *,
    scrip_code: str,
    from_date: datetime,
    to_date: datetime,
    category: str,
    page_sleep_seconds: float,
    request_timeout: float,
    max_retries: int,
) -> list[dict[str, Any]]:
    """Paginate one strCat for a scrip (no progress UI)."""
    rows: list[dict[str, Any]] = []
    page_no = 1
    total: Optional[int] = None

    while True:
        data = _fetch_announcements_page(
            http,
            scrip_code=scrip_code,
            from_date=from_date,
            to_date=to_date,
            page_no=page_no,
            timeout=request_timeout,
            max_retries=max_retries,
            category=category,
        )

        if page_no == 1 and data.get("Table1"):
            try:
                total = int(data["Table1"][0]["ROWCNT"])
            except (KeyError, TypeError, ValueError):
                total = None

        batch = data.get("Table") or []
        if not batch:
            break

        rows.extend(batch)
        if total is not None and len(rows) >= total:
            break

        page_no += 1
        if page_sleep_seconds > 0:
            time.sleep(page_sleep_seconds)

    return rows


def _fetch_announcements_for_category_verbose(
    http: requests.Session,
    *,
    scrip_code: str,
    from_date: datetime,
    to_date: datetime,
    category: str,
    page_sleep_seconds: float,
    request_timeout: float,
    max_retries: int,
    symbol: str,
    progress_position: Optional[int],
    pbar: Optional[tqdm],
    progress_with_total: bool = True,
) -> tuple[list[dict[str, Any]], Optional[tqdm]]:
    """Paginate one strCat with per-page tqdm (verbose mode)."""
    rows: list[dict[str, Any]] = []
    page_no = 1
    total: Optional[int] = None
    label = symbol or scrip_code
    cat_label = category if category != "-1" else "all"

    while True:
        data = _fetch_announcements_page(
            http,
            scrip_code=scrip_code,
            from_date=from_date,
            to_date=to_date,
            page_no=page_no,
            timeout=request_timeout,
            max_retries=max_retries,
            category=category,
        )

        if page_no == 1 and data.get("Table1"):
            try:
                total = int(data["Table1"][0]["ROWCNT"])
            except (KeyError, TypeError, ValueError):
                total = None

        batch = data.get("Table") or []
        if not batch:
            break

        if pbar is None:
            pbar = tqdm(
                total=total if progress_with_total else None,
                desc=f"{label} announcements",
                unit="ann",
                position=progress_position,
                leave=False,
                dynamic_ncols=True,
            )

        rows.extend(batch)
        if pbar is not None:
            pbar.update(len(batch))
            if progress_with_total and total is not None:
                remaining = max(0, total - len(rows))
                pbar.set_postfix(cat=cat_label, left=remaining, refresh=False)
            else:
                pbar.set_postfix(cat=cat_label, n=len(rows), refresh=False)

        if total is not None and len(rows) >= total:
            break

        page_no += 1
        if page_sleep_seconds > 0:
            time.sleep(page_sleep_seconds)

    return rows, pbar


def fetch_all_announcements(
    *,
    scrip_code: str,
    from_date: datetime,
    to_date: datetime,
    page_sleep_seconds: float = 0.3,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    max_retries: int = 4,
    session: Optional[requests.Session] = None,
    symbol: str = "",
    show_progress: bool = False,
    progress_position: Optional[int] = None,
    categories: Optional[Sequence[str]] = None,
    exclude_subcategories: Optional[Sequence[str]] = None,
    symbol_progress: Optional[SymbolProgress] = None,
) -> list[dict[str, Any]]:
    """Paginate BSE AnnSubCategoryGetData for one scrip (direct HTTP, not bse package)."""
    http = session or _bse_session()
    excludes = tuple(exclude_subcategories or ())
    cat_list = list(categories) if categories else []
    pbar: Optional[tqdm] = None
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    compact = symbol_progress is not None
    verbose = show_progress and not compact

    def _merge(batch: list[dict[str, Any]]) -> None:
        for row in batch:
            key = _announcement_row_key(row)
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)

    multi_cat = len(cat_list) > 1
    fetch_cats = cat_list if cat_list else ["-1"]
    cat_count = len(fetch_cats)

    for i, cat in enumerate(fetch_cats):
        if compact:
            batch = _paginate_announcements(
                http,
                scrip_code=scrip_code,
                from_date=from_date,
                to_date=to_date,
                category=cat,
                page_sleep_seconds=page_sleep_seconds,
                request_timeout=request_timeout,
                max_retries=max_retries,
            )
        elif verbose:
            batch, pbar = _fetch_announcements_for_category_verbose(
                http,
                scrip_code=scrip_code,
                from_date=from_date,
                to_date=to_date,
                category=cat,
                page_sleep_seconds=page_sleep_seconds,
                request_timeout=request_timeout,
                max_retries=max_retries,
                symbol=symbol,
                progress_position=progress_position,
                pbar=pbar,
                progress_with_total=not multi_cat and cat != "-1",
            )
        else:
            batch = _paginate_announcements(
                http,
                scrip_code=scrip_code,
                from_date=from_date,
                to_date=to_date,
                category=cat,
                page_sleep_seconds=page_sleep_seconds,
                request_timeout=request_timeout,
                max_retries=max_retries,
            )

        _merge(batch)
        if compact and symbol_progress is not None:
            symbol_progress.set_ann(
                "all" if cat == "-1" else cat,
                len(merged),
                i + 1,
                cat_count,
            )
        if page_sleep_seconds > 0 and i < cat_count - 1:
            time.sleep(page_sleep_seconds)

    if pbar is not None:
        if multi_cat:
            pbar.n = len(merged)
        elif pbar.total is not None:
            pbar.n = min(len(merged), int(pbar.total))
        pbar.close()

    if excludes:
        merged = filter_announcement_rows(
            merged, categories=None, exclude_subcategories=excludes
        )
    return merged


def attachment_urls(attachment_name: str, old_flag: Any) -> list[str]:
    """BSE stores PDFs under AttachLive or AttachHis."""
    name = (attachment_name or "").strip()
    if not name or name.lower() in ("null", "none"):
        return []
    folders = ("AttachLive", "AttachHis") if old_flag else ("AttachLive", "AttachHis")
    return [f"{BASE_URL}/xml-data/corpfiling/{folder}/{name}" for folder in folders]


def _safe_pdf_name(news_id: str, attachment_name: str) -> str:
    base = Path(attachment_name).name or "attachment.pdf"
    stem = re.sub(r"[^\w\-]+", "_", news_id)[:40]
    return f"{stem}_{base}"


def _download_one_pdf(
    row: dict[str, Any],
    out_dir: Path,
    *,
    sleep_seconds: float,
) -> str:
    """Download a single announcement PDF. Returns 'downloaded', 'skipped', or 'failed'."""
    attach = row.get("ATTACHMENTNAME")
    if not attach or str(attach).strip().lower() in ("null", "none", ""):
        return "skipped"

    if _pdf_already_saved(row, out_dir):
        return "skipped"
    dest = _pdf_dest_path(row, out_dir)

    headers = {
        "User-Agent": USER_AGENT,
        "Referer": f"{BASE_URL}/",
        "Accept": "application/pdf,*/*",
    }
    for url in attachment_urls(str(attach), row.get("OLD")):
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                dest.write_bytes(resp.content)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                return "downloaded"
        except requests.RequestException:
            continue
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return "failed"


def _pdf_rows_with_attachments(announcements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in announcements
        if r.get("ATTACHMENTNAME")
        and str(r.get("ATTACHMENTNAME")).strip().lower() not in ("null", "none", "")
    ]


def _pdf_dest_path(row: dict[str, Any], out_dir: Path) -> Path:
    attach = str(row.get("ATTACHMENTNAME", ""))
    news_id = str(row.get("NEWSID", "unknown"))
    return out_dir / _safe_pdf_name(news_id, attach)


def _pdf_already_saved(row: dict[str, Any], out_dir: Path) -> bool:
    dest = _pdf_dest_path(row, out_dir)
    return dest.is_file() and dest.stat().st_size > 0


def download_announcement_pdfs(
    announcements: list[dict[str, Any]],
    out_dir: Path,
    *,
    sleep_seconds: float = 0.2,
    pdf_workers: int = 1,
    symbol: str = "",
    show_progress: bool = False,
    progress_position: Optional[int] = None,
    symbol_progress: Optional[SymbolProgress] = None,
) -> tuple[int, int, int]:
    """Download PDF attachments; returns (downloaded, skipped, failed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _pdf_rows_with_attachments(announcements)
    if not rows:
        return 0, 0, 0

    workers = max(1, pdf_workers)
    downloaded = skipped = failed = 0
    label = symbol or "PDFs"
    pending = [r for r in rows if not _pdf_already_saved(r, out_dir)]
    already = len(rows) - len(pending)
    skipped = already
    compact = symbol_progress is not None
    pbar: Optional[tqdm] = None
    if show_progress and not compact:
        pbar = tqdm(
            total=len(rows),
            initial=already,
            desc=f"{label} PDFs",
            unit="pdf",
            position=progress_position,
            leave=False,
            dynamic_ncols=True,
        )
    pdf_update_every = max(1, len(rows) // 40)
    processed = already

    def _record(outcome: str) -> None:
        nonlocal downloaded, skipped, failed, processed
        if outcome == "downloaded":
            downloaded += 1
        elif outcome == "skipped":
            skipped += 1
        else:
            failed += 1
        processed += 1
        if pbar is not None:
            pbar.update(1)
            remaining = max(0, int(pbar.total) - int(pbar.n))
            pbar.set_postfix(
                new=downloaded,
                skip=skipped,
                fail=failed,
                left=remaining,
                refresh=False,
            )
        elif compact and symbol_progress is not None:
            if processed >= len(rows) or (processed - already) % pdf_update_every == 0:
                symbol_progress.set_pdf(processed, len(rows), failed)

    if workers == 1:
        for row in pending:
            _record(_download_one_pdf(row, out_dir, sleep_seconds=sleep_seconds))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_download_one_pdf, row, out_dir, sleep_seconds=sleep_seconds)
                for row in pending
            ]
            for fut in as_completed(futures):
                _record(fut.result())

    if pbar is not None:
        pbar.close()
    elif compact and symbol_progress is not None:
        symbol_progress.set_pdf(len(rows), len(rows), failed)

    return downloaded, skipped, failed


def _load_cached_announcements(
    sym_dir: Path,
    *,
    category_filter: Optional[CategoryFilter] = None,
) -> Optional[list[dict[str, Any]]]:
    path = sym_dir / "announcements.json"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    if category_filter is not None:
        meta_path = sym_dir / "_meta.json"
        if not meta_path.is_file():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not filters_match(meta, category_filter):
            return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list) and data:
        return data
    return None


def fetch_symbol_announcements(
    *,
    nse_symbol: str,
    output_dir: Path,
    from_date: str,
    to_date: str,
    download_pdfs: bool,
    scrip_cache: dict[str, str],
    download_folder: Path,
    page_sleep_seconds: float,
    pdf_sleep_seconds: float,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    max_retries: int = 4,
    pdf_workers: int = 1,
    cache_lock: Optional[threading.Lock] = None,
    bse_cache_dir: Optional[Path] = None,
    resume_existing_json: bool = True,
    show_progress: bool = True,
    progress_position: Optional[int] = None,
    category_filter: Optional[CategoryFilter] = None,
    progress_style: str = PROGRESS_STYLE_COMPACT,
) -> AnnouncementFetchResult:
    """Resolve scrip, fetch all announcement pages, save JSON, optional PDFs."""
    from_dt = _parse_date(from_date)
    to_dt = _parse_date(to_date)
    if to_dt < from_dt:
        raise ValueError("to_date must be on or after from_date")

    sym_dir = output_dir / nse_symbol
    cache_dir = bse_cache_dir or download_folder
    announcements: Optional[list[dict[str, Any]]] = None
    scrip_code: Optional[str] = None
    use_compact = show_progress and progress_style == PROGRESS_STYLE_COMPACT
    verbose = show_progress and progress_style == PROGRESS_STYLE_VERBOSE
    sym_progress = (
        SymbolProgress(nse_symbol, progress_position) if use_compact else None
    )

    if resume_existing_json:
        announcements = _load_cached_announcements(sym_dir, category_filter=category_filter)
        if announcements:
            key = nse_symbol.upper()
            if cache_lock:
                with cache_lock:
                    scrip_code = scrip_cache.get(key)
            else:
                scrip_code = scrip_cache.get(key)
            if not scrip_code and announcements[0].get("SCRIP_CD") is not None:
                scrip_code = str(announcements[0]["SCRIP_CD"])

    if announcements is None:
        with BSE(download_folder=str(cache_dir)) as bse:
            scrip_code = resolve_bse_scrip_code(
                bse, nse_symbol, scrip_cache, cache_lock=cache_lock
            )

        http = _bse_session()
        categories = (
            list(category_filter.categories) if category_filter and category_filter.categories else None
        )
        excludes = (
            list(category_filter.exclude_subcategories) if category_filter else None
        )
        if sym_progress is not None:
            sym_progress.set_status("scrip resolved, fetching")
        announcements = fetch_all_announcements(
            scrip_code=scrip_code,
            from_date=from_dt,
            to_date=to_dt,
            page_sleep_seconds=page_sleep_seconds,
            request_timeout=request_timeout,
            max_retries=max_retries,
            session=http,
            symbol=nse_symbol,
            show_progress=verbose,
            progress_position=progress_position,
            categories=categories,
            exclude_subcategories=excludes,
            symbol_progress=sym_progress,
        )

        sym_dir.mkdir(parents=True, exist_ok=True)
        (sym_dir / "announcements.json").write_text(
            json.dumps(announcements, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    else:
        sym_dir.mkdir(parents=True, exist_ok=True)
        if not scrip_code:
            with BSE(download_folder=str(cache_dir)) as bse:
                scrip_code = resolve_bse_scrip_code(
                    bse, nse_symbol, scrip_cache, cache_lock=cache_lock
                )

    assert announcements is not None and scrip_code is not None

    pdfs_dl = pdfs_skip = pdfs_fail = 0
    try:
        if sym_progress is not None:
            sym_progress.set_status(f"ann done ({len(announcements)} rows)")
        if download_pdfs and announcements:
            if sym_progress is not None:
                sym_progress.set_status("PDFs starting")
            pdfs_dl, pdfs_skip, pdfs_fail = download_announcement_pdfs(
                announcements,
                sym_dir / "attachments",
                sleep_seconds=pdf_sleep_seconds,
                pdf_workers=pdf_workers,
                symbol=nse_symbol,
                show_progress=verbose,
                progress_position=progress_position,
                symbol_progress=sym_progress,
            )
    finally:
        if sym_progress is not None:
            sym_progress.close()

    return AnnouncementFetchResult(
        nse_symbol=nse_symbol,
        bse_scrip_code=scrip_code,
        announcements=announcements,
        from_date=from_date,
        to_date=to_date,
        pdfs_downloaded=pdfs_dl,
        pdfs_skipped=pdfs_skip,
        pdfs_failed=pdfs_fail,
    )
