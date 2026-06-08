#!/usr/bin/env python3
"""
Extract a per-stock CSV of BSE announcements with structured fields.

For each symbol under  {dataset}/bse_announcements/{SYMBOL}/  this script
reads ``announcements.json`` (or falls back to ``announcements.csv``),
opens the matching PDF in ``attachments/`` for each row, and writes:

    {dataset}/bse_announcements/{SYMBOL}/announcements_extracted.csv

Output columns:
    date              ISO date  (YYYY-MM-DD) of the announcement
    time              HH:MM:SS  (when available)
    type              Normalised event type (e.g. results, dividend, bonus,
                      split, buyback, acquisition, board_meeting, ...)
    subcategory       Raw BSE SUBCATNAME
    subject           NEWSSUB  (announcement title)
    summary           Short human-readable summary built from HEADLINE and
                      the first informative lines of the PDF body
    key_figures       Pipe-separated "Label: value" pairs extracted from
                      both the subject/headline and the PDF body
                      (revenue, EBITDA, net profit, EPS, dividend, ratios,
                      record/ex dates, percentages, ...)
    pdf_found         1 / 0
    newsid            Original BSE NEWSID

Filename convention in attachments/  (already used by pdf_extract.py):
    {NEWSID}_{ATTACHMENTNAME-without-.pdf}.pdf
The lookup key is everything AFTER the first underscore (the ATTACHMENTNAME
field minus the ``.pdf`` suffix).  Note the ATTACHMENTNAME may itself contain
underscores, so the split is done only once on the first separator.

Usage:
    python scripts/extract_announcements_per_stock.py
    python scripts/extract_announcements_per_stock.py --dataset dataset_smallcap250
    python scripts/extract_announcements_per_stock.py --symbol AARTIIND
    python scripts/extract_announcements_per_stock.py --workers 4 --force
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

try:
    import fitz  # pymupdf
except ImportError:  # pragma: no cover
    print("pymupdf not found. Install with: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

# Silence MuPDF's C-level warnings (e.g. "format error: cannot find object in
# xref") that some malformed BSE PDFs trigger.  These are written directly to
# stderr by the C library and bypass Python's try/except; they don't affect
# the (partial) text we still recover, so we just stop them from spamming the
# progress log.  Guarded for older/newer PyMuPDF versions that may rename it.
try:
    fitz.TOOLS.mupdf_display_errors(False)  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass


# ── Output schema ─────────────────────────────────────────────────────────────

FIELDNAMES = [
    "date",
    "time",
    "type",
    "subcategory",
    "subject",
    "summary",
    "key_figures",
    "pdf_found",
    "newsid",
]

OUTPUT_FILENAME = "announcements_extracted.csv"


# ── High-level event type mapping ─────────────────────────────────────────────
# Order matters: first match wins.  The matcher checks SUBCATNAME first, then
# CATEGORYNAME, then keywords in NEWSSUB / HEADLINE.

_TYPE_RULES: list[tuple[str, list[str]]] = [
    ("results", [
        "financial results", "audited results", "unaudited results",
        "quarterly result", "quarterly results",
    ]),
    ("dividend", ["dividend"]),
    ("bonus", ["bonus issue", "bonus shares", "bonus"]),
    ("split", ["sub-division", "stock split", "subdivision", "share split"]),
    ("rights_issue", ["rights issue"]),
    ("buyback", ["buy back", "buyback", "buy-back"]),
    ("record_date", ["record date"]),
    ("book_closure", ["book closure"]),
    ("merger", ["amalgamation", "merger", "demerger", "scheme of arrangement"]),
    ("acquisition", ["acquisition", "stake acquisition", "share purchase"]),
    ("board_meeting", ["board meeting", "intimation of board meeting"]),
    ("agm_egm", ["annual general meeting", "agm", "extra ordinary general meeting", "egm", "postal ballot"]),
    ("change_in_directorate", ["change in directorate", "appointment of director", "resignation of director", "cessation"]),
    ("change_in_management", ["change in management", "appointment of ceo", "appointment of md", "appointment of chief"]),
    ("allotment", ["allotment of equity", "allotment of esop", "allotment of shares", "allotment of securities"]),
    ("press_release", ["press release", "media release"]),
    ("investor_presentation", ["investor presentation"]),
    ("earnings_call", ["earnings call", "earnings conference call", "conference call transcript", "earnings call transcript"]),
    ("analyst_meet", ["analyst / investor meet", "analyst meet", "investor meet"]),
    ("trading_window", ["closure of trading window", "trading window"]),
    ("credit_rating", ["credit rating"]),
    ("order_win", ["receipt of order", "order received", "letter of award", "loa"]),
    ("capacity_expansion", ["capacity expansion", "capacity enhancement", "commissioning"]),
    ("regulatory", ["sebi", "show cause", "penalty", "regulatory action"]),
    ("clarification", ["clarification"]),
    ("newspaper_publication", ["newspaper publication", "newspaper advertisement"]),
    ("compliance", ["compliance", "certificate under reg", "regulation 74", "regulation 7"]),
    ("disclosure_30", ["regulation 30", "lodr"]),
    ("sast", ["sast", "regulation 29", "regulation 10", "regulation 7(2)"]),
    ("insider_trading", ["insider trading", "regulation 7(2)", "pit regulations"]),
]


def classify_type(subcat: str, category: str, subject: str, headline: str) -> str:
    """Return a normalised type label (lower_snake_case)."""
    blob = " ".join([subcat or "", category or "", subject or "", headline or ""]).lower()
    for label, keywords in _TYPE_RULES:
        for kw in keywords:
            if kw in blob:
                return label
    return "other"


# ── Attachment lookup ─────────────────────────────────────────────────────────

def build_attachment_index(attachments_dir: Path) -> dict[str, Path]:
    """Map attachment UUID (lower-case, no .pdf) → full PDF path."""
    index: dict[str, Path] = {}
    if not attachments_dir.is_dir():
        return index
    for entry in attachments_dir.iterdir():
        name = entry.name
        if not name.lower().endswith(".pdf"):
            continue
        base = name[:-4]
        # On-disk files are named  {NEWSID}_{ATTACHMENTNAME-without-.pdf}.pdf
        # The lookup key is the ATTACHMENTNAME part, i.e. everything AFTER the
        # FIRST underscore.  The ATTACHMENTNAME itself may contain underscores
        # (e.g. "AB9DC053_CE7E_42EA_B870_7A9F920D977E_102926"), so we must
        # split only once on the first separator -- never rsplit.
        attach_key = base.split("_", 1)[1].lower() if "_" in base else base.lower()
        index[attach_key] = entry
    return index


def find_pdf(attach_name: str, index: dict[str, Path]) -> Path | None:
    if not attach_name:
        return None
    key = attach_name.lower().replace(".pdf", "").strip()
    return index.get(key)


# ── PDF extraction ────────────────────────────────────────────────────────────

_MAX_PDF_PAGES = 20  # cap for performance on huge filings (annual reports etc.)


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return f"[PDF_ERROR: {e}]"
    try:
        chunks: list[str] = []
        for i, page in enumerate(doc):
            if i >= _MAX_PDF_PAGES:
                break
            try:
                chunks.append(page.get_text())
            except Exception as e:
                # Encrypted / malformed page: keep whatever we have so far.
                chunks.append(f"[PAGE_ERROR: {e}]")
                break
        return "\n".join(chunks)
    finally:
        doc.close()


def clean_text(text: str) -> str:
    """Normalise whitespace and drop non-printable chars."""
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


# ── Number / figure extraction ────────────────────────────────────────────────

# Permissive numeric token (with thousands separators / decimals / negative)
_NUM = r"(?:\(?-?[\d,]+(?:\.\d+)?\)?)"

# Currency / unit suffix often following financial figures
_UNIT = r"(?:cr\.?|crore|crores|lakh|lakhs|lac|lacs|mn|million|bn|billion|rs\.?|inr|usd|₹|\$)"

# Each figure pattern: (label, compiled_regex, min_plausible_value).
# The regex must always have ONE capturing group (the numeric value).
# When the regex requires a unit (cr/lakh/mn/...) the min_plausible is 0.
# When the regex allows a bare number, min_plausible filters out tiny
# values that are almost always row labels / note refs / EPS digits.
#
# We try unit-anchored patterns first; only if they miss do we fall back to
# bare-number patterns.  This dramatically reduces false positives on
# financial-results PDFs where "Net Profit" first appears as a row label.
_FIGURE_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    # ── Revenue / turnover / net sales ──────────────────────────────────
    ("Revenue", re.compile(
        rf"(?:total\s+revenue|revenue\s+from\s+operations|turnover|net\s+sales)"
        rf"[^\n]{{0,80}}?({_NUM})\s*(?:{_UNIT})",
        re.IGNORECASE), 0.0),
    ("Revenue", re.compile(
        rf"(?:total\s+revenue|revenue\s+from\s+operations|turnover|net\s+sales)"
        rf"[^\n]{{0,80}}?({_NUM})",
        re.IGNORECASE), 100.0),
    # ── EBITDA ──────────────────────────────────────────────────────────
    ("EBITDA", re.compile(
        rf"\bebitda\b[^\n]{{0,80}}?({_NUM})\s*(?:{_UNIT})",
        re.IGNORECASE), 0.0),
    ("EBITDA", re.compile(
        rf"\bebitda\b[^\n]{{0,80}}?({_NUM})",
        re.IGNORECASE), 10.0),
    # ── PAT / Net profit / Net loss ────────────────────────────────────
    ("Net Profit", re.compile(
        rf"(?:profit\s+after\s+tax|net\s+profit|net\s+loss|\bpat\b)"
        rf"[^\n]{{0,80}}?({_NUM})\s*(?:{_UNIT})",
        re.IGNORECASE), 0.0),
    ("Net Profit", re.compile(
        rf"(?:profit\s+after\s+tax|net\s+profit|net\s+loss|\bpat\b)"
        rf"[^\n]{{0,80}}?({_NUM})",
        re.IGNORECASE), 10.0),
    # ── Profit before tax ──────────────────────────────────────────────
    ("PBT", re.compile(
        rf"profit\s+before\s+tax[^\n]{{0,80}}?({_NUM})\s*(?:{_UNIT})",
        re.IGNORECASE), 0.0),
    ("PBT", re.compile(
        rf"profit\s+before\s+tax[^\n]{{0,80}}?({_NUM})",
        re.IGNORECASE), 10.0),
    # ── Operating profit / income ──────────────────────────────────────
    ("Operating Profit", re.compile(
        rf"operating\s+(?:profit|income)[^\n]{{0,80}}?({_NUM})\s*(?:{_UNIT})",
        re.IGNORECASE), 0.0),
    ("Operating Profit", re.compile(
        rf"operating\s+(?:profit|income)[^\n]{{0,80}}?({_NUM})",
        re.IGNORECASE), 10.0),
    # ── EPS (small numbers OK — EPS is typically Rs. 0–100) ────────────
    ("EPS", re.compile(
        r"(?:basic|diluted)?\s*\beps\b[^\n]{0,60}?(-?\d+(?:\.\d+)?)",
        re.IGNORECASE), 0.0),
    # ── Total assets ───────────────────────────────────────────────────
    ("Total Assets", re.compile(
        rf"total\s+assets[^\n]{{0,80}}?({_NUM})\s*(?:{_UNIT})",
        re.IGNORECASE), 0.0),
    ("Total Assets", re.compile(
        rf"total\s+assets[^\n]{{0,80}}?({_NUM})",
        re.IGNORECASE), 100.0),
    # ── Total income ───────────────────────────────────────────────────
    ("Total Income", re.compile(
        rf"total\s+income[^\n]{{0,80}}?({_NUM})\s*(?:{_UNIT})",
        re.IGNORECASE), 0.0),
    ("Total Income", re.compile(
        rf"total\s+income[^\n]{{0,80}}?({_NUM})",
        re.IGNORECASE), 100.0),
    # ── Total expenses ─────────────────────────────────────────────────
    ("Total Expenses", re.compile(
        rf"total\s+expenses[^\n]{{0,80}}?({_NUM})\s*(?:{_UNIT})",
        re.IGNORECASE), 0.0),
    ("Total Expenses", re.compile(
        rf"total\s+expenses[^\n]{{0,80}}?({_NUM})",
        re.IGNORECASE), 100.0),
]

# Dividend amount per share — look for explicit Rs/Re/₹ amount, optionally with % nearby
_RE_DIV_AMT = re.compile(
    r"(?:dividend|@)[^\n]{0,40}?(?:rs\.?|re\.?|₹|inr)\s*([\d,]+(?:\.\d+)?)\s*/?\-?",
    re.IGNORECASE,
)
_RE_DIV_PCT = re.compile(r"dividend[^\n]{0,40}?@\s*([\d.]+)\s*%", re.IGNORECASE)
_RE_DIV_PER_SHARE = re.compile(
    r"(?:rs\.?|re\.?|₹|inr)\s*([\d,]+(?:\.\d+)?)\s*/?\-?\s*per\s+(?:equity\s+)?share",
    re.IGNORECASE,
)

# Bonus / split ratio
_RE_RATIO = re.compile(r"\b(\d+)\s*:\s*(\d+)\b")
_RE_FACE_VALUE = re.compile(
    r"face\s+value[^\n]{0,40}?(?:rs\.?\s*)?(\d+)[^\n]{0,20}?(?:to\s+)?(?:rs\.?\s*)?(\d+)",
    re.IGNORECASE,
)

# Buyback price / size
_RE_BUYBACK_PRICE = re.compile(
    r"buy[- ]?back[^\n]{0,80}?(?:price|@)\s*(?:rs\.?|₹|inr)?\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_RE_BUYBACK_SIZE = re.compile(
    rf"buy[- ]?back[^\n]{{0,80}}?(?:size|amount|aggregate)[^\n]{{0,30}}?({_NUM})\s*(?:{_UNIT})?",
    re.IGNORECASE,
)

# Order / contract value
_RE_ORDER_VALUE = re.compile(
    rf"(?:order|contract|loa|letter\s+of\s+award)[^\n]{{0,80}}?(?:value|worth|amount|aggregating)[^\n]{{0,30}}?({_NUM})\s*(?:{_UNIT})?",
    re.IGNORECASE,
)

# Acquisition stake
_RE_STAKE = re.compile(
    r"(?:acqui[a-z]+|purchase\s+of)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)

# Allotment of shares — number of shares
_RE_ALLOTMENT_SHARES = re.compile(
    r"allot(?:ment|ted)[^\n]{0,80}?([\d,]+)\s+(?:equity\s+)?shares?",
    re.IGNORECASE,
)

# Generic %  ("growth of X%", "increased by X%", "Y/Y +X%").
# Bare "up"/"down" require an explicit "by" to avoid matching unrelated
# prose like "...wind up. 5%..."; the directional verbs and Y/Y / Q/Q
# markers stay permissive.
_RE_PERCENT = re.compile(
    r"(?:growth|increase[d]?|decline[d]?|decrease[d]?|grew|rose|fell|y/?o/?y|q/?o/?q)\s*(?:of|by)?\s*([\d.]+)\s*%"
    r"|(?:up|down)\s+by\s+([\d.]+)\s*%",
    re.IGNORECASE,
)

# Record / ex / book-closure dates.  Re-uses month patterns from the existing
# corporate-actions extractor so the output stays consistent across scripts.
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_RE_DATE_DMY = re.compile(rf"\b(\d{{1,2}})[a-z]{{0,2}}\s+({_MONTHS})\s+(\d{{4}})\b", re.IGNORECASE)
_RE_DATE_MDY = re.compile(rf"\b({_MONTHS})\s+(\d{{1,2}})[a-z]{{0,2}}\,?\s+(\d{{4}})\b", re.IGNORECASE)
_RE_DATE_NUM = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
_RE_DATE_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _parse_iso_date(text: str) -> str:
    """Return the first plausible date in *text* as YYYY-MM-DD, or ''."""
    m = _RE_DATE_ISO.search(text)
    if m:
        try:
            datetime.strptime(m.group(1), "%Y-%m-%d")
            return m.group(1)
        except ValueError:
            pass
    m = _RE_DATE_DMY.search(text)
    if m:
        try:
            day = int(m.group(1))
            mon = _MONTH_MAP.get(m.group(2).lower()[:3])
            year = int(m.group(3))
            if mon and 1 <= day <= 31:
                return f"{year:04d}-{mon:02d}-{day:02d}"
        except (ValueError, KeyError):
            pass
    m = _RE_DATE_MDY.search(text)
    if m:
        try:
            mon = _MONTH_MAP.get(m.group(1).lower()[:3])
            day = int(m.group(2))
            year = int(m.group(3))
            if mon and 1 <= day <= 31:
                return f"{year:04d}-{mon:02d}-{day:02d}"
        except (ValueError, KeyError):
            pass
    m = _RE_DATE_NUM.search(text)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    return ""


def _clean_num(token: str) -> str:
    """Strip noise from a numeric capture and reject implausible values."""
    t = token.strip().replace(",", "").replace("(", "-").replace(")", "")
    # collapse trailing dot (artefact of sentence boundary)
    t = t.rstrip(".")
    try:
        float(t)
    except ValueError:
        return ""
    return t


# Phrases that, when they immediately precede a date, mark it as a
# financial-year / period boundary rather than the record/ex date itself.
_PERIOD_END_PREFIX = re.compile(
    r"(?:year|period|quarter|nine\s+months|half\s+year|ended|ending|"
    r"financial\s+year|fy)\s*$",
    re.IGNORECASE,
)

# Record / ex / book-closure dates must be TIGHTLY bound to explicit
# record-date phrasing.  A loose "first date near the keyword" heuristic
# grabs the letter header date or the financial-year-end date, which is
# worse than emitting nothing.  Each pattern captures a date string that is
# then re-parsed by _parse_iso_date.
#
# Matches forms like:
#   "record date is/shall be/fixed as/i.e. <date>"
#   "<date> as the record date"
#   "record date for ... : <date>"
_DATE_TOKEN = (
    rf"(?:\d{{1,2}}[a-z]{{0,2}}\s+(?:{_MONTHS})\s+\d{{4}}"
    rf"|(?:{_MONTHS})\s+\d{{1,2}}[a-z]{{0,2}},?\s+\d{{4}}"
    rf"|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{4}}"
    rf"|\d{{4}}-\d{{2}}-\d{{2}})"
)
_RE_RECORD_DATE_FWD = re.compile(
    rf"record\s+date\b[^\n]{{0,60}}?"
    rf"(?:is|shall\s+be|will\s+be|fixed\s+as|fixed\s+for|as|i\.?e\.?|of|:)\s*"
    rf"({_DATE_TOKEN})",
    re.IGNORECASE,
)
_RE_RECORD_DATE_BWD = re.compile(
    rf"({_DATE_TOKEN})\s+as\s+the\s+record\s+date",
    re.IGNORECASE,
)


def _extract_record_date(text: str) -> str:
    """Return a record/ex date only when tightly bound to explicit phrasing.

    Returns '' when no confidently-identified record date is present, which
    is preferable to emitting a guessed (and often wrong) nearby date such
    as the letter header date or the financial-year-end date.
    """
    for pat in (_RE_RECORD_DATE_FWD, _RE_RECORD_DATE_BWD):
        m = pat.search(text)
        if m:
            iso = _parse_iso_date(m.group(1))
            if iso:
                return iso
    return ""


# Event types whose announcement is genuinely *about* company financials.
# Only these get the generic line-item figures (Revenue / EBITDA / Net Profit
# / EPS / ...).  Many filings share a single board-meeting PDF across several
# announcement rows (results, dividend, change-in-management, ...), so without
# this gate the financial numbers leak onto unrelated rows.
_FINANCIALS_EVENT_TYPES = frozenset({
    "results",
    "earnings_call",
    "investor_presentation",
    "press_release",
})


def extract_key_figures(text: str, event_type: str) -> str:
    """Return a pipe-separated 'Label: value' string of figures found in text.

    Multiple matches for the same label keep only the first (which tends to be
    the headline / summary section in BSE filings).

    Generic financial line items (revenue, profit, ...) are extracted ONLY for
    event types in ``_FINANCIALS_EVENT_TYPES`` so that a shared board-meeting
    PDF does not spray those numbers onto unrelated announcements (dividend,
    change-in-management, regulatory, ...).  Action-specific figures (dividend
    per share, bonus ratio, buyback price, ...) remain gated on their own
    event type below.
    """
    if not text:
        return ""

    found: dict[str, str] = {}

    # Generic financial line items -- only for financials-bearing event types.
    # Patterns are listed unit-first, then bare-number with a min-value floor;
    # first acceptable match per label wins, so unit-anchored hits always beat
    # bare numbers.
    if event_type in _FINANCIALS_EVENT_TYPES:
        for label, pat, min_value in _FIGURE_PATTERNS:
            if label in found:
                continue
            m = pat.search(text)
            if not m:
                continue
            v = _clean_num(m.group(1))
            if not v:
                continue
            try:
                if abs(float(v)) < min_value:
                    continue
            except ValueError:
                continue
            found[label] = v

    # Dividend variants.  Per-share dividends are realistically small
    # (< ~1000); anything larger is almost certainly a mis-grabbed table
    # cell (e.g. total dividend payout in lakhs), so reject it.
    # Gated on dividend/record-date event types only -- the bare "dividend"
    # keyword appears in many board-meeting PDFs (which are also attached to
    # results / management-change rows), so keying off the text alone would
    # re-introduce cross-contamination.
    if event_type in {"dividend", "record_date"}:
        m = _RE_DIV_PER_SHARE.search(text) or _RE_DIV_AMT.search(text)
        if m:
            v = _clean_num(m.group(1))
            if v:
                try:
                    if 0 < float(v) <= 1000:
                        found.setdefault("Dividend/Share", v)
                except ValueError:
                    pass
        m_pct = _RE_DIV_PCT.search(text)
        if m_pct:
            v = _clean_num(m_pct.group(1))
            if v:
                found.setdefault("Dividend %", v)

    # Bonus / split ratio.  Only accept ratios that appear near a relevant
    # keyword so we don't pick up address PIN codes ("Mumbai 400 001") or
    # other incidental "N:M" tokens elsewhere in the filing.
    if event_type in {"bonus", "split", "record_date"}:
        for kw in ("bonus", "ratio", "sub-division", "subdivision", "split",
                   "every", "for every"):
            idx = text.lower().find(kw)
            if idx < 0:
                continue
            window = text[idx: idx + 120]
            m = _RE_RATIO.search(window)
            if m:
                found.setdefault("Ratio", f"{m.group(1)}:{m.group(2)}")
                break
        if event_type == "split":
            m_fv = _RE_FACE_VALUE.search(text)
            if m_fv:
                found.setdefault("Face Value", f"{m_fv.group(1)}->{m_fv.group(2)}")

    # Buyback
    if event_type == "buyback":
        m = _RE_BUYBACK_PRICE.search(text)
        if m:
            v = _clean_num(m.group(1))
            if v:
                found.setdefault("Buyback Price", v)
        m = _RE_BUYBACK_SIZE.search(text)
        if m:
            v = _clean_num(m.group(1))
            if v:
                found.setdefault("Buyback Size", v)

    # Order win / contract
    if event_type == "order_win":
        m = _RE_ORDER_VALUE.search(text)
        if m:
            v = _clean_num(m.group(1))
            if v:
                found.setdefault("Order Value", v)

    # Acquisition stake
    if event_type == "acquisition":
        m = _RE_STAKE.search(text)
        if m:
            v = _clean_num(m.group(1))
            if v:
                found.setdefault("Stake %", v)

    # Allotment count
    if event_type == "allotment":
        m = _RE_ALLOTMENT_SHARES.search(text)
        if m:
            v = _clean_num(m.group(1))
            if v:
                found.setdefault("Shares Allotted", v)

    # Y/Y or Q/Q growth %  — restricted to results, where a headline % is
    # most likely to be a genuine growth figure rather than incidental prose.
    if event_type == "results":
        m = _RE_PERCENT.search(text)
        if m:
            v = _clean_num(m.group(1) or m.group(2) or "")
            if v:
                found.setdefault("Growth %", v)

    # Record / ex date when relevant.  Only emit when a date is tightly bound
    # to explicit record-date phrasing (see _extract_record_date); a guessed
    # nearby date is worse than none for a trading dataset.
    if event_type in {"record_date", "dividend", "bonus", "split", "rights_issue", "buyback"}:
        rd = _extract_record_date(text)
        if rd:
            found.setdefault("Record/Ex Date", rd)

    if not found:
        return ""
    return " | ".join(f"{k}: {v}" for k, v in found.items())


# ── Summary builder ───────────────────────────────────────────────────────────

_SUMMARY_NOISE = re.compile(
    r"(?:please\s+find\s+(?:attached|enclosed)|pfa|kindly\s+refer|"
    r"with\s+reference\s+to|in\s+terms\s+of\s+regulation\s+\d+\s+of\s+sebi)\b[^\n.]{0,200}\.?\s*",
    re.IGNORECASE,
)

# Sentences that look like part of the BSE/NSE letter-address block
# ("To, Listing Dept, BSE Ltd, Phiroze Jeejeebhoy Towers, Dalal Street,
# Mumbai 400 001 / NSE Code: XYZ…") — these appear in nearly every
# filing and add no information.
_ADDRESS_BLOCK = re.compile(
    r"\b(?:"
    r"to\s*,|listing\s+(?:dep|department|/|compliance)|compliance\s+department|"
    r"bse\s*(?:ltd|limited)|bombay\s+stock\s+exchange|national\s+stock\s+exchange|"
    r"nse\s+(?:code|symbol)|bse\s+(?:scrip|code)|scrip\s+code|symbol\s*[:\-]\s*[a-z]|"
    r"phiroze\s+jeejeebhoy|dalal\s+street|exchange\s+plaza|"
    r"bandra[- ]kurla|bandra\s*\(\s*e\s*\)|mumbai\s+400\s*0?\d{2}|"
    r"c[/\-]\s*1\s*,?\s*g\s*block|isin\s*[:\-]|equity\s*isin"
    r")\b",
    re.IGNORECASE,
)

# Sentences that are pure salutations / framing.
_SALUTATION = re.compile(
    r"^(?:dear\s+sir|dear\s+ma|respected\s+sir|sir\s*/?\s*madam|"
    r"ref\s*[:\-]|subject\s*[:\-]|sub\s*[:\-]|kind\s+attn)",
    re.IGNORECASE,
)


def _is_informative_sentence(sentence: str) -> bool:
    """Return True if a candidate sentence is worth keeping in the summary."""
    s = sentence.strip()
    if len(s) < 25:
        return False
    if _SALUTATION.search(s):
        return False
    if _ADDRESS_BLOCK.search(s):
        return False
    # Drop fragments that are mostly uppercase shouting (e.g. table headers).
    letters = [c for c in s if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
        return False
    # Drop sentences that are mostly punctuation/digits (fragmented table rows).
    alnum = sum(1 for c in s if c.isalnum())
    if alnum < len(s) * 0.4:
        return False
    return True


def build_summary(headline: str, pdf_text: str, max_chars: int = 400) -> str:
    """Compose a short summary from the BSE HEADLINE and the PDF body.

    Strategy:
    1. Start with HEADLINE (already a curated description).
    2. Append the first 1–2 informative sentences from the PDF body, skipping
       boilerplate ("Please find attached…"), salutations, the BSE/NSE
       letter-address block, and all-caps fragments.
    """
    headline = (headline or "").strip()
    parts: list[str] = []
    if headline:
        parts.append(headline)

    if pdf_text:
        body = _SUMMARY_NOISE.sub("", pdf_text)
        body = re.sub(r"\s+", " ", body).strip()
        sentences = re.split(r"(?<=[.!?])\s+", body)
        picked = 0
        for s in sentences:
            s = s.strip().strip("-—–")
            if not _is_informative_sentence(s):
                continue
            parts.append(s)
            picked += 1
            if picked >= 2:
                break

    summary = " — ".join(parts) if parts else ""
    # Collapse any embedded newlines / carriage returns (BSE HEADLINE values
    # often contain literal \r\n) so the field stays on a single CSV row.
    summary = re.sub(r"\s+", " ", summary).strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
    return summary


# ── Per-stock worker ──────────────────────────────────────────────────────────

def _load_announcements(stock_dir: Path) -> list[dict]:
    """Prefer announcements.json (preserves all fields), fall back to CSV."""
    j = stock_dir / "announcements.json"
    if j.exists():
        try:
            with j.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[WARN] {j}: {e}", file=sys.stderr)

    c = stock_dir / "announcements.csv"
    if c.exists():
        try:
            with c.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            print(f"[WARN] {c}: {e}", file=sys.stderr)

    return []


def _split_dt(value: str) -> tuple[str, str]:
    """Split BSE timestamp into (YYYY-MM-DD, HH:MM:SS)."""
    if not value:
        return "", ""
    head = value.split(".")[0]
    try:
        dt = datetime.fromisoformat(head)
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
    except ValueError:
        # Fallback: try to slice
        return head[:10], head[11:19] if len(head) >= 19 else ""


def process_stock(args: tuple[Path, Path, bool]) -> tuple[str, int, int]:
    """Process one stock; return (symbol, rows_written, pdfs_extracted).

    Returns (symbol, -1, 0) when output already exists and *force* is False.
    """
    stock_dir, output_path, force = args
    symbol = stock_dir.name

    if output_path.exists() and not force:
        return symbol, -1, 0

    announcements = _load_announcements(stock_dir)
    if not announcements:
        return symbol, 0, 0

    attach_index = build_attachment_index(stock_dir / "attachments")

    rows: list[dict] = []
    pdfs_extracted = 0

    for ann in announcements:
        # JSON keys vs CSV column names are identical for BSE dumps,
        # but JSON values may be None whereas CSV gives "".
        def g(*keys: str) -> str:
            for k in keys:
                v = ann.get(k)
                if v not in (None, ""):
                    return str(v).strip()
            return ""

        date_str, time_str = _split_dt(g("DT_TM", "NEWS_DT"))
        subject     = g("NEWSSUB")
        headline    = g("HEADLINE")
        category    = g("CATEGORYNAME")
        subcategory = g("SUBCATNAME")
        attach_name = g("ATTACHMENTNAME")
        newsid      = g("NEWSID")

        event_type = classify_type(subcategory, category, subject, headline)

        pdf_path = find_pdf(attach_name, attach_index)
        pdf_text = ""
        if pdf_path is not None:
            raw = extract_pdf_text(pdf_path)
            if not raw.startswith("[PDF_ERROR"):
                pdf_text = clean_text(raw)
                pdfs_extracted += 1

        # Combine subject + headline + pdf text for figure extraction so we
        # capture numbers from BOTH the title and the body.
        combined_text = "\n".join(filter(None, [subject, headline, pdf_text]))
        key_figures = extract_key_figures(combined_text, event_type)

        summary = build_summary(headline or subject, pdf_text)

        rows.append({
            "date":        date_str,
            "time":        time_str,
            "type":        event_type,
            "subcategory": subcategory,
            "subject":     subject,
            "summary":     summary,
            "key_figures": key_figures,
            "pdf_found":   "1" if pdf_path is not None else "0",
            "newsid":      newsid,
        })

    # Sort newest first (matches BSE ordering); use (date, time) so rows on
    # the same day keep a deterministic order.
    rows.sort(key=lambda r: (r["date"], r["time"]), reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return symbol, len(rows), pdfs_extracted


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        default="dataset_smallcap250",
        help="Dataset root folder (default: dataset_smallcap250)",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Process only this symbol (e.g. AARTIIND). Default: all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N symbols (after filtering).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(cpu_count(), 8),
        help="Number of parallel workers (default: min(cpu_count, 8))",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process symbols even if announcements_extracted.csv exists.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    ann_base = dataset_root / "bse_announcements"
    if not ann_base.is_dir():
        print(f"[ERROR] {ann_base} not found", file=sys.stderr)
        return 1

    stock_dirs = sorted(
        d for d in ann_base.iterdir()
        if d.is_dir() and d.name not in {".cache", "meta"}
    )

    if args.symbol:
        stock_dirs = [d for d in stock_dirs if d.name == args.symbol]
        if not stock_dirs:
            print(f"[ERROR] Symbol {args.symbol} not found under {ann_base}",
                  file=sys.stderr)
            return 1

    if args.limit is not None:
        stock_dirs = stock_dirs[: args.limit]

    print(f"Dataset : {dataset_root}")
    print(f"Stocks  : {len(stock_dirs)}")
    print(f"Workers : {args.workers}")
    print(f"Force   : {args.force}")
    print("-" * 60, flush=True)

    work = [
        (d, d / OUTPUT_FILENAME, args.force)
        for d in stock_dirs
    ]

    total_rows = 0
    total_pdfs = 0
    skipped = 0

    if args.workers <= 1 or len(work) <= 1:
        for i, w in enumerate(work, 1):
            sym, n, p = process_stock(w)
            _print_progress(i, len(work), sym, n, p)
            if n < 0:
                skipped += 1
            else:
                total_rows += n
                total_pdfs += p
    else:
        with Pool(args.workers) as pool:
            for i, (sym, n, p) in enumerate(
                pool.imap_unordered(process_stock, work), 1
            ):
                _print_progress(i, len(work), sym, n, p)
                if n < 0:
                    skipped += 1
                else:
                    total_rows += n
                    total_pdfs += p

    print("-" * 60)
    print(f"Done. {len(work) - skipped} processed, {skipped} skipped (already had output).")
    print(f"Total rows  : {total_rows}")
    print(f"PDFs parsed : {total_pdfs}")
    print(f"Output file : <stock_dir>/{OUTPUT_FILENAME}")
    return 0


def _print_progress(i: int, total: int, symbol: str, rows: int, pdfs: int) -> None:
    if rows < 0:
        msg = f"  [{i:4d}/{total}] {symbol:20s} -> skipped (output exists, use --force)"
    else:
        msg = f"  [{i:4d}/{total}] {symbol:20s} -> {rows:5d} rows, {pdfs:5d} PDFs"
    print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
