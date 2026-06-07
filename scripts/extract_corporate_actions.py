#!/usr/bin/env python3
"""
Extract structured corporate action events from BSE announcement CSVs.

Reads  dataset_smallcap250/bse_announcements/{SYMBOL}/announcements.csv
and produces two output files:

  dataset_smallcap250/corporate_actions_extracted.csv
      Full table: symbol, ann_date, subcategory, event_type,
                  ratio, amount_per_share, record_date, headline

  dataset_smallcap250/corporate_actions.csv
      Price-adjustment table (same format as config/corporate_actions.csv.example):
      symbol, ex_date, action, ratio, notes
      — only rows where a record date AND a ratio could be parsed
      — use as input to historical_screener price-adjustment logic

Usage:
    python scripts/extract_corporate_actions.py [--dataset FOLDER]
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Target subcategories ──────────────────────────────────────────────────────

CORP_ACTION_SUBCATS = {
    "Bonus",
    "Sub-division / Stock Split",
    "Dividend",
    "Record Date",
    "Amalgamation / Merger / Demerger",
    "Book Closure",
    "Buyback",
    "Rights Issue",
}

# ── Regex helpers ─────────────────────────────────────────────────────────────

# Bonus ratio: "1:1", "2 : 1", "3:2"  → (new, old)
_RE_RATIO = re.compile(r"(\d+)\s*:\s*(\d+)")

# Dividend per share: "Rs. 5.40", "Re. 1", "Rs5/-", "₹2.50", "@ Rs. 3"
_RE_DIV_AMT = re.compile(
    r"(?:rs\.?|re\.?|₹)\s*([\d,]+(?:\.\d+)?)\s*/?\-?",
    re.IGNORECASE,
)

# Dividend percentage: "@20%", "@ 15 %"
_RE_DIV_PCT = re.compile(r"@\s*([\d.]+)\s*%", re.IGNORECASE)

# Record / ex-date patterns:
#   "Record Date - Friday, July 28, 2023"
#   "Record date Friday, June 26, 2026"
#   "record date: 28th July 2023"
#   "28 July 2023", "July 28, 2023", "28-07-2023", "28/07/2023"
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_RE_DATE_DMY = re.compile(
    rf"(\d{{1,2}})[thstndrd]*\s+(?:{_MONTHS})\s+(\d{{4}})", re.IGNORECASE
)
_RE_DATE_MDY = re.compile(
    rf"(?:{_MONTHS})\s+(\d{{1,2}})[thstndrd]*,?\s+(\d{{4}})", re.IGNORECASE
)
_RE_DATE_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_RE_DATE_NUM = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")

_MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _parse_date_from_text(text: str) -> str:
    """Return first plausible date found in text as YYYY-MM-DD, or ''."""
    # ISO first
    m = _RE_DATE_ISO.search(text)
    if m:
        try:
            datetime.strptime(m.group(1), "%Y-%m-%d")
            return m.group(1)
        except ValueError:
            pass

    # DD Month YYYY
    m = _RE_DATE_DMY.search(text)
    if m:
        day, year = int(m.group(1)), int(m.group(2))
        mon_str = re.search(_MONTHS, m.group(0), re.IGNORECASE).group(0).lower()[:3]
        if "sep" in mon_str[:3]:
            mon_str = "sep"
        month = _MONTH_MAP.get(mon_str, 0)
        if month:
            return f"{year:04d}-{month:02d}-{day:02d}"

    # Month DD, YYYY
    m = _RE_DATE_MDY.search(text)
    if m:
        day, year = int(m.group(1)), int(m.group(2))
        mon_str = re.search(_MONTHS, m.group(0), re.IGNORECASE).group(0).lower()[:3]
        if "sep" in mon_str[:3]:
            mon_str = "sep"
        month = _MONTH_MAP.get(mon_str, 0)
        if month:
            return f"{year:04d}-{month:02d}-{day:02d}"

    # DD/MM/YYYY or DD-MM-YYYY
    m = _RE_DATE_NUM.search(text)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"

    return ""


def parse_bonus(headline: str) -> dict:
    """Return {'ratio': float, 'notes': str} e.g. 1:1 → ratio=2.0 (1+1 new per 1 old)."""
    # Standard "N:M" ratio
    m = _RE_RATIO.search(headline)
    if m:
        new, old = int(m.group(1)), int(m.group(2))
        ratio = round((old + new) / old, 6)
        return {"ratio": ratio, "notes": f"{new}:{old} bonus"}

    # "1 new ... share ... for every ... 1 existing" / "1 new share for every 1"
    m2 = re.search(
        r"(\d+)\s+new\s+(?:fully\s+paid[- ]up\s+)?(?:equity\s+)?shares?\s+.*?for\s+every\s+(?:existing\s+)?(\d+)",
        headline, re.IGNORECASE,
    )
    if m2:
        new, old = int(m2.group(1)), int(m2.group(2))
        ratio = round((old + new) / old, 6)
        return {"ratio": ratio, "notes": f"{new}:{old} bonus (parsed from text)"}

    return {"ratio": None, "notes": ""}


def parse_split(headline: str) -> dict:
    """Return {'ratio': float, 'notes': str} e.g. face 10→1 → ratio=10.0."""
    # "face value from Rs. 10 to Rs. 1"
    m_fv = re.search(
        r"(?:face\s+value|fv)[^\d]*(\d+)[^\d]+(?:to\s+)?(?:rs\.?\s*)?(\d+)",
        headline, re.IGNORECASE,
    )
    if m_fv:
        old_fv, new_fv = int(m_fv.group(1)), int(m_fv.group(2))
        if new_fv > 0:
            ratio = round(old_fv / new_fv, 6)
            return {"ratio": ratio, "notes": f"FV {old_fv}→{new_fv}"}

    # plain ratio e.g. "10:1" means 10 new for 1 old
    m = _RE_RATIO.search(headline)
    if m:
        new, old = int(m.group(1)), int(m.group(2))
        ratio = round(new / old, 6) if old > 0 else None
        return {"ratio": ratio, "notes": f"{new}:{old} split"}

    return {"ratio": None, "notes": ""}


def parse_dividend(headline: str) -> dict:
    """Return {'amount_per_share': float or None, 'div_type': str}."""
    # Try rupee amount first
    m = _RE_DIV_AMT.search(headline)
    if m:
        amt_str = m.group(1).replace(",", "")
        try:
            amt = float(amt_str)
        except ValueError:
            amt = None
    else:
        amt = None

    # Try percentage (then amount is unknown)
    if amt is None:
        m_pct = _RE_DIV_PCT.search(headline)
        pct = float(m_pct.group(1)) if m_pct else None
    else:
        pct = None

    div_type = "final" if "final" in headline.lower() else (
        "interim" if "interim" in headline.lower() else "unknown"
    )

    return {
        "amount_per_share": amt,
        "div_pct": pct,
        "div_type": div_type,
    }


# ── Per-stock processor ───────────────────────────────────────────────────────

def process_stock(sym: str, csv_path: str) -> list[dict]:
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
    except Exception:
        return []

    df["_subcat"] = df.get("SUBCATNAME", pd.Series(dtype=str)).str.strip()
    corp = df[df["_subcat"].isin(CORP_ACTION_SUBCATS)]
    if corp.empty:
        return []

    rows = []
    for _, r in corp.iterrows():
        subcat = r["_subcat"]
        headline = r.get("HEADLINE", r.get("NEWSSUB", "")).strip()
        ann_date = r.get("NEWS_DT", "")[:10]

        row: dict = {
            "symbol": sym,
            "ann_date": ann_date,
            "subcategory": subcat,
            "event_type": subcat.lower().replace(" / ", "/").replace(" ", "_"),
            "ratio": None,
            "amount_per_share": None,
            "div_pct": None,
            "div_type": "",
            "record_date": "",
            "headline": headline[:200],
        }

        if subcat == "Bonus":
            parsed = parse_bonus(headline)
            row["ratio"] = parsed["ratio"]
            row["notes"] = parsed["notes"]

        elif subcat == "Sub-division / Stock Split":
            parsed = parse_split(headline)
            row["ratio"] = parsed["ratio"]
            row["notes"] = parsed["notes"]
            # Record date sometimes in headline
            row["record_date"] = _parse_date_from_text(headline)

        elif subcat == "Dividend":
            parsed = parse_dividend(headline)
            row["amount_per_share"] = parsed["amount_per_share"]
            row["div_pct"] = parsed["div_pct"]
            row["div_type"] = parsed["div_type"]

        elif subcat == "Record Date":
            # Parse the actual record date from headline
            row["record_date"] = _parse_date_from_text(headline)
            # Identify what the record date is for
            hl_lower = headline.lower()
            if "bonus" in hl_lower:
                row["event_type"] = "record_date_bonus"
            elif "split" in hl_lower or "subdivision" in hl_lower or "sub-division" in hl_lower:
                row["event_type"] = "record_date_split"
            elif "dividend" in hl_lower:
                row["event_type"] = "record_date_dividend"
            else:
                row["event_type"] = "record_date_other"

        elif subcat == "Amalgamation / Merger / Demerger":
            hl_lower = headline.lower()
            if "demerger" in hl_lower:
                row["event_type"] = "demerger"
            elif "amalgamation" in hl_lower:
                row["event_type"] = "amalgamation"
            else:
                row["event_type"] = "merger"
            row["record_date"] = _parse_date_from_text(headline)

        elif subcat == "Buyback":
            row["event_type"] = "buyback"

        elif subcat == "Rights Issue":
            row["event_type"] = "rights_issue"
            row["ratio"] = None  # ratio parsed separately if needed

        rows.append(row)

    return rows


# ── Link record dates back to bonus/split rows ────────────────────────────────

def enrich_with_record_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For bonus and split rows that lack a record_date, attempt to fill it from
    a nearby Record Date announcement for the same symbol.
    Matches on: same symbol + record_date_bonus/record_date_split within ±90 days.
    """
    df = df.copy()
    df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")

    record_rows = df[
        df["event_type"].isin(["record_date_bonus", "record_date_split", "record_date_dividend"])
        & df["record_date"].notna()
        & (df["record_date"] != "")
    ].copy()
    record_rows["record_date"] = pd.to_datetime(record_rows["record_date"], errors="coerce")

    for idx, row in df[
        df["event_type"].isin(["bonus", "sub-division/stock_split"])
        & (df["record_date"].isna() | (df["record_date"] == ""))
    ].iterrows():
        sym = row["symbol"]
        adate = row["ann_date"]
        if pd.isna(adate):
            continue
        matches = record_rows[
            (record_rows["symbol"] == sym)
            & (abs((record_rows["ann_date"] - adate).dt.days) <= 90)
        ]
        if not matches.empty:
            rd = matches.iloc[0]["record_date"]
            if pd.notna(rd):
                df.at[idx, "record_date"] = rd.strftime("%Y-%m-%d")

    df["ann_date"] = df["ann_date"].dt.strftime("%Y-%m-%d").fillna("")
    return df


# ── Build price-adjustment table ──────────────────────────────────────────────

def build_price_adj_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build corporate_actions.csv (price-adjustment format):
    symbol, ex_date, action, ratio, notes
    Only rows with a parsed (non-NaN) ratio AND a usable date.
    """
    rows = []
    for _, r in df.iterrows():
        if r["event_type"] in ("bonus", "sub-division/stock_split"):
            ratio = r.get("ratio")
            if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)):
                continue
            action = "bonus" if r["event_type"] == "bonus" else "split"
            ex_date = r.get("record_date") or r.get("ann_date") or ""
            if not ex_date or str(ex_date) in ("nan", "NaT", ""):
                continue
            rows.append({
                "symbol": r["symbol"],
                "ex_date": str(ex_date)[:10],
                "action": action,
                "ratio": round(float(ratio), 6),
                "notes": r.get("headline", "")[:100],
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["symbol", "ex_date", "action"])
        out = out.sort_values(["symbol", "ex_date"])
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="dataset_smallcap250",
        help="Dataset root folder (default: dataset_smallcap250)",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    ann_base = dataset / "bse_announcements"
    if not ann_base.is_dir():
        print(f"[ERROR] bse_announcements not found: {ann_base}")
        return

    paths = sorted(glob.glob(str(ann_base / "*" / "announcements.csv")))
    print(f"Processing {len(paths)} symbols from {ann_base} ...")

    all_rows: list[dict] = []
    for path in paths:
        sym = os.path.basename(os.path.dirname(path))
        rows = process_stock(sym, path)
        all_rows.extend(rows)

    if not all_rows:
        print("No corporate action rows found.")
        return

    df = pd.DataFrame(all_rows)
    df = enrich_with_record_dates(df)
    df = df.sort_values(["symbol", "ann_date"], na_position="last")

    # ── Full extracted table ──────────────────────────────────────────────────
    out_full = dataset / "corporate_actions_extracted.csv"
    cols = [
        "symbol", "ann_date", "subcategory", "event_type",
        "ratio", "amount_per_share", "div_pct", "div_type",
        "record_date", "headline",
    ]
    df[[c for c in cols if c in df.columns]].to_csv(out_full, index=False)
    print(f"\nWrote {len(df)} rows → {out_full}")

    # ── Summary by event type ─────────────────────────────────────────────────
    print("\nEvent counts:")
    for ev, cnt in df["event_type"].value_counts().items():
        print(f"  {ev:35s} {cnt:4d}")

    # ── Bonus/split detail ────────────────────────────────────────────────────
    bs = df[df["event_type"].isin(["bonus", "sub-division/stock_split"])]
    parsed = bs[bs["ratio"].notna()]
    print(f"\nBonus/split parsed ratio: {len(parsed)}/{len(bs)}")
    if not parsed.empty:
        print(parsed[["symbol", "ann_date", "event_type", "ratio", "record_date", "headline"]]
              .head(10).to_string(index=False))

    # ── Dividend detail ───────────────────────────────────────────────────────
    divs = df[df["event_type"] == "dividend"]
    div_parsed = divs[divs["amount_per_share"].notna()]
    print(f"\nDividends with parsed amount: {len(div_parsed)}/{len(divs)}")
    if not div_parsed.empty:
        print(div_parsed[["symbol", "ann_date", "amount_per_share", "div_type", "headline"]]
              .head(10).to_string(index=False))

    # ── Price-adjustment table ────────────────────────────────────────────────
    adj = build_price_adj_table(df)
    out_adj = dataset / "corporate_actions.csv"
    if not adj.empty:
        adj.to_csv(out_adj, index=False)
        print(f"\nWrote {len(adj)} price-adjustment rows → {out_adj}")
        print(adj.head(10).to_string(index=False))
    else:
        print("\nNo rows with both ratio and date for price-adjustment table.")


if __name__ == "__main__":
    main()
