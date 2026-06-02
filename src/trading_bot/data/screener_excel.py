"""Parse Screener.in Excel exports (dataset_smallcap250/screener_excel/)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

SCREENER_SUFFIX = "_consolidated.xlsx"
DATA_SHEET = "Data Sheet"

_SECTIONS = (
    ("PROFIT & LOSS", "annual_pl"),
    ("Quarters", "quarterly"),
    ("BALANCE SHEET", "annual_bs"),
    ("CASH FLOW:", "annual_cf"),
)

SECTION_LABELS = {
    "annual_pl": "Profit & Loss (annual)",
    "quarterly": "Quarters",
    "annual_bs": "Balance Sheet (annual)",
    "annual_cf": "Cash Flow (annual)",
}

_KEY_METRICS = {
    "Sales",
    "Net profit",
    "Operating Profit",
    "Borrowings",
    "Reserves",
    "Equity Share Capital",
    "Cash & Bank",
    "Receivables",
    "Cash from Operating Activity",
    "Profit before tax",
    "Expenses",
}


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def parse_dates(row: pd.Series) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    for val in row.iloc[1:]:
        if pd.isna(val):
            continue
        try:
            dates.append(pd.Timestamp(val).normalize())
        except (TypeError, ValueError):
            continue
    return dates


def find_sections(df: pd.DataFrame) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    labels = {title: period for title, period in _SECTIONS}
    for i in range(len(df)):
        cell = df.iloc[i, 0]
        if isinstance(cell, str) and cell.strip() in labels:
            found.append((i, labels[cell.strip()]))
    return found


def parse_screener_excel(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=DATA_SHEET, header=None)
    sections = find_sections(raw)
    rows: list[dict] = []

    for idx, (start_row, period_type) in enumerate(sections):
        end_row = sections[idx + 1][0] if idx + 1 < len(sections) else len(raw)
        block = raw.iloc[start_row:end_row]
        date_row_idx = None
        for j in range(len(block)):
            label = block.iloc[j, 0]
            if isinstance(label, str) and label.strip() == "Report Date":
                date_row_idx = j
                break
        if date_row_idx is None:
            continue

        dates = parse_dates(block.iloc[date_row_idx])
        if not dates:
            continue

        for j in range(date_row_idx + 1, len(block)):
            metric = block.iloc[j, 0]
            if not isinstance(metric, str) or not metric.strip():
                continue
            metric = metric.strip()
            if metric not in _KEY_METRICS:
                continue
            values = block.iloc[j, 1 : 1 + len(dates)]
            for dt, val in zip(dates, values):
                if pd.isna(val):
                    continue
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "period_type": period_type,
                        "report_date": dt,
                        "metric": _slug(metric),
                        "value": num,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["period_type", "report_date", "metric", "value"])
    return pd.DataFrame(rows)


def _pivot_fundamentals(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    wide = long_df.pivot_table(
        index=["period_type", "report_date"],
        columns="metric",
        values="value",
        aggfunc="last",
    )
    wide.columns = [f"f_{c}" for c in wide.columns]
    return wide.reset_index()


def _derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sales = out.get("f_sales")
    np_ = out.get("f_net_profit")
    reserves = out.get("f_reserves")
    equity = out.get("f_equity_share_capital")
    borrowings = out.get("f_borrowings")

    if sales is not None and np_ is not None:
        out["f_profit_margin"] = np_ / sales.replace(0, pd.NA)
    if np_ is not None and reserves is not None and equity is not None:
        book = equity + reserves
        out["f_roe"] = np_ / book.replace(0, pd.NA)
    if borrowings is not None and reserves is not None and equity is not None:
        book = equity + reserves
        out["f_debt_equity"] = borrowings / book.replace(0, pd.NA)

    for period in ("annual_pl", "quarterly"):
        mask = out["period_type"] == period
        if not mask.any() or "f_sales" not in out.columns:
            continue
        sub = out.loc[mask].sort_values("report_date")
        out.loc[mask, "f_sales_growth"] = sub["f_sales"].pct_change().values
    return out


def load_symbol_fundamentals(screener_dir: Path, symbol: str) -> pd.DataFrame:
    path = screener_dir / f"{symbol.upper()}{SCREENER_SUFFIX}"
    if not path.is_file():
        return pd.DataFrame()
    long_df = parse_screener_excel(path)
    wide = _pivot_fundamentals(long_df)
    if wide.empty:
        return wide
    return _derived_ratios(wide)


def list_screener_symbols(screener_dir: Path) -> list[str]:
    return sorted(
        p.name[: -len(SCREENER_SUFFIX)].upper()
        for p in screener_dir.glob(f"*{SCREENER_SUFFIX}")
    )


def screener_file(screener_dir: Path, symbol: str) -> Path:
    return screener_dir / f"{symbol.upper()}{SCREENER_SUFFIX}"


def _section_table(block: pd.DataFrame, date_row_idx: int, dates: List[pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict] = []
    date_cols = [d.strftime("%Y-%m-%d") for d in dates]
    for j in range(date_row_idx + 1, len(block)):
        metric = block.iloc[j, 0]
        if not isinstance(metric, str) or not metric.strip():
            continue
        metric = metric.strip()
        if metric == "Report Date":
            continue
        values = block.iloc[j, 1 : 1 + len(dates)]
        row = {"Metric": metric}
        for col, val in zip(date_cols, values):
            row[col] = val
        if any(pd.notna(v) and str(v) != "nan" for k, v in row.items() if k != "Metric"):
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Metric")


def load_section_tables(path: Path) -> Dict[str, pd.DataFrame]:
    if not path.is_file():
        return {}
    raw = pd.read_excel(path, sheet_name=DATA_SHEET, header=None)
    sections = find_sections(raw)
    tables: Dict[str, pd.DataFrame] = {}

    for idx, (start_row, period_type) in enumerate(sections):
        end_row = sections[idx + 1][0] if idx + 1 < len(sections) else len(raw)
        block = raw.iloc[start_row:end_row]
        date_row_idx = None
        for j in range(len(block)):
            label = block.iloc[j, 0]
            if isinstance(label, str) and label.strip() == "Report Date":
                date_row_idx = j
                break
        if date_row_idx is None:
            continue
        dates = parse_dates(block.iloc[date_row_idx])
        if not dates:
            continue
        table = _section_table(block, date_row_idx, dates)
        if not table.empty:
            tables[period_type] = table
    return tables


def load_meta(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    raw = pd.read_excel(path, sheet_name=DATA_SHEET, header=None)
    meta: Dict[str, str] = {}
    for i in range(min(12, len(raw))):
        key = raw.iloc[i, 0]
        val = raw.iloc[i, 1]
        if isinstance(key, str) and pd.notna(val) and str(val) != "nan":
            k = key.strip()
            if k in (
                "COMPANY NAME",
                "Current Price",
                "Market Capitalization",
                "Face Value",
                "Number of shares",
            ):
                if isinstance(val, str):
                    meta[k] = val
                elif isinstance(val, float):
                    meta[k] = f"{val:,.2f}"
                else:
                    meta[k] = str(val)
    return meta


def load_all_fundamentals(
    screener_dir: Path,
    symbols: Optional[Iterable[str]] = None,
) -> dict[str, pd.DataFrame]:
    if symbols is None:
        symbols = list_screener_symbols(screener_dir)
    return {sym: load_symbol_fundamentals(screener_dir, sym) for sym in symbols}
