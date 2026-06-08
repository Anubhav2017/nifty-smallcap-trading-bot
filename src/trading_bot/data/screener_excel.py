"""Parse Screener.in Excel exports (dataset_smallcap250/screener_excel/)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

SCREENER_SUFFIX = "_consolidated.xlsx"
SCREENER_SUFFIX_STANDALONE = "_standalone.xlsx"
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
    "Interest",
    "Depreciation",
}


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def parse_dates_with_positions(row: pd.Series) -> list[tuple[int, pd.Timestamp]]:
    """Return ``(column_position, date)`` for each parseable date cell in *row*.

    Screener sheets are right-aligned when a company has fewer reporting periods
    than the template's column count, so the date cells (and their value cells
    below) may start well after column 1.  Capturing the position lets callers
    read each metric's value from the *same* column as its date, instead of
    assuming dates occupy columns ``1..len(dates)`` (which silently produced
    all-NaN reads for short-history / Nov-FY companies such as PFIZER, TIMKEN,
    BAYERCROP, BLUEJET, FIVESTAR).
    """
    out: list[tuple[int, pd.Timestamp]] = []
    values = list(row)
    for pos in range(1, len(values)):  # column 0 is the label
        val = values[pos]
        if pd.isna(val):
            continue
        try:
            out.append((pos, pd.Timestamp(val).normalize()))
        except (TypeError, ValueError):
            continue
    return out


def parse_dates(row: pd.Series) -> list[pd.Timestamp]:
    return [dt for _pos, dt in parse_dates_with_positions(row)]


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

        date_pairs = parse_dates_with_positions(block.iloc[date_row_idx])
        if not date_pairs:
            continue
        cols = [pos for pos, _dt in date_pairs]
        dates = [dt for _pos, dt in date_pairs]

        for j in range(date_row_idx + 1, len(block)):
            metric = block.iloc[j, 0]
            if not isinstance(metric, str) or not metric.strip():
                continue
            metric = metric.strip()
            if metric not in _KEY_METRICS:
                continue
            row_vals = list(block.iloc[j])
            for dt, col in zip(dates, cols):
                val = row_vals[col] if col < len(row_vals) else None
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
        out.loc[mask, "f_sales_growth"] = sub["f_sales"].pct_change(fill_method=None).values
    return out


def load_symbol_fundamentals(screener_dir: Path, symbol: str) -> pd.DataFrame:
    path = screener_file(screener_dir, symbol)
    if not path.is_file():
        return pd.DataFrame()
    long_df = parse_screener_excel(path)
    wide = _pivot_fundamentals(long_df)
    if wide.empty:
        return wide
    return _derived_ratios(wide)


def list_screener_symbols(screener_dir: Path) -> list[str]:
    found: set[str] = set()
    for p in screener_dir.glob(f"*{SCREENER_SUFFIX}"):
        found.add(p.name[: -len(SCREENER_SUFFIX)].upper())
    for p in screener_dir.glob(f"*{SCREENER_SUFFIX_STANDALONE}"):
        found.add(p.name[: -len(SCREENER_SUFFIX_STANDALONE)].upper())
    return sorted(found)


def screener_file(screener_dir: Path, symbol: str) -> Path:
    """Return the path to the screener Excel file for *symbol*.

    Prefers ``_consolidated.xlsx``; falls back to ``_standalone.xlsx``.
    """
    consolidated = screener_dir / f"{symbol.upper()}{SCREENER_SUFFIX}"
    if consolidated.is_file():
        return consolidated
    standalone = screener_dir / f"{symbol.upper()}{SCREENER_SUFFIX_STANDALONE}"
    if standalone.is_file():
        return standalone
    return consolidated  # caller checks .is_file()


def _section_table(block: pd.DataFrame, date_row_idx: int, dates: List[pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict] = []
    date_pairs = parse_dates_with_positions(block.iloc[date_row_idx])
    cols = [pos for pos, _dt in date_pairs]
    date_cols = [dt.strftime("%Y-%m-%d") for _pos, dt in date_pairs]
    for j in range(date_row_idx + 1, len(block)):
        metric = block.iloc[j, 0]
        if not isinstance(metric, str) or not metric.strip():
            continue
        metric = metric.strip()
        if metric == "Report Date":
            continue
        row_vals = list(block.iloc[j])
        row = {"Metric": metric}
        for col_name, col in zip(date_cols, cols):
            row[col_name] = row_vals[col] if col < len(row_vals) else None
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


_META_COLUMNS = {
    "COMPANY NAME": "company_name",
    "Current Price": "current_price",
    "Market Capitalization": "market_cap",
    "Face Value": "face_value",
    "Number of shares": "shares",
}


def load_symbol_meta(screener_dir: Path, symbol: str) -> dict[str, str]:
    path = screener_file(screener_dir, symbol)
    meta = load_meta(path)
    row = {"symbol": symbol.upper()}
    for src, dst in _META_COLUMNS.items():
        row[dst] = meta.get(src, "")
    return row


def consolidate_screener_directory(
    screener_dir: Path,
    symbols: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Parse every ``*{SCREENER_SUFFIX}`` file under *screener_dir*.

    Returns ``(meta, fundamentals_wide, fundamentals_long, errors)`` where
    *errors* lists ``SYMBOL: reason`` for files that failed to parse.
    """
    screener_dir = Path(screener_dir)
    if symbols is None:
        paths = sorted(screener_dir.glob(f"*{SCREENER_SUFFIX}"))
    else:
        paths = [screener_file(screener_dir, sym) for sym in symbols]

    meta_rows: list[dict] = []
    wide_parts: list[pd.DataFrame] = []
    long_parts: list[pd.DataFrame] = []
    errors: list[str] = []

    for path in paths:
        if not path.is_file():
            errors.append(f"{path.stem}: file not found")
            continue
        symbol = path.name[: -len(SCREENER_SUFFIX)].upper()
        try:
            meta_rows.append(load_symbol_meta(screener_dir, symbol))
            long_df = parse_screener_excel(path)
            if not long_df.empty:
                tagged = long_df.copy()
                tagged.insert(0, "symbol", symbol)
                long_parts.append(tagged)
            wide = load_symbol_fundamentals(screener_dir, symbol)
            if not wide.empty:
                tagged_wide = wide.copy()
                tagged_wide.insert(0, "symbol", symbol)
                wide_parts.append(tagged_wide)
        except Exception as exc:  # noqa: BLE001 — collect per-symbol failures
            errors.append(f"{symbol}: {exc}")

    meta = pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame(columns=["symbol"])
    fundamentals_wide = (
        pd.concat(wide_parts, ignore_index=True) if wide_parts else pd.DataFrame()
    )
    fundamentals_long = (
        pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
    )

    if not fundamentals_wide.empty:
        fundamentals_wide["report_date"] = pd.to_datetime(
            fundamentals_wide["report_date"]
        ).dt.normalize()
        fundamentals_wide = fundamentals_wide.sort_values(
            ["symbol", "period_type", "report_date"]
        ).reset_index(drop=True)
    if not fundamentals_long.empty:
        fundamentals_long["report_date"] = pd.to_datetime(
            fundamentals_long["report_date"]
        ).dt.normalize()
        fundamentals_long = fundamentals_long.sort_values(
            ["symbol", "period_type", "report_date", "metric"]
        ).reset_index(drop=True)
    if not meta.empty:
        meta = meta.sort_values("symbol").reset_index(drop=True)

    return meta, fundamentals_wide, fundamentals_long, errors


def write_consolidated_screener(
    output: Path,
    meta: pd.DataFrame,
    fundamentals_wide: pd.DataFrame,
    fundamentals_long: pd.DataFrame,
    *,
    fmt: str = "xlsx",
) -> None:
    """Write consolidated tables to a single file (xlsx) or parquet/csv bundle."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()

    if fmt == "xlsx":
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            meta.to_excel(writer, sheet_name="meta", index=False)
            fundamentals_wide.to_excel(writer, sheet_name="fundamentals", index=False)
            fundamentals_long.to_excel(writer, sheet_name="fundamentals_long", index=False)
        return

    if fmt == "parquet":
        # Single parquet file: wide fundamentals (primary analysis table).
        # Meta and long form as sibling files with shared stem.
        fundamentals_wide.to_parquet(output, index=False)
        if not meta.empty:
            meta.to_parquet(output.with_name(output.stem + "_meta.parquet"), index=False)
        if not fundamentals_long.empty:
            fundamentals_long.to_parquet(
                output.with_name(output.stem + "_long.parquet"), index=False
            )
        return

    if fmt == "csv":
        fundamentals_wide.to_csv(output, index=False)
        if not meta.empty:
            meta.to_csv(output.with_name(output.stem + "_meta.csv"), index=False)
        if not fundamentals_long.empty:
            fundamentals_long.to_csv(
                output.with_name(output.stem + "_long.csv"), index=False
            )
        return

    raise ValueError(f"Unsupported format: {fmt!r} (use xlsx, parquet, or csv)")


def load_balance_sheet_extended(path: Path) -> pd.DataFrame:
    """
    Extended annual balance-sheet history from the Data Sheet block.

    Returns columns: report_date, total_liabilities, total_assets, other_liabilities,
    bonus_shares, shares (when present).
    """
    if not path.is_file():
        return pd.DataFrame(
            columns=[
                "report_date",
                "total_liabilities",
                "total_assets",
                "other_liabilities",
                "bonus_shares",
                "shares",
            ]
        )

    raw = pd.read_excel(path, sheet_name=DATA_SHEET, header=None)
    sections = find_sections(raw)
    bs_start = next((i for i, pt in sections if pt == "annual_bs"), None)
    if bs_start is None:
        return pd.DataFrame(
            columns=[
                "report_date",
                "total_liabilities",
                "total_assets",
                "other_liabilities",
                "bonus_shares",
                "shares",
            ]
        )

    bs_end = next((i for i, pt in sections if i > bs_start), len(raw))
    block = raw.iloc[bs_start:bs_end]

    date_row_idx = None
    for j in range(len(block)):
        label = block.iloc[j, 0]
        if isinstance(label, str) and label.strip() == "Report Date":
            date_row_idx = j
            break
    if date_row_idx is None:
        return pd.DataFrame(
            columns=[
                "report_date",
                "total_liabilities",
                "total_assets",
                "other_liabilities",
                "bonus_shares",
                "shares",
            ]
        )

    date_pairs = parse_dates_with_positions(block.iloc[date_row_idx])
    if not date_pairs:
        return pd.DataFrame(
            columns=[
                "report_date",
                "total_liabilities",
                "total_assets",
                "other_liabilities",
                "bonus_shares",
                "shares",
            ]
        )
    cols = [pos for pos, _dt in date_pairs]
    dates = [dt for _pos, dt in date_pairs]

    metric_rows: dict[str, list[float | None]] = {
        "other_liabilities": [None] * len(dates),
        "total_liabilities": [None] * len(dates),
        "total_assets": [None] * len(dates),
        "bonus_shares": [None] * len(dates),
        "shares": [None] * len(dates),
    }
    total_seen = 0

    for j in range(date_row_idx + 1, len(block)):
        metric = block.iloc[j, 0]
        if not isinstance(metric, str) or not metric.strip():
            continue
        name = metric.strip()
        row_vals = list(block.iloc[j])
        nums: list[float | None] = []
        for col in cols:
            val = row_vals[col] if col < len(row_vals) else None
            if pd.isna(val):
                nums.append(None)
                continue
            try:
                nums.append(float(val))
            except (TypeError, ValueError):
                nums.append(None)

        if name == "Other Liabilities":
            metric_rows["other_liabilities"] = nums
        elif name == "Total":
            total_seen += 1
            key = "total_liabilities" if total_seen == 1 else "total_assets"
            metric_rows[key] = nums
        elif name == "New Bonus Shares":
            metric_rows["bonus_shares"] = nums
        elif name == "No. of Equity Shares":
            metric_rows["shares"] = nums

    rows: list[dict] = []
    for i, dt in enumerate(dates):
        rows.append(
            {
                "report_date": dt,
                "total_liabilities": metric_rows["total_liabilities"][i],
                "total_assets": metric_rows["total_assets"][i],
                "other_liabilities": metric_rows["other_liabilities"][i],
                "bonus_shares": metric_rows["bonus_shares"][i],
                "shares": metric_rows["shares"][i],
            }
        )
    return pd.DataFrame(rows).sort_values("report_date").reset_index(drop=True)


def load_bonus_shares_history(path: Path) -> pd.DataFrame:
    """Bonus share issuances from extended balance sheet parse."""
    ext = load_balance_sheet_extended(path)
    if ext.empty:
        return pd.DataFrame(columns=["report_date", "bonus_shares"])
    sub = ext[ext["bonus_shares"].notna() & (ext["bonus_shares"] > 0)][
        ["report_date", "bonus_shares"]
    ]
    return sub.reset_index(drop=True)
