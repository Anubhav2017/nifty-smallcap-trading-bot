"""Tests for Screener consolidation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_bot.data.screener_excel import (
    SCREENER_SUFFIX,
    consolidate_screener_directory,
    write_consolidated_screener,
)


def _write_minimal_screener(path: Path, symbol: str = "TESTCO") -> None:
    """Minimal Screener-style workbook for parser tests."""
    rows = [
        ["COMPANY NAME", "Test Co Ltd"],
        ["Current Price", 100.0],
        ["Market Capitalization", "1,000 Cr."],
        ["PROFIT & LOSS", None],
        ["Report Date", "2024-03-31", "2023-03-31"],
        ["Sales", 1000.0, 900.0],
        ["Net profit", 100.0, 80.0],
        ["Quarters", None],
        ["Report Date", "2024-06-30"],
        ["Sales", 260.0],
        ["Net profit", 28.0],
    ]
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data Sheet", header=False, index=False)


@pytest.fixture
def screener_dir(tmp_path: Path) -> Path:
    d = tmp_path / "screener_excel"
    _write_minimal_screener(d / f"AAA{SCREENER_SUFFIX}", "AAA")
    _write_minimal_screener(d / f"BBB{SCREENER_SUFFIX}", "BBB")
    return d


def test_consolidate_screener_directory(screener_dir: Path) -> None:
    meta, wide, long_df, errors = consolidate_screener_directory(screener_dir)
    assert not errors
    assert set(meta["symbol"]) == {"AAA", "BBB"}
    assert "company_name" in meta.columns
    assert set(wide["symbol"]) == {"AAA", "BBB"}
    assert "f_sales" in wide.columns
    assert "f_profit_margin" in wide.columns
    assert set(long_df["symbol"]) == {"AAA", "BBB"}
    assert "metric" in long_df.columns


def test_write_consolidated_xlsx(screener_dir: Path, tmp_path: Path) -> None:
    meta, wide, long_df, _ = consolidate_screener_directory(screener_dir)
    out = tmp_path / "all.xlsx"
    write_consolidated_screener(out, meta, wide, long_df, fmt="xlsx")
    assert out.is_file()
    sheets = pd.read_excel(out, sheet_name=None)
    assert "meta" in sheets
    assert "fundamentals" in sheets
    assert len(sheets["meta"]) == 2


def test_write_consolidated_parquet(screener_dir: Path, tmp_path: Path) -> None:
    meta, wide, long_df, _ = consolidate_screener_directory(screener_dir)
    out = tmp_path / "all.parquet"
    write_consolidated_screener(out, meta, wide, long_df, fmt="parquet")
    assert out.is_file()
    assert out.with_name("all_meta.parquet").is_file()
    loaded = pd.read_parquet(out)
    assert len(loaded) == len(wide)
