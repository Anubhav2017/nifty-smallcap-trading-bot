"""Tests for historical point-in-time screener."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_bot.data.screener_excel import (
    SCREENER_SUFFIX,
    load_balance_sheet_extended,
    load_symbol_fundamentals,
)
from trading_bot.screener.historical import (
    HistoricalScreener,
    fundamentals_as_of,
    load_shares_history,
    technicals_as_of,
)


def _write_screener(path: Path) -> None:
    rows = [
        ["COMPANY NAME", "Test Co"],
        ["PROFIT & LOSS", None],
        ["Report Date", "2023-03-31", "2024-03-31"],
        ["Sales", 900.0, 1000.0],
        ["Net profit", 80.0, 100.0],
        ["Operating Profit", 120.0, 150.0],
        ["Quarters", None],
        ["Report Date", "2024-03-31", "2024-06-30"],
        ["Sales", 240.0, 260.0],
        ["Operating Profit", 30.0, 35.0],
        ["Net profit", 20.0, 22.0],
        ["BALANCE SHEET", None],
        ["Report Date", "2023-03-31", "2024-03-31"],
        ["Equity Share Capital", 100.0, 100.0],
        ["Reserves", 400.0, 500.0],
        ["Borrowings", 200.0, 250.0],
        ["Other Liabilities", 80.0, 90.0],
        ["Total", 880.0, 990.0],
        ["Net Block", 300.0, 350.0],
        ["Other Assets", 580.0, 640.0],
        ["Total", 880.0, 990.0],
        ["Cash & Bank", 50.0, 60.0],
        ["No. of Equity Shares", 10000000.0, 10000000.0],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Data Sheet", header=False, index=False)


@pytest.fixture
def mini_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    sym = "TESTCO"
    # OHLCV
    day_dir = root / "ohlcv" / "day"
    day_dir.mkdir(parents=True)
    dates = pd.bdate_range("2024-01-01", periods=300)
    n = len(dates)
    close = [100.0 + (i % 7) for i in range(n)]
    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": [1_000_000] * n,
        }
    ).to_csv(day_dir / f"{sym}.csv", index=False)

    _write_screener(root / "screener_excel" / f"{sym}{SCREENER_SUFFIX}")
    return root


def test_technicals_as_of_truncates_future(mini_dataset: Path) -> None:
    bars = pd.read_csv(mini_dataset / "ohlcv/day/TESTCO.csv")
    tech = technicals_as_of(bars, pd.Timestamp("2024-06-15"))
    assert tech["close"] == pytest.approx(100.0 + (len(bars[bars["date"] <= "2024-06-15"]) - 1) % 7)
    assert tech["volume_avg_252d"] == pytest.approx(1_000_000)


def test_fundamentals_as_of_uses_latest_filing(mini_dataset: Path) -> None:
    path = mini_dataset / "screener_excel" / f"TESTCO{SCREENER_SUFFIX}"
    fund = load_symbol_fundamentals(mini_dataset / "screener_excel", "TESTCO")
    shares = load_shares_history(path)
    bs_ext = load_balance_sheet_extended(path)
    out = fundamentals_as_of(
        fund, shares, bs_ext, pd.Timestamp("2024-08-01"), close=110.0, price_for_mcap=110.0
    )
    assert out["report_date_pl"] == "2024-03-31"
    assert out["debt_to_equity"] == pytest.approx(250 / 600)
    assert out["market_cap_cr"] == pytest.approx(110.0 * 10_000_000 / 1e7)


def test_snapshot_integration(mini_dataset: Path) -> None:
    screener = HistoricalScreener(mini_dataset)
    snap = screener.snapshot("TESTCO", "2024-08-01")
    assert snap.symbol == "TESTCO"
    assert snap.close is not None
    assert snap.rsi_14 is not None
    assert snap.market_cap_cr is not None
