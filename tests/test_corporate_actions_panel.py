"""Tests for corporate actions and panel cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_bot.data.corporate_actions import adjust_ohlcv, infer_actions_from_shares
from trading_bot.data.screener_excel import SCREENER_SUFFIX, load_balance_sheet_extended
from trading_bot.screener.panel_cache import build_panel_cache, load_panel
from tests.test_historical_screener import mini_dataset  # noqa: F401 — pytest fixture


def test_infer_bonus_from_share_jump():
    shares = pd.DataFrame(
        {
            "report_date": pd.to_datetime(["2019-03-31", "2020-03-31"]),
            "shares": [100.0, 200.0],
        }
    )
    actions = infer_actions_from_shares("TEST", shares)
    assert len(actions) == 1
    assert actions.iloc[0]["ratio"] == pytest.approx(2.0)


def test_adjust_ohlcv_backward():
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-04-01"]),
            "open": [100.0, 50.0],
            "high": [101.0, 51.0],
            "low": [99.0, 49.0],
            "close": [100.0, 50.0],
            "volume": [1000, 2000],
        }
    )
    actions = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "ex_date": pd.to_datetime(["2020-04-01"]),
            "action": ["bonus"],
            "ratio": [2.0],
            "notes": [""],
        }
    )
    adj = adjust_ohlcv(bars, actions, "TEST")
    assert adj.iloc[0]["close_adj"] == pytest.approx(50.0)
    assert adj.iloc[1]["close_adj"] == pytest.approx(50.0)
    assert adj.iloc[0]["volume_adj"] == pytest.approx(2000.0)


def test_load_balance_sheet_extended(mini_screener_path: Path) -> None:
    ext = load_balance_sheet_extended(mini_screener_path)
    assert not ext.empty
    assert ext["total_assets"].notna().any()
    assert ext["shares"].notna().any()


@pytest.fixture
def mini_screener_path(tmp_path: Path) -> Path:
    rows = [
        ["BALANCE SHEET", None],
        ["Report Date", "2023-03-31", "2024-03-31"],
        ["Other Liabilities", 100.0, 110.0],
        ["Total", 500.0, 600.0],
        ["Net Block", 200.0, 220.0],
        ["Other Assets", 300.0, 380.0],
        ["Total", 500.0, 600.0],
        ["No. of Equity Shares", 10_000_000.0, 20_000_000.0],
        ["New Bonus Shares", None, 10_000_000.0],
    ]
    path = tmp_path / f"TEST{SCREENER_SUFFIX}"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Data Sheet", header=False, index=False)
    return path


def test_panel_cache_roundtrip(mini_dataset: Path) -> None:
    df, manifest = build_panel_cache(
        mini_dataset,
        "2024-06-01",
        "2024-08-01",
        freq="W-FRI",
        symbols=["TESTCO"],
        incremental=False,
    )
    assert manifest["rows"] == len(df)
    assert df["symbol"].nunique() == 1
    loaded = load_panel(mini_dataset)
    assert len(loaded) >= len(df)
