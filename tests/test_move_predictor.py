"""Tests for move predictor (no lookahead)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from trading_bot.strategies.move_predictor.features import (
    LABEL_BIG_UP_COL,
    build_lagged_panel,
    symbol_lagged_frame,
)
from trading_bot.strategies.move_predictor.walk_forward import (
    quarterly_walk_forward_folds,
    quarter_key,
)
from trading_bot.types import Instrument


def _bars(n: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(range(100, 100 + n), index=dates, dtype=float)
    return pd.DataFrame(
        {
            "date": [d.date() for d in dates],
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 2_000_000.0,
        }
    )


def test_big_up_label_threshold():
    inst = Instrument(symbol="TEST", isin="INE", instrument_token=1)
    frame = symbol_lagged_frame(_bars(), inst, label_min_move_pct=0.02)
    row = frame.dropna(subset=["fwd_ret_1d"]).iloc[0]
    if row["fwd_ret_1d"] >= 0.02:
        assert row[LABEL_BIG_UP_COL] == 1.0
    else:
        assert row[LABEL_BIG_UP_COL] == 0.0


def test_last_row_has_no_forward_label():
    inst = Instrument(symbol="TEST", isin="INE", instrument_token=1)
    frame = symbol_lagged_frame(_bars(), inst)
    assert pd.isna(frame.iloc[-1]["fwd_ret_1d"])


def test_quarterly_folds_no_overlap():
    dates = pd.bdate_range("2025-01-01", "2025-06-15").date.tolist()
    folds = quarterly_walk_forward_folds(dates, date(2024, 1, 1))
    assert len(folds) >= 2
    assert folds[0]["train_end"] < folds[0]["oos_start"]
    assert folds[1]["train_end"] >= folds[0]["oos_end"] or folds[1]["train_end"] > folds[0]["train_end"]


def test_quarter_key():
    assert quarter_key(date(2025, 4, 15)) == (2025, 1)
