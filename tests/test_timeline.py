"""Tests for dashboard timeline helpers."""

from __future__ import annotations

import pandas as pd

from dashboard.timeline import (
    fundamental_events,
    merge_timeline_events,
    significant_price_moves,
)


def _sample_bars(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(100.0, index=dates)
    close.iloc[50] = 112.0  # ~+12% spike
    close.iloc[80] = 88.0  # ~-12% drop
    return pd.DataFrame(
        {
            "date": dates,
            "open": close.shift(1).fillna(100),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        }
    )


def test_significant_price_moves_detects_spikes():
    df = _sample_bars()
    moves = significant_price_moves(df, window=40, z_threshold=2.0, min_abs_return=0.05)
    assert not moves.empty
    assert set(moves["direction"]) <= {"up", "down"}
    assert (moves["z_score"].abs() >= 2.0).all()


def test_fundamental_events_from_wide_frame():
    fund = pd.DataFrame(
        {
            "period_type": ["quarterly", "quarterly"],
            "report_date": ["2024-03-31", "2024-06-30"],
            "f_sales": [1e9, 1.1e9],
            "f_net_profit": [1e8, 1.2e8],
            "f_sales_growth": [0.1, 0.15],
        }
    )
    events = fundamental_events(fund)
    assert len(events) == 2
    assert events.iloc[0]["category"] == "fundamental"
    assert "Sales" in events.iloc[0]["detail"]


def test_merge_timeline_events_sorts_newest_first():
    fund = fundamental_events(
        pd.DataFrame(
            {
                "period_type": ["quarterly"],
                "report_date": ["2024-06-30"],
                "f_sales": [1e9],
            }
        )
    )
    moves = significant_price_moves(_sample_bars(), window=40, z_threshold=2.0)
    table = merge_timeline_events(fund, moves)
    assert not table.empty
    dates = pd.to_datetime(table["date"])
    assert dates.is_monotonic_decreasing


def test_significant_price_moves_requires_daily_series():
    """Move detection operates on one row per calendar day."""
    daily = _sample_bars()
    moves = significant_price_moves(daily, window=20, z_threshold=2.0, min_abs_return=0.05)
    assert moves["date"].dt.normalize().nunique() == len(moves)
