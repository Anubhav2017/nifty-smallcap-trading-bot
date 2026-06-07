"""Tests for significant move factor correlation."""

from __future__ import annotations

import pandas as pd

from dashboard.timeline import significant_price_moves
from trading_bot.analysis.move_correlation import (
    attach_fundamental_context,
    build_move_analysis,
    enrich_technical_factors,
)


def _sample_bars(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(100.0, index=dates)
    close.iloc[50] = 112.0
    close.iloc[80] = 88.0
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close.shift(1).fillna(100),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
        }
    )
    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_20d"] = df["close"].pct_change(20)
    df["ret_60d"] = df["close"].pct_change(60)
    df["volatility_20d"] = df["ret_1d"].rolling(20).std()
    df["volume_ratio_20d"] = 1.0
    df["rsi_14"] = 50.0
    df["close_sma_20d"] = 0.0
    df["close_sma_50d"] = 0.0
    df["high_low_range"] = 0.02
    return df


def _minimal_bars(n: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(100.0, index=dates)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
        }
    )


def test_attach_fundamental_context_asof():
    bars = _minimal_bars(40)
    fund = pd.DataFrame(
        {
            "period_type": ["quarterly", "quarterly"],
            "report_date": ["2024-01-15", "2024-03-31"],
            "f_sales_growth": [0.10, 0.20],
            "f_roe": [0.15, 0.18],
            "f_debt_equity": [0.5, 0.4],
            "f_profit_margin": [0.12, 0.14],
        }
    )
    out = attach_fundamental_context(bars, fund)
    assert "days_since_filing" in out.columns
    assert out.loc[out["date"] == pd.Timestamp("2024-02-01"), "f_roe_asof"].iloc[0] == 0.15


def test_factor_correlations_on_moves():
    bars = enrich_technical_factors(_sample_bars())
    moves = significant_price_moves(bars, window=40, z_threshold=2.0, min_abs_return=0.05)
    analysis = build_move_analysis(bars, pd.DataFrame(), moves)
    corr = analysis["factor_correlations"]
    assert not corr.empty
    assert "corr_z_score_moves" in corr.columns
    assert len(analysis["move_features"]) == len(moves)


def test_simple_summary_flags_volume():
    bars = _sample_bars()
    bars = enrich_technical_factors(bars)
    move_idx = bars.index[50]
    bars.loc[move_idx, "vol_surge_20d"] = 5.0
    bars.loc[bars.index[80], "vol_surge_20d"] = 4.5
    bars.loc[bars.index.difference([move_idx, bars.index[80]]), "vol_surge_20d"] = 1.0

    moves = significant_price_moves(bars, window=40, z_threshold=2.0, min_abs_return=0.05)
    analysis = build_move_analysis(bars, pd.DataFrame(), moves)
    summary = analysis["simple_summary"]
    assert not summary.empty
    assert "pattern" in summary.columns
    vol_row = summary[summary["indicator"] == "Volume surge"]
    assert not vol_row.empty
    assert vol_row.iloc[0]["pattern"] == "Higher on big days"
