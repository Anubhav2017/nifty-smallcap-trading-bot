"""Tests for move predictor fundamental screens."""

from __future__ import annotations

import pandas as pd

from trading_bot.strategies.move_predictor.fundamental_screen import (
    FundamentalScreenConfig,
    passes_fundamental_screen,
)


def _row(**kwargs) -> pd.Series:
    base = {
        "f_roce_lag1": 0.20,
        "f_debt_equity_lag1": 0.5,
        "f_profit_growth_yoy_lag1": 0.10,
        "f_profit_growth_qtr_lag1": 0.08,
        "f_pe_lag1": 15.0,
        "above_dma_lag1": 1.0,
        "above_trend_dma_lag1": 1.0,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_passes_all_screens():
    screen = FundamentalScreenConfig()
    assert passes_fundamental_screen(_row(), screen)


def test_rejects_low_roce():
    screen = FundamentalScreenConfig(min_roce=0.15)
    assert not passes_fundamental_screen(_row(f_roce_lag1=0.10), screen)


def test_rejects_high_debt():
    screen = FundamentalScreenConfig(max_debt_equity=1.0)
    assert not passes_fundamental_screen(_row(f_debt_equity_lag1=1.5), screen)


def test_rejects_negative_profit_growth():
    screen = FundamentalScreenConfig(min_profit_growth_yoy=0.0)
    assert not passes_fundamental_screen(_row(f_profit_growth_yoy_lag1=-0.05), screen)


def test_rejects_negative_quarterly_profit_growth():
    screen = FundamentalScreenConfig(min_profit_growth_qtr=0.0)
    assert not passes_fundamental_screen(_row(f_profit_growth_qtr_lag1=-0.05), screen)


def test_rejects_high_pe():
    screen = FundamentalScreenConfig(max_pe=20.0)
    assert not passes_fundamental_screen(_row(f_pe_lag1=25.0), screen)


def test_rejects_below_dma():
    screen = FundamentalScreenConfig(require_price_above_dma=True)
    assert not passes_fundamental_screen(_row(above_dma_lag1=0.0), screen)


def test_rejects_below_trend_dma():
    screen = FundamentalScreenConfig(require_price_above_trend_dma=True)
    assert not passes_fundamental_screen(_row(above_trend_dma_lag1=0.0), screen)


def test_skips_missing_when_configured():
    screen = FundamentalScreenConfig(skip_missing=True)
    assert not passes_fundamental_screen(_row(f_roce_lag1=float("nan")), screen)
