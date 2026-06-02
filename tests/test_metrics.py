"""Unit tests for performance metrics."""

import numpy as np
import pandas as pd
import pytest


def test_sortino_positive_returns():
    from trading_bot.backtest.metrics import sortino_ratio

    returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01])
    s = sortino_ratio(returns)
    assert s > 0


def test_sortino_all_positive():
    from trading_bot.backtest.metrics import sortino_ratio

    returns = pd.Series([0.01] * 100)
    s = sortino_ratio(returns)
    assert s >= 0


def test_max_drawdown():
    from trading_bot.backtest.metrics import max_drawdown

    equity = pd.Series([100, 110, 105, 95, 100, 115])
    dd = max_drawdown(equity)
    assert 0.0 < dd < 1.0
    # peak=110, trough=95, DD = 15/110 ≈ 0.136
    assert abs(dd - 15 / 110) < 0.001


def test_max_drawdown_monotone():
    from trading_bot.backtest.metrics import max_drawdown

    equity = pd.Series([100, 110, 120, 130])
    assert max_drawdown(equity) == 0.0


def test_win_rate():
    from trading_bot.backtest.metrics import win_rate
    from trading_bot.types import Position, Signal, Instrument, Horizon, TradeStatus
    from datetime import date

    def make_pos(pnl):
        inst = Instrument("X", "INE0", 1)
        sig = Signal(inst, Horizon.SWING, 100, 95, 110, 0.6, 0.5, 0.8, date(2024, 1, 2))
        p = Position(sig, 10, date(2024, 1, 2), 100.0, TradeStatus.CLOSED_TP)
        p.net_pnl = pnl
        return p

    positions = [make_pos(50), make_pos(-20), make_pos(30), make_pos(-10)]
    assert win_rate(positions) == 0.5
