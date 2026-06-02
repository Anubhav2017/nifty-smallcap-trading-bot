"""Unit tests for position sizer."""

import pytest


def test_compute_shares_basic():
    from trading_bot.risk.sizer import compute_shares

    equity = 1_000_000.0
    risk_pct = 0.75
    entry = 100.0
    sl = 95.0
    shares = compute_shares(equity, risk_pct, entry, sl)
    expected = int((1_000_000 * 0.0075) / 5.0)
    assert shares == expected


def test_compute_shares_zero_risk():
    from trading_bot.risk.sizer import compute_shares

    assert compute_shares(1_000_000.0, 0.75, 100.0, 100.0) == 0


def test_compute_shares_sl_above_entry():
    from trading_bot.risk.sizer import compute_shares

    assert compute_shares(1_000_000.0, 0.75, 100.0, 105.0) == 0
