"""Tests for trading calendar helpers."""

from __future__ import annotations

from datetime import date

import pytest

from trading_bot.data.trading_calendar import resolve_period, resolve_trading_day


def test_resolve_weekend_to_monday():
    saturday = date(2025, 12, 6)
    resolved, note = resolve_trading_day(saturday)
    assert resolved == date(2025, 12, 8)
    assert note is not None
    assert "not a trading day" in note


def test_resolve_trading_day_unchanged_on_weekday():
    monday = date(2025, 12, 8)
    resolved, note = resolve_trading_day(monday)
    assert resolved == monday
    assert note is None


def test_resolve_with_known_sessions():
    known = {date(2025, 12, 8), date(2025, 12, 9), date(2025, 12, 10)}
    holiday = date(2025, 12, 9)
    resolved, note = resolve_trading_day(holiday, known)
    assert resolved == holiday
    assert note is None

    missing = date(2025, 12, 7)  # Sunday
    resolved, note = resolve_trading_day(missing, known)
    assert resolved == date(2025, 12, 8)
    assert note is not None


def test_resolve_period_adjusts_boundaries():
    start, end, notes = resolve_period(date(2025, 12, 6), date(2025, 12, 7))
    assert start == date(2025, 12, 8)
    assert end == date(2025, 12, 8)
    assert len(notes) == 2


def test_resolve_period_invalid_after_adjustment():
    known = {date(2025, 12, 8), date(2025, 12, 9)}
    with pytest.raises(ValueError, match="Invalid period"):
        resolve_period(date(2025, 12, 9), date(2025, 12, 8), known)
