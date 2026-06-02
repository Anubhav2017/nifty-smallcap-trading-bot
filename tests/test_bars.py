"""Tests for dataset-backed bar store."""

from __future__ import annotations

from datetime import date

from trading_bot.data.bars import BarStore
from tests.dataset_fixtures import write_test_dataset


def test_get_and_list_by_date(tmp_path):
    write_test_dataset(tmp_path, symbol="TESTCO", token=123)
    store = BarStore(dataset_root=tmp_path)

    day = date(2025, 12, 15)
    loaded = store.get_bars("TESTCO", day)
    assert len(loaded) >= 1
    assert store.list_symbols(day) == ["TESTCO"]
    assert day in store.list_dates("TESTCO")


def test_get_bars_resolved_weekend(tmp_path):
    write_test_dataset(tmp_path, symbol="TESTCO", token=123, day=date(2025, 12, 15))
    store = BarStore(dataset_root=tmp_path)

    session = date(2025, 12, 15)
    weekend = date(2025, 12, 13)
    loaded, resolved, note = store.get_bars_resolved("TESTCO", weekend)
    assert resolved == session
    assert note is not None
    assert len(loaded) >= 1
