"""Integration tests for backtest engine and dataset loading."""

from __future__ import annotations

from datetime import date

import pandas as pd

from trading_bot.config import Config
from trading_bot.risk.engine import RiskEngine
from trading_bot.types import Horizon, Instrument, Signal, TradeStatus
from tests.dataset_fixtures import write_test_dataset


def test_backtest_engine_with_risk_engine():
    from trading_bot.backtest.costs import CostModel
    from trading_bot.backtest.engine import BacktestEngine
    from trading_bot.types import Position

    cfg = Config(None)
    risk = RiskEngine(cfg)
    engine = BacktestEngine(cfg, CostModel(cfg), risk)

    d0 = date(2024, 1, 2)
    d1 = date(2024, 1, 3)
    token = 42
    inst = Instrument("TEST", "INE000TEST", token)

    ohlcv = pd.DataFrame(
        [
            {"date": d0, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1e6},
            {"date": d1, "open": 100, "high": 101, "low": 94, "close": 95, "volume": 1e6},
        ]
    )

    sig = Signal(
        instrument=inst,
        horizon=Horizon.SWING,
        entry_price=100.0,
        stop_loss=95.0,
        target=110.0,
        win_prob=0.6,
        expected_value=0.5,
        rank_score=1.0,
        signal_date=d0,
    )

    closed, curve = engine.run({d0: [sig]}, {token: ohlcv}, initial_equity=1_000_000.0)
    assert len(closed) == 1
    assert closed[0].status == TradeStatus.CLOSED_SL
    assert not curve.empty


def test_load_ohlcv_reads_dataset(tmp_path, monkeypatch):
    from trading_bot.data import loader

    write_test_dataset(tmp_path, symbol="TESTCO", token=999001)
    monkeypatch.setattr(loader, "dataset_root_from_config", lambda cfg: tmp_path)
    monkeypatch.setattr("trading_bot.data.universe.dataset_root_from_config", lambda cfg: tmp_path)

    result = loader.load_ohlcv(date(2024, 6, 10), date(2024, 6, 20), Config(None))
    assert 999001 in result
    assert len(result[999001]) >= 5
