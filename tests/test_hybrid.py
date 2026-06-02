"""Tests for hybrid intraday simulation and paper ledger."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from trading_bot.backtest.costs import CostModel
from trading_bot.backtest.intraday_sim import simulate_intraday_session
from trading_bot.config import Config
from trading_bot.data.bars import BarStore
from trading_bot.features.intraday_build import build_intraday_matrix
from trading_bot.features.intraday_features import add_intraday_bar_features
from trading_bot.features.intraday_labels import label_intraday_tp_before_sl
from trading_bot.risk.engine import RiskEngine
from trading_bot.types import Horizon, Instrument, OHLCVBar, Position, Signal, TradeStatus
from tests.dataset_fixtures import write_test_dataset


def test_intraday_label_and_features(tmp_path):
    session = date(2025, 12, 15)
    write_test_dataset(tmp_path, symbol="TESTCO", token=1, day=session)
    store = BarStore(dataset_root=tmp_path)
    bars = store.get_bars("TESTCO", session)

    daily_row = pd.Series(
        {
            "symbol": "TESTCO",
            "instrument_token": 1,
            "mom_20d": 0.02,
            "rs_20d": 0.01,
            "atr_pct_14": 0.02,
            "atr_14": 2.0,
        }
    )
    feats = add_intraday_bar_features(bars, daily_row)
    assert "minutes_from_open" in feats.columns
    labels = label_intraday_tp_before_sl(bars, atr=2.0, cfg=Config(None), horizon=Horizon.SWING)
    assert len(labels) >= 1


def test_build_intraday_matrix(tmp_path):
    session = date(2025, 12, 15)
    write_test_dataset(tmp_path, symbol="TESTCO", token=1, day=session)
    store = BarStore(dataset_root=tmp_path)

    daily = pd.DataFrame(
        [
            {
                "date": session,
                "symbol": "TESTCO",
                "instrument_token": 1,
                "mom_20d": 0.02,
                "rs_20d": 0.01,
                "atr_pct_14": 0.02,
                "atr_14": 2.0,
                "close": 100.0,
                "volume": 1_000_000.0,
            }
        ]
    )
    cfg = Config(None)
    matrix = build_intraday_matrix(cfg, daily, [session], bar_store=store)
    assert not matrix.empty
    assert "label_timing" in matrix.columns


def test_intraday_sl_exit_on_5m_bar():
    cfg = Config(None)
    risk = RiskEngine(cfg)
    cost = CostModel(cfg)
    session = date(2025, 12, 15)
    entry_dt = datetime(2025, 12, 15, 9, 35)

    inst = Instrument(symbol="TESTCO", isin="INE000", instrument_token=42, exchange="NSE")
    sig = Signal(
        instrument=inst,
        horizon=Horizon.SWING,
        entry_price=100.0,
        stop_loss=98.0,
        target=106.0,
        win_prob=0.6,
        expected_value=0.01,
        rank_score=0.5,
        signal_date=session,
    )
    sig.features["entry_datetime"] = entry_dt.isoformat()

    pos = Position(
        signal=sig,
        shares=10,
        entry_date=session,
        entry_price=100.0,
        entry_datetime=entry_dt,
        status=TradeStatus.OPEN,
    )

    bars = pd.DataFrame(
        {
            "datetime": [
                datetime(2025, 12, 15, 9, 30),
                datetime(2025, 12, 15, 9, 35),
                datetime(2025, 12, 15, 9, 40),
            ],
            "open": [100.0, 100.0, 99.0],
            "high": [101.0, 101.0, 100.0],
            "low": [99.5, 99.5, 97.5],
            "close": [100.5, 100.5, 98.0],
            "volume": [1000.0, 1000.0, 1000.0],
        }
    )

    still_open, closed, equity = simulate_intraday_session(
        session,
        [pos],
        [],
        [],
        {"TESTCO": bars},
        all_trading_dates=[session],
        equity=1_000_000.0,
        risk_engine=risk,
        cost_model=cost,
    )

    assert len(still_open) == 0
    assert len(closed) == 1
    assert closed[0].status == TradeStatus.CLOSED_SL
    assert closed[0].exit_price == 98.0
    assert equity < 1_000_000.0


def test_risk_engine_skips_exit_on_entry_bar():
    cfg = Config(None)
    risk = RiskEngine(cfg)
    session = date(2025, 12, 15)
    entry_dt = datetime(2025, 12, 15, 9, 35)

    inst = Instrument(symbol="X", isin="INE000", instrument_token=1, exchange="NSE")
    sig = Signal(
        instrument=inst,
        horizon=Horizon.SWING,
        entry_price=100.0,
        stop_loss=95.0,
        target=110.0,
        win_prob=0.6,
        expected_value=0.01,
        rank_score=0.5,
        signal_date=session,
    )
    pos = Position(
        signal=sig,
        shares=1,
        entry_date=session,
        entry_price=100.0,
        entry_datetime=entry_dt,
    )
    bar = OHLCVBar(
        date=session,
        open=100.0,
        high=100.0,
        low=94.0,
        close=96.0,
        volume=100.0,
        instrument_token=1,
        bar_time=entry_dt,
    )
    should_exit, status, _ = risk.check_exits(pos, bar, session_number=0)
    assert not should_exit
    assert status == TradeStatus.OPEN
