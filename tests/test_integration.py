"""Integration tests for walk-forward pipeline wiring."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from trading_bot.config import Config
from trading_bot.models.ranker import StockRanker
from trading_bot.models.training import train
from trading_bot.risk.engine import RiskEngine
from trading_bot.risk.signals import generate_signals
from trading_bot.types import Horizon, Instrument, Signal, TradeStatus
from tests.dataset_fixtures import write_test_dataset


def _make_feature_row(d: date, token: int, close: float, fwd: float, label: float) -> dict:
    return {
        "date": d,
        "instrument_token": token,
        "symbol": f"STK{token}",
        "isin": f"INE{token:06d}",
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1_000_000.0,
        "mom_20d": 0.02,
        "mom_50d": 0.03,
        "rs_20d": 0.01,
        "vol_surge_20d": 1.2,
        "hl_position_260d": 0.6,
        "gap_risk": 0.005,
        "atr_pct_14": 0.02,
        "atr_14": close * 0.02,
        "fwd_ret_swing": fwd,
        "rank_label": int(fwd * 100),
        f"label_tp_{Horizon.SWING.value}": label,
        f"label_tp_{Horizon.POSITIONAL.value}": label,
    }


def _synthetic_features(n_days: int = 30, n_stocks: int = 5) -> pd.DataFrame:
    start = date(2024, 1, 2)
    rows: list[dict] = []
    for day in range(n_days):
        d = start + timedelta(days=day)
        if d.weekday() >= 5:
            continue
        for token in range(1, n_stocks + 1):
            close = 100.0 + token + day * 0.1
            fwd = 0.01 * token
            label = 1.0 if token % 2 == 0 else 0.0
            rows.append(_make_feature_row(d, token, close, fwd, label))
    df = pd.DataFrame(rows)
    df["rank_label"] = (
        df.groupby("date")["fwd_ret_swing"]
        .rank(pct=True, na_option="bottom")
        .mul(4)
        .fillna(0)
        .astype(int)
    )
    return df


def test_train_and_generate_signals():
    cfg = Config(None)
    feature_df = _synthetic_features()
    train_dates = sorted(feature_df["date"].unique())[:20]
    val_date = train_dates[-1]

    bundle = train(cfg, feature_df, train_dates)
    assert bundle.ranker._fitted
    assert bundle.classifiers[Horizon.SWING]._fitted

    signals = generate_signals(bundle, feature_df, val_date, RiskEngine(cfg))
    assert isinstance(signals, list)
    for sig in signals:
        assert isinstance(sig, Signal)
        assert sig.expected_value > 0


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
