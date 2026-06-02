"""Tests for period-based train/evaluate."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from trading_bot.config import Config
from trading_bot.learning.period_runner import PeriodRunner
from trading_bot.models.training import load
from tests.dataset_fixtures import write_test_dataset


def test_train_and_evaluate_synthetic(tmp_path, monkeypatch):
    token = 9001
    write_test_dataset(tmp_path, symbol="TESTCO", token=token)

    monkeypatch.setattr(
        "trading_bot.data.loader.dataset_root_from_config",
        lambda cfg: tmp_path,
    )
    monkeypatch.setattr(
        "trading_bot.data.universe.dataset_root_from_config",
        lambda cfg: tmp_path,
    )
    monkeypatch.setattr(
        "trading_bot.data.loader.load_index_ohlcv",
        lambda start, end, cfg=None: pd.DataFrame({"date": [start], "close": [100.0]}),
    )

    def _fake_build(cfg, ohlcv_by_token, dates, include_labels=True):
        rows = []
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "instrument_token": token,
                    "symbol": "TESTCO",
                    "isin": "TESTCO",
                    "close": 100.0,
                    "volume": 2_000_000.0,
                    "mom_20d": 0.02,
                    "mom_50d": 0.03,
                    "rs_20d": 0.01,
                    "vol_surge_20d": 1.2,
                    "hl_position_260d": 0.6,
                    "gap_risk": 0.005,
                    "atr_pct_14": 0.02,
                    "atr_14": 2.0,
                    "fwd_ret_swing": 0.01,
                    "rank_label": 2,
                    "label_tp_swing": 1.0,
                    "label_tp_positional": 0.0,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr("trading_bot.learning.period_runner.build", _fake_build)

    cfg = Config(None)
    runner = PeriodRunner(cfg)
    model_dir = tmp_path / "models" / "run_a"
    out = runner.train("2024-08-01", "2024-09-30", model_dir)
    assert (out / "run_manifest.json").exists()
    assert (out / "ranker.lgb").exists()

    bundle = load(cfg, out)
    assert bundle.ranker._fitted

    report_dir = tmp_path / "reports"
    metrics = runner.evaluate(out, "2024-10-01", "2024-10-31", report_dir)
    assert (report_dir / "evaluation_metrics.csv").exists()
    assert metrics.oos_start == date(2024, 10, 1)


def test_resolve_period_weekend_start():
    runner = PeriodRunner(Config(None))
    start, end, notes = runner._resolve_period("2024-08-03", "2024-08-03")
    assert start == date(2024, 8, 5)
    assert end == date(2024, 8, 5)
    assert len(notes) == 2


def test_discover_saved_runs(tmp_path):
    named = tmp_path / "dec2025"
    named.mkdir()
    (named / "ranker.lgb").write_text("stub")
    (named / "classifier_swing.lgb").write_text("stub")
    (named / "run_manifest.json").write_text(
        json.dumps(
            {
                "train_start": "2025-12-01",
                "train_end": "2025-12-31",
                "feature_rows": 100,
                "symbols": 42,
            }
        )
    )

    runs = PeriodRunner.discover_saved_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["name"] == "dec2025"
    assert runs[0]["symbols"] == 42
