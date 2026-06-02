"""Walk-forward splitter window resolution."""

from datetime import date
from pathlib import Path

import pandas as pd

from trading_bot.config import Config
from trading_bot.learning.walk_forward import WalkForwardSplitter

CONFIG_PATH = Path(__file__).parents[1] / "config" / "strategy.yaml"


def test_weekly_windows_from_config():
    cfg = Config(CONFIG_PATH)
    splitter = WalkForwardSplitter(cfg)
    assert splitter._train_days == 63
    assert splitter._validate_days == 5
    assert splitter._step_days == 5


def test_weekly_folds_are_non_overlapping():
    cfg = Config(CONFIG_PATH)
    splitter = WalkForwardSplitter(cfg)
    dates = [ts.date() for ts in pd.bdate_range("2024-11-01", "2026-05-31")]
    folds = splitter.generate_folds(dates)
    assert len(folds) > 1
    assert folds[0][1][-1] < folds[1][1][0]
