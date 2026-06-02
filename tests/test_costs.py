"""Unit tests for the Indian cost model."""

import pytest
from pathlib import Path

from trading_bot.config import Config
from trading_bot.types import Horizon

CONFIG_PATH = Path(__file__).parents[1] / "config" / "strategy.yaml"


def test_cost_model_positive():
    from trading_bot.backtest.costs import CostModel

    cfg = Config(CONFIG_PATH)
    cm = CostModel(cfg)
    cost = cm.compute(
        entry_price=100.0,
        exit_price=110.0,
        shares=100,
        horizon=Horizon.SWING,
    )
    assert cost > 0


def test_cost_model_pct_reasonable():
    from trading_bot.backtest.costs import CostModel

    cfg = Config(CONFIG_PATH)
    cm = CostModel(cfg)
    pct = cm.compute_pct(
        entry_price=100.0,
        exit_price=100.0,
        shares=100,
        horizon=Horizon.POSITIONAL,
    )
    # round-trip cost on a flat trade should be roughly 0.1% to 0.5%
    assert 0.05 < pct < 1.0
