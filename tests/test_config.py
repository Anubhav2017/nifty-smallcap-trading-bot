"""Smoke tests for config loading."""

from pathlib import Path

import pytest

from trading_bot.config import Config, load_config


CONFIG_PATH = Path(__file__).parents[1] / "config" / "move_predictor.yaml"


def test_load_config():
    cfg_dict = load_config(CONFIG_PATH)
    assert "universe" in cfg_dict
    assert "risk" in cfg_dict
    assert "costs" in cfg_dict
    assert "move_predictor" in cfg_dict


def test_config_class():
    cfg = Config(CONFIG_PATH)
    assert cfg.risk["max_daily_entries"] == 2
    assert cfg.costs["stt_delivery_pct"] == 0.1
    assert cfg.horizons["swing"]["max_hold_days"] == 10
    assert cfg.horizons["positional"]["max_hold_days"] == 60


def test_config_move_predictor_section():
    cfg = Config(CONFIG_PATH)
    mp = cfg._raw["move_predictor"]
    assert mp["horizon"] == "swing"
    assert mp["walk_forward_quarters"] is True
    assert cfg.exit["swing"]["reward_risk_ratio"] == 2.0
