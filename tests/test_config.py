"""Smoke tests for config loading."""

from pathlib import Path

import pytest

from trading_bot.config import Config, load_config


CONFIG_PATH = Path(__file__).parents[1] / "config" / "strategy.yaml"


def test_load_config():
    cfg_dict = load_config(CONFIG_PATH)
    assert "universe" in cfg_dict
    assert "risk" in cfg_dict
    assert "costs" in cfg_dict


def test_config_class():
    cfg = Config(CONFIG_PATH)
    assert cfg.risk["max_daily_entries"] == 10
    assert cfg.costs["stt_delivery_pct"] == 0.1
    assert cfg.walk_forward["train_years"] == 0.25
    assert cfg.walk_forward["step_weeks"] == 1
    assert cfg.walk_forward["validate_weeks"] == 1
    assert cfg.objective["alpha"] == 2.0


def test_config_retrain():
    cfg = Config(CONFIG_PATH)
    assert cfg.retrain["schedule_weeks"] == 1
    assert cfg.retrain["degradation"]["win_rate_drop_pct"] == 15.0
    assert cfg.retrain["degradation"]["cooldown_sessions"] == 10
