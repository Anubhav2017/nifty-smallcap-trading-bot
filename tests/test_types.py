"""Smoke tests for shared types."""

from datetime import date

from trading_bot.types import (
    Horizon,
    Instrument,
    Signal,
    TradeStatus,
    DegradationState,
    RetrainTrigger,
)


def test_horizon_enum():
    assert Horizon.SWING.value == "swing"
    assert Horizon.POSITIONAL.value == "positional"


def test_signal_properties():
    inst = Instrument(symbol="ABCD", isin="INE000A01234", instrument_token=12345)
    sig = Signal(
        instrument=inst,
        horizon=Horizon.SWING,
        entry_price=100.0,
        stop_loss=95.0,
        target=110.0,
        win_prob=0.6,
        expected_value=0.5,
        rank_score=0.8,
        signal_date=date(2024, 1, 2),
    )
    assert sig.risk_per_share == 5.0
    assert sig.reward_per_share == 10.0
    assert sig.reward_risk_ratio == 2.0


def test_degradation_state_defaults():
    state = DegradationState()
    assert state.triggered is False
    assert state.paused is False
    assert state.trigger_reason is None
