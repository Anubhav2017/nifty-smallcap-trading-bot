"""Shared dataclasses and enums used across all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional


class Horizon(str, Enum):
    SWING = "swing"
    POSITIONAL = "positional"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED_TP = "closed_tp"
    CLOSED_SL = "closed_sl"
    CLOSED_TIME = "closed_time"
    CLOSED_MANUAL = "closed_manual"


class RetrainTrigger(str, Enum):
    SCHEDULED = "scheduled"
    SORTINO_FLOOR = "sortino_floor"
    WIN_RATE_DROP = "win_rate_drop"
    DRAWDOWN_MULTIPLIER = "drawdown_multiplier"


@dataclass
class Instrument:
    symbol: str
    isin: str
    instrument_token: int
    exchange: str = "NSE"


@dataclass
class OHLCVBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    instrument_token: int
    bar_time: datetime | None = None


@dataclass
class Signal:
    instrument: Instrument
    horizon: Horizon
    entry_price: float
    stop_loss: float
    target: float
    win_prob: float
    expected_value: float
    rank_score: float
    signal_date: date
    features: dict = field(default_factory=dict)

    @property
    def risk_per_share(self) -> float:
        return self.entry_price - self.stop_loss

    @property
    def reward_per_share(self) -> float:
        return self.target - self.entry_price

    @property
    def reward_risk_ratio(self) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        return self.reward_per_share / self.risk_per_share


@dataclass
class Position:
    signal: Signal
    shares: int
    entry_date: date
    entry_price: float
    status: TradeStatus = TradeStatus.OPEN
    entry_datetime: datetime | None = None
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    cost: Optional[float] = None

    @property
    def r_multiple(self) -> Optional[float]:
        if self.net_pnl is None or self.signal.risk_per_share <= 0:
            return None
        risk = self.signal.risk_per_share * self.shares
        return self.net_pnl / risk if risk > 0 else None


@dataclass
class FoldMetrics:
    fold_id: int
    train_start: date
    train_end: date
    oos_start: date
    oos_end: date
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    expectancy_r: float
    total_trades: int
    swing_trades: int
    positional_trades: int
    avg_daily_entries: float
    turnover_cost_pct: float
    objective_j: float
    beats_baseline: bool
    swing_win_rate: float = 0.0
    positional_win_rate: float = 0.0


@dataclass
class DegradationState:
    triggered: bool = False
    trigger_reason: Optional[RetrainTrigger] = None
    trigger_date: Optional[date] = None
    last_retrain_date: Optional[date] = None
    sessions_since_retrain: int = 0
    paused: bool = False
