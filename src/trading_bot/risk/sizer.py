"""Position sizing based on fixed-fractional risk per trade."""

from __future__ import annotations

import math

from trading_bot.config import Config
from trading_bot.types import Signal


def compute_shares(
    equity: float,
    risk_pct: float,
    entry: float,
    stop_loss: float,
) -> int:
    """Return share count sized so that a full stop-out risks exactly risk_pct of equity.

    Returns 0 when entry <= stop_loss or the computed count is less than 1.
    """
    if entry <= stop_loss:
        return 0
    risk_amount = equity * risk_pct / 100.0
    shares = math.floor(risk_amount / (entry - stop_loss))
    return shares if shares >= 1 else 0


def compute_position_value(shares: int, entry: float) -> float:
    """Gross notional value of a position."""
    return shares * entry


class PositionSizer:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def size(self, signal: Signal, equity: float) -> int:
        """Compute share count for *signal* given current *equity*."""
        risk_pct: float = self._cfg.risk["risk_per_trade_pct"]
        return compute_shares(equity, risk_pct, signal.entry_price, signal.stop_loss)
