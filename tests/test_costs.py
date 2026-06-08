
"""Model package: ATR-based exit policy.









Provides :class:`ExitPolicy`, which converts a raw entry intent (instrument +
ATR + win probability) into a fully-specified :class:`Signal` with an ATR-based
stop-loss, an R-multiple target, and an expected-value estimate used for
optional EV gating.
"""

from __future__ import annotations

from trading_bot.models.exit_policy import ExitPolicy

__all__ = ["ExitPolicy"]
