"""Daily-entry and gross-exposure caps for the risk engine."""

from __future__ import annotations

from trading_bot.config import Config
from trading_bot.types import Horizon, Position, Signal


class RiskCaps:
    def __init__(self, cfg: Config) -> None:
        risk = cfg.risk
        self._max_daily_entries: int = int(risk.get("max_daily_entries", 10))
        self._max_gross_exposure_pct: float = float(risk.get("max_gross_exposure_pct", 80.0))
        self._swing_max_positions: int = int(risk.get("swing_max_positions", 5))
        self._positional_max_positions: int = int(risk.get("positional_max_positions", 3))

    # ── Public API ─────────────────────────────────────────────────────────────

    def can_enter(
        self,
        signal: Signal,
        open_positions: list[Position],
        daily_entry_count: int,
        equity: float,
    ) -> tuple[bool, str]:
        """Return (True, "") when all risk gates pass, or (False, reason) on first failure.

        Pure predicate — does not mutate any state.
        """
        if daily_entry_count >= self._max_daily_entries:
            return False, "daily_cap_reached"

        by_horizon = self.count_open_by_horizon(open_positions)
        horizon_val = signal.horizon.value

        if signal.horizon == Horizon.SWING:
            if by_horizon.get(Horizon.SWING.value, 0) >= self._swing_max_positions:
                return False, "horizon_cap_swing"
        elif signal.horizon == Horizon.POSITIONAL:
            if by_horizon.get(Horizon.POSITIONAL.value, 0) >= self._positional_max_positions:
                return False, "horizon_cap_positional"

        # Exposure check: would adding this position breach the gross exposure limit?
        new_position_value = signal.entry_price * 1  # 1 share proxy; actual check uses pct
        # We compare against the hypothetical exposure after adding *any* position.
        # The caller must pass shares-adjusted value; here we compute exposure as a pct of
        # equity using a single share and let the engine re-check with actual shares.
        # For the cap gate we use the current exposure plus the minimum 1-share increment.
        current_pct = self.current_exposure_pct(open_positions, equity)
        incremental_pct = (signal.entry_price / equity) * 100.0
        if current_pct + incremental_pct > self._max_gross_exposure_pct:
            return False, "exposure_limit"

        return True, ""

    def count_open_by_horizon(self, open_positions: list[Position]) -> dict[str, int]:
        """Count open positions keyed by horizon value string."""
        counts: dict[str, int] = {}
        for pos in open_positions:
            key = pos.signal.horizon.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def current_exposure_pct(self, open_positions: list[Position], equity: float) -> float:
        """Gross exposure as a percentage of equity across all open positions."""
        if equity <= 0:
            return 0.0
        total_notional = sum(pos.shares * pos.entry_price for pos in open_positions)
        return (total_notional / equity) * 100.0
