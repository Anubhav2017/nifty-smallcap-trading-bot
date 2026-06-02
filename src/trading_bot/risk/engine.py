"""Combined risk engine: signal approval, position sizing, exit logic."""

from __future__ import annotations

from datetime import date

from trading_bot.config import Config
from trading_bot.types import Horizon, OHLCVBar, Position, Signal, TradeStatus

from .caps import RiskCaps
from .sizer import PositionSizer, compute_position_value


class RiskEngine:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self.sizer = PositionSizer(cfg)
        self.caps = RiskCaps(cfg)

    # ── Signal approval ────────────────────────────────────────────────────────

    def evaluate_signal(
        self,
        signal: Signal,
        open_positions: list[Position],
        daily_entry_count: int,
        equity: float,
    ) -> tuple[bool, int, str]:
        """Gate a signal through all risk checks.

        Returns (approved, shares, rejection_reason).
        rejection_reason is "" when approved is True.
        """
        if signal.expected_value <= 0:
            return False, 0, "negative_ev"

        approved, reason = self.caps.can_enter(signal, open_positions, daily_entry_count, equity)
        if not approved:
            return False, 0, reason

        shares = self.sizer.size(signal, equity)
        if shares <= 0:
            return False, 0, "zero_shares"

        # Re-check exposure with actual share count
        current_pct = self.caps.current_exposure_pct(open_positions, equity)
        new_notional = compute_position_value(shares, signal.entry_price)
        incremental_pct = (new_notional / equity) * 100.0
        if current_pct + incremental_pct > self.caps._max_gross_exposure_pct:
            return False, 0, "exposure_limit"

        return True, shares, ""

    # ── Exit logic ─────────────────────────────────────────────────────────────

    def check_exits(
        self,
        position: Position,
        current_bar: OHLCVBar,
        session_number: int,
    ) -> tuple[bool, TradeStatus, float]:
        """Evaluate whether *position* should be exited on *current_bar*.

        Returns (should_exit, exit_status, exit_price).

        Priority order when multiple triggers fire on the same bar:
          SL > TP > Time stop

        Gap-through handling: when open < stop_loss the position gapped down through
        the SL level; exit_price is the bar open (realised fill), not the SL level.
        """
        sl = position.signal.stop_loss
        tp = position.signal.target
        horizon = position.signal.horizon

        horizon_cfg = self._cfg.horizons[horizon.value]
        max_hold: int = int(horizon_cfg["max_hold_days"])

        sl_hit = current_bar.low <= sl
        tp_hit = current_bar.high >= tp
        time_hit = session_number >= max_hold

        if current_bar.bar_time and position.entry_datetime:
            if current_bar.bar_time <= position.entry_datetime:
                return False, TradeStatus.OPEN, 0.0

        # SL priority (gap scenario: if open is already below SL, use open as fill)
        if sl_hit:
            exit_price = min(current_bar.open, sl) if current_bar.open < sl else sl
            return True, TradeStatus.CLOSED_SL, exit_price

        if tp_hit:
            return True, TradeStatus.CLOSED_TP, tp

        if time_hit:
            return True, TradeStatus.CLOSED_TIME, current_bar.close

        return False, TradeStatus.OPEN, 0.0

    # ── Session counting ───────────────────────────────────────────────────────

    def sessions_held(
        self,
        position: Position,
        current_date: date,
        trading_dates: list[date],
    ) -> int:
        """Number of trading sessions from entry_date (exclusive) to current_date (inclusive)."""
        entry = position.entry_date
        return sum(1 for d in trading_dates if entry < d <= current_date)
