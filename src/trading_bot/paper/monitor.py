"""Degradation threshold monitor. Checks daily after market close."""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.config import Config
from trading_bot.types import DegradationState, RetrainTrigger

logger = logging.getLogger(__name__)

_RETRAIN_LOG_COLS = [
    "timestamp",
    "trigger",
    "old_j",
    "new_j",
    "accepted",
    "status",
]

_FOLD_SUMMARY_COLS = [
    "fold_id",
    "win_rate",
    "max_drawdown",
    "sortino",
]


class DegradationMonitor:
    """Monitors live paper-trading performance for degradation vs OOS backtest.

    Checks three thresholds and writes events to a retrain_log.csv when
    any threshold is breached and sufficient cooldown has elapsed.
    """

    def __init__(self, cfg: Config, ledger_path: Path) -> None:
        self.cfg = cfg
        self.ledger_path = Path(ledger_path)
        self._report_dir = Path("hermes/reports")
        self._retrain_log = self._report_dir / "retrain_log.csv"
        self._fold_summary = self._report_dir / "fold_summary.csv"

    # ── Public API ─────────────────────────────────────────────────────────

    def is_paused(self) -> bool:
        """Return True if trading is currently paused pending a retrain."""
        if not self._retrain_log.exists():
            return False
        try:
            df = pd.read_csv(self._retrain_log)
            if df.empty:
                return False
            last_status = str(df.iloc[-1].get("status", ""))
            return last_status.strip().lower() == "paused"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read retrain_log: %s", exc)
            return False

    def check_and_flag(self) -> DegradationState:
        """Evaluate degradation triggers against the current paper ledger.

        Trigger checks (applied when enough data is present):
        1. Rolling 14-session Sortino < sortino_floor.
        2. Win rate over last 20 trades drops > 15 pp below backtest baseline.
        3. Peak-to-trough drawdown > 2× OOS MaxDD from last fold.

        If any trigger fires **and** cooldown has elapsed since the last
        retrain, a paused event is written to retrain_log.csv.
        """
        ledger = self._load_ledger()
        degradation_cfg = self.cfg.retrain.get("degradation", {})
        sortino_floor: float = degradation_cfg.get("sortino_floor", 0.0)
        win_rate_drop_pct: float = degradation_cfg.get("win_rate_drop_pct", 15.0)
        dd_multiplier: float = degradation_cfg.get("drawdown_multiplier", 2.0)
        rolling_window: int = int(degradation_cfg.get("rolling_window_sessions", 14))
        min_trades_winrate: int = int(degradation_cfg.get("min_trades_for_winrate_check", 20))
        cooldown: int = int(degradation_cfg.get("cooldown_sessions", 10))

        last_retrain, sessions_since = self._sessions_since_last_retrain(ledger, cooldown)
        state = DegradationState(
            last_retrain_date=last_retrain,
            sessions_since_retrain=sessions_since,
        )

        schedule_sessions = self._scheduled_retrain_sessions()
        schedule_elapsed = self._sessions_for_schedule(ledger, last_retrain)
        if schedule_elapsed >= schedule_sessions:
            logger.info(
                "Scheduled retrain due: %d / %d sessions since last retrain.",
                schedule_elapsed,
                schedule_sessions,
            )
            state.triggered = True
            state.trigger_reason = RetrainTrigger.SCHEDULED
            state.trigger_date = date.today()
            state.paused = True
            self._write_trigger_event(RetrainTrigger.SCHEDULED)
            return state

        if sessions_since < cooldown:
            logger.debug(
                "Cooldown active: %d / %d sessions since last retrain.",
                sessions_since,
                cooldown,
            )
            return state

        fold_win_rate, fold_max_dd = self._latest_fold_expectations()
        triggered: RetrainTrigger | None = None

        # ── Trigger 1: Sortino ────────────────────────────────────────────
        daily_pnl = self._compute_daily_pnl(ledger)
        if len(daily_pnl) >= rolling_window:
            recent_pnl = daily_pnl.iloc[-rolling_window:]
            sortino = self._sortino(recent_pnl)
            if sortino < sortino_floor:
                logger.warning(
                    "Sortino trigger fired: rolling-%d Sortino=%.3f < floor=%.3f",
                    rolling_window,
                    sortino,
                    sortino_floor,
                )
                triggered = RetrainTrigger.SORTINO_FLOOR
        else:
            logger.debug(
                "Skipping Sortino check: only %d sessions available (need %d).",
                len(daily_pnl),
                rolling_window,
            )

        # ── Trigger 2: Win-rate drop ──────────────────────────────────────
        if triggered is None:
            closed = ledger[ledger["action"] == "exit"] if not ledger.empty else pd.DataFrame()
            if len(closed) >= min_trades_winrate and fold_win_rate is not None:
                recent_wins = closed.iloc[-min_trades_winrate:]["pnl"].gt(0).mean() * 100.0
                if (fold_win_rate - recent_wins) > win_rate_drop_pct:
                    logger.warning(
                        "Win-rate trigger fired: live=%.1f%% backtest=%.1f%% drop=%.1f%%",
                        recent_wins,
                        fold_win_rate,
                        fold_win_rate - recent_wins,
                    )
                    triggered = RetrainTrigger.WIN_RATE_DROP
            elif len(closed) < min_trades_winrate:
                logger.debug(
                    "Skipping win-rate check: only %d closed trades (need %d).",
                    len(closed),
                    min_trades_winrate,
                )

        # ── Trigger 3: Drawdown ───────────────────────────────────────────
        if triggered is None and fold_max_dd is not None:
            equity_series = ledger["equity"].dropna() if not ledger.empty else pd.Series(dtype=float)
            if not equity_series.empty:
                live_dd = self._peak_to_trough_dd(equity_series)
                threshold = dd_multiplier * abs(fold_max_dd)
                if live_dd > threshold:
                    logger.warning(
                        "Drawdown trigger fired: live_dd=%.4f > %.1f× OOS_MaxDD=%.4f",
                        live_dd,
                        dd_multiplier,
                        abs(fold_max_dd),
                    )
                    triggered = RetrainTrigger.DRAWDOWN_MULTIPLIER

        if triggered is not None:
            state.triggered = True
            state.trigger_reason = triggered
            state.trigger_date = date.today()
            state.paused = True
            self._write_trigger_event(triggered)

        return state

    def record_retrain(
        self,
        trigger: RetrainTrigger,
        old_j: float,
        new_j: float,
        accepted: bool,
    ) -> None:
        """Append a retrain outcome row to retrain_log.csv.

        If *accepted*, the status is set to "retrained" (trading resumes).
        If not accepted, status is "rejected" (remains paused until reviewed).
        """
        status = "retrained" if accepted else "rejected"
        row = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat(),
            "trigger": trigger.value,
            "old_j": old_j,
            "new_j": new_j,
            "accepted": accepted,
            "status": status,
        }])
        self._append_retrain_log(row)
        logger.info(
            "Retrain recorded: trigger=%s old_j=%.4f new_j=%.4f accepted=%s",
            trigger.value,
            old_j,
            new_j,
            accepted,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_ledger(self) -> pd.DataFrame:
        if not self.ledger_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.ledger_path, parse_dates=["date"])
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load ledger from %s: %s", self.ledger_path, exc)
            return pd.DataFrame()

    def _compute_daily_pnl(self, ledger: pd.DataFrame) -> pd.Series:
        """Aggregate net PnL per calendar session from the ledger."""
        if ledger.empty or "pnl" not in ledger.columns:
            return pd.Series(dtype=float)
        return (
            ledger.groupby("date")["pnl"]
            .sum()
            .sort_index()
        )

    @staticmethod
    def _sortino(pnl: pd.Series) -> float:
        """Annualised Sortino ratio from a session-level PnL series."""
        mean = pnl.mean()
        downside = pnl[pnl < 0]
        if len(downside) == 0:
            return np.inf if mean > 0 else 0.0
        downside_std = float(np.sqrt((downside ** 2).mean()))
        if downside_std == 0:
            return np.inf if mean > 0 else 0.0
        return float(mean / downside_std * np.sqrt(252))

    @staticmethod
    def _peak_to_trough_dd(equity: pd.Series) -> float:
        """Maximum peak-to-trough drawdown as a positive fraction."""
        roll_max = equity.cummax()
        drawdowns = (equity - roll_max) / roll_max.replace(0, np.nan)
        return float(drawdowns.min() * -1)

    def _latest_fold_expectations(self) -> tuple[float | None, float | None]:
        """Return (win_rate_pct, max_drawdown) from the most recent fold summary."""
        if not self._fold_summary.exists():
            return None, None
        try:
            df = pd.read_csv(self._fold_summary)
            if df.empty:
                return None, None
            last = df.iloc[-1]
            win_rate = float(last["win_rate"]) * 100.0 if "win_rate" in last else None
            max_dd = float(last["max_drawdown"]) if "max_drawdown" in last else None
            return win_rate, max_dd
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read fold_summary: %s", exc)
            return None, None

    def _scheduled_retrain_sessions(self) -> int:
        """Return trading sessions between scheduled retrains."""
        retrain_cfg = self.cfg.retrain
        if retrain_cfg.get("schedule_weeks") is not None:
            return round(float(retrain_cfg["schedule_weeks"]) * 5)
        return round(float(retrain_cfg.get("schedule_months", 1)) * 21)

    @staticmethod
    def _sessions_for_schedule(ledger: pd.DataFrame, last_retrain: date | None) -> int:
        """Count trading sessions since the last accepted retrain."""
        if ledger.empty or "date" not in ledger.columns:
            return 0
        session_dates = sorted(ledger["date"].unique())
        if last_retrain is None:
            return len(session_dates)
        return sum(1 for d in session_dates if d > last_retrain)

    def _sessions_since_last_retrain(
        self, ledger: pd.DataFrame, cooldown: int
    ) -> tuple[date | None, int]:
        """Return (last_retrain_date, sessions_elapsed) based on retrain_log."""
        if not self._retrain_log.exists():
            return None, cooldown  # No log → treat cooldown as elapsed

        try:
            log = pd.read_csv(self._retrain_log, parse_dates=["timestamp"])
            retrained = log[log["status"].isin(["retrained", "rejected"])]
            if retrained.empty:
                return None, cooldown

            last_ts = pd.to_datetime(retrained.iloc[-1]["timestamp"])
            last_date = last_ts.date()

            if ledger.empty or "date" not in ledger.columns:
                return last_date, cooldown

            session_dates = sorted(ledger["date"].unique())
            sessions_after = sum(1 for d in session_dates if d > last_date)
            return last_date, sessions_after
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse retrain_log for cooldown check: %s", exc)
            return None, cooldown

    def _write_trigger_event(self, trigger: RetrainTrigger) -> None:
        """Append a paused event to retrain_log.csv."""
        row = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat(),
            "trigger": trigger.value,
            "old_j": None,
            "new_j": None,
            "accepted": None,
            "status": "paused",
        }])
        self._append_retrain_log(row)
        logger.warning("Degradation trigger '%s' fired — trading paused.", trigger.value)

    def _append_retrain_log(self, row: pd.DataFrame) -> None:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        write_header = not self._retrain_log.exists()
        row.to_csv(
            self._retrain_log,
            mode="a",
            header=write_header,
            index=False,
        )
