"""Event-driven daily backtest loop for Indian equity delivery trades."""

from __future__ import annotations

import logging
from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

from trading_bot.backtest.costs import CostModel
from trading_bot.config import Config
from trading_bot.types import OHLCVBar, Position, Signal, TradeStatus

logger = logging.getLogger(__name__)


@runtime_checkable
class RiskEngineProtocol(Protocol):
    """Minimal interface the BacktestEngine expects from the risk layer."""

    def check_exits(
        self,
        position: Position,
        current_bar: OHLCVBar,
        session_number: int,
    ) -> tuple[bool, TradeStatus, float]:
        ...

    def evaluate_signal(
        self,
        signal: Signal,
        open_positions: list[Position],
        daily_entry_count: int,
        equity: float,
    ) -> tuple[bool, int, str]:
        ...

    def sessions_held(
        self,
        position: Position,
        current_date: date,
        trading_dates: list[date],
    ) -> int:
        ...


class BacktestEngine:
    """Event-driven daily backtest engine."""

    def __init__(
        self,
        cfg: Config,
        cost_model: CostModel,
        risk_engine: RiskEngineProtocol,
    ) -> None:
        self._cfg = cfg
        self._cost_model = cost_model
        self._risk_engine = risk_engine

    def run(
        self,
        signals_by_date: dict[date, list[Signal]],
        ohlcv_by_token: dict[int, pd.DataFrame],
        initial_equity: float = 1_000_000.0,
    ) -> tuple[list[Position], pd.Series]:
        all_dates = self._build_trading_calendar(signals_by_date, ohlcv_by_token)
        if not all_dates:
            empty_curve = pd.Series(dtype=float, name="equity")
            return [], empty_curve

        open_positions: list[Position] = []
        closed_positions: list[Position] = []
        equity = initial_equity
        equity_history: dict[date, float] = {}

        for current_date in all_dates:
            still_open: list[Position] = []
            for pos in open_positions:
                token = pos.signal.instrument.instrument_token
                bar_row = self._get_bar(ohlcv_by_token.get(token), current_date)

                exited = False
                if bar_row is not None:
                    try:
                        ohlcv_bar = self._to_ohlcv_bar(bar_row, token, current_date)
                        sessions = self._risk_engine.sessions_held(
                            pos, current_date, all_dates
                        )
                        should_exit, exit_status, exit_price = self._risk_engine.check_exits(
                            pos, ohlcv_bar, sessions
                        )
                    except Exception as exc:
                        logger.warning(
                            "check_exits raised for token %d on %s: %s",
                            token,
                            current_date,
                            exc,
                        )
                        should_exit = False

                    if should_exit:
                        cost = self._cost_model.compute(
                            pos.entry_price,
                            exit_price,
                            pos.shares,
                            pos.signal.horizon,
                        )
                        net_pnl = self._compute_net_pnl(pos, exit_price, cost)

                        pos.exit_date = current_date
                        pos.exit_price = exit_price
                        pos.gross_pnl = (exit_price - pos.entry_price) * pos.shares
                        pos.net_pnl = net_pnl
                        pos.cost = cost
                        pos.status = exit_status

                        equity += net_pnl
                        closed_positions.append(pos)
                        exited = True

                if not exited:
                    still_open.append(pos)

            open_positions = still_open

            daily_signals = signals_by_date.get(current_date, [])
            daily_signals = sorted(daily_signals, key=lambda s: s.rank_score, reverse=True)
            daily_entry_count = 0

            for signal in daily_signals:
                try:
                    can_enter, shares, _reason = self._risk_engine.evaluate_signal(
                        signal, open_positions, daily_entry_count, equity
                    )
                except Exception as exc:
                    logger.warning(
                        "evaluate_signal raised for %s on %s: %s",
                        signal.instrument.symbol,
                        current_date,
                        exc,
                    )
                    can_enter, shares = False, 0

                if can_enter and shares > 0:
                    open_positions.append(
                        Position(
                            signal=signal,
                            shares=shares,
                            entry_date=current_date,
                            entry_price=signal.entry_price,
                            status=TradeStatus.OPEN,
                        )
                    )
                    daily_entry_count += 1

            equity_history[current_date] = equity

        dt_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        values = [equity_history[d] for d in all_dates]
        equity_curve = pd.Series(values, index=dt_index, name="equity", dtype=float)
        return closed_positions, equity_curve

    @staticmethod
    def _compute_net_pnl(position: Position, exit_price: float, cost: float) -> float:
        return (exit_price - position.entry_price) * position.shares - cost

    @staticmethod
    def _to_ohlcv_bar(bar: pd.Series, token: int, current_date: date) -> OHLCVBar:
        return OHLCVBar(
            date=current_date,
            open=float(bar["open"]),
            high=float(bar["high"]),
            low=float(bar["low"]),
            close=float(bar["close"]),
            volume=float(bar.get("volume", 0.0)),
            instrument_token=token,
        )

    @staticmethod
    def _build_trading_calendar(
        signals_by_date: dict[date, list[Signal]],
        ohlcv_by_token: dict[int, pd.DataFrame],
    ) -> list[date]:
        all_dates_set: set[date] = set(signals_by_date.keys())

        for df in ohlcv_by_token.values():
            if df is None or df.empty:
                continue
            if "date" in df.columns:
                all_dates_set.update(df["date"].tolist())
            else:
                for idx_val in df.index:
                    if isinstance(idx_val, pd.Timestamp):
                        all_dates_set.add(idx_val.date())
                    elif isinstance(idx_val, date):
                        all_dates_set.add(idx_val)
            break

        return sorted(all_dates_set)

    @staticmethod
    def _get_bar(
        df: pd.DataFrame | None,
        d: date,
    ) -> pd.Series | None:
        if df is None or df.empty:
            return None

        if "date" in df.columns:
            rows = df[df["date"] == d]
            if rows.empty:
                return None
            return rows.iloc[-1]

        try:
            key: date | pd.Timestamp = (
                pd.Timestamp(d) if isinstance(df.index, pd.DatetimeIndex) else d
            )
            return df.loc[key]
        except KeyError:
            return None
