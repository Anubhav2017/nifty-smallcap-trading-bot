"""Shared intraday session simulation (5m exits and timed entries)."""

from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd

from trading_bot.backtest.costs import CostModel
from trading_bot.config import Config
from trading_bot.data.bars import BarStore
from trading_bot.risk.engine import RiskEngine
from trading_bot.types import OHLCVBar, Position, Signal, TradeStatus

logger = logging.getLogger(__name__)


def parse_entry_datetime(signal: Signal) -> datetime | None:
    raw = signal.features.get("entry_datetime")
    if not raw:
        return None
    return pd.to_datetime(raw).to_pydatetime()


def load_session_bars(
    store: BarStore,
    symbols: set[str],
    session: date,
) -> dict[str, pd.DataFrame]:
    """Load 5m bars for *symbols* on *session*."""
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        bars = store.get_bars(symbol.upper(), session)
        if bars.empty:
            continue
        bars = bars.copy()
        bars["datetime"] = pd.to_datetime(bars["datetime"])
        out[symbol.upper()] = bars.sort_values("datetime").reset_index(drop=True)
    return out


def session_timeline(bars_by_symbol: dict[str, pd.DataFrame]) -> list[datetime]:
    times: set[datetime] = set()
    for bars in bars_by_symbol.values():
        times.update(pd.to_datetime(bars["datetime"]).tolist())
    return sorted(times)


def row_to_ohlcv(row: pd.Series, instrument_token: int, session: date) -> OHLCVBar:
    dt = pd.to_datetime(row["datetime"]).to_pydatetime()
    return OHLCVBar(
        date=session,
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row.get("volume", 0.0)),
        instrument_token=instrument_token,
        bar_time=dt,
    )


def _close_position(
    pos: Position,
    exit_price: float,
    exit_status: TradeStatus,
    session: date,
    cost_model: CostModel,
) -> tuple[Position, float]:
    cost = cost_model.compute(
        pos.entry_price,
        exit_price,
        pos.shares,
        pos.signal.horizon,
    )
    net_pnl = (exit_price - pos.entry_price) * pos.shares - cost
    pos.exit_date = session
    pos.exit_price = exit_price
    pos.gross_pnl = (exit_price - pos.entry_price) * pos.shares
    pos.net_pnl = net_pnl
    pos.cost = cost
    pos.status = exit_status
    return pos, net_pnl


def simulate_intraday_session(
    session: date,
    open_positions: list[Position],
    closed_positions: list[Position],
    pending_signals: list[Signal],
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    all_trading_dates: list[date],
    equity: float,
    risk_engine: RiskEngine,
    cost_model: CostModel,
    daily_ohlcv_by_token: dict[int, pd.DataFrame] | None = None,
) -> tuple[list[Position], list[Position], float]:
    """Walk 5m bars: intraday SL/TP/time exits, then timed entries."""
    timeline = session_timeline(bars_by_symbol)
    still_open = list(open_positions)
    entered_keys: set[tuple[str, str]] = set()
    daily_entry_count = 0

    if not timeline:
        still_open, closed_positions, equity = _daily_exit_fallback(
            session,
            still_open,
            closed_positions,
            equity,
            all_trading_dates,
            risk_engine,
            cost_model,
            daily_ohlcv_by_token,
        )
        return still_open, closed_positions, equity

    pending = sorted(
        pending_signals,
        key=lambda s: parse_entry_datetime(s) or datetime.combine(session, datetime.min.time()),
    )

    for bar_dt in timeline:
        next_open: list[Position] = []
        for pos in still_open:
            symbol = pos.signal.instrument.symbol.upper()
            token = pos.signal.instrument.instrument_token
            bars = bars_by_symbol.get(symbol)
            if bars is None:
                next_open.append(pos)
                continue

            rows = bars[pd.to_datetime(bars["datetime"]) == pd.Timestamp(bar_dt)]
            if rows.empty:
                next_open.append(pos)
                continue

            ohlcv_bar = row_to_ohlcv(rows.iloc[0], token, session)
            sessions = risk_engine.sessions_held(pos, session, all_trading_dates)
            should_exit, exit_status, exit_price = risk_engine.check_exits(
                pos, ohlcv_bar, sessions
            )
            if should_exit:
                pos, net_pnl = _close_position(pos, exit_price, exit_status, session, cost_model)
                equity += net_pnl
                closed_positions.append(pos)
            else:
                next_open.append(pos)
        still_open = next_open

        for signal in pending:
            key = (signal.instrument.symbol.upper(), signal.horizon.value)
            if key in entered_keys:
                continue
            entry_dt = parse_entry_datetime(signal)
            if entry_dt is None or pd.Timestamp(bar_dt) < pd.Timestamp(entry_dt):
                continue
            if pd.Timestamp(bar_dt) != pd.Timestamp(entry_dt):
                continue

            try:
                can_enter, shares, _reason = risk_engine.evaluate_signal(
                    signal, still_open, daily_entry_count, equity
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "evaluate_signal failed for %s on %s: %s",
                    signal.instrument.symbol,
                    session,
                    exc,
                )
                can_enter, shares = False, 0

            if can_enter and shares > 0:
                still_open.append(
                    Position(
                        signal=signal,
                        shares=shares,
                        entry_date=session,
                        entry_price=signal.entry_price,
                        entry_datetime=entry_dt,
                        status=TradeStatus.OPEN,
                    )
                )
                entered_keys.add(key)
                daily_entry_count += 1

    return still_open, closed_positions, equity


def _daily_exit_fallback(
    session: date,
    open_positions: list[Position],
    closed_positions: list[Position],
    equity: float,
    all_trading_dates: list[date],
    risk_engine: RiskEngine,
    cost_model: CostModel,
    daily_ohlcv_by_token: dict[int, pd.DataFrame] | None,
) -> tuple[list[Position], list[Position], float]:
    """When 5m data is missing, fall back to one daily bar exit check."""
    if not daily_ohlcv_by_token:
        return open_positions, closed_positions, equity

    still_open: list[Position] = []
    for pos in open_positions:
        token = pos.signal.instrument.instrument_token
        df = daily_ohlcv_by_token.get(token)
        if df is None or df.empty:
            still_open.append(pos)
            continue
        rows = df[df["date"] == session] if "date" in df.columns else df
        if rows.empty:
            still_open.append(pos)
            continue
        row = rows.iloc[-1]
        bar = OHLCVBar(
            date=session,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
            instrument_token=token,
        )
        sessions = risk_engine.sessions_held(pos, session, all_trading_dates)
        should_exit, exit_status, exit_price = risk_engine.check_exits(pos, bar, sessions)
        if should_exit:
            pos, net_pnl = _close_position(pos, exit_price, exit_status, session, cost_model)
            equity += net_pnl
            closed_positions.append(pos)
        else:
            still_open.append(pos)
    return still_open, closed_positions, equity
