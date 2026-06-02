"""Hybrid backtest: 5m timed entries and 5m SL/TP/time exits."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.backtest.intraday_sim import load_session_bars, simulate_intraday_session
from trading_bot.data.bars import BarStore
from trading_bot.types import Position, Signal

logger = logging.getLogger(__name__)


class HybridBacktestEngine(BacktestEngine):
    """Bar-level hybrid simulation using dataset ``ohlcv/minute/`` bars."""

    def __init__(self, cfg, cost_model, risk_engine, *, bar_store: BarStore | None = None) -> None:
        super().__init__(cfg, cost_model, risk_engine)
        self._bar_store = bar_store or BarStore(cfg=cfg)

    def run(
        self,
        signals_by_date: dict[date, list[Signal]],
        ohlcv_by_token: dict[int, pd.DataFrame],
        initial_equity: float = 1_000_000.0,
    ) -> tuple[list[Position], pd.Series]:
        all_dates = self._build_trading_calendar(signals_by_date, ohlcv_by_token)
        if not all_dates:
            return [], pd.Series(dtype=float, name="equity")

        open_positions: list[Position] = []
        closed_positions: list[Position] = []
        equity = initial_equity
        equity_history: dict[date, float] = {}

        for current_date in all_dates:
            symbols: set[str] = set()
            for pos in open_positions:
                symbols.add(pos.signal.instrument.symbol.upper())
            for sig in signals_by_date.get(current_date, []):
                symbols.add(sig.instrument.symbol.upper())

            bars_by_symbol = load_session_bars(self._bar_store, symbols, current_date)
            open_positions, closed_positions, equity = simulate_intraday_session(
                current_date,
                open_positions,
                closed_positions,
                signals_by_date.get(current_date, []),
                bars_by_symbol,
                all_trading_dates=all_dates,
                equity=equity,
                risk_engine=self._risk_engine,
                cost_model=self._cost_model,
                daily_ohlcv_by_token=ohlcv_by_token,
            )
            equity_history[current_date] = equity

        dt_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        values = [equity_history[d] for d in all_dates]
        equity_curve = pd.Series(values, index=dt_index, name="equity", dtype=float)
        return closed_positions, equity_curve
