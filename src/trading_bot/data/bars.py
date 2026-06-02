"""Intraday bar access from dataset ``ohlcv/minute/{SYMBOL}.csv`` (resampled to 5m)."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from trading_bot.config import Config
from trading_bot.data.dataset_store import (
    dataset_root_from_config,
    list_symbols,
    load_minute_session,
    load_universe,
    minute_session_dates,
)
from trading_bot.data.intraday import OHLCV_COLS
from trading_bot.data.trading_calendar import resolve_trading_day

logger = logging.getLogger(__name__)


class BarStore:
    """Read intraday bars from the active dataset (1m CSV → 5m bars)."""

    def __init__(self, cfg: Config | None = None, dataset_root=None) -> None:
        if dataset_root is not None:
            from pathlib import Path

            self._root = Path(dataset_root).resolve()
        elif cfg is not None:
            self._root = dataset_root_from_config(cfg)
        else:
            self._root = dataset_root_from_config(Config(None))

        self._token_to_symbol: dict[int, str] | None = None
        self._symbol_to_token: dict[str, int] | None = None

    def interval_key(self, interval: str) -> str:
        return "minute" if interval in ("minute", "5minute") else interval

    def _load_symbol_maps(self) -> tuple[dict[int, str], dict[str, int]]:
        if self._token_to_symbol is not None and self._symbol_to_token is not None:
            return self._token_to_symbol, self._symbol_to_token

        token_to_symbol: dict[int, str] = {}
        symbol_to_token: dict[str, int] = {}
        enriched = load_universe(self._root)
        for _, row in enriched.iterrows():
            if "found" in row.index and not bool(row.get("found", True)):
                continue
            symbol = str(row["symbol"]).strip().upper()
            try:
                token = int(row["instrument_token"])
            except (TypeError, ValueError):
                continue
            token_to_symbol[token] = symbol
            symbol_to_token[symbol] = token

        self._token_to_symbol = token_to_symbol
        self._symbol_to_token = symbol_to_token
        return token_to_symbol, symbol_to_token

    def symbol_for_token(self, token: int) -> str | None:
        return self._load_symbol_maps()[0].get(token)

    def token_for_symbol(self, symbol: str) -> int | None:
        return self._load_symbol_maps()[1].get(symbol.upper())

    def get_bars(
        self,
        symbol: str,
        day: date,
        interval: str = "5minute",
    ) -> pd.DataFrame:
        resample = interval != "minute"
        return load_minute_session(
            symbol.upper(),
            day,
            self._root,
            resample_5m=resample,
        )

    def resolve_session_date(
        self,
        symbol: str,
        day: date,
        interval: str = "5minute",
    ) -> tuple[date, str | None]:
        available = self.list_dates(symbol, interval)
        if available:
            return resolve_trading_day(day, set(available))
        return resolve_trading_day(day)

    def get_bars_resolved(
        self,
        symbol: str,
        day: date,
        interval: str = "5minute",
    ) -> tuple[pd.DataFrame, date, str | None]:
        session, note = self.resolve_session_date(symbol, day, interval=interval)
        return self.get_bars(symbol, session, interval=interval), session, note

    def market_trading_days(self, interval: str = "5minute") -> set[date]:
        days: set[date] = set()
        for symbol in list_symbols("minute", self._root):
            days.update(minute_session_dates(symbol, self._root))
        return days

    def resolve_market_session(
        self,
        day: date,
        interval: str = "5minute",
    ) -> tuple[date, str | None]:
        known = self.market_trading_days(interval)
        if known:
            return resolve_trading_day(day, known)
        return resolve_trading_day(day)

    def list_symbols(self, day: date, interval: str = "5minute") -> list[str]:
        return self.list_symbols_resolved(day, interval=interval)[0]

    def list_symbols_resolved(
        self,
        day: date,
        interval: str = "5minute",
    ) -> tuple[list[str], date, str | None]:
        session, note = self.resolve_market_session(day, interval=interval)
        found = [
            sym
            for sym in list_symbols("minute", self._root)
            if session in minute_session_dates(sym, self._root)
        ]
        return sorted(found), session, note

    def list_dates(self, symbol: str, interval: str = "5minute") -> list[date]:
        return minute_session_dates(symbol.upper(), self._root)

    def list_stock_symbols(self, interval: str = "5minute") -> list[str]:
        return list_symbols("minute", self._root)
