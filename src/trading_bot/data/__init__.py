"""Data layer for the trading bot: Kite OHLCV client, universe management, and corporate actions."""

from trading_bot.data.corporate_actions import is_ex_dividend_day, load_dividend_dates

__all__ = [
    "is_ex_dividend_day",
    "load_dividend_dates",
]
