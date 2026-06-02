"""Zerodha Kite Connect OHLCV fetcher (live API only, no local cache)."""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime

import pandas as pd

from trading_bot.config import Config
from trading_bot.data.intraday import INTERVAL_CHUNK_DAYS, iter_chunks

try:
    from kiteconnect import KiteConnect
except ImportError as _kite_import_err:
    KiteConnect = None  # type: ignore[assignment,misc]
    _KITE_MISSING_MSG = (
        "kiteconnect is not installed. "
        "Run `pip install kiteconnect` to enable live data fetching."
    )
else:
    _KITE_MISSING_MSG = ""

logger = logging.getLogger(__name__)

_RATE_LIMIT_SLEEP = 0.35
_MAX_RETRIES = 3
_RETRY_SLEEP_BASE = 1.0


class KiteDataClient:
    """Fetch OHLCV bars from Kite Connect (historical datasets live under dataset_*)."""

    def __init__(self, cfg: Config) -> None:
        if KiteConnect is None:
            raise ImportError(_KITE_MISSING_MSG)

        api_key = os.environ.get("KITE_API_KEY")
        access_token = os.environ.get("KITE_ACCESS_TOKEN")

        if not api_key:
            raise EnvironmentError("KITE_API_KEY environment variable is not set.")
        if not access_token:
            raise EnvironmentError("KITE_ACCESS_TOKEN environment variable is not set.")

        self._cfg = cfg
        self._kite = KiteConnect(api_key=api_key)
        self._kite.set_access_token(access_token)

    def _fetch_ohlcv_interval_chunk(
        self,
        instrument_token: int,
        from_dt: datetime,
        to_dt: datetime,
        interval: str,
    ) -> list[dict]:
        """Fetch one intraday window from Kite (single API call)."""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                time.sleep(_RATE_LIMIT_SLEEP)
                return self._kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=from_dt,
                    to_date=to_dt,
                    interval=interval,
                )

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    sleep_time = _RETRY_SLEEP_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "Kite intraday fetch failed (attempt %d/%d) token=%s: %s",
                        attempt,
                        _MAX_RETRIES,
                        instrument_token,
                        exc,
                    )
                    time.sleep(sleep_time)

        raise RuntimeError(
            f"Could not fetch {interval} for token={instrument_token} "
            f"[{from_dt}, {to_dt}] after {_MAX_RETRIES} attempts."
        ) from last_exc

    def fetch_ohlcv_interval_range(
        self,
        instrument_token: int,
        start: date,
        end: date,
        interval: str,
    ) -> pd.DataFrame:
        """Fetch intraday OHLCV in API-safe chunks and return combined bars."""
        if interval not in INTERVAL_CHUNK_DAYS:
            raise ValueError(f"Unsupported interval: {interval}")

        frames: list[pd.DataFrame] = []
        for from_dt, to_dt in iter_chunks(start, end, interval):
            raw = self._fetch_ohlcv_interval_chunk(instrument_token, from_dt, to_dt, interval)
            if not raw:
                continue
            df = pd.DataFrame(raw)
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"])
            frames.append(df)

        if not frames:
            return self._empty_ohlcv()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date")
        combined["date"] = pd.to_datetime(combined["date"])
        return combined.reset_index(drop=True)

    def fetch_ohlcv(
        self,
        instrument_token: int,
        from_date: date,
        to_date: date,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV from Kite Connect with rate limiting and retries."""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                time.sleep(_RATE_LIMIT_SLEEP)
                raw = self._kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=from_date,
                    to_date=to_date,
                    interval="day",
                )
                df = pd.DataFrame(raw)
                if df.empty:
                    logger.warning(
                        "Kite returned empty data for token=%s [%s, %s]",
                        instrument_token,
                        from_date,
                        to_date,
                    )
                    return self._empty_ohlcv()

                df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df[["date", "open", "high", "low", "close", "volume"]]
                return df.sort_values("date").reset_index(drop=True)

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    sleep_time = _RETRY_SLEEP_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "Kite fetch failed (attempt %d/%d) for token=%s: %s. Retrying in %.1fs.",
                        attempt,
                        _MAX_RETRIES,
                        instrument_token,
                        exc,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        "Kite fetch failed after %d attempts for token=%s: %s",
                        _MAX_RETRIES,
                        instrument_token,
                        exc,
                    )

        raise RuntimeError(
            f"Could not fetch OHLCV for token={instrument_token} after {_MAX_RETRIES} attempts."
        ) from last_exc

    def flag_corporate_action_gaps(
        self,
        df: pd.DataFrame,
        index_df: pd.DataFrame,
        threshold_pct: float = 40.0,
    ) -> pd.DataFrame:
        """Add bool column ``corp_action_gap`` for likely unadjusted price gaps."""
        df = df.copy()
        df["_pct_chg"] = df["close"].pct_change().abs() * 100.0

        index_df = index_df.copy()
        index_df["_idx_pct_chg"] = index_df["close"].pct_change().abs() * 100.0
        index_lookup = index_df.set_index("date")["_idx_pct_chg"]

        def _is_gap(row: pd.Series) -> bool:
            if row["_pct_chg"] <= threshold_pct:
                return False
            idx_move = index_lookup.get(row["date"], 0.0)
            return float(idx_move) <= 10.0

        df["corp_action_gap"] = df.apply(_is_gap, axis=1)
        return df.drop(columns=["_pct_chg"])

    @staticmethod
    def _empty_ohlcv() -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
