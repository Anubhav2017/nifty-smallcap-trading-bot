"""Build 5m feature matrix for hybrid timing model training."""

from __future__ import annotations

import logging
from datetime import date, time

import pandas as pd

from trading_bot.config import Config
from trading_bot.data.bars import BarStore
from trading_bot.features.intraday_features import INTRADAY_FEATURE_COLS, add_intraday_bar_features
from trading_bot.features.intraday_labels import label_intraday_tp_before_sl
from trading_bot.types import Horizon

logger = logging.getLogger(__name__)


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def build_intraday_matrix(
    cfg: Config,
    daily_feature_df: pd.DataFrame,
    dates: list[date],
    *,
    bar_store: BarStore | None = None,
) -> pd.DataFrame:
    """Build 5m rows for hybrid timing training.

    Uses daily feature rows for context (momentum, ATR) and 5m bars from ``BarStore``.
    """
    if daily_feature_df.empty or not dates:
        return pd.DataFrame()

    store = bar_store or BarStore(cfg=cfg)
    hybrid = cfg.hybrid or {}
    start_t = _parse_time(hybrid.get("entry_start", "09:30"))
    cutoff_t = _parse_time(hybrid.get("entry_cutoff", "14:30"))
    date_set = set(dates)

    daily = daily_feature_df[daily_feature_df["date"].isin(date_set)].copy()
    if daily.empty:
        return pd.DataFrame()

    total_rows = len(daily)
    log_every = max(500, total_rows // 20)
    frames: list[pd.DataFrame] = []
    horizon = Horizon.SWING
    processed = built = skipped_no_bars = 0

    for _, day_row in daily.iterrows():
        processed += 1
        symbol = str(day_row["symbol"]).upper()
        session = day_row["date"]
        if isinstance(session, pd.Timestamp):
            session = session.date()

        bars = store.get_bars(symbol, session)
        if bars.empty:
            skipped_no_bars += 1
            if processed % log_every == 0 or processed == total_rows:
                logger.info(
                    "  intraday matrix: %d/%d symbol-days scanned, %d built, %d missing 5m",
                    processed,
                    total_rows,
                    built,
                    skipped_no_bars,
                )
            continue

        bars = bars.copy()
        bars["datetime"] = pd.to_datetime(bars["datetime"])
        bar_times = bars["datetime"].dt.time
        bars = bars[(bar_times >= start_t) & (bar_times <= cutoff_t)].copy()
        if bars.empty:
            if processed % log_every == 0 or processed == total_rows:
                logger.info(
                    "  intraday matrix: %d/%d symbol-days scanned, %d built",
                    processed,
                    total_rows,
                    built,
                )
            continue

        atr = float(day_row.get("atr_14", 0.0) or 0.0)
        if atr <= 0:
            continue

        enriched = add_intraday_bar_features(bars, day_row)
        enriched["label_timing"] = label_intraday_tp_before_sl(bars, atr, cfg, horizon)
        enriched["date"] = session
        enriched["symbol"] = symbol
        enriched["instrument_token"] = int(day_row["instrument_token"])
        enriched["atr_14"] = atr
        frames.append(enriched)
        built += 1

        if processed % log_every == 0 or processed == total_rows:
            logger.info(
                "  intraday matrix: %d/%d symbol-days scanned, %d built, %d missing 5m",
                processed,
                total_rows,
                built,
                skipped_no_bars,
            )

    if not frames:
        logger.warning("No intraday training rows built for %d dates.", len(dates))
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=INTRADAY_FEATURE_COLS + ["label_timing"])
    logger.info(
        "  intraday matrix: complete — %d bars from %d symbol-days",
        len(out),
        built,
    )
    return out.sort_values(["date", "symbol", "datetime"]).reset_index(drop=True)


def intraday_features_for_session(
    cfg: Config,
    symbol: str,
    session: date,
    daily_row: pd.Series,
    *,
    bar_store: BarStore | None = None,
) -> pd.DataFrame:
    """Feature rows for one symbol-day (inference)."""
    store = bar_store or BarStore(cfg=cfg)
    hybrid = cfg.hybrid or {}
    start_t = _parse_time(hybrid.get("entry_start", "09:30"))
    cutoff_t = _parse_time(hybrid.get("entry_cutoff", "14:30"))

    bars = store.get_bars(symbol.upper(), session)
    if bars.empty:
        return pd.DataFrame()

    bars = bars.copy()
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    bar_times = bars["datetime"].dt.time
    bars = bars[(bar_times >= start_t) & (bar_times <= cutoff_t)].copy()
    if bars.empty:
        return pd.DataFrame()

    enriched = add_intraday_bar_features(bars, daily_row)
    enriched["date"] = session
    enriched["symbol"] = symbol.upper()
    enriched["datetime"] = pd.to_datetime(enriched["datetime"])
    return enriched.dropna(subset=INTRADAY_FEATURE_COLS, how="any")
