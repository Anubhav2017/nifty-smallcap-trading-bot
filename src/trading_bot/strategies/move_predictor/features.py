"""Lagged features for move predictor — no same-day leakage."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from trading_bot.analysis.move_correlation import (
    BSE_ANNOUNCEMENT_COLS,
    SIMPLE_FACTOR_COLS,
    enrich_technical_factors,
)
from trading_bot.features.bse_events import build_bse_event_features
from trading_bot.features.chart_indicators import add_chart_indicators
from trading_bot.types import Instrument

# Features known at market open on day T (computed through T-1 close).
# Features known at market open on day T (computed through T-1 close).
# high_52w_ratio_lag1 is added explicitly — it's computed after SIMPLE_FACTOR_COLS lagging
# to ensure it's available in the panel before the filter lag loop.
LAGGED_FEATURE_COLS = [f"{c}_lag1" for c in SIMPLE_FACTOR_COLS] + ["atr_14_lag1", "high_52w_ratio_lag1"]

# BSE event features are already point-in-time (they look backward from each date),
# so they need a 1-day lag so the model only sees yesterday's flags at entry.
_BSE_FEATURE_COLS = BSE_ANNOUNCEMENT_COLS  # alias for clarity inside this module

# Add BSE lag1 columns to the feature set used by the model.
LAGGED_FEATURE_COLS = LAGGED_FEATURE_COLS + [f"{c}_lag1" for c in _BSE_FEATURE_COLS]

# Extra lagged columns used by entry filters (not model features — no leakage risk)
_FILTER_LAG_COLS = ["close_sma_200d", "ret_20d", "high_52w_ratio"]

LABEL_COL = "fwd_ret_1d"
LABEL_UP_COL = "label_up_1d"
LABEL_BIG_UP_COL = "label_big_up_1d"
DEFAULT_LABEL_MIN_MOVE_PCT = 0.015


def _prepare_symbol_bars(df: pd.DataFrame) -> pd.DataFrame:
    out = add_chart_indicators(df.sort_values("date").copy())
    out = enrich_technical_factors(out)
    if "atr_14" not in out.columns and "atr_pct_14" in out.columns:
        out["atr_14"] = out["close"] * out["atr_pct_14"]
    return out


def symbol_lagged_frame(
    df: pd.DataFrame,
    instrument: Instrument,
    *,
    label_min_move_pct: float = DEFAULT_LABEL_MIN_MOVE_PCT,
) -> pd.DataFrame:
    """One row per session with lagged inputs and next-day labels (training only)."""
    if df.empty:
        return pd.DataFrame()

    work = _prepare_symbol_bars(df)
    work["date"] = pd.to_datetime(work["date"]).dt.date

    for col in SIMPLE_FACTOR_COLS:
        if col in work.columns:
            work[f"{col}_lag1"] = work[col].shift(1)
    if "atr_14" in work.columns:
        work["atr_14_lag1"] = work["atr_14"].shift(1)

    # 52-week high ratio: close / rolling-252-day high — used by breakout signal filter
    work["high_52w_ratio"] = work["close"] / work["high"].rolling(252, min_periods=20).max()

    # Extra filter lags (not model features)
    for col in _FILTER_LAG_COLS:
        if col in work.columns and f"{col}_lag1" not in work.columns:
            work[f"{col}_lag1"] = work[col].shift(1)

    work["entry_open"] = work["open"]
    work["fwd_ret_1d"] = work["close"].shift(-1) / work["close"] - 1.0
    work["label_up_1d"] = (work["fwd_ret_1d"] > 0).astype(float)
    work[LABEL_BIG_UP_COL] = (work["fwd_ret_1d"] >= label_min_move_pct).astype(float)
    work["instrument_token"] = instrument.instrument_token
    work["symbol"] = instrument.symbol
    work["isin"] = instrument.isin
    return work


def build_lagged_panel(
    ohlcv_by_token: dict[int, pd.DataFrame],
    instruments: list[Instrument],
    *,
    label_min_move_pct: float = DEFAULT_LABEL_MIN_MOVE_PCT,
    ann_dir: "str | Path | None" = None,
) -> pd.DataFrame:
    token_map = {i.instrument_token: i for i in instruments}
    frames: list[pd.DataFrame] = []
    for token, ohlcv in ohlcv_by_token.items():
        inst = token_map.get(token)
        if inst is None or ohlcv.empty:
            continue
        frame = symbol_lagged_frame(
            ohlcv, inst, label_min_move_pct=label_min_move_pct
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])

    # ── BSE announcement features ──────────────────────────────────────────
    # Applied after full concat so vectorised date ops work across all symbols.
    # Features default to 0.0 if no ann_dir provided or directory is missing.
    if ann_dir is not None:
        panel = build_bse_event_features(panel, ann_dir)

    # Lag BSE features by 1 day per symbol to match the other *_lag1 features.
    # (BSE features already use backward-looking windows, so this adds
    #  conservative 1-day buffer — a result filed on D-1 shows up at D+1.)
    for col in _BSE_FEATURE_COLS:
        if col in panel.columns:
            panel[f"{col}_lag1"] = (
                panel.groupby("symbol")[col].shift(1).fillna(0.0)
            )

    return panel


def panel_for_dates(panel: pd.DataFrame, dates: list[date]) -> pd.DataFrame:
    if panel.empty:
        return panel
    dset = set(dates)
    return panel[panel["date"].isin(dset)].copy()


def apply_liquidity_filter(
    panel: pd.DataFrame,
    *,
    adtv_cr_min: float,
    lookback_days: int,
    as_of_date: "date | None" = None,
) -> pd.DataFrame:
    """Filter out illiquid stocks based on recent ADTV.

    Args:
        as_of_date: If provided, only use data up to this date when computing
                    ADTV. This prevents look-ahead bias when the panel contains
                    data beyond the backtest end date.
    """
    if panel.empty:
        return panel
    work = panel.copy()
    work["_dv"] = work["close"] * work["volume"] / 1e7
    keep: set[int] = set()
    for token, grp in work.groupby("instrument_token"):
        if as_of_date is not None:
            grp = grp[grp["date"] <= as_of_date]
        recent = grp.sort_values("date").tail(lookback_days)
        if recent["_dv"].mean() >= adtv_cr_min:
            keep.add(int(token))
    return work[work["instrument_token"].isin(keep)].drop(columns=["_dv"])
