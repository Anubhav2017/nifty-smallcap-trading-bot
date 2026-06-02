"""Technical indicators computed on OHLCV DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range over `period` bars. Column: atr_{period}."""
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out[f"atr_{period}"] = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return out


def add_momentum(df: pd.DataFrame, periods: tuple[int, ...] = (20, 50)) -> pd.DataFrame:
    """Log-return momentum over each lookback period. Columns: mom_{n}d."""
    out = df.copy()
    log_close = np.log(out["close"])
    for n in periods:
        out[f"mom_{n}d"] = log_close - log_close.shift(n)
    return out


def add_relative_strength(
    df: pd.DataFrame, index_df: pd.DataFrame, period: int = 20
) -> pd.DataFrame:
    """Stock log-return minus index log-return over `period` days. Column: rs_{period}d."""
    out = df.copy()
    stock_ret = np.log(out["close"]) - np.log(out["close"].shift(period))
    # Align index on date; reindex to match df's index
    idx_close = index_df.set_index("date")["close"].reindex(out["date"]).values
    idx_series = pd.Series(idx_close, index=out.index)
    index_ret = np.log(idx_series) - np.log(idx_series.shift(period))
    out[f"rs_{period}d"] = stock_ret - index_ret
    return out


def add_volume_surge(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Current volume divided by rolling mean volume. Column: vol_surge_{period}d."""
    out = df.copy()
    rolling_mean = out["volume"].rolling(period, min_periods=period).mean()
    out[f"vol_surge_{period}d"] = out["volume"] / rolling_mean
    return out


def add_distance_from_high_low(df: pd.DataFrame, period: int = 52 * 5) -> pd.DataFrame:
    """Position of close within rolling [min, max] range. Column: hl_position_{period}d."""
    out = df.copy()
    rolling_min = out["low"].rolling(period, min_periods=period).min()
    rolling_max = out["high"].rolling(period, min_periods=period).max()
    denom = rolling_max - rolling_min
    out[f"hl_position_{period}d"] = (out["close"] - rolling_min) / denom.where(denom != 0)
    return out


def add_gap_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Overnight gap as abs(open - prev_close) / prev_close. Column: gap_risk."""
    out = df.copy()
    prev_close = out["close"].shift(1)
    out["gap_risk"] = (out["open"] - prev_close).abs() / prev_close
    return out


def add_atr_pct(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR as percentage of close. Column: atr_pct_{period}."""
    out = df.copy()
    atr_col = f"atr_{period}"
    if atr_col not in out.columns:
        out = add_atr(out, period)
    out[f"atr_pct_{period}"] = out[atr_col] / out["close"]
    return out


def add_all_features(df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """Apply all indicators with default parameters and return combined DataFrame."""
    out = df.copy()
    out = add_atr(out)
    out = add_momentum(out)
    out = add_relative_strength(out, index_df)
    out = add_volume_surge(out)
    out = add_distance_from_high_low(out)
    out = add_gap_risk(out)
    out = add_atr_pct(out)
    return out
