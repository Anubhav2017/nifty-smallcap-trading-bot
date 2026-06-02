"""Look-forward labels for supervised training. Never use on live data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.config import Config
from trading_bot.types import Horizon


def label_tp_before_sl(
    df: pd.DataFrame, horizon: Horizon, cfg: Config
) -> pd.Series:
    """Return 1.0 if TP is hit before SL within the hold window, else 0.0.

    Uses vectorized numpy operations over the price matrix.
    Last `max_hold_days` rows are NaN (incomplete future window).
    """
    exit_cfg = cfg.exit[horizon.value]
    k: float = exit_cfg["atr_sl_multiple"]
    r: float = exit_cfg["reward_risk_ratio"]
    max_hold: int = cfg.horizons[horizon.value]["max_hold_days"]

    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr_14"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    sl_dist = k * atr
    sl = close - sl_dist
    tp = close + r * sl_dist

    result = np.full(n, np.nan)

    # Build (n, max_hold) matrices of future highs and lows.
    # future_high[i, j] = high[i + j + 1], future_low[i, j] = low[i + j + 1]
    valid_rows = n - max_hold  # rows with a complete future window

    if valid_rows <= 0:
        return pd.Series(result, index=df.index, name=f"label_tp_{horizon.value}")

    # Shape: (valid_rows, max_hold)
    row_idx = np.arange(valid_rows)[:, None]          # (valid_rows, 1)
    col_idx = np.arange(1, max_hold + 1)[None, :]     # (1, max_hold)
    future_idx = row_idx + col_idx                    # (valid_rows, max_hold)

    future_high = high[future_idx]   # (valid_rows, max_hold)
    future_low = low[future_idx]     # (valid_rows, max_hold)

    tp_hits = future_high >= tp[:valid_rows, None]    # bool (valid_rows, max_hold)
    sl_hits = future_low <= sl[:valid_rows, None]     # bool (valid_rows, max_hold)

    # For each row, find the first bar where TP or SL is touched.
    # argmax returns 0 when no True is found; guard with `.any()`.
    tp_any = tp_hits.any(axis=1)
    sl_any = sl_hits.any(axis=1)

    # First hit index (0-based within the window).
    tp_first = np.where(tp_any, tp_hits.argmax(axis=1), max_hold)
    sl_first = np.where(sl_any, sl_hits.argmax(axis=1), max_hold)

    label = np.where(
        tp_any & (tp_first <= sl_first),
        1.0,
        np.where(sl_any | tp_any, 0.0, 0.0),  # neither hit also counts as 0
    )

    result[:valid_rows] = label
    return pd.Series(result, index=df.index, name=f"label_tp_{horizon.value}")


def label_forward_return(
    df: pd.DataFrame, horizon: Horizon, cfg: Config
) -> pd.Series:
    """Log return from close[i] to close[i + label_day]. NaN for last label_day rows."""
    label_day: int = cfg.horizons[horizon.value]["label_day"]
    log_close = np.log(df["close"])
    fwd = log_close.shift(-label_day)
    series = fwd - log_close
    series.name = f"fwd_ret_{horizon.value}"
    return series


def label_mae(
    df: pd.DataFrame, horizon: Horizon, cfg: Config
) -> pd.Series:
    """Max adverse excursion over the hold window: max((entry - low_j) / entry).

    Last `max_hold_days` rows are NaN.
    """
    max_hold: int = cfg.horizons[horizon.value]["max_hold_days"]
    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    result = np.full(n, np.nan)
    valid_rows = n - max_hold

    if valid_rows <= 0:
        return pd.Series(result, index=df.index, name=f"mae_{horizon.value}")

    row_idx = np.arange(valid_rows)[:, None]
    col_idx = np.arange(1, max_hold + 1)[None, :]
    future_idx = row_idx + col_idx

    future_low = low[future_idx]                               # (valid_rows, max_hold)
    entry = close[:valid_rows, None]                           # (valid_rows, 1)
    adverse = (entry - future_low) / entry                     # (valid_rows, max_hold)
    result[:valid_rows] = adverse.max(axis=1)

    return pd.Series(result, index=df.index, name=f"mae_{horizon.value}")


def add_all_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add all labels for both horizons to the DataFrame."""
    out = df.copy()
    for horizon in Horizon:
        out[f"label_tp_{horizon.value}"] = label_tp_before_sl(out, horizon, cfg)
        out[f"fwd_ret_{horizon.value}"] = label_forward_return(out, horizon, cfg)
        out[f"mae_{horizon.value}"] = label_mae(out, horizon, cfg)
    return out
