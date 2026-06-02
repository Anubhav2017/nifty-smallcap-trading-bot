"""5-minute bar features for hybrid entry-timing model."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

INTRADAY_FEATURE_COLS: list[str] = [
    "minutes_from_open",
    "bar_ret_1",
    "bar_ret_3",
    "vol_ratio_session",
    "vwap_dist",
    "open_dist",
    "session_range_pos",
    "mom_20d",
    "rs_20d",
    "atr_pct_14",
]

SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)


def _minutes_from_open(ts: pd.Series) -> pd.Series:
    open_minutes = SESSION_OPEN.hour * 60 + SESSION_OPEN.minute
    return (ts.dt.hour * 60 + ts.dt.minute - open_minutes).astype(float)


def add_intraday_bar_features(
    bars: pd.DataFrame,
    daily_row: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute per-bar features for one symbol-day 5m series."""
    if bars.empty:
        return pd.DataFrame(columns=INTRADAY_FEATURE_COLS)

    out = bars.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values("datetime").reset_index(drop=True)

    close = out["close"].astype(float)
    volume = out["volume"].astype(float).replace(0, np.nan)
    session_open = float(close.iloc[0])
    session_high = out["high"].astype(float).cummax()
    session_low = out["low"].astype(float).cummin()
    cum_vol = volume.cumsum()
    vwap = (close * volume.fillna(0)).cumsum() / cum_vol.replace(0, np.nan)

    out["minutes_from_open"] = _minutes_from_open(out["datetime"])
    out["bar_ret_1"] = (np.log(close / close.shift(1))).fillna(0.0)
    out["bar_ret_3"] = (np.log(close / close.shift(3))).fillna(0.0)
    out["vol_ratio_session"] = volume / volume.expanding().mean()
    out["vwap_dist"] = (close - vwap) / close
    out["open_dist"] = (close - session_open) / session_open
    span = (session_high - session_low).replace(0, np.nan)
    out["session_range_pos"] = (close - session_low) / span

    for col in ("mom_20d", "rs_20d", "atr_pct_14"):
        out[col] = float(daily_row[col]) if daily_row is not None and col in daily_row else np.nan

    return out
