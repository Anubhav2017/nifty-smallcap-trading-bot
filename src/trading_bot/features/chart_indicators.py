"""Technical indicators for charting (dashboard)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_chart_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    close = out["close"]
    volume = out["volume"]

    for window in (1, 5, 20, 60):
        out[f"ret_{window}d"] = close.pct_change(window)

    out["volatility_20d"] = out["ret_1d"].rolling(20).std()
    vol_ma = volume.rolling(20).mean()
    out["volume_ratio_20d"] = volume / vol_ma.replace(0, np.nan)
    out["rsi_14"] = _rsi(close, 14)

    for window in (20, 50):
        sma = close.rolling(window).mean()
        out[f"sma_{window}"] = sma
        out[f"close_sma_{window}d"] = close / sma.replace(0, np.nan) - 1.0

    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["high_low_range"] = (out["high"] - out["low"]) / close.replace(0, np.nan)
    return out
