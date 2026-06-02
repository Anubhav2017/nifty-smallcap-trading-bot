"""Intraday entry-timing labels on 5-minute bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.config import Config
from trading_bot.types import Horizon


def label_intraday_tp_before_sl(
    bars: pd.DataFrame,
    atr: float,
    cfg: Config,
    horizon: Horizon = Horizon.SWING,
) -> pd.Series:
    """1.0 if TP is hit before SL from bar close through session end, else 0.0."""
    name = "label_timing"
    n = len(bars)
    if n == 0 or atr <= 0:
        return pd.Series(dtype=float, name=name)

    exit_cfg = cfg.exit[horizon.value]
    k = float(exit_cfg["atr_sl_multiple"])
    r = float(exit_cfg["reward_risk_ratio"])

    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)

    sl_dist = k * atr
    sl = close - sl_dist
    tp = close + r * sl_dist

    result = np.zeros(n, dtype=float)
    for i in range(n):
        future_high = high[i + 1 :]
        future_low = low[i + 1 :]
        if future_high.size == 0:
            result[i] = 0.0
            continue

        tp_hits = future_high >= tp[i]
        sl_hits = future_low <= sl[i]
        if not tp_hits.any() and not sl_hits.any():
            result[i] = 0.0
            continue

        tp_first = np.argmax(tp_hits) if tp_hits.any() else len(future_high)
        sl_first = np.argmax(sl_hits) if sl_hits.any() else len(future_high)
        result[i] = 1.0 if tp_hits.any() and tp_first <= sl_first else 0.0

    return pd.Series(result, index=bars.index, name=name)
