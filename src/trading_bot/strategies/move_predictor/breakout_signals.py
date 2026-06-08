"""52-week breakout signal generator.

Complementary to the LightGBM momentum signal.  Captures trend-start moves
that the model misses because they lack the volume/RSI pattern the model was
trained on.

Entry criteria (all using T-1 data — no lookahead):
  1. close ≥ (52-week high × threshold)  e.g. ≥ 97% of 52-week high
  2. volume ratio ≥ min_vol (default 1.5×)
  3. price above 200 DMA  (close_sma_200d > 0)
  4. 20-day return ≤ max_20d_ret (default 20%) — avoid chasing exhausted moves
  5. Not in the SL-cooldown exclusion set

Exit: same ATR-based SL / 2R target as the momentum signal.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from trading_bot.config import Config
from trading_bot.models.exit_policy import ExitPolicy
from trading_bot.backtest.costs import estimate_cost_per_share
from trading_bot.types import Horizon, Instrument, Signal

logger = logging.getLogger(__name__)


def generate_breakout_signals(
    cfg: Config,
    panel: pd.DataFrame,
    as_of_date: date,
    instruments: list[Instrument],
    *,
    excluded_symbols: set[str] | None = None,
    already_picked: set[str] | None = None,
    breadth: float = 1.0,
) -> list[Signal]:
    """Return 52-week breakout signals for *as_of_date*.

    Args:
        excluded_symbols: Symbols in SL-cooldown.
        already_picked:   Symbols already chosen by the momentum model today
                          (we don't double-up on the same stock).
    """
    bo = cfg._raw.get("breakout_signal", {})
    if not bo.get("enabled", True):
        return []

    # ── Breadth gate: breakout needs a stronger market than momentum ──────────
    # Breakout into a declining market (low breadth) reliably fails.
    # Uses a separate, higher threshold than the regime filter minimum.
    min_breadth = float(bo.get("min_breadth_pct", 0.0))
    if min_breadth > 0.0 and breadth < min_breadth:
        logger.debug(
            "Breakout suppressed on %s — breadth=%.0f%% < %.0f%%",
            as_of_date, breadth * 100, min_breadth * 100,
        )
        return []

    top_n      = int(bo.get("top_n", 3))
    min_vol    = float(bo.get("min_volume_ratio", 1.5))
    min_ratio  = float(bo.get("min_52w_high_ratio", 0.97))   # ≥ 97% of 52w high
    max_20d    = float(bo.get("max_20d_return", 0.20))        # not over-extended
    require_above_200 = bool(bo.get("require_above_200dma", True))
    rr_override = float(bo.get("reward_risk_ratio", 2.0))     # wider TP for breakout (default 2.5R)

    ef = cfg._raw.get("entry_filters", {})
    sl_cooldown_days = int(ef.get("sl_cooldown_days", 0))

    today = panel[panel["date"] == as_of_date].copy()
    if today.empty:
        return []

    # ── Exclusions ───────────────────────────────────────────────────────────
    if excluded_symbols:
        today = today[~today["symbol"].isin(excluded_symbols)]
    if already_picked:
        today = today[~today["symbol"].isin(already_picked)]
    if today.empty:
        return []

    # ── Filter: 52-week high proximity (lagged — uses prior close vs prior 52w high) ──
    ratio_col = "high_52w_ratio_lag1"
    if ratio_col not in today.columns:
        return []
    today = today[today[ratio_col].fillna(0) >= min_ratio]
    if today.empty:
        return []

    # ── Filter: volume surge ─────────────────────────────────────────────────
    vol_col = "volume_ratio_20d_lag1"
    if vol_col in today.columns:
        today = today[today[vol_col].fillna(0) >= min_vol]
    if today.empty:
        return []

    # ── Filter: above 200 DMA ────────────────────────────────────────────────
    if require_above_200 and "close_sma_200d_lag1" in today.columns:
        today = today[today["close_sma_200d_lag1"].fillna(-1) > 0]
    if today.empty:
        return []

    # ── Filter: not over-extended (20d return cap) ───────────────────────────
    if "ret_20d_lag1" in today.columns:
        today = today[today["ret_20d_lag1"].fillna(0) <= max_20d]
    if today.empty:
        return []

    # ── BSE result blackout: skip stocks that filed earnings recently ─────────
    if "bse_result_blackout_lag1" in today.columns:
        today = today[today["bse_result_blackout_lag1"].fillna(0.0) < 1.0]
    if today.empty:
        return []

    # ── Rank by proximity to 52-week high (closest = strongest breakout) ─────
    candidates = today.nlargest(top_n, ratio_col)
    token_map  = {i.instrument_token: i for i in instruments}
    exit_policy = ExitPolicy(cfg)
    signals: list[Signal] = []

    horizon_key = str(cfg._raw.get("move_predictor", {}).get("horizon", "swing")).lower()
    horizon = Horizon.POSITIONAL if horizon_key == "positional" else Horizon.SWING

    for _, row in candidates.iterrows():
        token = int(row["instrument_token"])
        inst  = token_map.get(token)
        if inst is None:
            continue

        entry_price = float(row.get("entry_open", 0.0))
        atr = float(row.get("atr_14_lag1", 0.0))
        if entry_price <= 0 or atr <= 0:
            continue

        cost = estimate_cost_per_share(cfg, entry_price)
        sig = exit_policy.build_signal(
            instrument=inst,
            horizon=horizon,
            entry_price=entry_price,
            atr=atr,
            win_prob=0.45,         # fixed prior — breakout tends to work ~45% of time
            rank_score=float(row[ratio_col]),
            signal_date=as_of_date,
            cost_per_share=cost,
            require_positive_ev=False,
            reward_risk_override=rr_override,
        )
        if sig is not None:
            sig.features["strategy"] = "breakout_52w"
            sig.features["high_52w_ratio"] = round(float(row[ratio_col]), 4)
            sig.features["volume_ratio_lag1"] = round(float(row.get(vol_col, float("nan"))), 2)
            sig.features["ret_20d"] = round(float(row.get("ret_20d_lag1", float("nan"))), 4)
            sig.features["close_sma_200d"] = round(float(row.get("close_sma_200d_lag1", float("nan"))), 4)
            signals.append(sig)

    return signals
