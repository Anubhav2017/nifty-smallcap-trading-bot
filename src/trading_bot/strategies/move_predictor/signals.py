"""Signal generation for volume-momentum move predictor."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from trading_bot.config import Config
from trading_bot.models.exit_policy import ExitPolicy
from trading_bot.strategies.move_predictor.fundamental_screen import (
    FundamentalScreenConfig,
    passes_fundamental_screen,
)
from trading_bot.backtest.costs import estimate_cost_per_share
from trading_bot.strategies.move_predictor.model import MovePredictorModel
from trading_bot.types import Horizon, Instrument, Signal

logger = logging.getLogger(__name__)


def generate_move_predictor_signals(
    cfg: Config,
    panel: pd.DataFrame,
    model: MovePredictorModel,
    as_of_date: date,
    instruments: list[Instrument],
    *,
    excluded_symbols: set[str] | None = None,
) -> list[Signal]:
    """
    Generate long signals for *as_of_date*.

    Timing (no lookahead):
    - Features use data through previous close (``*_lag1`` columns on row *as_of_date*).
    - Entry price is today's open (``entry_open``).

    Args:
        excluded_symbols: Symbols currently in SL-cooldown; excluded from candidates.
    """
    mp = cfg._raw.get("move_predictor", {})
    top_n = int(mp.get("top_n", cfg.entry.get("top_n_candidates", 5)))
    min_vol = float(mp.get("min_volume_ratio", 1.0))
    min_score = float(mp.get("min_model_score", 0.0))
    require_positive_ev = bool(mp.get("require_positive_ev", False))
    horizon_key = str(mp.get("horizon", "swing")).lower()
    horizon = Horizon.POSITIONAL if horizon_key == "positional" else Horizon.SWING

    # Entry-filter thresholds (post-analysis improvements)
    ef = cfg._raw.get("entry_filters", {})
    max_20d_ret: float | None = float(ef["max_20d_return"]) if "max_20d_return" in ef else None
    max_ext_200sma: float | None = float(ef["max_ext_from_200sma"]) if "max_ext_from_200sma" in ef else None

    today = panel[panel["date"] == as_of_date].copy()
    if today.empty or not model.fitted:
        return []

    # SL-cooldown: drop symbols that recently stopped out
    if excluded_symbols:
        today = today[~today["symbol"].isin(excluded_symbols)]
    if today.empty:
        return []

    fund_screen = FundamentalScreenConfig.from_config(cfg)
    if fund_screen.enabled:
        mask = today.apply(lambda row: passes_fundamental_screen(row, fund_screen), axis=1)
        today = today.loc[mask]
    if today.empty:
        return []

    vol_col = "volume_ratio_20d_lag1"
    if vol_col in today.columns:
        today = today[today[vol_col] >= min_vol]
    if today.empty:
        return []

    # ── Extension filters (uses only prior-day data → no lookahead) ──────────
    if max_20d_ret is not None and "ret_20d_lag1" in today.columns:
        today = today[today["ret_20d_lag1"].fillna(0.0) <= max_20d_ret]
    if max_ext_200sma is not None and "close_sma_200d_lag1" in today.columns:
        today = today[today["close_sma_200d_lag1"].fillna(0.0) <= max_ext_200sma]
    if today.empty:
        return []

    # ── BSE result blackout: skip stocks that filed earnings in the last 3 days ──
    # Post-results sell-offs are a common source of loss; avoid entry until the
    # dust settles.  Feature defaults to 0 when BSE data is unavailable.
    if "bse_result_blackout_lag1" in today.columns:
        today = today[today["bse_result_blackout_lag1"].fillna(0.0) < 1.0]
    if today.empty:
        return []

    try:
        scores = model.predict_proba(today)
    except Exception as exc:
        logger.warning("predict_proba failed on %s: %s", as_of_date, exc)
        return []

    today = today.copy()
    today["_score"] = scores
    if min_score > 0:
        today = today[today["_score"] >= min_score]
    if today.empty:
        return []

    candidates = today.nlargest(top_n, "_score")
    token_map = {i.instrument_token: i for i in instruments}
    exit_policy = ExitPolicy(cfg)
    signals: list[Signal] = []

    for _, row in candidates.iterrows():
        token = int(row["instrument_token"])
        inst = token_map.get(token)
        if inst is None:
            continue

        entry_price = float(row.get("entry_open", 0.0))
        atr = float(row.get("atr_14_lag1", 0.0))
        if entry_price <= 0 or atr <= 0:
            continue

        win_prob = float(row["_score"])
        cost = estimate_cost_per_share(cfg, entry_price)
        sig = exit_policy.build_signal(
            instrument=inst,
            horizon=horizon,
            entry_price=entry_price,
            atr=atr,
            win_prob=win_prob,
            rank_score=win_prob,
            signal_date=as_of_date,
            cost_per_share=cost,
            require_positive_ev=require_positive_ev,
        )
        if sig is not None:
            sig.features["strategy"] = "move_predictor"
            sig.features["model_score"] = round(win_prob, 4)
            sig.features["volume_ratio_lag1"] = round(float(row.get("volume_ratio_20d_lag1", float("nan"))), 2)
            sig.features["rsi_lag1"] = round(float(row.get("rsi_14_lag1", float("nan"))), 1)
            sig.features["ret_20d_lag1"] = round(float(row.get("ret_20d_lag1", float("nan"))), 4)
            sig.features["close_sma20_lag1"] = round(float(row.get("close_sma_20d_lag1", float("nan"))), 4)
            sig.features["volatility_20d_lag1"] = round(float(row.get("volatility_20d_lag1", float("nan"))), 4)
            sig.features["gap_risk_lag1"] = round(float(row.get("gap_risk_lag1", float("nan"))), 4)
            sig.features["roce"] = row.get("f_roce_lag1")
            sig.features["debt_equity"] = row.get("f_debt_equity_lag1")
            sig.features["profit_growth_yoy"] = row.get("f_profit_growth_yoy_lag1")
            sig.features["profit_growth_qtr"] = row.get("f_profit_growth_qtr_lag1")
            sig.features["pe"] = row.get("f_pe_lag1")
            sig.features["above_dma50"] = row.get("above_dma_lag1")
            sig.features["above_dma200"] = row.get("above_trend_dma_lag1")
            sig.features["ret_20d"] = round(float(row.get("ret_20d_lag1", float("nan"))), 4)
            sig.features["ext_200sma"] = round(float(row.get("close_sma_200d_lag1", float("nan"))), 4)
            signals.append(sig)

    signals.sort(key=lambda s: s.rank_score, reverse=True)
    return signals
