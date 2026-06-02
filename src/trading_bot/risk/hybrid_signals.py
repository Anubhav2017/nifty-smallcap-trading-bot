"""Hybrid signal generation: daily shortlist + 5m entry timing."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from trading_bot.features.intraday_build import intraday_features_for_session
from trading_bot.models.classifier import EntryClassifier
from trading_bot.models.ranker import StockRanker
from trading_bot.models.training import ModelBundle
from trading_bot.risk.engine import RiskEngine
from trading_bot.risk.signals import _estimate_cost_per_share, generate_signals
from trading_bot.types import Horizon, Instrument, Signal

logger = logging.getLogger(__name__)


def _daily_candidates(
    model: ModelBundle,
    feature_df: pd.DataFrame,
    as_of_date: date,
) -> pd.DataFrame:
    """Return daily-ranked rows that pass the swing classifier (stage 1)."""
    if not model.ranker._fitted or model.exit_policy is None:
        return pd.DataFrame()

    today = feature_df[feature_df["date"] == as_of_date].copy()
    if today.empty or not all(c in today.columns for c in StockRanker.FEATURE_COLS):
        return pd.DataFrame()

    today = today.copy()
    today["_rank_score"] = model.ranker.predict(today)
    cfg = model.ranker.cfg
    top_n = int(cfg.entry.get("top_n_candidates", 10))
    min_win_prob = float(cfg.entry.get("min_win_prob", 0.55))
    candidates = today.nlargest(top_n, "_rank_score")

    clf = model.classifiers.get(Horizon.SWING)
    if clf is None or not clf._fitted:
        return candidates

    approved_rows: list[pd.Series] = []
    for _, row in candidates.iterrows():
        row_df = pd.DataFrame([row])
        if not all(c in row_df.columns for c in EntryClassifier.FEATURE_COLS):
            continue
        if float(clf.predict_proba(row_df)[0]) >= min_win_prob:
            approved_rows.append(row)

    if not approved_rows:
        return pd.DataFrame()
    return pd.DataFrame(approved_rows)


def generate_hybrid_signals(
    model: ModelBundle,
    feature_df: pd.DataFrame,
    as_of_date: date,
    risk_engine: RiskEngine,
) -> list[Signal]:
    """Stage 1: daily ranker/classifier. Stage 2: 5m timing model for entry bar."""
    timing = model.intraday_timing
    if timing is None or not timing._fitted or model.exit_policy is None:
        logger.warning("Hybrid timing model missing; falling back to daily signals.")
        return generate_signals(model, feature_df, as_of_date, risk_engine)

    cfg = model.ranker.cfg
    hybrid = cfg.hybrid or {}
    min_timing_prob = float(hybrid.get("min_timing_prob", 0.55))
    pick = str(hybrid.get("pick", "best")).lower()

    candidates = _daily_candidates(model, feature_df, as_of_date)
    if candidates.empty:
        return []

    signals: list[Signal] = []
    for _, row in candidates.iterrows():
        symbol = str(row["symbol"]).upper()
        token = int(row["instrument_token"])
        inst = Instrument(symbol=symbol, isin=str(row["isin"]), instrument_token=token)

        intraday_df = intraday_features_for_session(cfg, symbol, as_of_date, row)
        if intraday_df.empty:
            continue

        probs = timing.predict_proba(intraday_df)
        intraday_df = intraday_df.copy()
        intraday_df["_timing_prob"] = probs

        if pick == "first":
            hits = intraday_df[intraday_df["_timing_prob"] >= min_timing_prob]
            if hits.empty:
                continue
            entry_row = hits.iloc[0]
        else:
            best = intraday_df.loc[intraday_df["_timing_prob"].idxmax()]
            if float(best["_timing_prob"]) < min_timing_prob:
                continue
            entry_row = best

        entry_price = float(entry_row["close"])
        atr = float(row.get("atr_14", 0.0) or 0.0)
        if entry_price <= 0 or atr <= 0:
            continue

        rank_score = float(row["_rank_score"])
        win_prob = float(entry_row["_timing_prob"])
        cost_per_share = _estimate_cost_per_share(cfg, entry_price)

        sig = model.exit_policy.build_signal(
            instrument=inst,
            horizon=Horizon.SWING,
            entry_price=entry_price,
            atr=atr,
            win_prob=win_prob,
            rank_score=rank_score,
            signal_date=as_of_date,
            cost_per_share=cost_per_share,
        )
        if sig is not None:
            sig.features["entry_datetime"] = entry_row["datetime"].isoformat()
            sig.features["timing_prob"] = win_prob
            sig.features["hybrid"] = True
            signals.append(sig)

    signals.sort(key=lambda s: s.features.get("entry_datetime", ""))
    return signals
