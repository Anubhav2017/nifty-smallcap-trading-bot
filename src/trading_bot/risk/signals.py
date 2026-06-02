"""Signal generation for walk-forward backtests."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from trading_bot.config import Config
from trading_bot.models.classifier import EntryClassifier
from trading_bot.models.ranker import StockRanker
from trading_bot.models.training import ModelBundle
from trading_bot.risk.engine import RiskEngine
from trading_bot.types import Horizon, Instrument, Signal

logger = logging.getLogger(__name__)


def _estimate_cost_per_share(cfg: Config, price: float) -> float:
    costs = cfg.costs
    brokerage = price * costs.get("brokerage_pct", 0.03) / 100.0
    stt = price * costs.get("stt_delivery_pct", 0.1) / 100.0
    stamp = price * costs.get("stamp_duty_pct", 0.015) / 100.0
    exchange = price * costs.get("exchange_txn_charge_pct", 0.00345) / 100.0
    sebi = price * costs.get("sebi_turnover_fee_pct", 0.0001) / 100.0
    gst = (brokerage + exchange) * costs.get("gst_on_charges_pct", 18.0) / 100.0
    slippage = price * costs.get("slippage_min_pct", 0.05) / 100.0
    return brokerage + stt + stamp + exchange + sebi + gst + slippage


def generate_signals(
    model: ModelBundle,
    feature_df: pd.DataFrame,
    as_of_date: date,
    risk_engine: RiskEngine,
) -> list[Signal]:
    """Score candidates for *as_of_date* using a trained :class:`ModelBundle`."""
    del risk_engine  # gating happens in BacktestEngine after signal generation

    if not model.ranker._fitted or model.exit_policy is None:
        return []

    today = feature_df[feature_df["date"] == as_of_date].copy()
    if today.empty:
        return []

    if not all(c in today.columns for c in StockRanker.FEATURE_COLS):
        return []

    scores = model.ranker.predict(today)
    today = today.copy()
    today["_rank_score"] = scores

    cfg = model.ranker.cfg
    top_n = int(cfg.entry.get("top_n_candidates", 10))
    candidates = today.nlargest(top_n, "_rank_score")
    min_win_prob = float(cfg.entry.get("min_win_prob", 0.55))

    token_to_inst: dict[int, Instrument] = {}
    for _, row in today.iterrows():
        token = int(row["instrument_token"])
        token_to_inst[token] = Instrument(
            symbol=str(row["symbol"]),
            isin=str(row["isin"]),
            instrument_token=token,
        )

    signals: list[Signal] = []
    for _, row in candidates.iterrows():
        token = int(row["instrument_token"])
        inst = token_to_inst.get(token)
        if inst is None:
            continue

        entry_price = float(row.get("close", 0.0))
        atr = float(row.get("atr_14", 0.0))
        if entry_price <= 0 or atr <= 0:
            continue

        rank_score = float(row["_rank_score"])
        cost_per_share = _estimate_cost_per_share(cfg, entry_price)
        row_df = pd.DataFrame([row])

        for horizon in Horizon:
            clf = model.classifiers.get(horizon)
            if clf is None or not clf._fitted:
                continue
            if not all(c in row_df.columns for c in EntryClassifier.FEATURE_COLS):
                continue

            win_prob = float(clf.predict_proba(row_df)[0])
            if win_prob < min_win_prob:
                continue

            sig = model.exit_policy.build_signal(
                instrument=inst,
                horizon=horizon,
                entry_price=entry_price,
                atr=atr,
                win_prob=win_prob,
                rank_score=rank_score,
                signal_date=as_of_date,
                cost_per_share=cost_per_share,
            )
            if sig is not None:
                signals.append(sig)

    signals.sort(key=lambda s: s.expected_value, reverse=True)
    return signals
