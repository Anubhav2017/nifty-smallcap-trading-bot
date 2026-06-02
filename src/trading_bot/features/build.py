"""Module-level feature builder used by walk-forward and training."""

from __future__ import annotations

from datetime import date

import pandas as pd

from trading_bot.config import Config
from trading_bot.data.loader import instruments_for_range, load_index_ohlcv
from trading_bot.data.universe import Universe
from trading_bot.features.pipeline import FeaturePipeline


def _add_rank_label(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional relevance grades (0–4) for LambdaRank training."""
    out = df.copy()
    label_col = "fwd_ret_swing"
    if label_col not in out.columns or out.empty:
        out["rank_label"] = 0
        return out

    out["rank_label"] = (
        out.groupby("date")[label_col]
        .rank(pct=True, na_option="bottom")
        .mul(4)
        .fillna(0)
        .astype(int)
    )
    return out


def _synthetic_index_from_stocks(ohlcv_by_token: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight mean close by date when index benchmark cache is missing."""
    parts = [df[["date", "close"]] for df in ohlcv_by_token.values() if not df.empty]
    if not parts:
        return pd.DataFrame(columns=["date", "close"])
    combined = pd.concat(parts, ignore_index=True)
    return combined.groupby("date", as_index=False)["close"].mean()


def _normalize_index_df(index_df: pd.DataFrame | None) -> pd.DataFrame:
    """Ensure index OHLCV has ``date`` and ``close`` columns."""
    if index_df is None or index_df.empty:
        return pd.DataFrame(columns=["date", "close"])
    out = index_df.reset_index()
    if "date" not in out.columns:
        if out.index.name == "date":
            out = out.reset_index()
        elif "index" in out.columns:
            out = out.rename(columns={"index": "date"})
    if "close" not in out.columns:
        return pd.DataFrame(columns=["date", "close"])
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out[["date", "close"]].sort_values("date").reset_index(drop=True)


def build(
    cfg: Config,
    ohlcv_by_token: dict[int, pd.DataFrame],
    dates: list[date],
    include_labels: bool = True,
) -> pd.DataFrame:
    """Build features (and labels) for *dates* from cached OHLCV."""
    if not ohlcv_by_token or not dates:
        return pd.DataFrame()

    start, end = min(dates), max(dates)
    universe = Universe(cfg)
    kite_df = universe.load_kite_instruments()
    instruments = instruments_for_range(universe, start, end, kite_df)
    if not instruments:
        return pd.DataFrame()

    index_df = _normalize_index_df(load_index_ohlcv(start, end))
    if index_df.empty:
        index_df = _synthetic_index_from_stocks(ohlcv_by_token)

    pipeline = FeaturePipeline(cfg)
    feature_df = pipeline.build(
        ohlcv_by_token,
        index_df,
        instruments,
        include_labels=include_labels,
    )
    if feature_df.empty:
        return feature_df

    adtv_min = float(cfg.universe.get("liquidity_filter_adtv_cr", 2.0))
    lookback = int(cfg.universe.get("liquidity_lookback_days", 20))
    feature_df = pipeline.apply_liquidity_filter(feature_df, adtv_min, lookback)

    if include_labels:
        feature_df = _add_rank_label(feature_df)

    date_set = set(dates)
    feature_df = feature_df[feature_df["date"].isin(date_set)].copy()
    return feature_df.reset_index(drop=True)
