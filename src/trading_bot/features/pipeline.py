"""Orchestrates feature and label computation across the universe."""

from __future__ import annotations

import pandas as pd

from trading_bot.config import Config
from trading_bot.types import Instrument
from trading_bot.features.indicators import add_all_features
from trading_bot.features.labels import add_all_labels


class FeaturePipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def build(
        self,
        ohlcv_by_token: dict[int, pd.DataFrame],
        index_df: pd.DataFrame,
        instruments: list[Instrument],
        include_labels: bool = True,
    ) -> pd.DataFrame:
        """Build the full feature (and optionally label) DataFrame for all instruments.

        Returns a long DataFrame sorted by (date, instrument_token).
        """
        token_to_instrument: dict[int, Instrument] = {
            inst.instrument_token: inst for inst in instruments
        }

        frames: list[pd.DataFrame] = []
        for token, ohlcv in ohlcv_by_token.items():
            inst = token_to_instrument.get(token)
            if inst is None:
                continue

            frame = add_all_features(ohlcv, index_df)
            if include_labels:
                frame = add_all_labels(frame, self.cfg)

            frame = frame.copy()
            frame["instrument_token"] = token
            frame["symbol"] = inst.symbol
            frame["isin"] = inst.isin
            frames.append(frame)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["date", "instrument_token"]).reset_index(drop=True)
        return combined

    def apply_liquidity_filter(
        self,
        feature_df: pd.DataFrame,
        adtv_cr_min: float,
        lookback_days: int,
    ) -> pd.DataFrame:
        """Remove instruments whose recent ADTV (in Crore INR) is below the threshold.

        ADTV = mean(close * volume / 1e7) over the last `lookback_days` sessions
        per instrument, computed on whatever rows are present in feature_df.
        """
        feature_df = feature_df.copy()

        feature_df["_dv"] = feature_df["close"] * feature_df["volume"] / 1e7

        adtv = (
            feature_df.groupby("instrument_token")["_dv"]
            .apply(lambda s: s.iloc[-lookback_days:].mean())
        )

        liquid_tokens = adtv[adtv >= adtv_cr_min].index
        feature_df = feature_df[feature_df["instrument_token"].isin(liquid_tokens)].copy()
        feature_df = feature_df.drop(columns=["_dv"])
        return feature_df
