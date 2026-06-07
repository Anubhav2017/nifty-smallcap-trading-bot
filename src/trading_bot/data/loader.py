"""OHLCV loading for backtest and training pipelines."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from trading_bot.config import Config
from trading_bot.data.dataset_store import dataset_root_from_config, load_daily_bars
from trading_bot.data.universe import Universe
from trading_bot.types import Instrument

logger = logging.getLogger(__name__)


def instruments_for_range(
    universe: Universe,
    start: date,
    end: date,
    kite_instruments: pd.DataFrame,
) -> list[Instrument]:
    """Union of index constituents between *start* and *end* (point-in-time)."""
    isins: set[str] = set()
    for ts in pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end)):
        isins.update(universe.get_constituents(ts.date()))

    if not isins or kite_instruments.empty:
        return []

    lookup = kite_instruments.set_index("isin")
    instruments: list[Instrument] = []
    for isin in sorted(isins):
        if isin not in lookup.index:
            continue
        rows = lookup.loc[[isin]]
        nse_eq = rows[
            (rows.get("exchange", pd.Series(dtype=str)) == "NSE")
            & (rows.get("instrument_type", pd.Series(dtype=str)) == "EQ")
        ]
        row = nse_eq.iloc[0] if not nse_eq.empty else rows.iloc[0]
        instruments.append(
            Instrument(
                symbol=str(row.get("tradingsymbol", row.get("symbol", ""))),
                isin=isin,
                instrument_token=int(row["instrument_token"]),
                exchange=str(row.get("exchange", "NSE")),
            )
        )
    return instruments


def load_ohlcv(start: date, end: date, cfg: Config | None = None) -> dict[int, pd.DataFrame]:
    """Load daily OHLCV for all universe instruments in [start, end].

    Reads ``{dataset_root}/ohlcv/day/{SYMBOL}.csv``.
    """
    cfg = cfg or Config(None)
    root = dataset_root_from_config(cfg)
    universe = Universe(cfg)
    kite_df = universe.load_kite_instruments()
    instruments = instruments_for_range(universe, start, end, kite_df)

    if not instruments:
        logger.warning("No universe instruments found for [%s, %s].", start, end)
        return {}

    fetch_start = start - timedelta(days=400)
    ohlcv_by_token: dict[int, pd.DataFrame] = {}

    for inst in instruments:
        df = load_daily_bars(inst.symbol, fetch_start, end, root)
        if not df.empty:
            ohlcv_by_token[inst.instrument_token] = df

    logger.info(
        "Loaded OHLCV for %d/%d instruments [%s, %s] from %s.",
        len(ohlcv_by_token),
        len(instruments),
        start,
        end,
        root,
    )
    return ohlcv_by_token


def load_index_ohlcv(start: date, end: date, cfg: Config | None = None) -> pd.DataFrame | None:
    """Load index benchmark daily OHLCV when an index symbol exists in the dataset."""
    cfg = cfg or Config(None)
    root = dataset_root_from_config(cfg)
    universe = Universe(cfg)
    kite_df = universe.load_kite_instruments()
    index_token = universe.get_index_instrument_token(kite_df)
    if index_token is None:
        return None

    row = kite_df[kite_df["instrument_token"] == index_token]
    if row.empty:
        return None
    symbol = str(row.iloc[0]["tradingsymbol"])

    fetch_start = start - timedelta(days=30)
    df = load_daily_bars(symbol, fetch_start, end, root)
    if df.empty:
        return None

    return df.set_index("date").sort_index()
