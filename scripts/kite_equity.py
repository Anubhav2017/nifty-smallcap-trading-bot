"""NSE cash-equity helpers for Kite Connect (no F&O)."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from kiteconnect import KiteConnect


def nse_eq_instruments_df(kite: KiteConnect) -> pd.DataFrame:
    """All live NSE equity instruments from Kite instrument dump."""
    rows = kite.instruments("NSE")
    eq = [
        item
        for item in rows
        if item.get("segment") == "NSE" and item.get("instrument_type") == "EQ"
    ]
    return pd.DataFrame(eq)


def build_nse_eq_token_map(kite: KiteConnect) -> Dict[str, int]:
    df = nse_eq_instruments_df(kite)
    return dict(zip(df["tradingsymbol"].astype(str), df["instrument_token"].astype(int)))


def enrich_universe(symbols: List[str], kite: KiteConnect) -> pd.DataFrame:
    """Map symbols to instrument_token and company name."""
    df = nse_eq_instruments_df(kite)
    df["tradingsymbol"] = df["tradingsymbol"].astype(str).str.upper()
    lookup = df.set_index("tradingsymbol")
    records = []
    for sym in symbols:
        sym = sym.upper()
        if sym not in lookup.index:
            records.append(
                {
                    "symbol": sym,
                    "instrument_token": None,
                    "name": None,
                    "exchange": "NSE",
                    "found": False,
                }
            )
            continue
        row = lookup.loc[sym]
        records.append(
            {
                "symbol": sym,
                "instrument_token": int(row["instrument_token"]),
                "name": row.get("name"),
                "exchange": "NSE",
                "found": True,
            }
        )
    return pd.DataFrame(records)
