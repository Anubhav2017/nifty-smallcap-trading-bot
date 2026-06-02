"""Point-in-time index universe lookup backed by dataset_*/universe/."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from trading_bot.config import Config
from trading_bot.data.dataset_store import (
    dataset_root_from_config,
    load_instruments,
    load_manifest,
    load_universe,
)
from trading_bot.data.universe_registry import IndexSpec, get_index_spec
from trading_bot.types import Instrument

logger = logging.getLogger(__name__)

_MEMBERSHIP_COLS = ["effective_date", "symbol", "isin", "action"]
_INSTRUMENTS_COLS = [
    "instrument_token",
    "exchange_token",
    "tradingsymbol",
    "name",
    "last_price",
    "expiry",
    "strike",
    "tick_size",
    "lot_size",
    "instrument_type",
    "segment",
    "exchange",
    "isin",
]


class Universe:
    """Manages point-in-time membership from a built dataset (``dataset_*``)."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._spec: IndexSpec = get_index_spec(cfg)
        self._root = dataset_root_from_config(cfg)
        self._membership = self._load_membership()

    def index_spec(self) -> IndexSpec:
        return self._spec

    def dataset_root(self) -> Path:
        return self._root

    def get_constituents(self, as_of: date) -> list[str]:
        """Return ISINs in the index on *as_of* (point-in-time membership)."""
        if self._membership.empty:
            return []

        relevant = self._membership[self._membership["effective_date"] <= as_of].copy()
        relevant = relevant.sort_values("effective_date")

        active: set[str] = set()
        for _, row in relevant.iterrows():
            isin: str = str(row["isin"])
            action: str = str(row["action"]).strip().lower()
            if action == "add":
                active.add(isin)
            elif action == "remove":
                active.discard(isin)
            else:
                logger.warning("Unknown membership action '%s' for ISIN %s", action, isin)

        return sorted(active)

    def get_instruments(
        self,
        as_of: date,
        kite_instruments: pd.DataFrame,
    ) -> list[Instrument]:
        """Return ``Instrument`` objects for all constituents on *as_of*."""
        isins = self.get_constituents(as_of)
        if not isins:
            return []

        if kite_instruments.empty or "isin" not in kite_instruments.columns:
            logger.warning(
                "Instruments DataFrame is empty or missing 'isin' column; returning no instruments."
            )
            return []

        lookup = kite_instruments.set_index("isin")
        instruments: list[Instrument] = []

        for isin in isins:
            if isin not in lookup.index:
                logger.warning("No instrument row for ISIN/symbol %s", isin)
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

    def load_kite_instruments(self) -> pd.DataFrame:
        """Load instrument metadata from the active dataset."""
        enriched = load_universe(self._root)
        nse_eq = load_instruments(self._root)

        if enriched.empty:
            logger.warning("Universe enriched CSV missing under %s", self._root)
            return pd.DataFrame(columns=_INSTRUMENTS_COLS)

        rows: list[dict] = []
        nse_lookup = (
            nse_eq.set_index("tradingsymbol") if not nse_eq.empty and "tradingsymbol" in nse_eq.columns else None
        )

        for _, row in enriched.iterrows():
            if "found" in row.index and not bool(row.get("found", True)):
                continue
            symbol = str(row["symbol"]).strip().upper()
            token = int(row["instrument_token"])
            inst_row: dict = {
                "instrument_token": token,
                "exchange_token": row.get("exchange_token", 0),
                "tradingsymbol": symbol,
                "name": row.get("name", symbol),
                "last_price": 0,
                "expiry": "",
                "strike": 0,
                "tick_size": 0.05,
                "lot_size": 1,
                "instrument_type": "EQ",
                "segment": "NSE",
                "exchange": str(row.get("exchange", "NSE")),
                "isin": symbol,
            }
            if nse_lookup is not None and symbol in nse_lookup.index:
                extra = nse_lookup.loc[symbol]
                if isinstance(extra, pd.DataFrame):
                    extra = extra.iloc[0]
                for col in _INSTRUMENTS_COLS:
                    if col in extra.index and pd.notna(extra[col]):
                        inst_row[col] = extra[col]
                if "isin" not in extra.index or not str(extra.get("isin", "")).strip():
                    inst_row["isin"] = symbol
            rows.append(inst_row)

        return pd.DataFrame(rows)[_INSTRUMENTS_COLS] if rows else pd.DataFrame(columns=_INSTRUMENTS_COLS)

    def get_index_instrument_token(self, kite_instruments: pd.DataFrame) -> int | None:
        """Return benchmark index token if present in instrument metadata."""
        if kite_instruments.empty or "tradingsymbol" not in kite_instruments.columns:
            return None

        ts_col = kite_instruments["tradingsymbol"].astype(str).str.upper()
        for pattern in self._spec.index_patterns:
            mask = ts_col.str.contains(pattern.upper(), regex=False)
            matches = kite_instruments[mask]
            if not matches.empty:
                token = int(matches.iloc[0]["instrument_token"])
                logger.info(
                    "Found %s index token=%d (tradingsymbol=%s)",
                    self._spec.key,
                    token,
                    matches.iloc[0]["tradingsymbol"],
                )
                return token

        logger.debug(
            "No %s index instrument in dataset (patterns: %s).",
            self._spec.key,
            self._spec.index_patterns,
        )
        return None

    def _load_membership(self) -> pd.DataFrame:
        try:
            enriched = load_universe(self._root)
            manifest = load_manifest(self._root)
        except FileNotFoundError as exc:
            logger.warning("Dataset not found at %s: %s", self._root, exc)
            return pd.DataFrame(columns=_MEMBERSHIP_COLS)

        if enriched.empty:
            return pd.DataFrame(columns=_MEMBERSHIP_COLS)

        effective = date.fromisoformat(str(manifest["date_range"]["from"]))
        rows = []
        for _, row in enriched.iterrows():
            if "found" in row.index and not bool(row.get("found", True)):
                continue
            symbol = str(row["symbol"]).strip().upper()
            rows.append(
                {
                    "effective_date": effective,
                    "symbol": symbol,
                    "isin": symbol,
                    "action": "add",
                }
            )

        df = pd.DataFrame(rows)
        return df[_MEMBERSHIP_COLS] if not df.empty else pd.DataFrame(columns=_MEMBERSHIP_COLS)
