"""Thin helper for corporate action bookkeeping."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DIVIDEND_COLS = ["ex_date", "isin", "amount"]


def load_dividend_dates(path: Path | None = None) -> pd.DataFrame:
    """Load ex-dividend dates from CSV.

    When *path* is omitted or the file is missing, returns an empty DataFrame.
    """
    if path is None:
        return pd.DataFrame(columns=_DIVIDEND_COLS)

    csv_path = Path(path)

    if not csv_path.exists():
        logger.info(
            "Dividend dates file not found at %s; returning empty DataFrame.",
            csv_path,
        )
        return pd.DataFrame(columns=_DIVIDEND_COLS)

    try:
        df = pd.read_csv(csv_path, dtype={"isin": str})
        df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.date
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        missing = [c for c in _DIVIDEND_COLS if c not in df.columns]
        if missing:
            logger.error(
                "Dividend CSV is missing expected columns: %s. Returning empty DataFrame.",
                missing,
            )
            return pd.DataFrame(columns=_DIVIDEND_COLS)
        return df[_DIVIDEND_COLS].reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load dividend dates from %s: %s", csv_path, exc)
        return pd.DataFrame(columns=_DIVIDEND_COLS)


def is_ex_dividend_day(isin: str, check_date: date, dividend_df: pd.DataFrame) -> bool:
    """Return True if *check_date* is an ex-dividend date for *isin*.

    Parameters
    ----------
    isin:
        ISIN of the security to check.
    check_date:
        The date to look up.
    dividend_df:
        DataFrame as returned by :func:`load_dividend_dates`.
    """
    if dividend_df.empty:
        return False

    mask = (dividend_df["isin"] == isin) & (dividend_df["ex_date"] == check_date)
    return bool(mask.any())
