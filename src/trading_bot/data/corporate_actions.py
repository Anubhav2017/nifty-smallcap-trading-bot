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


_ACTION_COLS = ["symbol", "ex_date", "action", "ratio", "notes"]


def load_corporate_actions(path: Path | None = None) -> pd.DataFrame:
    """
    Load split/bonus actions from CSV.

    Columns: symbol, ex_date, action (bonus|split), ratio (new_shares/old_shares), notes
    """
    if path is None:
        return pd.DataFrame(columns=_ACTION_COLS)

    csv_path = Path(path)
    if not csv_path.is_file():
        return pd.DataFrame(columns=_ACTION_COLS)

    df = pd.read_csv(csv_path, dtype={"symbol": str}, comment="#")
    missing = [c for c in _ACTION_COLS[:4] if c not in df.columns]
    if missing:
        raise ValueError(f"Corporate actions CSV missing columns: {missing}")
    df["symbol"] = df["symbol"].str.upper()
    df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.normalize()
    df["action"] = df["action"].str.lower()
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    if "notes" not in df.columns:
        df["notes"] = ""
    return df[_ACTION_COLS].dropna(subset=["symbol", "ex_date", "ratio"]).reset_index(drop=True)


def infer_actions_from_shares(
    symbol: str,
    shares_df: pd.DataFrame,
    bonus_df: pd.DataFrame | None = None,
    *,
    min_ratio: float = 1.05,
) -> pd.DataFrame:
    """Infer bonus/split events when share count jumps between filings."""
    if shares_df.empty or len(shares_df) < 2:
        return pd.DataFrame(columns=_ACTION_COLS)

    work = shares_df.copy()
    work["report_date"] = pd.to_datetime(work["report_date"]).dt.normalize()
    work = work.sort_values("report_date").reset_index(drop=True)
    rows: list[dict] = []

    for i in range(1, len(work)):
        prev, curr = work.iloc[i - 1], work.iloc[i]
        if prev["shares"] <= 0:
            continue
        ratio = float(curr["shares"]) / float(prev["shares"])
        if ratio < min_ratio:
            continue
        action = "bonus" if ratio <= 3.0 else "split"
        rows.append(
            {
                "symbol": symbol.upper(),
                "ex_date": curr["report_date"],
                "action": action,
                "ratio": ratio,
                "notes": f"inferred:shares {prev['shares']:,.0f}->{curr['shares']:,.0f}",
            }
        )

    if bonus_df is not None and not bonus_df.empty:
        bonus_df = bonus_df.copy()
        bonus_df["report_date"] = pd.to_datetime(bonus_df["report_date"]).dt.normalize()
        for _, brow in bonus_df.iterrows():
            bonus_shares = brow.get("bonus_shares")
            if pd.isna(bonus_shares) or bonus_shares <= 0:
                continue
            match_idx = work.index[work["report_date"] == brow["report_date"]]
            if len(match_idx) == 0 or match_idx[0] == 0:
                continue
            idx = match_idx[0]
            prev_shares = float(work.iloc[idx - 1]["shares"])
            curr_shares = float(work.iloc[idx]["shares"])
            if prev_shares <= 0:
                continue
            ratio = curr_shares / prev_shares
            if ratio >= min_ratio:
                rows.append(
                    {
                        "symbol": symbol.upper(),
                        "ex_date": brow["report_date"],
                        "action": "bonus",
                        "ratio": ratio,
                        "notes": f"screener_bonus:{bonus_shares:,.0f}",
                    }
                )

    if not rows:
        return pd.DataFrame(columns=_ACTION_COLS)
    out = pd.DataFrame(rows).drop_duplicates(subset=["symbol", "ex_date"], keep="last")
    return out.sort_values("ex_date").reset_index(drop=True)


def merge_corporate_actions(manual: pd.DataFrame, inferred: pd.DataFrame) -> pd.DataFrame:
    """Manual CSV overrides inferred rows on the same symbol+ex_date."""
    parts = [df for df in (inferred, manual) if not df.empty]
    if not parts:
        return pd.DataFrame(columns=_ACTION_COLS)
    merged = pd.concat(parts, ignore_index=True)
    merged["ex_date"] = pd.to_datetime(merged["ex_date"]).dt.normalize()
    merged = merged.sort_values(["symbol", "ex_date"])
    return merged.drop_duplicates(subset=["symbol", "ex_date"], keep="last").reset_index(drop=True)


def adjust_ohlcv(bars: pd.DataFrame, actions: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Add ``close_adj``, ``open_adj``, ``high_adj``, ``low_adj``, ``volume_adj``.

    Backward adjustment: latest raw prices unchanged; history scaled for splits/bonus.
    """
    out = bars.copy()
    for col in ("close_adj", "open_adj", "high_adj", "low_adj", "volume_adj"):
        out[col] = pd.NA

    if bars.empty:
        return out

    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out.sort_values("date").reset_index(drop=True)

    sub = actions[actions["symbol"] == symbol.upper()].copy()
    if sub.empty:
        for raw, adj in (("open", "open_adj"), ("high", "high_adj"), ("low", "low_adj"), ("close", "close_adj")):
            out[adj] = out[raw]
        out["volume_adj"] = out["volume"]
        return out

    sub["ex_date"] = pd.to_datetime(sub["ex_date"]).dt.normalize()
    sub = sub.sort_values("ex_date")

    price_mult = pd.Series(1.0, index=out.index)
    vol_mult = pd.Series(1.0, index=out.index)
    for _, row in sub.iterrows():
        ratio = float(row["ratio"])
        if ratio <= 0:
            continue
        mask = out["date"] < row["ex_date"]
        price_mult.loc[mask] /= ratio
        vol_mult.loc[mask] *= ratio

    for raw, adj in (("open", "open_adj"), ("high", "high_adj"), ("low", "low_adj"), ("close", "close_adj")):
        out[adj] = out[raw] * price_mult
    out["volume_adj"] = out["volume"] * vol_mult
    return out


def load_dataset_corporate_actions(
    root: Path,
    symbol: str,
    shares_df: pd.DataFrame,
    bonus_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ``corporate_actions.csv``, per-symbol CSV, and inferred share-count events."""
    root = Path(root)
    manual = load_corporate_actions(root / "corporate_actions.csv")
    per_symbol = root / "corporate_actions" / f"{symbol.upper()}.csv"
    if per_symbol.is_file():
        manual = pd.concat([manual, load_corporate_actions(per_symbol)], ignore_index=True)
    inferred = infer_actions_from_shares(symbol, shares_df, bonus_df)
    return merge_corporate_actions(manual, inferred)
