"""BSE announcement-based features for the move predictor.

All features use ANNOUNCEMENT DATE (NEWS_DT) — the date BSE published the
disclosure.  Since we only look backward (news published before trade date),
there is zero lookahead bias.

Five new binary features per symbol per day:

  bse_result_blackout   – Financial result filed in last 3 calendar days (0/1)
                          → entry avoidance: stocks sell off post-results
  bse_bulk_buy_last5d   – Bulk acquisition disclosure (SAST Reg 10) in
                          last 5 calendar days                     (0/1)
                          → +3.5 % average next-day move observed
  bse_promoter_buy_7d   – Promoter stake increase (SAST Reg 29(1)) in
                          last 7 calendar days                     (0/1)
                          → +0.88 % average next-day move observed
  bse_corp_action_5d    – Bonus/dividend/record-date/split announced
                          in last 5 calendar days                  (0/1)
  bse_window_closed     – Trading-window closure announcement in
                          last 10 calendar days (pre-results blackout) (0/1)
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Category / subcategory keyword maps ──────────────────────────────────────

_RESULT_SUBCATS = {"Financial Results"}

_BULK_BUY_KEYWORDS = (
    r"10\(7\)", r"10\(5\)", r"10\(6\)", "creeping", "open offer", "acquisition of shares",
)

_PROMOTER_BUY_KEYWORDS = (r"29\(1\)",)

_CORP_ACTION_SUBCATS = {
    "Bonus", "Dividend", "Record Date",
    "Sub-division / Stock Split", "Amalgamation / Merger / Demerger",
}

_WINDOW_CLOSURE_SUBCATS = {"Closure of Trading Window"}


def _load_announcements(ann_dir: Path, symbol: str) -> pd.DataFrame:
    path = ann_dir / symbol / "announcements.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, usecols=["NEWS_DT", "SUBCATNAME"], dtype=str)
        df["ann_date"] = pd.to_datetime(df["NEWS_DT"], format="mixed", errors="coerce").dt.normalize()
        return df.dropna(subset=["ann_date"])
    except Exception:
        return pd.DataFrame()


def _rolling_flag(
    panel_dates: pd.Series,  # Series of Timestamps (datetime64)
    event_dates: pd.Series,  # Series of Timestamps (sorted)
    window_days: int,
) -> pd.Series:
    """Return 1.0 if ANY event_date falls in [d - window_days, d-1], else 0.0.

    Closed on both ends: an event on exactly (d - window_days) qualifies.
    """
    if event_dates.empty:
        return pd.Series(0.0, index=panel_dates.index)

    # Convert to integer day counts (days since epoch) for unit-safe arithmetic
    def _to_day_int(s: pd.Series) -> np.ndarray:
        """Convert Series of Timestamps → integer number of days since epoch."""
        return pd.to_datetime(s).values.astype("datetime64[D]").astype(np.int32)

    ev_days = np.sort(_to_day_int(event_dates))
    pd_days = _to_day_int(panel_dates)

    def _check(d_day: int) -> float:
        lo = d_day - window_days   # inclusive start
        hi = d_day - 1             # inclusive end (yesterday)
        i = np.searchsorted(ev_days, lo, side="left")
        j = np.searchsorted(ev_days, hi, side="right")
        return 1.0 if i < j else 0.0

    return pd.Series([_check(d) for d in pd_days], index=panel_dates.index, dtype=float)


def build_bse_event_features(
    panel: pd.DataFrame,
    ann_dir: str | Path,
) -> pd.DataFrame:
    """Compute BSE announcement features and left-join onto the OHLCV panel.

    Parameters
    ----------
    panel : DataFrame
        Full lagged-features panel with columns ``symbol`` and ``date``
        (Python ``datetime.date`` objects).
    ann_dir : path
        Root of bse_announcements, e.g.
        ``dataset_smallcap250/bse_announcements``.

    Returns
    -------
    DataFrame
        Same panel with five new float columns (0.0 / 1.0).
    """
    ann_dir = Path(ann_dir)

    new_cols = [
        "bse_result_blackout",
        "bse_bulk_buy_last5d",
        "bse_promoter_buy_7d",
        "bse_corp_action_5d",
        "bse_window_closed",
    ]

    panel = panel.copy()
    for col in new_cols:
        panel[col] = 0.0

    if not ann_dir.exists():
        return panel

    symbols = panel["symbol"].unique().tolist()

    for sym in symbols:
        ann = _load_announcements(ann_dir, sym)
        if ann.empty:
            continue

        sym_mask = panel["symbol"] == sym
        sym_rows = panel.loc[sym_mask].copy()

        # Convert date column to DatetimeIndex for vectorised ops
        sym_rows["_dt"] = pd.to_datetime(sym_rows["date"])
        dt_series = sym_rows["_dt"]

        def _ev_dates(subcats: set[str] | None, keywords: tuple[str, ...] | None) -> pd.Series:
            mask = pd.Series(False, index=ann.index)
            if subcats:
                mask |= ann["SUBCATNAME"].isin(subcats)
            if keywords:
                sub_lower = ann["SUBCATNAME"].str.lower().fillna("")
                for kw in keywords:
                    mask |= sub_lower.str.contains(kw.lower(), na=False, regex=True)
            return ann.loc[mask, "ann_date"].drop_duplicates().sort_values()

        result_ev = _ev_dates(_RESULT_SUBCATS, None)
        bulk_ev   = _ev_dates(None, _BULK_BUY_KEYWORDS)
        prom_ev   = _ev_dates(None, _PROMOTER_BUY_KEYWORDS)
        corp_ev   = _ev_dates(_CORP_ACTION_SUBCATS, None)
        window_ev = _ev_dates(_WINDOW_CLOSURE_SUBCATS, None)

        panel.loc[sym_mask, "bse_result_blackout"] = _rolling_flag(dt_series, result_ev, 3).values
        panel.loc[sym_mask, "bse_bulk_buy_last5d"] = _rolling_flag(dt_series, bulk_ev,   5).values
        panel.loc[sym_mask, "bse_promoter_buy_7d"] = _rolling_flag(dt_series, prom_ev,   7).values
        panel.loc[sym_mask, "bse_corp_action_5d"]  = _rolling_flag(dt_series, corp_ev,   5).values
        panel.loc[sym_mask, "bse_window_closed"]   = _rolling_flag(dt_series, window_ev, 10).values

    return panel
