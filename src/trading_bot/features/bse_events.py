"""BSE announcement-based features for the move predictor.

All features use the ANNOUNCEMENT DATE (the date BSE published the disclosure).
Since we only look backward (news published strictly before the trade date),
there is zero lookahead bias.  ``build_lagged_panel`` additionally lags every
column by one more day, so the model only ever sees yesterday's flags at entry.

Source of truth
---------------
Features are computed from the normalised, hardened extraction file produced by
``scripts/extract_announcements_per_stock.py``:

    {ann_dir}/{SYMBOL}/announcements_extracted.csv   (columns: date, type, ...)

The ``type`` column is a clean lower_snake_case event label (results, dividend,
order_win, acquisition, sast, ...) and is far more reliable than the raw BSE
``SUBCATNAME`` string.  If the extracted file is missing for a symbol we fall
back to the raw ``announcements.csv`` (NEWS_DT / SUBCATNAME) so the pipeline
keeps working on partially-extracted datasets.

Feature catalogue (one float column per symbol per day, 0.0 / 1.0 unless noted)
------------------------------------------------------------------------------
  bse_result_blackout         results filed in last 3 days  -> post-results sell-off zone
  bse_bulk_buy_last5d         bulk/creeping acquisition (SAST) in last 5 days
  bse_promoter_buy_7d         promoter stake increase (SAST) in last 7 days
  bse_corp_action_5d          bonus / dividend / split / record-date in last 5 days
  bse_window_closed           trading-window closure in last 10 days (pre-results)
  bse_results_5d              results filed in last 5 days  (broader than blackout)
  bse_earnings_call_5d        earnings call / investor pres / analyst meet in last 5 days
  bse_order_win_10d           order / contract win in last 10 days
  bse_acquisition_10d         acquisition / merger in last 10 days
  bse_capacity_expansion_15d  capacity expansion / commissioning in last 15 days
  bse_credit_rating_10d       credit-rating action in last 10 days
  bse_ann_count_5d            NUMBER of announcements (any type) in last 5 days
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Event-type → feature mapping (against the normalised ``type`` column) ─────
# Each entry: feature column → (set of normalised types, lookback window days).
# bse_ann_count_5d is handled separately because it is a count, not a flag.
_TYPE_FLAG_SPECS: dict[str, tuple[frozenset[str], int]] = {
    "bse_result_blackout":        (frozenset({"results"}), 3),
    "bse_bulk_buy_last5d":        (frozenset({"sast", "acquisition"}), 5),
    "bse_promoter_buy_7d":        (frozenset({"sast", "insider_trading"}), 7),
    "bse_corp_action_5d":         (frozenset({"dividend", "bonus", "split",
                                              "record_date", "rights_issue"}), 5),
    "bse_window_closed":          (frozenset({"trading_window"}), 10),
    "bse_results_5d":             (frozenset({"results"}), 5),
    "bse_earnings_call_5d":       (frozenset({"earnings_call",
                                              "investor_presentation",
                                              "analyst_meet"}), 5),
    "bse_order_win_10d":          (frozenset({"order_win"}), 10),
    "bse_acquisition_10d":        (frozenset({"acquisition", "merger"}), 10),
    "bse_capacity_expansion_15d": (frozenset({"capacity_expansion"}), 15),
    "bse_credit_rating_10d":      (frozenset({"credit_rating"}), 10),
}

# Disclosure-intensity count feature: (column name, lookback days).
_COUNT_FEATURE = ("bse_ann_count_5d", 5)

# All feature column names, in stable order, exported for the model column list.
BSE_FEATURE_COLS: list[str] = list(_TYPE_FLAG_SPECS.keys()) + [_COUNT_FEATURE[0]]


# ── Raw-CSV fallback: map SUBCATNAME keywords → normalised type ───────────────
# Only used when announcements_extracted.csv is absent.  Mirrors the classifier
# in scripts/extract_announcements_per_stock.py closely enough for the feature
# windows that matter here.
_FALLBACK_SUBCAT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("results", ("financial results", "quarterly result")),
    ("dividend", ("dividend",)),
    ("bonus", ("bonus",)),
    ("split", ("sub-division", "stock split", "subdivision")),
    ("rights_issue", ("rights issue",)),
    ("record_date", ("record date",)),
    ("merger", ("amalgamation", "merger", "demerger")),
    ("acquisition", ("acquisition", "open offer")),
    ("sast", ("sast", "29(1)", "10(5)", "10(6)", "10(7)", "creeping")),
    ("insider_trading", ("7(2)", "insider")),
    ("trading_window", ("closure of trading window", "trading window")),
    ("earnings_call", ("earnings call", "conference call", "investor presentation")),
    ("analyst_meet", ("analyst", "investor meet")),
    ("order_win", ("order", "letter of award", "loa", "contract")),
    ("capacity_expansion", ("capacity", "commissioning")),
    ("credit_rating", ("credit rating",)),
]


def _classify_subcat(subcat: str) -> str:
    blob = (subcat or "").lower()
    for label, kws in _FALLBACK_SUBCAT_RULES:
        for kw in kws:
            if kw in blob:
                return label
    return "other"


def _load_announcements(ann_dir: Path, symbol: str) -> pd.DataFrame:
    """Return a frame with columns ``ann_date`` (Timestamp) and ``type`` (str).

    Prefers the hardened ``announcements_extracted.csv`` (normalised ``type``);
    falls back to the raw ``announcements.csv`` (NEWS_DT / SUBCATNAME) when the
    extracted file is unavailable.
    """
    extracted = ann_dir / symbol / "announcements_extracted.csv"
    if extracted.exists():
        try:
            df = pd.read_csv(extracted, usecols=["date", "type"], dtype=str)
            df["ann_date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            df["type"] = df["type"].fillna("other").str.strip()
            return df.dropna(subset=["ann_date"])[["ann_date", "type"]]
        except Exception:
            pass  # fall through to raw CSV

    raw = ann_dir / symbol / "announcements.csv"
    if raw.exists():
        try:
            df = pd.read_csv(raw, usecols=["NEWS_DT", "SUBCATNAME"], dtype=str)
            df["ann_date"] = pd.to_datetime(
                df["NEWS_DT"], format="mixed", errors="coerce"
            ).dt.normalize()
            df["type"] = df["SUBCATNAME"].map(_classify_subcat)
            return df.dropna(subset=["ann_date"])[["ann_date", "type"]]
        except Exception:
            return pd.DataFrame(columns=["ann_date", "type"])

    return pd.DataFrame(columns=["ann_date", "type"])


def _to_day_int(s: pd.Series) -> np.ndarray:
    """Convert a Series of Timestamps → integer number of days since epoch."""
    return pd.to_datetime(s).values.astype("datetime64[D]").astype(np.int64)


def _rolling_flag(panel_days: np.ndarray, event_days: np.ndarray, window: int) -> np.ndarray:
    """1.0 if ANY event falls in [d - window, d - 1] (inclusive), else 0.0."""
    if event_days.size == 0:
        return np.zeros(panel_days.shape[0], dtype=float)
    ev = np.sort(event_days)
    out = np.empty(panel_days.shape[0], dtype=float)
    for k, d in enumerate(panel_days):
        i = np.searchsorted(ev, d - window, side="left")
        j = np.searchsorted(ev, d - 1, side="right")
        out[k] = 1.0 if i < j else 0.0
    return out


def _rolling_count(panel_days: np.ndarray, event_days: np.ndarray, window: int) -> np.ndarray:
    """COUNT of events in [d - window, d - 1] (inclusive), as float."""
    if event_days.size == 0:
        return np.zeros(panel_days.shape[0], dtype=float)
    ev = np.sort(event_days)
    out = np.empty(panel_days.shape[0], dtype=float)
    for k, d in enumerate(panel_days):
        i = np.searchsorted(ev, d - window, side="left")
        j = np.searchsorted(ev, d - 1, side="right")
        out[k] = float(j - i)
    return out


def build_bse_event_features(
    panel: pd.DataFrame,
    ann_dir: str | Path,
) -> pd.DataFrame:
    """Compute BSE announcement features and attach them to the OHLCV panel.

    Parameters
    ----------
    panel : DataFrame
        Lagged-features panel with columns ``symbol`` and ``date``
        (Python ``datetime.date`` objects).
    ann_dir : path
        Root of bse_announcements, e.g. ``dataset_smallcap250/bse_announcements``.

    Returns
    -------
    DataFrame
        Same panel with the ``BSE_FEATURE_COLS`` float columns added.
        All features default to 0.0 when no announcement data is available.
    """
    ann_dir = Path(ann_dir)

    panel = panel.copy()
    for col in BSE_FEATURE_COLS:
        panel[col] = 0.0

    if not ann_dir.exists() or panel.empty:
        return panel

    count_col, count_window = _COUNT_FEATURE

    for sym in panel["symbol"].unique().tolist():
        ann = _load_announcements(ann_dir, sym)
        if ann.empty:
            continue

        sym_mask = (panel["symbol"] == sym).values
        panel_days = _to_day_int(panel.loc[sym_mask, "date"])

        # Pre-group event days by type once, then build each feature window.
        days_by_type: dict[str, np.ndarray] = {}
        for t, grp in ann.groupby("type"):
            days_by_type[t] = _to_day_int(grp["ann_date"].drop_duplicates())

        all_days = _to_day_int(ann["ann_date"])  # keep duplicates → true count

        for col, (types, window) in _TYPE_FLAG_SPECS.items():
            present = [days_by_type[t] for t in types if t in days_by_type]
            event_days = (
                np.concatenate(present) if present else np.array([], dtype=np.int64)
            )
            panel.loc[sym_mask, col] = _rolling_flag(panel_days, event_days, window)

        panel.loc[sym_mask, count_col] = _rolling_count(panel_days, all_days, count_window)

    return panel
