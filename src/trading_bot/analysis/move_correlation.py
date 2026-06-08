"""Significant price moves: context enrichment and factor correlation."""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd

from trading_bot.features.bse_events import BSE_FEATURE_COLS as BSE_ANNOUNCEMENT_COLS
from trading_bot.features.indicators import add_atr_pct, add_gap_risk, add_volume_surge

TargetKind = Literal["abs_return", "signed_return", "is_significant", "z_score"]

TECHNICAL_FACTOR_COLS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "volatility_20d",
    "volume_ratio_20d",
    "vol_surge_20d",
    "rsi_14",
    "close_sma_20d",
    "close_sma_50d",
    "high_low_range",
    "atr_pct_14",
    "gap_risk",
]

FUNDAMENTAL_FACTOR_COLS = [
    "days_since_filing",
    "filing_within_5d",
    "f_sales_growth_asof",
    "f_roe_asof",
    "f_debt_equity_asof",
    "f_profit_margin_asof",
    "f_sales_growth_delta",
]

FACTOR_LABELS = {
    "ret_1d_prior": "Prior day return",
    "ret_5d": "5-day return",
    "ret_20d": "20-day return",
    "ret_60d": "60-day return",
    "volatility_20d": "20-day volatility",
    "volume_ratio_20d": "Volume vs average",
    "vol_surge_20d": "Volume surge",
    "rsi_14": "RSI",
    "close_sma_20d": "Price vs 20-day average",
    "close_sma_50d": "Price vs 50-day average",
    "high_low_range": "Daily range",
    "atr_pct_14": "ATR (volatility)",
    "gap_risk": "Overnight gap",
    "days_since_filing": "Days since results filed",
    "filing_within_5d": "Results filed this week",
    "f_sales_growth_asof": "Sales growth",
    "f_roe_asof": "ROE",
    "f_debt_equity_asof": "Debt / equity",
    "f_profit_margin_asof": "Profit margin",
    "f_sales_growth_delta": "Sales growth change",
    "abs_return": "|Daily return| on move day",
    "signed_return": "Signed return on move day",
    "is_significant": "Significant move (0/1)",
    "z_score": "Move z-score",
    # BSE announcement features
    "bse_result_blackout": "Results filed (last 3d)",
    "bse_bulk_buy_last5d": "Bulk/creeping buy (last 5d)",
    "bse_promoter_buy_7d": "Promoter stake buy (last 7d)",
    "bse_corp_action_5d": "Corp action (last 5d)",
    "bse_window_closed": "Trading window closed (last 10d)",
    "bse_results_5d": "Results filed (last 5d)",
    "bse_earnings_call_5d": "Earnings call / investor meet (last 5d)",
    "bse_order_win_10d": "Order / contract win (last 10d)",
    "bse_acquisition_10d": "Acquisition / merger (last 10d)",
    "bse_capacity_expansion_15d": "Capacity expansion (last 15d)",
    "bse_credit_rating_10d": "Credit-rating action (last 10d)",
    "bse_ann_count_5d": "Announcement count (last 5d)",
}

# Small set shown in the dashboard (easy to scan).
SIMPLE_FACTOR_COLS = [
    "rsi_14",
    "volume_ratio_20d",
    "ret_5d",               # short-term momentum (5-day return)
    "ret_20d",
    "volatility_20d",
    "vol_surge_20d",
    "close_sma_20d",
    "close_sma_50d",        # medium-term trend position (% above 50D SMA)
    "gap_risk",
    "filing_within_5d",
]

# BSE announcement features — computed post-concat in build_lagged_panel,
# so kept separate from SIMPLE_FACTOR_COLS (not available in symbol_lagged_frame).
# The canonical list (BSE_ANNOUNCEMENT_COLS) is imported at the top of this
# module from trading_bot.features.bse_events so the feature builder and the
# model column set never drift apart.


def enrich_technical_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add training-style indicators on top of chart indicators already in *df*."""
    if df.empty:
        return df.copy()
    out = df.sort_values("date").copy()
    if "atr_pct_14" not in out.columns:
        out = add_atr_pct(out, period=14)
    if "gap_risk" not in out.columns:
        out = add_gap_risk(out)
    if "vol_surge_20d" not in out.columns:
        out = add_volume_surge(out, period=20)
    # Prior-day return avoids leaking the move day's close into "prior" momentum reads.
    if "ret_1d" in out.columns:
        out["ret_1d_prior"] = out["ret_1d"].shift(1)
    return out


def _fundamental_asof_table(fund_df: pd.DataFrame) -> pd.DataFrame:
    """Latest quarterly (or any) filing metrics as-of each report date."""
    if fund_df.empty:
        return pd.DataFrame()

    fund = fund_df.copy()
    fund["report_date"] = pd.to_datetime(fund["report_date"]).dt.normalize()
    fund = fund.sort_values("report_date")

    quarterly = fund[fund["period_type"] == "quarterly"]
    base = quarterly if not quarterly.empty else fund

    ratio_cols = [c for c in base.columns if c.startswith("f_")]
    if not ratio_cols:
        return pd.DataFrame()

    rows: list[dict] = []
    prev_growth: float | None = None
    for _, row in base.iterrows():
        growth = row.get("f_sales_growth")
        delta = np.nan
        if pd.notna(growth) and prev_growth is not None and pd.notna(prev_growth):
            delta = float(growth) - float(prev_growth)
        if pd.notna(growth):
            prev_growth = float(growth)

        entry: dict = {"report_date": row["report_date"]}
        for col in ratio_cols:
            entry[f"{col}_asof"] = row.get(col)
        entry["f_sales_growth_delta"] = delta
        rows.append(entry)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("report_date")


def attach_fundamental_context(
    bars: pd.DataFrame,
    fund_df: pd.DataFrame,
    *,
    filing_window_days: int = 5,
) -> pd.DataFrame:
    """Point-in-time fundamental context for each bar date."""
    out = bars.sort_values("date").copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["filing_within_5d"] = 0.0

    asof = _fundamental_asof_table(fund_df)
    if asof.empty:
        for col in FUNDAMENTAL_FACTOR_COLS:
            if col != "filing_within_5d":
                out[col] = np.nan
        return out

    asof = asof.sort_values("report_date")
    merged = pd.merge_asof(
        out,
        asof,
        left_on="date",
        right_on="report_date",
        direction="backward",
    )
    merged["days_since_filing"] = (merged["date"] - merged["report_date"]).dt.days
    merged["filing_within_5d"] = merged["days_since_filing"].between(
        0, filing_window_days
    ).astype(float)
    return merged.sort_values("date").reset_index(drop=True)


def label_move_days(
    bars: pd.DataFrame,
    moves: pd.DataFrame,
) -> pd.DataFrame:
    """Add move flags and z-scores onto the daily bar series."""
    out = bars.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["signed_return"] = out["close"].pct_change()
    out["abs_return"] = out["signed_return"].abs()
    out["is_significant"] = 0.0
    out["z_score"] = np.nan

    if moves.empty:
        return out

    move_map = moves.copy()
    move_map["date"] = pd.to_datetime(move_map["date"]).dt.normalize()
    for _, row in move_map.iterrows():
        mask = out["date"] == row["date"]
        out.loc[mask, "is_significant"] = 1.0
        out.loc[mask, "z_score"] = row.get("z_score", np.nan)
        if mask.any() and pd.notna(row.get("return")):
            out.loc[mask, "signed_return"] = row["return"]
            out.loc[mask, "abs_return"] = abs(float(row["return"]))
    return out


def available_factor_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in TECHNICAL_FACTOR_COLS:
        if c == "ret_1d":
            if "ret_1d_prior" in df.columns:
                cols.append("ret_1d_prior")
            elif "ret_1d" in df.columns:
                cols.append("ret_1d")
        elif c in df.columns:
            cols.append(c)
    cols.extend([c for c in FUNDAMENTAL_FACTOR_COLS if c in df.columns])
    return cols


def _safe_corr(a: pd.Series, b: pd.Series, method: str = "spearman") -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def factor_move_correlations(
    enriched: pd.DataFrame,
    factor_cols: Iterable[str] | None = None,
    *,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Correlate each factor with move outcomes.

    Returns one row per factor with correlations to |return|, signed return,
    z-score (move days), and point-biserial-style association with is_significant.
    """
    factors = list(factor_cols or available_factor_cols(enriched))
    if not factors:
        return pd.DataFrame()

    work = enriched.copy()
    move_mask = work["is_significant"] == 1.0 if "is_significant" in work.columns else pd.Series(
        False, index=work.index
    )

    rows: list[dict] = []
    for col in factors:
        series = work[col]
        row = {
            "factor": col,
            "label": FACTOR_LABELS.get(col, col),
            "corr_abs_return_all": _safe_corr(series, work.get("abs_return", pd.Series()), method),
            "corr_signed_return_all": _safe_corr(
                series, work.get("signed_return", pd.Series()), method
            ),
            "corr_is_significant": _safe_corr(series, work.get("is_significant", pd.Series()), method),
        }
        if move_mask.any():
            sub = work.loc[move_mask]
            row["corr_abs_return_moves"] = _safe_corr(
                sub[col], sub.get("abs_return", pd.Series()), method
            )
            row["corr_z_score_moves"] = _safe_corr(
                sub[col], sub.get("z_score", pd.Series()), method
            )
            row["mean_on_moves"] = float(sub[col].mean(skipna=True))
            non = work.loc[~move_mask]
            row["mean_on_other"] = float(non[col].mean(skipna=True)) if not non.empty else np.nan
            if pd.notna(row["mean_on_moves"]) and pd.notna(row["mean_on_other"]):
                pooled_std = float(work[col].std(skipna=True))
                if pooled_std and pooled_std > 0:
                    row["mean_diff_std"] = (row["mean_on_moves"] - row["mean_on_other"]) / pooled_std
                else:
                    row["mean_diff_std"] = np.nan
            else:
                row["mean_diff_std"] = np.nan
        else:
            row["corr_abs_return_moves"] = np.nan
            row["corr_z_score_moves"] = np.nan
            row["mean_on_moves"] = np.nan
            row["mean_on_other"] = np.nan
            row["mean_diff_std"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out["abs_corr_moves"] = out["corr_z_score_moves"].abs()
    return out.sort_values("abs_corr_moves", ascending=False, na_position="last").reset_index(
        drop=True
    )


def factor_correlation_matrix(
    enriched: pd.DataFrame,
    factor_cols: Iterable[str] | None = None,
    *,
    on_moves_only: bool = True,
    method: str = "spearman",
) -> pd.DataFrame:
    """Square correlation matrix among factors (move days only by default)."""
    factors = list(factor_cols or available_factor_cols(enriched))
    if len(factors) < 2:
        return pd.DataFrame()

    work = enriched.copy()
    if on_moves_only and "is_significant" in work.columns:
        work = work.loc[work["is_significant"] == 1.0]
    if len(work) < 3:
        return pd.DataFrame()

    mat = work[factors].corr(method=method)
    mat.index = [FACTOR_LABELS.get(c, c) for c in mat.index]
    mat.columns = [FACTOR_LABELS.get(c, c) for c in mat.columns]
    return mat


def _format_factor_value(col: str, value: float) -> str:
    if pd.isna(value):
        return "—"
    if col == "rsi_14":
        return f"{value:.0f}"
    if col in ("volume_ratio_20d", "vol_surge_20d"):
        return f"{value:.1f}× avg"
    if col in (
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "close_sma_20d",
        "close_sma_50d",
        "volatility_20d",
        "gap_risk",
        "atr_pct_14",
        "high_low_range",
        "ret_1d_prior",
    ):
        return f"{value * 100:+.1f}%"
    if col == "filing_within_5d":
        return "Yes" if value >= 0.5 else "No"
    if col == "days_since_filing":
        return f"{int(value)} days"
    if col == "bse_ann_count_5d":
        return f"{value:.1f}"
    if col.startswith("bse_"):
        return "Yes" if value >= 0.5 else "No"
    if col.startswith("f_"):
        return f"{value * 100:.1f}%" if abs(value) < 2 else f"{value:.2f}"
    return f"{value:.2f}"


def summarize_move_factors(
    enriched: pd.DataFrame,
    factor_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Plain comparison: indicator averages on big-move days vs normal days."""
    factors = [c for c in (factor_cols or SIMPLE_FACTOR_COLS) if c in enriched.columns]
    stats = factor_move_correlations(enriched, factors)

    rows: list[dict] = []
    for _, row in stats.iterrows():
        diff = row.get("mean_diff_std")
        if pd.isna(diff):
            pattern = "—"
        elif diff > 0.25:
            pattern = "Higher on big days"
        elif diff < -0.25:
            pattern = "Lower on big days"
        else:
            pattern = "About the same"

        col = row["factor"]
        rows.append(
            {
                "indicator": row["label"],
                "on_big_days": _format_factor_value(col, row.get("mean_on_moves")),
                "usually": _format_factor_value(col, row.get("mean_on_other")),
                "pattern": pattern,
                "_rank": abs(diff) if pd.notna(diff) else 0.0,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["indicator", "on_big_days", "usually", "pattern"])
    return (
        pd.DataFrame(rows)
        .sort_values("_rank", ascending=False)
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )


def simple_move_table(
    enriched: pd.DataFrame,
    moves: pd.DataFrame,
) -> pd.DataFrame:
    """Short list of big move days with a few key readings."""
    if moves.empty:
        return pd.DataFrame(columns=["date", "move", "rsi", "volume", "near_filing"])

    work = enriched.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    rows: list[dict] = []
    for _, mv in moves.sort_values("date", ascending=False).iterrows():
        d = pd.Timestamp(mv["date"]).normalize()
        bar = work.loc[work["date"] == d]
        if bar.empty:
            continue
        bar = bar.iloc[-1]
        near = bar.get("filing_within_5d", 0)
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "move": f"{float(mv['return']) * 100:+.1f}%",
                "rsi": _format_factor_value("rsi_14", bar.get("rsi_14")),
                "volume": _format_factor_value("volume_ratio_20d", bar.get("volume_ratio_20d")),
                "near_filing": "Yes" if pd.notna(near) and near >= 0.5 else "No",
            }
        )
    return pd.DataFrame(rows)


def move_day_feature_table(
    enriched: pd.DataFrame,
    moves: pd.DataFrame,
    factor_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """One row per significant move with return stats and factor readings."""
    if moves.empty:
        return pd.DataFrame()

    factors = list(factor_cols or available_factor_cols(enriched))
    move_dates = pd.to_datetime(moves["date"]).dt.normalize()
    work = enriched.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()

    rows: list[dict] = []
    for _, mv in moves.iterrows():
        d = pd.Timestamp(mv["date"]).normalize()
        bar = work.loc[work["date"] == d]
        if bar.empty:
            continue
        bar = bar.iloc[-1]
        row = {
            "date": d.strftime("%Y-%m-%d"),
            "return_pct": float(mv["return"]) * 100.0,
            "z_score": float(mv["z_score"]),
            "direction": mv["direction"],
        }
        for col in factors:
            val = bar.get(col)
            row[col] = float(val) if pd.notna(val) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def build_move_analysis(
    bars: pd.DataFrame,
    fund_df: pd.DataFrame,
    moves: pd.DataFrame,
) -> dict:
    """Full analysis bundle for dashboard / export."""
    enriched = enrich_technical_factors(bars)
    enriched = attach_fundamental_context(enriched, fund_df)
    enriched = label_move_days(enriched, moves)
    factors = available_factor_cols(enriched)
    simple_factors = [c for c in SIMPLE_FACTOR_COLS if c in enriched.columns]

    return {
        "enriched": enriched,
        "moves": moves,
        "simple_moves": simple_move_table(enriched, moves),
        "simple_summary": summarize_move_factors(enriched, simple_factors),
        "factor_correlations": factor_move_correlations(enriched, factors),
        "factor_matrix_moves": factor_correlation_matrix(enriched, factors, on_moves_only=True),
        "factor_matrix_all": factor_correlation_matrix(enriched, factors, on_moves_only=False),
        "move_features": move_day_feature_table(enriched, moves, factors),
        "factor_cols": factors,
    }


def pool_universe_move_stats(
    symbol_frames: list[pd.DataFrame],
    *,
    factor_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Combine per-symbol enriched day rows and compare big-move vs normal days.

    Each frame must include ``is_significant`` and factor columns.
    """
    factors = list(factor_cols or SIMPLE_FACTOR_COLS)
    move_parts: list[pd.DataFrame] = []
    other_parts: list[pd.DataFrame] = []

    for df in symbol_frames:
        if df.empty:
            continue
        cols = [c for c in factors if c in df.columns]
        if not cols or "is_significant" not in df.columns:
            continue
        sub = df[cols + ["is_significant"]].copy()
        move_parts.append(sub.loc[sub["is_significant"] == 1.0])
        other_parts.append(sub.loc[sub["is_significant"] != 1.0])

    if not move_parts:
        return pd.DataFrame()

    moves = pd.concat(move_parts, ignore_index=True)
    others = pd.concat(other_parts, ignore_index=True)
    rows: list[dict] = []

    for col in [c for c in factors if c in moves.columns]:
        move_mean = float(moves[col].mean(skipna=True))
        other_mean = float(others[col].mean(skipna=True))
        pooled_std = float(pd.concat([moves[col], others[col]]).std(skipna=True))
        diff_std = (
            (move_mean - other_mean) / pooled_std
            if pooled_std and pooled_std > 0
            else float("nan")
        )
        if pd.isna(diff_std):
            pattern = "—"
        elif diff_std > 0.25:
            pattern = "Higher on big days"
        elif diff_std < -0.25:
            pattern = "Lower on big days"
        else:
            pattern = "About the same"

        rows.append(
            {
                "factor": col,
                "indicator": FACTOR_LABELS.get(col, col),
                "on_big_days": _format_factor_value(col, move_mean),
                "usually": _format_factor_value(col, other_mean),
                "pattern": pattern,
                "mean_diff_std": diff_std,
                "pct_symbols_higher": float("nan"),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("mean_diff_std", key=lambda s: s.abs(), ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def build_universe_playbook(
    pooled: pd.DataFrame,
    *,
    symbols_scanned: int,
    symbols_with_moves: int,
    total_moves: int,
    up_moves: int,
    down_moves: int,
    pct_near_filing: float,
    median_move_pct: float,
    per_symbol_votes: pd.DataFrame | None = None,
) -> str:
    """Render a plain-English playbook from pooled universe stats."""
    lines = [
        "# What to look for before big price moves",
        "",
        f"Based on **{total_moves:,}** big daily moves across **{symbols_with_moves}** "
        f"stocks (of {symbols_scanned} scanned) in the smallcap dataset.",
        "",
        "## Quick checklist",
        "",
    ]

    if not pooled.empty:
        higher = pooled[pooled["pattern"] == "Higher on big days"]
        lower = pooled[pooled["pattern"] == "Lower on big days"]
        for _, row in higher.head(5).iterrows():
            lines.append(f"- **Watch for elevated {row['indicator'].lower()}** "
                         f"(avg {row['on_big_days']} on big days vs {row['usually']} normally)")
        for _, row in lower.head(3).iterrows():
            lines.append(f"- **Big moves often come after lower {row['indicator'].lower()}** "
                         f"({row['on_big_days']} vs {row['usually']} normally)")
        lines.append("")

    lines.extend(
        [
            "## Context numbers",
            "",
            f"- **Up vs down:** {up_moves:,} up days / {down_moves:,} down days "
            f"({up_moves / total_moves * 100:.0f}% up)" if total_moves else "- No moves found",
            f"- **Median move size:** {median_move_pct:.1f}%",
            f"- **Near a results filing (within 5 days):** {pct_near_filing:.0f}% of big moves",
            "",
            "## How to use this",
            "",
            "1. **Volume** — unusually high vs the 20-day average often coincides with "
            "the largest moves; treat volume spikes as a warning light, not a direction signal.",
            "2. **Trend / extension** — check 20-day return and price vs 20-day average; "
            "big moves frequently happen when the stock is already stretched (momentum) "
            "or after a sharp gap.",
            "3. **Volatility** — stocks already volatile (high 20d vol or ATR) tend to "
            "produce larger daily swings.",
            "4. **Results week** — a meaningful share of moves land within a few days of "
            "quarterly filings; cross-check the Timeline tab for filing dates.",
            "5. **RSI alone is weak** — overbought/oversold readings are noisy predictors "
            "of direction; combine with volume and trend.",
            "",
            "## Caveats",
            "",
            "- These are **historical associations** across the universe, not rules for any single stock.",
            "- Same-day indicators (volume, gap) overlap with the move itself — use prior-day "
            "readings when building entry models.",
            "- Fundamentals use filing dates, not announcement dates.",
            "",
        ]
    )

    if per_symbol_votes is not None and not per_symbol_votes.empty:
        lines.extend(["## Consistency across stocks", ""])
        for _, row in per_symbol_votes.head(8).iterrows():
            pct = row.get("pct_symbols_higher")
            if pd.notna(pct):
                lines.append(
                    f"- **{row['indicator']}** — higher on big days in "
                    f"{pct:.0f}% of stocks with enough moves"
                )
        lines.append("")

    if not pooled.empty:
        lines.extend(["## Full pooled comparison", ""])
        lines.append("| Indicator | On big days | Usually | Pattern |")
        lines.append("|-----------|-------------|---------|---------|")
        for _, row in pooled.iterrows():
            lines.append(
                f"| {row['indicator']} | {row['on_big_days']} | {row['usually']} | {row['pattern']} |"
            )
        lines.append("")

    return "\n".join(lines)
