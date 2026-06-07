"""Point-in-time fundamental screens for move predictor stock selection.

Design:
  All ratios are computed per-symbol at the **filing-date level** then joined
  to the panel with ``merge_asof`` (backward lookup = last known value).
  Lagged by one calendar day so day-T rows only see data known by day T-1.
  This is O(symbols × filings), not O(symbols × backtest_days).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.config import Config
from trading_bot.data.dataset_store import dataset_root_from_config
from trading_bot.data.screener_excel import load_symbol_fundamentals

logger = logging.getLogger(__name__)

FUNDAMENTAL_LAG_COLS = [
    "f_roce_lag1",
    "f_debt_equity_lag1",
    "f_profit_growth_yoy_lag1",
    "f_profit_growth_qtr_lag1",
    "f_pe_lag1",
    "above_dma_lag1",
    "above_trend_dma_lag1",
]


@dataclass(frozen=True)
class FundamentalScreenConfig:
    enabled: bool = True
    min_roce: float = 0.15
    max_debt_equity: float = 1.0
    min_profit_growth_yoy: float = 0.0
    min_profit_growth_qtr: float = 0.0
    max_pe: float = 20.0
    dma_period: int = 50
    require_price_above_dma: bool = True
    trend_dma_period: int = 200
    require_price_above_trend_dma: bool = True
    skip_missing: bool = True

    @classmethod
    def from_config(cls, cfg: Config) -> FundamentalScreenConfig:
        raw: dict[str, Any] = cfg._raw.get("fundamental_screener", {})
        return cls(
            enabled=bool(raw.get("enabled", True)),
            min_roce=float(raw.get("min_roce", 0.15)),
            max_debt_equity=float(raw.get("max_debt_equity", 1.0)),
            min_profit_growth_yoy=float(raw.get("min_profit_growth_yoy", 0.0)),
            min_profit_growth_qtr=float(raw.get("min_profit_growth_qtr", 0.0)),
            max_pe=float(raw.get("max_pe", 20.0)),
            dma_period=int(raw.get("dma_period", 50)),
            require_price_above_dma=bool(raw.get("require_price_above_dma", True)),
            trend_dma_period=int(raw.get("trend_dma_period", 200)),
            require_price_above_trend_dma=bool(raw.get("require_price_above_trend_dma", True)),
            skip_missing=bool(raw.get("skip_missing", True)),
        )

    def dma_column(self) -> str:
        return f"close_sma_{self.dma_period}d"

    def trend_dma_column(self) -> str:
        return f"close_sma_{self.trend_dma_period}d"


def _is_finite(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def passes_fundamental_screen(row: pd.Series, screen: FundamentalScreenConfig) -> bool:
    """Return True when *row* satisfies all enabled fundamental screens."""
    if not screen.enabled:
        return True

    roce = row.get("f_roce_lag1")
    de = row.get("f_debt_equity_lag1")
    profit_g = row.get("f_profit_growth_yoy_lag1")
    profit_q = row.get("f_profit_growth_qtr_lag1")
    pe = row.get("f_pe_lag1")
    above_dma = row.get("above_dma_lag1")
    above_trend_dma = row.get("above_trend_dma_lag1")

    required = [roce, de, profit_g, profit_q, pe]
    if screen.require_price_above_dma:
        required.append(above_dma)
    if screen.require_price_above_trend_dma:
        required.append(above_trend_dma)

    if screen.skip_missing and any(not _is_finite(v) for v in required):
        return False

    if _is_finite(roce) and float(roce) < screen.min_roce:
        return False
    if _is_finite(de) and float(de) > screen.max_debt_equity:
        return False
    if _is_finite(profit_g) and float(profit_g) < screen.min_profit_growth_yoy:
        return False
    if _is_finite(profit_q) and float(profit_q) < screen.min_profit_growth_qtr:
        return False
    if _is_finite(pe):
        pe_val = float(pe)
        if pe_val <= 0 or pe_val > screen.max_pe:
            return False
    elif screen.skip_missing:
        return False

    if screen.require_price_above_dma:
        if not _is_finite(above_dma):
            if screen.skip_missing:
                return False
            # else: insufficient history — allow through
        elif float(above_dma) <= 0:
            return False

    if screen.require_price_above_trend_dma:
        if not _is_finite(above_trend_dma):
            if screen.skip_missing:
                return False
        elif float(above_trend_dma) <= 0:
            return False

    return True


def _ratio_timeseries(fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-filing-date ratio table for one symbol.

    Returns DataFrame with columns:
        report_date, f_roce, f_debt_equity, f_profit_growth_yoy, f_net_profit
    sorted by report_date.  All ratios use only data available at that report_date.
    """
    if fund_df.empty:
        return pd.DataFrame()

    fund = fund_df.copy()
    fund["report_date"] = pd.to_datetime(fund["report_date"]).dt.normalize()

    annual = fund[fund["period_type"] == "annual_pl"].sort_values("report_date")
    bs = fund[fund["period_type"] == "annual_bs"].sort_values("report_date")
    quarterly = fund[fund["period_type"] == "quarterly"].sort_values("report_date")

    rows: list[dict] = []
    all_report_dates = pd.concat(
        [annual[["report_date"]], bs[["report_date"]], quarterly[["report_date"]]],
        ignore_index=True,
    )["report_date"].drop_duplicates().sort_values()

    for rdt in all_report_dates:
        row: dict = {"report_date": rdt}

        # ── D/E  (annual BS) ────────────────────────────────────────────────
        bs_row = bs[bs["report_date"] <= rdt]
        if not bs_row.empty:
            bs_latest = bs_row.iloc[-1]
            eq = bs_latest.get("f_equity_share_capital")
            res = bs_latest.get("f_reserves")
            bor = bs_latest.get("f_borrowings")
            if pd.notna(eq) and pd.notna(res):
                book = float(eq) + float(res)
                if book > 0 and pd.notna(bor):
                    row["f_debt_equity"] = float(bor) / book
                else:
                    row["f_debt_equity"] = np.nan
            else:
                row["f_debt_equity"] = np.nan
        else:
            row["f_debt_equity"] = np.nan

        # ── Profit growth YoY (annual P&L, with TTM-quarterly fallback) ────────
        pl_row = annual[annual["report_date"] <= rdt]
        row["f_net_profit"] = np.nan
        row["f_profit_growth_yoy"] = np.nan
        if len(pl_row) >= 2 and "f_net_profit" in pl_row.columns:
            cur_np = pl_row.iloc[-1]["f_net_profit"]
            prev_np = pl_row.iloc[-2]["f_net_profit"]
            if pd.notna(cur_np):
                row["f_net_profit"] = float(cur_np)
            if pd.notna(cur_np) and pd.notna(prev_np) and float(prev_np) != 0.0:
                row["f_profit_growth_yoy"] = (float(cur_np) - float(prev_np)) / abs(
                    float(prev_np)
                )
        elif len(pl_row) == 1 and "f_net_profit" in pl_row.columns:
            cur_np = pl_row.iloc[-1]["f_net_profit"]
            if pd.notna(cur_np):
                row["f_net_profit"] = float(cur_np)

        # TTM net profit fallback from quarterly when annual P&L is absent
        if pd.isna(row["f_net_profit"]) and "f_net_profit" in quarterly.columns:
            q_avail = quarterly[quarterly["report_date"] <= rdt].dropna(subset=["f_net_profit"])
            if len(q_avail) >= 4:
                row["f_net_profit"] = float(q_avail.tail(4)["f_net_profit"].sum())
            elif not q_avail.empty:
                row["f_net_profit"] = float(q_avail["f_net_profit"].sum()) * 4.0 / len(q_avail)

        # TTM YoY growth fallback: compare last-4Q sum vs same 4Q one year prior
        if pd.isna(row["f_profit_growth_yoy"]) and "f_net_profit" in quarterly.columns:
            q_avail = quarterly[quarterly["report_date"] <= rdt].dropna(subset=["f_net_profit"])
            if len(q_avail) >= 8:
                ttm_cur = float(q_avail.tail(4)["f_net_profit"].sum())
                ttm_prior = float(q_avail.iloc[-8:-4]["f_net_profit"].sum())
                if ttm_prior != 0.0:
                    row["f_profit_growth_yoy"] = (ttm_cur - ttm_prior) / abs(ttm_prior)

        # ── Quarterly profit growth (same quarter YoY) ──────────────────────
        # Compare the latest quarter's net profit to the same quarter one year ago.
        row["f_profit_growth_qtr"] = np.nan
        if "f_net_profit" in quarterly.columns:
            q_before = quarterly[quarterly["report_date"] <= rdt].sort_values("report_date")
            if len(q_before) >= 5:
                cur_q = q_before.iloc[-1]["f_net_profit"]
                prior_q = q_before.iloc[-5]["f_net_profit"]  # same quarter last year
                if pd.notna(cur_q) and pd.notna(prior_q) and float(prior_q) != 0.0:
                    row["f_profit_growth_qtr"] = (float(cur_q) - float(prior_q)) / abs(
                        float(prior_q)
                    )
            elif len(q_before) >= 2:
                # Fallback: QoQ sequential growth
                cur_q = q_before.iloc[-1]["f_net_profit"]
                prev_q = q_before.iloc[-2]["f_net_profit"]
                if pd.notna(cur_q) and pd.notna(prev_q) and float(prev_q) != 0.0:
                    row["f_profit_growth_qtr"] = (float(cur_q) - float(prev_q)) / abs(
                        float(prev_q)
                    )

        # ── ROCE (TTM operating profit / capital employed) ──────────────────
        row["f_roce"] = np.nan
        if "f_operating_profit" not in quarterly.columns:
            q_sub = pd.DataFrame()
        else:
            q_sub = quarterly[
                (quarterly["report_date"] <= rdt)
                & quarterly["f_operating_profit"].notna()
            ]
        ttm_op: float | None = None
        if len(q_sub) >= 4:
            ttm_op = float(q_sub.tail(4)["f_operating_profit"].sum())
        elif not q_sub.empty:
            n = len(q_sub)
            ttm_op = float(q_sub["f_operating_profit"].sum()) * 4.0 / n

        if ttm_op is not None and not bs_row.empty:
            bs_latest = bs_row.iloc[-1]
            ta = bs_latest.get("total_assets")
            ol = bs_latest.get("other_liabilities")
            eq2 = bs_latest.get("f_equity_share_capital")
            res2 = bs_latest.get("f_reserves")
            bor2 = bs_latest.get("f_borrowings")
            cash = bs_latest.get("f_cash_bank")
            capital: float | None = None
            if pd.notna(ta) and pd.notna(ol):
                capital = float(ta) - float(ol)
            elif pd.notna(eq2) and pd.notna(res2):
                capital = float(eq2) + float(res2)
                if pd.notna(bor2):
                    capital += float(bor2)
                if pd.notna(cash):
                    capital -= float(cash)
            if capital and capital > 0:
                row["f_roce"] = ttm_op / capital

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("report_date").reset_index(drop=True)
    return result


def _build_fundamental_ratios(
    panel_grp: pd.DataFrame,
    fund_df: pd.DataFrame,
    screener_dir: Any,
    symbol: str,
    dma_col: str,
    trend_dma_col: str,
) -> pd.DataFrame:
    """
    Attach lagged fundamental ratio columns to one symbol's panel rows.

    All fundamental values are lagged by 1 day so row T only sees data from T-1.
    P/E = (lagged_close × shares_asof) / annual_net_profit, in INR Cr / INR Cr terms.
    """
    from trading_bot.screener.historical import load_shares_history, screener_file

    grp = panel_grp.copy()
    grp["date"] = pd.to_datetime(grp["date"])
    ratio_ts = _ratio_timeseries(fund_df)

    null_cols = ("f_roce_lag1", "f_debt_equity_lag1", "f_profit_growth_yoy_lag1", "f_pe_lag1")
    if ratio_ts.empty:
        for col in null_cols:
            grp[col] = np.nan
        grp["above_dma_lag1"] = np.nan
        return grp.sort_values("date")

    ratio_ts["report_date"] = pd.to_datetime(ratio_ts["report_date"])

    # Lagged date: fundamentals known on or before the *previous* session
    grp_lookup = grp[["date"]].copy()
    grp_lookup["lookup_date"] = grp_lookup["date"] - pd.Timedelta(days=1)
    grp_lookup = grp_lookup.sort_values("lookup_date")

    ratio_sorted = ratio_ts.sort_values("report_date")
    merged = pd.merge_asof(
        grp_lookup,
        ratio_sorted[
            [
                "report_date",
                "f_roce",
                "f_debt_equity",
                "f_profit_growth_yoy",
                "f_profit_growth_qtr",
                "f_net_profit",
            ]
        ],
        left_on="lookup_date",
        right_on="report_date",
        direction="backward",
    )

    merged = merged.set_index("date")
    grp = grp.set_index("date")
    grp["f_roce_lag1"] = merged["f_roce"]
    grp["f_debt_equity_lag1"] = merged["f_debt_equity"]
    grp["f_profit_growth_yoy_lag1"] = merged["f_profit_growth_yoy"]
    grp["f_profit_growth_qtr_lag1"] = merged["f_profit_growth_qtr"]

    # P/E:  market cap (₹Cr) / annual net profit (₹Cr)
    # market cap = lagged_close (₹) × shares_asof / 1e7
    # net profit already in ₹Cr from screener
    grp["f_pe_lag1"] = np.nan
    if "close" in grp.columns and "f_net_profit" in merged.columns:
        close_lag = grp["close"].shift(1)
        np_cr = merged["f_net_profit"]
        sdir = Path(screener_dir)
        shares_df = load_shares_history(screener_file(sdir, symbol))
        if not shares_df.empty:
            shares_df["report_date"] = pd.to_datetime(shares_df["report_date"])
            sh_lookup = grp[[]].copy().reset_index()
            sh_lookup["lookup_date"] = sh_lookup["date"] - pd.Timedelta(days=1)
            sh_merged = pd.merge_asof(
                sh_lookup.sort_values("lookup_date"),
                shares_df.sort_values("report_date"),
                left_on="lookup_date",
                right_on="report_date",
                direction="backward",
            ).set_index("date")
            shares_lag = sh_merged["shares"]
            mcap_cr = close_lag * shares_lag / 1e7
            valid = np_cr > 0
            grp.loc[valid, "f_pe_lag1"] = mcap_cr[valid] / np_cr[valid]

    # DMA above/below: use *lagged* DMA column already in the panel
    if dma_col in grp.columns:
        grp["above_dma_lag1"] = (grp[dma_col].shift(1) > 0).astype(float)
    else:
        grp["above_dma_lag1"] = np.nan

    # Trend DMA (200-day by default): long-term trend gate
    if trend_dma_col in grp.columns:
        grp["above_trend_dma_lag1"] = (grp[trend_dma_col].shift(1) > 0).astype(float)
    else:
        grp["above_trend_dma_lag1"] = np.nan

    return grp.reset_index().sort_values("date")


def enrich_panel_fundamentals(
    panel: pd.DataFrame,
    cfg: Config,
    *,
    screen: FundamentalScreenConfig | None = None,
) -> pd.DataFrame:
    """
    Attach lagged fundamental ratio columns to *panel* using vectorised merge_asof.

    Fast: O(symbols × filings), not O(symbols × backtest_days).
    """
    if panel.empty:
        return panel

    screen = screen or FundamentalScreenConfig.from_config(cfg)
    if not screen.enabled:
        return panel

    root = dataset_root_from_config(cfg)
    screener_dir = root / "screener_excel"
    dma_col = screen.dma_column()
    trend_dma_col = screen.trend_dma_column()

    parts: list[pd.DataFrame] = []
    symbols = panel["symbol"].unique()
    logger.info(
        "Enriching fundamentals for %d symbols (vectorised merge_asof, DMA=%d, trend_DMA=%d)",
        len(symbols),
        screen.dma_period,
        screen.trend_dma_period,
    )

    no_data = 0
    for sym in symbols:
        grp = panel[panel["symbol"] == sym].copy()
        fund_df = load_symbol_fundamentals(screener_dir, str(sym))
        if fund_df.empty:
            no_data += 1
            for col in FUNDAMENTAL_LAG_COLS:
                grp[col] = np.nan
            parts.append(grp)
            continue
        enriched_grp = _build_fundamental_ratios(grp, fund_df, screener_dir, str(sym), dma_col, trend_dma_col)
        parts.append(enriched_grp)

    if no_data:
        logger.warning(
            "%d / %d symbols had no screener Excel data — fundamental cols will be NaN",
            no_data,
            len(symbols),
        )

    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["date", "symbol"])
    n_pass = out.apply(lambda row: passes_fundamental_screen(row, screen), axis=1).sum()
    logger.info(
        "Fundamental screen: %d / %d symbol-days pass (%.1f%%)",
        n_pass,
        len(out),
        100.0 * n_pass / len(out) if len(out) else 0.0,
    )
    return out


def merge_fundamentals_into_panel(
    panel: pd.DataFrame,
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join lagged fundamental columns onto the full training panel."""
    if enriched.empty:
        return panel
    available = [c for c in FUNDAMENTAL_LAG_COLS if c in enriched.columns]
    if not available:
        return panel
    cols = ["symbol", "date", *available]
    panel["date"] = pd.to_datetime(panel["date"])
    enriched = enriched.copy()
    enriched["date"] = pd.to_datetime(enriched["date"])
    result = panel.merge(enriched[cols], on=["symbol", "date"], how="left")
    # restore date as date objects to stay compatible with the rest of the pipeline
    if not result.empty:
        result["date"] = result["date"].dt.date
    return result


def clear_fundamental_cache() -> None:
    """No-op; kept for API compatibility."""
    pass
