"""Point-in-time screener snapshots (technical + fundamental)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from trading_bot.data.corporate_actions import adjust_ohlcv, load_dataset_corporate_actions
from trading_bot.data.dataset_store import dataset_root, list_symbols, load_ohlcv
from trading_bot.data.screener_excel import (
    load_balance_sheet_extended,
    load_bonus_shares_history,
    load_symbol_fundamentals,
    list_screener_symbols,
    screener_file,
)
from trading_bot.features.chart_indicators import _rsi


@dataclass
class HistoricalSnapshot:
    """Metrics available as-of a calendar date (no future data)."""

    symbol: str
    as_of_date: str
    company_name: str = ""

    close: Optional[float] = None
    close_adj: Optional[float] = None
    volume: Optional[float] = None
    volume_adj: Optional[float] = None
    volume_avg_252d: Optional[float] = None
    rsi_14: Optional[float] = None

    market_cap_cr: Optional[float] = None
    pe: Optional[float] = None
    pb: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    roce: Optional[float] = None
    sales_growth_5y: Optional[float] = None
    sales_growth_yoy: Optional[float] = None
    profit_growth_yoy: Optional[float] = None

    report_date_pl: Optional[str] = None
    report_date_bs: Optional[str] = None
    shares_as_of: Optional[float] = None
    missing: list[str] = field(default_factory=list)
    approximations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_ts(d: date | pd.Timestamp | str) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def load_shares_history(path: Path) -> pd.DataFrame:
    """Historical share count from extended balance sheet parse."""
    ext = load_balance_sheet_extended(path)
    if ext.empty or ext["shares"].isna().all():
        return pd.DataFrame(columns=["report_date", "shares"])
    sub = ext[ext["shares"].notna()][["report_date", "shares"]].copy()
    return sub.sort_values("report_date").reset_index(drop=True)


def _latest_on_or_before(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    period_type: str | None = None,
    date_col: str = "report_date",
) -> Optional[pd.Series]:
    if df.empty:
        return None
    sub = df[df[date_col] <= as_of] if date_col in df.columns else df
    if period_type is not None and "period_type" in sub.columns:
        sub = sub[sub["period_type"] == period_type]
    sub = sub.sort_values(date_col)
    if sub.empty:
        return None
    return sub.iloc[-1]


def _shares_on_or_before(shares_df: pd.DataFrame, as_of: pd.Timestamp) -> Optional[float]:
    if shares_df.empty:
        return None
    sub = shares_df[shares_df["report_date"] <= as_of]
    if sub.empty:
        return None
    return float(sub.iloc[-1]["shares"])


def _sales_growth_5y(annual_pl: pd.DataFrame, as_of: pd.Timestamp) -> Optional[float]:
    sub = annual_pl[annual_pl["report_date"] <= as_of].sort_values("report_date")
    if len(sub) < 2 or "f_sales" not in sub.columns:
        return None
    latest = sub.iloc[-1]
    target = latest["report_date"] - pd.DateOffset(years=5)
    prior = sub.iloc[(sub["report_date"] - target).abs().argsort().values[0]]
    if prior["report_date"] >= latest["report_date"]:
        return None
    years = (latest["report_date"] - prior["report_date"]).days / 365.25
    if years < 1 or prior["f_sales"] in (0, None) or pd.isna(prior["f_sales"]):
        return None
    ratio = float(latest["f_sales"]) / float(prior["f_sales"])
    if ratio <= 0:
        return None
    return ratio ** (1.0 / years) - 1.0


def _ttm_operating_profit(quarterly: pd.DataFrame, as_of: pd.Timestamp) -> Optional[float]:
    sub = quarterly[
        (quarterly["report_date"] <= as_of) & quarterly["f_operating_profit"].notna()
    ].sort_values("report_date")
    if len(sub) >= 4:
        return float(sub.tail(4)["f_operating_profit"].sum())
    return None


def _operating_profit_as_of(
    quarterly: pd.DataFrame,
    pl: pd.Series | None,
    as_of: pd.Timestamp,
) -> tuple[Optional[float], list[str]]:
    """TTM OP from quarters, else annualize partial quarters, else EBIT proxy from P&L."""
    notes: list[str] = []
    ttm = _ttm_operating_profit(quarterly, as_of)
    if ttm is not None:
        return ttm, notes

    sub = quarterly[
        (quarterly["report_date"] <= as_of) & quarterly["f_operating_profit"].notna()
    ].sort_values("report_date")
    if not sub.empty:
        n = len(sub)
        est = float(sub["f_operating_profit"].sum()) * 4.0 / n
        notes.append(f"op_annualized_from_{n}_quarters")
        return est, notes

    if pl is not None and pd.notna(pl.get("f_profit_before_tax")):
        ebit = float(pl["f_profit_before_tax"])
        if pd.notna(pl.get("f_interest")):
            ebit += float(pl["f_interest"])
            notes.append("roce_ebit_proxy_pbt_plus_interest")
            return ebit, notes
    return None, notes


def _compute_roce(
    operating_profit: float,
    bs_row: pd.Series,
    pl_row: pd.Series | None,
) -> tuple[Optional[float], list[str]]:
    """ROCE from operating profit and balance-sheet capital employed."""
    notes: list[str] = []
    equity = bs_row.get("f_equity_share_capital")
    if pd.isna(equity) and "equity" in bs_row.index:
        equity = bs_row.get("equity")
    reserves = bs_row.get("f_reserves")
    borrowings = bs_row.get("f_borrowings")
    cash = bs_row.get("f_cash_bank")
    total_assets = bs_row.get("total_assets")
    other_liab = bs_row.get("other_liabilities")

    capital = None
    if pd.notna(total_assets) and pd.notna(other_liab):
        capital = float(total_assets) - float(other_liab)
        notes.append("roce_capital_total_assets_minus_other_liabilities")
    elif pd.notna(equity) and pd.notna(reserves):
        capital = float(equity) + float(reserves)
        if pd.notna(borrowings):
            capital += float(borrowings)
        if pd.notna(cash):
            capital -= float(cash)
        notes.append("roce_capital_equity_plus_debt_minus_cash")

    if capital is None or capital <= 0:
        return None, notes
    return operating_profit / capital, notes


def technicals_as_of(
    bars: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    use_adjusted: bool = True,
) -> dict[str, Optional[float]]:
    """Compute technical fields using only bars with date <= as_of."""
    out: dict[str, Optional[float]] = {
        "close": None,
        "close_adj": None,
        "volume": None,
        "volume_adj": None,
        "volume_avg_252d": None,
        "rsi_14": None,
    }
    if bars.empty:
        return out

    work = bars.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.tz_localize(None).dt.normalize()
    work = work[work["date"] <= as_of].sort_values("date")
    if work.empty:
        return out

    last = work.iloc[-1]
    out["close"] = float(last["close"])
    out["volume"] = float(last["volume"])
    if "close_adj" in work.columns and pd.notna(last.get("close_adj")):
        out["close_adj"] = float(last["close_adj"])
    else:
        out["close_adj"] = out["close"]
    if "volume_adj" in work.columns and pd.notna(last.get("volume_adj")):
        out["volume_adj"] = float(last["volume_adj"])
    else:
        out["volume_adj"] = out["volume"]

    price_col = "close_adj" if use_adjusted and "close_adj" in work.columns else "close"
    vol_col = "volume_adj" if use_adjusted and "volume_adj" in work.columns else "volume"

    vol_window = work[vol_col].tail(252)
    if not vol_window.empty:
        out["volume_avg_252d"] = float(vol_window.mean())

    if len(work) >= 15:
        rsi = _rsi(work[price_col], 14)
        val = rsi.iloc[-1]
        if pd.notna(val):
            out["rsi_14"] = float(val)

    return out


def fundamentals_as_of(
    fund: pd.DataFrame,
    shares_df: pd.DataFrame,
    bs_ext: pd.DataFrame,
    as_of: pd.Timestamp,
    close: Optional[float],
    *,
    price_for_mcap: Optional[float] = None,
) -> dict[str, Any]:
    """Fundamental ratios using filings known on or before *as_of*."""
    result: dict[str, Any] = {
        "market_cap_cr": None,
        "pe": None,
        "pb": None,
        "debt_to_equity": None,
        "roe": None,
        "roce": None,
        "sales_growth_5y": None,
        "sales_growth_yoy": None,
        "profit_growth_yoy": None,
        "report_date_pl": None,
        "report_date_bs": None,
        "shares_as_of": None,
        "missing": [],
        "approximations": [],
    }
    if fund.empty:
        result["missing"].append("screener_fundamentals")
        return result

    fund = fund.copy()
    fund["report_date"] = pd.to_datetime(fund["report_date"]).dt.normalize()

    pl = _latest_on_or_before(fund, as_of, "annual_pl")
    bs = _latest_on_or_before(fund, as_of, "annual_bs")
    bs_x = _latest_on_or_before(bs_ext, as_of, date_col="report_date")
    annual_pl = fund[fund["period_type"] == "annual_pl"]
    quarterly = fund[fund["period_type"] == "quarterly"]

    if pl is not None:
        result["report_date_pl"] = pl["report_date"].strftime("%Y-%m-%d")
        if pd.notna(pl.get("f_sales_growth")):
            result["sales_growth_yoy"] = float(pl["f_sales_growth"])
    else:
        result["missing"].append("annual_pl")

    annual_pl_asof = annual_pl[annual_pl["report_date"] <= as_of].sort_values("report_date")
    if len(annual_pl_asof) >= 2 and "f_net_profit" in annual_pl_asof.columns:
        cur_np = annual_pl_asof.iloc[-1]["f_net_profit"]
        prev_np = annual_pl_asof.iloc[-2]["f_net_profit"]
        if pd.notna(cur_np) and pd.notna(prev_np) and float(prev_np) != 0.0:
            result["profit_growth_yoy"] = (float(cur_np) - float(prev_np)) / abs(
                float(prev_np)
            )
            result["approximations"].append("profit_growth_yoy_from_annual_pl")

    if bs is not None:
        result["report_date_bs"] = bs["report_date"].strftime("%Y-%m-%d")
    elif bs_x is not None:
        result["report_date_bs"] = bs_x["report_date"].strftime("%Y-%m-%d")
    else:
        result["missing"].append("annual_bs")

    shares = _shares_on_or_before(shares_df, as_of)
    result["shares_as_of"] = shares

    book = None
    bs_for_roce: pd.Series = pd.Series(dtype=float)
    if bs is not None:
        bs_for_roce = bs.copy()
    if bs_x is not None:
        if bs_for_roce.empty:
            bs_for_roce = pd.Series(dtype=float)
        for col in ("total_assets", "other_liabilities"):
            if pd.notna(bs_x.get(col)):
                bs_for_roce[col] = bs_x[col]
        if bs is not None:
            equity = bs.get("f_equity_share_capital")
            reserves = bs.get("f_reserves")
            borrowings = bs.get("f_borrowings")
            cash = bs.get("f_cash_bank")
            if pd.notna(equity) and pd.notna(reserves):
                book = float(equity) + float(reserves)
                if pd.notna(borrowings) and book > 0:
                    result["debt_to_equity"] = float(borrowings) / book
            if book and book > 0 and pl is not None and pd.notna(pl.get("f_net_profit")):
                result["roe"] = float(pl["f_net_profit"]) / book

    if bs is None and bs_x is None:
        result["missing"].append("balance_sheet_for_debt_book")

    ttm_op, op_notes = _operating_profit_as_of(quarterly, pl, as_of)
    result["approximations"].extend(op_notes)
    if ttm_op is not None and not bs_for_roce.empty:
        roce, roce_notes = _compute_roce(ttm_op, bs_for_roce, pl)
        if roce is not None:
            result["roce"] = roce
            result["approximations"].extend(roce_notes)
        else:
            result["missing"].append("roce_capital_employed")
    elif ttm_op is None:
        result["missing"].append("roce_operating_profit")

    result["sales_growth_5y"] = _sales_growth_5y(annual_pl, as_of)

    px = price_for_mcap if price_for_mcap is not None else close
    if px is not None and shares is not None:
        result["market_cap_cr"] = px * shares / 1e7
        result["approximations"].append("market_cap_from_shares_x_price")
    elif px is not None:
        result["missing"].append("shares_for_market_cap")

    mcap = result["market_cap_cr"]
    if mcap and pl is not None and pd.notna(pl.get("f_net_profit")):
        np_ = float(pl["f_net_profit"])
        if np_ > 0:
            result["pe"] = mcap / np_
            result["approximations"].append("pe_uses_latest_annual_not_ttm")
        else:
            result["missing"].append("negative_or_zero_earnings_for_pe")

    if mcap and book and book > 0:
        result["pb"] = mcap / book

    return result


class HistoricalScreener:
    """Point-in-time screener for one symbol or a universe."""

    def __init__(self, root: Path | str, *, use_adjusted: bool = True) -> None:
        self.root = dataset_root(Path(root))
        self.screener_dir = self.root / "screener_excel"
        self.use_adjusted = use_adjusted
        self._fund_cache: dict[str, pd.DataFrame] = {}
        self._shares_cache: dict[str, pd.DataFrame] = {}
        self._bs_ext_cache: dict[str, pd.DataFrame] = {}
        self._bonus_cache: dict[str, pd.DataFrame] = {}
        self._bars_cache: dict[str, pd.DataFrame] = {}
        self._actions_cache: dict[str, pd.DataFrame] = {}

    def _screener_path(self, symbol: str) -> Path:
        return screener_file(self.screener_dir, symbol)

    def _fundamentals(self, symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        if sym not in self._fund_cache:
            self._fund_cache[sym] = load_symbol_fundamentals(self.screener_dir, sym)
        return self._fund_cache[sym]

    def _shares(self, symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        if sym not in self._shares_cache:
            path = self._screener_path(sym)
            self._shares_cache[sym] = load_shares_history(path) if path.is_file() else pd.DataFrame()
        return self._shares_cache[sym]

    def _bs_extended(self, symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        if sym not in self._bs_ext_cache:
            path = self._screener_path(sym)
            self._bs_ext_cache[sym] = (
                load_balance_sheet_extended(path) if path.is_file() else pd.DataFrame()
            )
        return self._bs_ext_cache[sym]

    def _bonus(self, symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        if sym not in self._bonus_cache:
            path = self._screener_path(sym)
            self._bonus_cache[sym] = (
                load_bonus_shares_history(path) if path.is_file() else pd.DataFrame()
            )
        return self._bonus_cache[sym]

    def _actions(self, symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        if sym not in self._actions_cache:
            self._actions_cache[sym] = load_dataset_corporate_actions(
                self.root, sym, self._shares(sym), self._bonus(sym)
            )
        return self._actions_cache[sym]

    def _bars(self, symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        if sym not in self._bars_cache:
            try:
                raw = load_ohlcv(sym, "day", self.root)
            except FileNotFoundError:
                raw = pd.DataFrame()
            if raw.empty or not self.use_adjusted:
                self._bars_cache[sym] = raw
            else:
                self._bars_cache[sym] = adjust_ohlcv(raw, self._actions(sym), sym)
        return self._bars_cache[sym]

    def snapshot(self, symbol: str, as_of: date | str) -> HistoricalSnapshot:
        sym = symbol.upper()
        as_of_ts = _as_ts(as_of)
        tech = technicals_as_of(self._bars(sym), as_of_ts, use_adjusted=self.use_adjusted)
        price_for_mcap = tech.get("close_adj") if self.use_adjusted else tech.get("close")
        fund = fundamentals_as_of(
            self._fundamentals(sym),
            self._shares(sym),
            self._bs_extended(sym),
            as_of_ts,
            tech.get("close"),
            price_for_mcap=price_for_mcap,
        )

        missing = list(fund.pop("missing", []))
        approximations = list(fund.pop("approximations", []))
        if tech["close"] is None:
            missing.append("ohlcv")
        if tech["rsi_14"] is None and tech["close"] is not None:
            missing.append("rsi_insufficient_history")
        if tech["volume_avg_252d"] is None and tech["volume"] is not None:
            approximations.append("volume_avg_shorter_than_252d")
        if self.use_adjusted and not self._actions(sym).empty:
            approximations.append("price_volume_adjusted_for_corporate_actions")

        return HistoricalSnapshot(
            symbol=sym,
            as_of_date=as_of_ts.strftime("%Y-%m-%d"),
            close=tech["close"],
            close_adj=tech["close_adj"],
            volume=tech["volume"],
            volume_adj=tech["volume_adj"],
            volume_avg_252d=tech["volume_avg_252d"],
            rsi_14=tech["rsi_14"],
            missing=missing,
            approximations=approximations,
            **fund,
        )

    def screen(
        self,
        as_of: date | str,
        symbols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        if symbols is None:
            symbols = self._default_symbols()
        rows = [self.snapshot(sym, as_of).to_dict() for sym in symbols]
        return pd.DataFrame(rows)

    def build_panel(
        self,
        start: date | str,
        end: date | str,
        symbols: Optional[list[str]] = None,
        freq: str = "B",
        *,
        use_adjusted: bool | None = None,
    ) -> pd.DataFrame:
        """Symbol × date panel for dashboard backtests."""
        if use_adjusted is not None:
            self.use_adjusted = use_adjusted
        dates = pd.date_range(_as_ts(start), _as_ts(end), freq=freq)
        if symbols is None:
            symbols = self._default_symbols()

        parts: list[dict] = []
        for sym in symbols:
            for dt in dates:
                parts.append(self.snapshot(sym, dt.date()).to_dict())
        return pd.DataFrame(parts)

    def _default_symbols(self) -> list[str]:
        ohlcv = set(list_symbols("day", self.root))
        screener = set(list_screener_symbols(self.screener_dir))
        return sorted(ohlcv & screener)


DATA_REQUIREMENTS: dict[str, str] = {
    "daily_ohlcv": (
        "Required. ohlcv/day/{SYMBOL}.csv through each as-of date plus ~252 prior "
        "sessions for RSI and 1Y volume average."
    ),
    "screener_excel": (
        "Required. {SYMBOL}_consolidated.xlsx per symbol for P&L, balance sheet, "
        "and cash-flow history (report dates)."
    ),
    "shares_history": (
        "Required for historical market cap, P/E, P/B. Parsed from Balance Sheet "
        "'No. of Equity Shares' in Screener exports."
    ),
    "corporate_actions_csv": (
        "Optional but recommended. dataset/corporate_actions.csv or "
        "corporate_actions/{SYMBOL}.csv with columns: symbol, ex_date, action, ratio. "
        "Inferred automatically from share-count jumps when omitted."
    ),
    "filing_announcement_dates": (
        "NOT in repo. We use report_date <= as_of (instant availability). "
        "Add result announcement dates to avoid look-ahead in live-accurate backtests."
    ),
    "ttm_earnings": (
        "NOT stored. P/E uses latest annual net profit; ROCE prefers TTM operating "
        "profit from last 4 quarters when available."
    ),
    "roce": (
        "Computed from TTM operating profit / capital employed. Capital employed uses "
        "total_assets - other_liabilities when available, else equity + debt - cash."
    ),
    "index_ohlcv": (
        "Optional. For relative-strength vs Nifty; not used by this script yet."
    ),
}
