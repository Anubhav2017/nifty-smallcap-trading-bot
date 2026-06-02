"""Three baseline strategies for OOS objective J comparison.

Expected DataFrame formats
--------------------------
``index_ohlcv`` (buy-and-hold):
    DatetimeIndex or date-indexed DataFrame with a ``close`` column (case-
    insensitive).  Represents the Nifty Smallcap 100 index or a proxy ETF.

``feature_df`` (momentum / random-entry baselines):
    Wide-format DataFrame where:
    - Index: dates (``date`` objects or ``pd.DatetimeIndex``).
    - Columns: stock identifiers (any hashable; typically instrument tokens).
    - Values: adjusted close prices in INR.
    The random-entry baseline additionally uses a ``pct_change`` rolling window
    to approximate ATR.  If a column is missing for a given date it is skipped.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from trading_bot.config import Config
from trading_bot.types import FoldMetrics, Horizon

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_fold_metrics(start: date, end: date) -> FoldMetrics:
    """Return a zero-filled FoldMetrics sentinel when data is unavailable."""
    return FoldMetrics(
        fold_id=-1,
        train_start=start,
        train_end=start,
        oos_start=start,
        oos_end=end,
        sortino=0.0,
        max_drawdown=0.0,
        calmar=0.0,
        win_rate=0.0,
        expectancy_r=float("nan"),
        total_trades=0,
        swing_trades=0,
        positional_trades=0,
        avg_daily_entries=0.0,
        turnover_cost_pct=0.0,
        objective_j=0.0,
        beats_baseline=False,
    )


def _normalise_ohlcv_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase column names and a ``date``-typed index."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.date  # type: ignore[assignment]
    return df


def _normalise_wide_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a wide-format close-price frame with a ``date`` index."""
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.date  # type: ignore[assignment]
    return df


def _metrics_from_curve(
    equity_curve: pd.Series,
    start: date,
    end: date,
    total_trades: int = 0,
    turnover_cost_pct: float = 0.0,
    alpha: float = 2.0,
    beta: float = 1.0,
) -> FoldMetrics:
    """Compute FoldMetrics from an equity curve (no individual Position objects)."""
    from trading_bot.backtest.metrics import (
        calmar_ratio,
        compute_objective_j,
        max_drawdown as calc_mdd,
        sortino_ratio,
    )

    daily_returns = equity_curve.pct_change().dropna()

    sortino = sortino_ratio(daily_returns)
    mdd = calc_mdd(equity_curve)
    calmar = calmar_ratio(equity_curve)

    n_days = max(len(equity_curve), 1)
    avg_daily_entries = total_trades / n_days

    fm = FoldMetrics(
        fold_id=-1,
        train_start=start,
        train_end=start,
        oos_start=start,
        oos_end=end,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        win_rate=0.0,
        expectancy_r=float("nan"),
        total_trades=total_trades,
        swing_trades=0,
        positional_trades=0,
        avg_daily_entries=avg_daily_entries,
        turnover_cost_pct=turnover_cost_pct,
        objective_j=0.0,
        beats_baseline=False,
    )
    fm.objective_j = compute_objective_j(fm, alpha=alpha, beta=beta)
    return fm


# ── Baselines ─────────────────────────────────────────────────────────────────

def buy_and_hold_baseline(
    index_ohlcv: pd.DataFrame,
    start: date,
    end: date,
    initial_equity: float,
) -> FoldMetrics:
    """Buy the index at *start*, hold to *end*.

    Args:
        index_ohlcv:    OHLCV DataFrame for the index / proxy ETF with a
                        ``close`` column (case-insensitive).
        start:          First date of the OOS period.
        end:            Last date of the OOS period.
        initial_equity: Starting capital (INR).

    Returns:
        :class:`FoldMetrics` with ``fold_id=-1``.  Win-rate and expectancy_r
        are not meaningful (set to 0 / NaN) because no individual trades exist.
    """
    if index_ohlcv.empty:
        return _empty_fold_metrics(start, end)

    df = _normalise_ohlcv_index(index_ohlcv)
    df = df[(df.index >= start) & (df.index <= end)]

    if df.empty or "close" not in df.columns:
        logger.warning("buy_and_hold_baseline: no 'close' data in [%s, %s]", start, end)
        return _empty_fold_metrics(start, end)

    close = df["close"].dropna()
    if close.empty:
        return _empty_fold_metrics(start, end)

    equity_values = initial_equity * close / close.iloc[0]
    dt_index = pd.DatetimeIndex([pd.Timestamp(d) for d in close.index])
    equity_curve = pd.Series(equity_values.values, index=dt_index, name="equity", dtype=float)

    actual_start: date = close.index[0]  # type: ignore[assignment]
    actual_end: date = close.index[-1]  # type: ignore[assignment]

    # B&H has no rebalancing so turnover cost is effectively 0 (one-time ignored here)
    return _metrics_from_curve(equity_curve, actual_start, actual_end, turnover_cost_pct=0.0)


def equal_weight_momentum_baseline(
    feature_df: pd.DataFrame,
    cfg: Config,
    initial_equity: float,
) -> FoldMetrics:
    """Equal-weight top-10 stocks by 20-day momentum, rebalanced monthly.

    Args:
        feature_df:     Wide-format close-price DataFrame.  Index: dates;
                        columns: stock identifiers; values: close prices (INR).
                        Needs at least 20 rows of history.
        cfg:            Strategy config (used for objective weights and risk caps).
        initial_equity: Starting capital (INR).

    Returns:
        :class:`FoldMetrics` with ``fold_id=-1``.
    """
    if feature_df.empty:
        today = date.today()
        return _empty_fold_metrics(today, today)

    df = _normalise_wide_index(feature_df)
    sorted_dates: list[date] = sorted(df.index)  # type: ignore[arg-type]

    if len(sorted_dates) < 21:
        logger.warning("equal_weight_momentum_baseline: insufficient history (< 21 days)")
        return _empty_fold_metrics(sorted_dates[0], sorted_dates[-1])

    momentum_20 = df.pct_change(20)
    daily_returns = df.pct_change()

    # Identify first trading day of each calendar month
    month_first: dict[tuple[int, int], date] = {}
    for d in sorted_dates:
        key = (d.year, d.month)
        if key not in month_first:
            month_first[key] = d

    current_holdings: list = []
    portfolio_rets: list[float] = []
    n_rebalances = 0

    for d in sorted_dates:
        month_key = (d.year, d.month)
        # Rebalance on the first trading day of each month
        if month_first.get(month_key) == d:
            mom_row = momentum_20.loc[d].dropna()
            if not mom_row.empty:
                top_n = min(10, len(mom_row))
                current_holdings = mom_row.nlargest(top_n).index.tolist()
                n_rebalances += 1

        if current_holdings:
            valid = [s for s in current_holdings if s in daily_returns.columns]
            if valid:
                day_rets = daily_returns.loc[d, valid].dropna()
                port_ret = float(day_rets.mean()) if not day_rets.empty else 0.0
            else:
                port_ret = 0.0
        else:
            port_ret = 0.0

        portfolio_rets.append(port_ret)

    ret_series = pd.Series(portfolio_rets, index=sorted_dates, dtype=float)
    equity_values = initial_equity * (1.0 + ret_series).cumprod()
    dt_index = pd.DatetimeIndex([pd.Timestamp(d) for d in sorted_dates])
    equity_curve = pd.Series(equity_values.values, index=dt_index, name="equity", dtype=float)

    # Approximate turnover cost: each rebalance costs ~0.5% of portfolio
    cost_per_rebalance_pct = 0.5
    total_turnover_cost_pct = (n_rebalances * cost_per_rebalance_pct) / max(len(sorted_dates), 1) * 252

    alpha = float(cfg.objective.get("alpha", 2.0))
    beta = float(cfg.objective.get("beta", 1.0))

    return _metrics_from_curve(
        equity_curve,
        start=sorted_dates[0],
        end=sorted_dates[-1],
        total_trades=n_rebalances,
        turnover_cost_pct=total_turnover_cost_pct,
        alpha=alpha,
        beta=beta,
    )


def random_entry_baseline(
    feature_df: pd.DataFrame,
    cfg: Config,
    initial_equity: float,
    seed: int = 42,
) -> FoldMetrics:
    """Random entries with the same swing SL/TP rules as the strategy.

    Entries are random so there is no skill premium; after costs the strategy
    should yield negative J, providing a lower-bound sanity check.

    Args:
        feature_df:     Wide-format close-price DataFrame (same format as
                        ``equal_weight_momentum_baseline``).
        cfg:            Strategy config for SL/TP multipliers and cost model.
        initial_equity: Starting capital (INR).
        seed:           NumPy random seed for reproducibility.

    Returns:
        :class:`FoldMetrics` with ``fold_id=-1``.
    """
    if feature_df.empty:
        today = date.today()
        return _empty_fold_metrics(today, today)

    df = _normalise_wide_index(feature_df)
    sorted_dates: list[date] = sorted(df.index)  # type: ignore[arg-type]

    if len(sorted_dates) < 16:
        logger.warning("random_entry_baseline: insufficient history (< 16 days)")
        return _empty_fold_metrics(sorted_dates[0], sorted_dates[-1])

    # Config parameters
    swing_exit = cfg.exit["swing"]
    atr_sl_mult: float = float(swing_exit["atr_sl_multiple"])
    rr_ratio: float = float(swing_exit["reward_risk_ratio"])
    max_hold_days: int = int(cfg.horizons["swing"]["max_hold_days"])
    risk_pct: float = float(cfg.risk["risk_per_trade_pct"])
    max_daily_entries: int = int(cfg.risk["max_daily_entries"])

    from trading_bot.backtest.costs import CostModel
    cost_model = CostModel(cfg)

    # ATR proxy: 14-day rolling std of daily returns × price
    daily_pct_returns = df.pct_change()
    atr_pct_rolling = daily_pct_returns.rolling(14).std()

    rng = np.random.default_rng(seed)
    date_to_idx: dict[date, int] = {d: i for i, d in enumerate(sorted_dates)}
    stocks = list(df.columns)

    equity = initial_equity
    total_trades = 0
    wins = 0
    total_cost_inr = 0.0
    equity_history: dict[date, float] = {}

    for i, entry_date in enumerate(sorted_dates):
        equity_history[entry_date] = equity

        if i < 15:  # Need ATR history
            continue

        n_candidates = min(max_daily_entries, max(1, len(stocks)))
        chosen = rng.choice(len(stocks), size=n_candidates, replace=False)

        for idx in chosen:
            stock = stocks[idx]
            entry_price = df.loc[entry_date, stock]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            atr_pct_val = atr_pct_rolling.loc[entry_date, stock]
            if pd.isna(atr_pct_val) or atr_pct_val <= 0:
                atr_pct_val = 0.015  # 1.5% fallback

            atr_abs = atr_pct_val * entry_price * atr_sl_mult
            if atr_abs <= 0:
                continue

            sl = entry_price - atr_abs
            tp = entry_price + rr_ratio * atr_abs

            risk_amount = equity * risk_pct / 100.0
            shares = max(1, int(risk_amount / atr_abs))

            # Simulate forward exit
            exit_price = entry_price
            exit_date = entry_date
            exited = False

            for j in range(i + 1, min(i + max_hold_days + 1, len(sorted_dates))):
                fwd_date = sorted_dates[j]
                fwd_price = df.loc[fwd_date, stock]
                if pd.isna(fwd_price):
                    continue
                if fwd_price <= sl:
                    exit_price = sl
                    exit_date = fwd_date
                    exited = True
                    break
                if fwd_price >= tp:
                    exit_price = tp
                    exit_date = fwd_date
                    wins += 1
                    exited = True
                    break

            if not exited:
                last_j = min(i + max_hold_days, len(sorted_dates) - 1)
                exit_date = sorted_dates[last_j]
                last_price = df.loc[exit_date, stock]
                exit_price = last_price if not pd.isna(last_price) else entry_price

            cost = cost_model.compute(entry_price, exit_price, shares, Horizon.SWING)
            net_pnl = (exit_price - entry_price) * shares - cost

            equity += net_pnl
            total_cost_inr += cost
            total_trades += 1

            # Record updated equity on exit date
            if exit_date in equity_history:
                equity_history[exit_date] = equity
            else:
                equity_history[exit_date] = equity

    # Rebuild continuous equity curve (carry-forward on non-exit days)
    eq_values: list[float] = []
    running = initial_equity
    for d in sorted_dates:
        if d in equity_history:
            running = equity_history[d]
        eq_values.append(running)

    dt_index = pd.DatetimeIndex([pd.Timestamp(d) for d in sorted_dates])
    equity_curve = pd.Series(eq_values, index=dt_index, name="equity", dtype=float)

    # Turnover cost as % of mean equity, annualised
    mean_equity = float(equity_curve.mean()) if not equity_curve.empty else initial_equity
    if mean_equity > 0 and total_trades > 0:
        turnover_cost_pct = (total_cost_inr / mean_equity) / max(len(sorted_dates), 1) * 252
    else:
        turnover_cost_pct = 0.0

    win_rate_val = wins / total_trades if total_trades > 0 else 0.0

    alpha = float(cfg.objective.get("alpha", 2.0))
    beta = float(cfg.objective.get("beta", 1.0))

    fm = _metrics_from_curve(
        equity_curve,
        start=sorted_dates[0],
        end=sorted_dates[-1],
        total_trades=total_trades,
        turnover_cost_pct=turnover_cost_pct,
        alpha=alpha,
        beta=beta,
    )
    fm.win_rate = win_rate_val
    return fm
