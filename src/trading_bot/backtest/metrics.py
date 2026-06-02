"""Performance metrics and OOS objective J computation."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from trading_bot.types import FoldMetrics, Position, TradeStatus


# ── Risk-adjusted return metrics ───────────────────────────────────────────────

def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    annualisation: int = 252,
) -> float:
    """Sortino ratio using downside deviation of daily returns.

    Args:
        returns:       Daily return series (decimal, not percent).
        risk_free:     Annual risk-free rate (decimal). Divided by *annualisation*
                       to convert to a per-period rate.
        annualisation: Number of trading periods per year.

    Returns:
        Sortino ratio. Returns 0.0 when the series is empty or has no downside.
    """
    if returns.empty:
        return 0.0

    daily_rf = risk_free / annualisation
    excess = returns - daily_rf

    downside = excess[excess < 0]
    if downside.empty:
        return 0.0

    downside_variance = (downside**2).mean()
    downside_std_ann = math.sqrt(downside_variance * annualisation)
    if downside_std_ann == 0.0:
        return 0.0

    mean_excess_ann = excess.mean() * annualisation
    return mean_excess_ann / downside_std_ann


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction.

    Returns:
        Max drawdown in [0, 1]. E.g. 0.25 means a 25 % drawdown.
        Returns 0.0 for empty or flat curves.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0

    rolling_peak = equity_curve.cummax()
    # Avoid division by zero on zero-equity curves
    safe_peak = rolling_peak.replace(0.0, np.nan)
    drawdowns = (equity_curve - rolling_peak) / safe_peak
    worst = float(drawdowns.min())
    return max(0.0, -worst)


def calmar_ratio(equity_curve: pd.Series, annualisation: int = 252) -> float:
    """Calmar ratio: CAGR / max_drawdown.

    Returns:
        Calmar ratio. Returns 0.0 when max_drawdown is 0 or curve is too short.
    """
    mdd = max_drawdown(equity_curve)
    if mdd == 0.0 or equity_curve.empty or len(equity_curve) < 2:
        return 0.0

    n_years = len(equity_curve) / annualisation
    if n_years <= 0:
        return 0.0

    start_val = equity_curve.iloc[0]
    end_val = equity_curve.iloc[-1]
    if start_val <= 0:
        return 0.0

    cagr = (end_val / start_val) ** (1.0 / n_years) - 1.0
    return cagr / mdd


# ── Trade-level metrics ────────────────────────────────────────────────────────

def expectancy_r(trades: list[Position]) -> float:
    """Mean R-multiple across all closed trades.

    Returns:
        Mean R-multiple. Returns ``float('nan')`` when there are no closed trades
        with a computable R-multiple.
    """
    r_multiples: list[float] = [
        t.r_multiple
        for t in trades
        if t.status != TradeStatus.OPEN and t.r_multiple is not None
    ]  # type: ignore[misc]
    if not r_multiples:
        return float("nan")
    return float(np.mean(r_multiples))


def win_rate(trades: list[Position]) -> float:
    """Fraction of closed trades where net_pnl > 0.

    Returns:
        Win rate in [0, 1]. Returns 0.0 when no closed trades exist.
    """
    closed = [
        t for t in trades
        if t.status != TradeStatus.OPEN and t.net_pnl is not None
    ]
    if not closed:
        return 0.0
    wins = sum(1 for t in closed if t.net_pnl > 0)
    return wins / len(closed)


# ── Objective J ───────────────────────────────────────────────────────────────

def compute_objective_j(
    fold_metrics: FoldMetrics,
    alpha: float = 2.0,
    beta: float = 1.0,
) -> float:
    """Compute OOS objective J for hyperparameter and model selection.

    J = Sortino − α × MaxDD − β × turnover_cost_pct

    Hard rejects (returns ``-inf``) when avg_daily_entries > 10 to prevent
    over-trading strategies from being promoted.

    Args:
        fold_metrics:      Populated FoldMetrics for the OOS fold.
        alpha:             Max-drawdown penalty weight (default 2.0 per YAML).
        beta:              Turnover-cost penalty weight (default 1.0 per YAML).

    Returns:
        Scalar J. Lower is worse; ``-inf`` means hard-rejected.
    """
    if fold_metrics.avg_daily_entries > 10:
        return float("-inf")

    return (
        fold_metrics.sortino
        - alpha * fold_metrics.max_drawdown
        - beta * fold_metrics.turnover_cost_pct
    )


# ── Equity curve reconstruction ────────────────────────────────────────────────

def build_equity_curve(
    trades: list[Position],
    initial_equity: float,
    all_dates: list[date],
) -> pd.Series:
    """Reconstruct a daily equity curve from a list of closed positions.

    The curve starts at *initial_equity* on the first date. Each closed trade's
    ``net_pnl`` is credited on its ``exit_date``. Dates with no exits keep the
    previous day's equity level.

    Args:
        trades:         All positions (open or closed; open ones are ignored).
        initial_equity: Starting capital in INR.
        all_dates:      Ordered list of every trading date in the period.

    Returns:
        ``pd.Series`` with a ``pd.DatetimeIndex`` and ``float`` values (INR equity).
    """
    if not all_dates:
        return pd.Series(dtype=float, name="equity")

    # Aggregate net PnL by exit date
    pnl_by_date: dict[date, float] = {}
    for t in trades:
        if (
            t.status != TradeStatus.OPEN
            and t.exit_date is not None
            and t.net_pnl is not None
        ):
            pnl_by_date[t.exit_date] = pnl_by_date.get(t.exit_date, 0.0) + t.net_pnl

    equity = initial_equity
    values: list[float] = []
    for d in all_dates:
        equity += pnl_by_date.get(d, 0.0)
        values.append(equity)

    dt_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    return pd.Series(values, index=dt_index, name="equity", dtype=float)
