"""Quarterly walk-forward helpers for move predictor."""

from __future__ import annotations

from datetime import date, timedelta

from trading_bot.data.trading_calendar import trading_days_between


def quarter_key(d: date) -> tuple[int, int]:
    return d.year, (d.month - 1) // 3


def quarter_label(key: tuple[int, int]) -> str:
    year, q = key
    return f"{year}-Q{q + 1}"


def group_by_quarter(dates: list[date]) -> dict[tuple[int, int], list[date]]:
    groups: dict[tuple[int, int], list[date]] = {}
    for d in sorted(dates):
        groups.setdefault(quarter_key(d), []).append(d)
    return groups


def quarterly_walk_forward_folds(
    bt_dates: list[date],
    train_start: date,
) -> list[dict]:
    """
    Expanding-window quarterly folds for backtest dates.

    Each fold retrains on ``[train_start, last day before OOS quarter]`` and
    generates signals only for that quarter's sessions.
    """
    folds: list[dict] = []
    for q_key in sorted(group_by_quarter(bt_dates).keys()):
        oos_dates = group_by_quarter(bt_dates)[q_key]
        oos_start = min(oos_dates)
        prior = trading_days_between(train_start, oos_start - timedelta(days=1))
        if not prior:
            continue
        train_end = prior[-1]
        train_dates = trading_days_between(train_start, train_end)
        if not train_dates:
            continue
        folds.append(
            {
                "quarter": quarter_label(q_key),
                "train_start": train_start,
                "train_end": train_end,
                "oos_start": oos_start,
                "oos_end": max(oos_dates),
                "train_dates": train_dates,
                "oos_dates": oos_dates,
            }
        )
    return folds
