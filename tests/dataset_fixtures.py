"""Shared helpers for synthetic dataset fixtures in tests."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd


def write_test_dataset(
    root: Path,
    *,
    symbol: str = "TESTCO",
    token: int = 9001,
    day: date | None = None,
    n_daily: int = 120,
) -> None:
    """Create a minimal dataset_nifty50-style tree under *root*."""
    day = day or date(2025, 12, 15)
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": 1,
        "universe": {"name": "test", "symbol_count": 1, "enriched_csv": "universe/universe_enriched.csv"},
        "date_range": {"from": "2024-01-01", "to": "2026-01-01"},
        "intervals": ["day"],
        "instruments": {"latest": "instruments/nse_eq_latest.csv"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))

    (root / "universe").mkdir(exist_ok=True)
    pd.DataFrame(
        [{"symbol": symbol, "instrument_token": token, "name": symbol, "exchange": "NSE", "found": True}]
    ).to_csv(root / "universe" / "universe_enriched.csv", index=False)

    (root / "instruments").mkdir(exist_ok=True)
    pd.DataFrame(
        [
            {
                "instrument_token": token,
                "exchange_token": 1,
                "tradingsymbol": symbol,
                "name": symbol,
                "last_price": 100,
                "expiry": "",
                "strike": 0,
                "tick_size": 0.05,
                "lot_size": 1,
                "instrument_type": "EQ",
                "segment": "NSE",
                "exchange": "NSE",
            }
        ]
    ).to_csv(root / "instruments" / "nse_eq_latest.csv", index=False)

    dates = pd.bdate_range("2024-06-01", periods=n_daily)
    daily = pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in dates],
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 2_000_000.0,
        }
    )
    (root / "ohlcv" / "day").mkdir(parents=True, exist_ok=True)
    daily.to_csv(root / "ohlcv" / "day" / f"{symbol}.csv", index=False)
