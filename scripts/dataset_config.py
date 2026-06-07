"""Load equity dataset build configuration from JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from download_config import _parse_intervals, load_symbols_from_csv, resolve_date_range


@dataclass(frozen=True)
class DatasetBuildConfig:
    dataset_root: Path
    universe_name: str
    symbols_csv: Path
    symbol_column: str
    from_date: str
    to_date: str
    intervals: List[str]
    chunk_days: Optional[int]
    sleep_seconds: float
    skip_existing: bool
    refresh_universe_from_nse: bool
    save_instruments: bool
    nse_constituents_url: Optional[str]

    @property
    def from_dt(self) -> datetime:
        return datetime.strptime(self.from_date, "%Y-%m-%d")

    @property
    def to_dt(self) -> datetime:
        return datetime.strptime(self.to_date, "%Y-%m-%d")

    @property
    def universe_dir(self) -> Path:
        return self.dataset_root / "universe"

    @property
    def instruments_dir(self) -> Path:
        return self.dataset_root / "instruments"

    @property
    def ohlcv_dir(self) -> Path:
        return self.dataset_root / "ohlcv"

    @property
    def meta_dir(self) -> Path:
        return self.dataset_root / "meta"


def load_dataset_config(path: Path) -> DatasetBuildConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    base = path.parent.resolve()
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object.")

    dataset_root = (base / str(raw.get("dataset_root", "dataset"))).resolve()
    universe_name = str(raw.get("universe_name", "default")).strip() or "default"
    symbols_csv = raw.get("symbols_csv")
    if not symbols_csv:
        symbols_csv = f"universe/{universe_name}.csv"

    symbol_column = str(raw.get("symbol_column", "symbol"))
    from_date, to_date = resolve_date_range(
        raw,
        default_from="2021-01-01",
        default_to=datetime.now().strftime("%Y-%m-%d"),
    )
    if to_date < from_date:
        raise ValueError("to_date must be on or after from_date.")

    intervals = _parse_intervals(raw)

    chunk_days = raw.get("chunk_days")
    if chunk_days is not None:
        chunk_days = int(chunk_days)
        if chunk_days <= 0:
            raise ValueError("chunk_days must be positive.")

    nse_url = raw.get("nse_constituents_url")
    if nse_url is not None:
        nse_url = str(nse_url).strip() or None

    return DatasetBuildConfig(
        dataset_root=dataset_root,
        universe_name=universe_name,
        symbols_csv=(base / symbols_csv).resolve(),
        symbol_column=symbol_column,
        from_date=from_date,
        to_date=to_date,
        intervals=intervals,
        chunk_days=chunk_days,
        sleep_seconds=float(raw.get("sleep_seconds", 0.4)),
        skip_existing=bool(raw.get("skip_existing", True)),
        refresh_universe_from_nse=bool(raw.get("refresh_universe_from_nse", False)),
        save_instruments=bool(raw.get("save_instruments", True)),
        nse_constituents_url=nse_url,
    )


def load_universe_symbols(cfg: DatasetBuildConfig) -> List[str]:
    return load_symbols_from_csv(cfg.symbols_csv, cfg.symbol_column)
