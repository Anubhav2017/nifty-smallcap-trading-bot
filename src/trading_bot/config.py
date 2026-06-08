"""Load and expose strategy config from config/move_predictor.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "config" / "move_predictor.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)


class Config:
    """Thin wrapper around the YAML config dict with typed attribute access."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._raw = load_config(path)

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    @property
    def data(self) -> dict:
        return self._raw.get("data", {})

    @property
    def universe(self) -> dict:
        return self._raw["universe"]

    @property
    def horizons(self) -> dict:
        return self._raw["horizons"]

    @property
    def entry(self) -> dict:
        return self._raw["entry"]

    @property
    def exit(self) -> dict:
        return self._raw["exit"]

    @property
    def risk(self) -> dict:
        return self._raw["risk"]

    @property
    def costs(self) -> dict:
        return self._raw["costs"]
