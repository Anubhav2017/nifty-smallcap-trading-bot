"""NSE index metadata (universe membership lives in dataset_*/universe/)."""

from __future__ import annotations

from dataclasses import dataclass

from trading_bot.config import Config


@dataclass(frozen=True)
class IndexSpec:
    key: str
    slug: str
    index_patterns: tuple[str, ...]
    description: str


INDEX_SPECS: dict[str, IndexSpec] = {
    "NIFTY_50": IndexSpec(
        key="NIFTY_50",
        slug="nifty50",
        index_patterns=("NIFTY 50", "NIFTY_50", "NIFTY50"),
        description="Nifty 50",
    ),
    "NIFTY_SMALLCAP_100": IndexSpec(
        key="NIFTY_SMALLCAP_100",
        slug="nifty_smallcap100",
        index_patterns=("NIFTY_SMLCAP_100", "NIFTY SMLCAP 100", "NIFTYSMLCAP100"),
        description="Nifty Smallcap 100",
    ),
    "NIFTY_SMALLCAP_250": IndexSpec(
        key="NIFTY_SMALLCAP_250",
        slug="nifty_smallcap250",
        index_patterns=("NIFTY_SMLCAP_250", "NIFTY SMLCAP 250", "NIFTYSMLCAP250"),
        description="Nifty Smallcap 250",
    ),
}

INDEX_ALIASES: dict[str, str] = {
    "NIFTY_SMALLCAP_200": "NIFTY_SMALLCAP_250",
    "SMALLCAP_200": "NIFTY_SMALLCAP_250",
    "SMALLCAP_250": "NIFTY_SMALLCAP_250",
}


def resolve_index_key(raw: str) -> str:
    key = raw.strip().upper()
    return INDEX_ALIASES.get(key, key)


def get_index_spec(cfg: Config) -> IndexSpec:
    key = resolve_index_key(str(cfg.universe.get("index", "NIFTY_SMALLCAP_250")))
    if key not in INDEX_SPECS:
        known = ", ".join(sorted(INDEX_SPECS))
        raise ValueError(f"Unknown universe index '{key}'. Known: {known}")
    return INDEX_SPECS[key]
