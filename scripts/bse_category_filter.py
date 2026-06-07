"""BSE corporate announcement category filters for download configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

# Names must match BSE API / website dropdown (strCat parameter).
CATEGORY_ALL = "all"
CATEGORY_RELEVANT = "relevant"

ALL_CATEGORY_NAMES: tuple[str, ...] = (
    "AGM/EGM",
    "Board Meeting",
    "Company Update",
    "Corp. Action",
    "Insider Trading / SAST",
    "Integrated Filing",
    "New Listing",
    "Result",
    "Others",
)

# Categories most useful for price / fundamental event models.
RELEVANT_CATEGORY_NAMES: tuple[str, ...] = (
    "Result",
    "Corp. Action",
    "Company Update",
    "Insider Trading / SAST",
    "Board Meeting",
)

# Routine board-meeting notices; outcomes and material agendas are kept.
DEFAULT_EXCLUDE_SUBCATEGORIES: tuple[str, ...] = (
    "Board Meeting Intimation",
    "Prior Intimation of Board Meeting pursuant to Reg 29 of SEBI (LODR)",
)


@dataclass(frozen=True)
class CategoryFilter:
    """Resolved category filter for one download run."""

    mode: str  # "all" | "relevant" | "custom"
    categories: Optional[tuple[str, ...]]  # None = fetch all categories in one API call
    exclude_subcategories: tuple[str, ...]

    def is_all(self) -> bool:
        return self.categories is None

    def meta_dict(self) -> dict[str, Any]:
        return {
            "category_filter": self.mode,
            "categories": list(self.categories) if self.categories else None,
            "exclude_subcategories": list(self.exclude_subcategories),
        }


def _normalize_name(name: str) -> str:
    return name.strip().casefold()


def subcategory_excluded(
    subcat_name: Optional[str],
    exclude_subcategories: Sequence[str],
) -> bool:
    if not subcat_name or not exclude_subcategories:
        return False
    sub = _normalize_name(str(subcat_name))
    return any(_normalize_name(ex) == sub for ex in exclude_subcategories)


def filter_announcement_rows(
    rows: list[dict[str, Any]],
    *,
    categories: Optional[Sequence[str]],
    exclude_subcategories: Sequence[str],
) -> list[dict[str, Any]]:
    """Post-filter rows by category allow-list and subcategory block-list."""
    allowed: Optional[set[str]] = None
    if categories:
        allowed = {_normalize_name(c) for c in categories}

    out: list[dict[str, Any]] = []
    for row in rows:
        cat = row.get("CATEGORYNAME")
        if allowed is not None:
            if not cat or _normalize_name(str(cat)) not in allowed:
                continue
        if subcategory_excluded(row.get("SUBCATNAME"), exclude_subcategories):
            continue
        out.append(row)
    return out


def _parse_categories_value(raw: Any) -> Optional[tuple[str, ...]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        key = raw.strip().casefold()
        if key in ("", CATEGORY_ALL, "none", "false"):
            return None
        if key == CATEGORY_RELEVANT:
            return RELEVANT_CATEGORY_NAMES
        raise ValueError(
            f"Unknown categories preset {raw!r}. Use 'all', 'relevant', or a list of category names."
        )
    if isinstance(raw, list):
        if not raw:
            return None
        names = [str(x).strip() for x in raw if str(x).strip()]
        if not names:
            return None
        return tuple(names)
    raise ValueError("'categories' must be 'all', 'relevant', or a list of category names.")


def _parse_exclude_subcategories(raw: Any, *, preset: str) -> tuple[str, ...]:
    if raw is None:
        if preset == CATEGORY_RELEVANT:
            return DEFAULT_EXCLUDE_SUBCATEGORIES
        return ()
    if isinstance(raw, list):
        return tuple(str(x).strip() for x in raw if str(x).strip())
    if isinstance(raw, bool) and not raw:
        return ()
    raise ValueError("'exclude_subcategories' must be a list of subcategory names or omitted.")


def parse_category_filter(raw: dict[str, Any]) -> CategoryFilter:
    """
    Config keys:
      categories: "all" | "relevant" | ["Result", ...]  (default "relevant")
      exclude_subcategories: list of SUBCATNAME values to drop (default for relevant preset)
    """
    cat_raw = raw.get("categories", CATEGORY_RELEVANT)
    categories = _parse_categories_value(cat_raw)

    if isinstance(cat_raw, str) and cat_raw.strip().casefold() == CATEGORY_RELEVANT:
        preset = CATEGORY_RELEVANT
    elif categories is None:
        preset = CATEGORY_ALL
    else:
        preset = "custom"

    excludes = _parse_exclude_subcategories(
        raw.get("exclude_subcategories"),
        preset=preset,
    )

    if categories is not None:
        unknown = [
            c for c in categories if _normalize_name(c) not in {_normalize_name(x) for x in ALL_CATEGORY_NAMES}
        ]
        if unknown:
            raise ValueError(
                f"Unknown BSE categories: {unknown}. "
                f"Known: {', '.join(ALL_CATEGORY_NAMES)}"
            )

    return CategoryFilter(mode=preset, categories=categories, exclude_subcategories=excludes)


def filters_match(meta: dict[str, Any], filt: CategoryFilter) -> bool:
    """True if cached symbol meta used the same category filter."""
    return meta.get("category_filter") == filt.meta_dict()["category_filter"] and meta.get(
        "categories"
    ) == filt.meta_dict()["categories"] and meta.get("exclude_subcategories") == list(
        filt.exclude_subcategories
    )
