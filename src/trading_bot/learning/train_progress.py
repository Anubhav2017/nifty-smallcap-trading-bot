"""Progress logging helpers for the training CLI."""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger("trading_bot.train")


def configure_train_logging() -> None:
    """Ensure train progress lines are visible when invoked from the CLI."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(logging.INFO)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("trading_bot").setLevel(logging.INFO)


def train_step(step: int, total: int, message: str, **stats: Any) -> None:
    """Emit a numbered training progress line."""
    suffix = ", ".join(f"{k}={v}" for k, v in stats.items())
    line = f"[train {step}/{total}] {message}"
    if suffix:
        line = f"{line} ({suffix})"
    logger.info(line)
