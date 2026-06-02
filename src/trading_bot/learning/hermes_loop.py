"""Hermes Agent integration — reads OOS reports and proposes strategy patches via LLM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from trading_bot.config import Config
from trading_bot.types import FoldMetrics

logger = logging.getLogger(__name__)

_HERMES_CONFIG_PATH = Path("hermes/hermes_config.yaml")


class HermesLoop:
    """Reads OOS fold reports and calls an LLM to propose strategy improvements.

    The LLM receives structured fold metrics, SHAP feature importance, and the
    current ``strategy.yaml`` parameters, then proposes ONE concrete change as
    a YAML diff or new Python function.

    If the ``anthropic`` package is not installed, or if ``ANTHROPIC_API_KEY``
    is not set, the loop logs a warning and returns ``None`` without crashing.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._hermes_cfg: dict[str, Any] = self._load_hermes_config()
        self._report_dir = Path(
            self._hermes_cfg.get("reports", {}).get("directory", "hermes/reports")
        )
        self._report_dir.mkdir(parents=True, exist_ok=True)

    # ── Config loading ──────────────────────────────────────────────────────

    @staticmethod
    def _load_hermes_config() -> dict[str, Any]:
        if _HERMES_CONFIG_PATH.exists():
            try:
                with open(_HERMES_CONFIG_PATH) as f:
                    return yaml.safe_load(f) or {}
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load hermes_config.yaml: %s", exc)
        return {}

    # ── Public API ──────────────────────────────────────────────────────────

    def should_run(self, fold_id: int) -> bool:
        """Return True once enough folds have accumulated for Hermes to act."""
        min_folds: int = (
            self._hermes_cfg.get("skills", {}).get("min_folds_before_creation", 3)
        )
        return fold_id >= min_folds

    def build_prompt(
        self,
        fold_metrics: FoldMetrics,
        shap_path: Path,
        prev_shap_path: Path | None,
    ) -> str:
        """Construct the LLM prompt for proposing a strategy improvement.

        Includes:
        - Fold performance metrics
        - Top-5 and bottom-5 features by mean |SHAP|
        - Feature importance delta vs previous fold (when available)
        - Current ``strategy.yaml`` parameters
        - Clear task instructions with output format requirements
        """
        sections: list[str] = []

        # ── Fold metrics ────────────────────────────────────────────────
        sections.append(self._format_fold_metrics(fold_metrics))

        # ── SHAP feature importance ─────────────────────────────────────
        shap_text, shap_df = self._format_shap(shap_path)
        sections.append(shap_text)

        # ── Feature importance delta ────────────────────────────────────
        if prev_shap_path is not None and prev_shap_path.exists() and shap_df is not None:
            delta_text = self._format_shap_delta(shap_df, prev_shap_path)
            sections.append(delta_text)

        # ── Current strategy params ─────────────────────────────────────
        sections.append(self._format_strategy_yaml())

        # ── Task instructions ───────────────────────────────────────────
        sections.append(self._task_instructions())

        return "\n\n".join(sections)

    def run_fold(
        self,
        fold_id: int,
        fold_metrics: FoldMetrics,
        report_dir: Path,
    ) -> str | None:
        """Run Hermes for *fold_id* if enough folds have accumulated.

        Saves the raw LLM response to ``hermes/reports/hermes_fold_{fold_id}.txt``.
        Returns the response string, or None if skipped/unavailable.
        """
        if not self.should_run(fold_id):
            logger.info(
                "Hermes skipping fold %d (min folds not yet reached).", fold_id
            )
            return None

        shap_prefix: str = self._hermes_cfg.get("reports", {}).get("shap_prefix", "shap_fold_")
        shap_path = Path(report_dir) / f"{shap_prefix}{fold_id}.csv"
        prev_shap_path = Path(report_dir) / f"{shap_prefix}{fold_id - 1}.csv" if fold_id > 0 else None

        if not shap_path.exists():
            logger.warning(
                "Hermes: SHAP file not found at %s; skipping fold %d.", shap_path, fold_id
            )
            return None

        prompt = self.build_prompt(fold_metrics, shap_path, prev_shap_path)

        backend: str = (
            self._hermes_cfg.get("llm", {}).get("backend", "claude")
            or self.cfg.hermes.get("llm_backend", "claude")
        )

        if backend.lower() != "claude":
            logger.info("Hermes backend=%s; skipping LLM call (only 'claude' implemented).", backend)
            return None

        response = self._call_claude(prompt)
        if response is None:
            return None

        out_path = self._report_dir / f"hermes_fold_{fold_id}.txt"
        try:
            out_path.write_text(response, encoding="utf-8")
            logger.info("Hermes response for fold %d saved to %s", fold_id, out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save Hermes response: %s", exc)

        return response

    def _call_claude(self, prompt: str) -> str | None:
        """Call the Anthropic Claude API with *prompt*.

        Returns the response text, or None on any failure (import error,
        missing API key, or API error).
        """
        try:
            import anthropic
        except ImportError:
            logger.warning(
                "Hermes: 'anthropic' package not installed. "
                "Run `pip install anthropic` to enable LLM calls. Returning None."
            )
            return None

        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning(
                "Hermes: ANTHROPIC_API_KEY environment variable is not set. Returning None."
            )
            return None

        llm_cfg = self._hermes_cfg.get("llm", {})
        model: str = llm_cfg.get("claude_model", "claude-sonnet-4-5")
        max_tokens: int = int(llm_cfg.get("max_tokens", 4096))
        temperature: float = float(llm_cfg.get("temperature", 0.3))

        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hermes: Claude API call failed: %s", exc)
            return None

    # ── Prompt construction helpers ─────────────────────────────────────────

    @staticmethod
    def _format_fold_metrics(m: FoldMetrics) -> str:
        return (
            f"## Fold {m.fold_id} Out-of-Sample Performance\n"
            f"- Period: {m.oos_start} → {m.oos_end}\n"
            f"- Sortino ratio:        {m.sortino:.3f}\n"
            f"- Max drawdown:         {m.max_drawdown:.2%}\n"
            f"- Expectancy (R):       {m.expectancy_r:.3f}\n"
            f"- Win rate:             {m.win_rate:.1%}  "
            f"(swing: {m.swing_win_rate:.1%}, positional: {m.positional_win_rate:.1%})\n"
            f"- Total trades:         {m.total_trades}  "
            f"(swing: {m.swing_trades}, positional: {m.positional_trades})\n"
            f"- Avg daily entries:    {m.avg_daily_entries:.2f}\n"
            f"- Turnover cost:        {m.turnover_cost_pct:.3f}%\n"
            f"- Objective J:          {m.objective_j:.4f}  "
            f"(beats baseline: {m.beats_baseline})\n"
        )

    @staticmethod
    def _format_shap(shap_path: Path) -> tuple[str, pd.DataFrame | None]:
        try:
            df = pd.read_csv(shap_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read SHAP file %s: %s", shap_path, exc)
            return "## Feature Importance (SHAP)\nUnavailable.\n", None

        top5 = df.nsmallest(5, "rank")[["feature", "mean_abs_shap", "rank"]].to_string(index=False)
        bottom5 = df.nlargest(5, "rank")[["feature", "mean_abs_shap", "rank"]].to_string(index=False)

        text = (
            f"## Feature Importance (mean |SHAP|)\n\n"
            f"### Top 5 features (most predictive)\n```\n{top5}\n```\n\n"
            f"### Bottom 5 features (least predictive)\n```\n{bottom5}\n```\n"
        )
        return text, df

    @staticmethod
    def _format_shap_delta(current_df: pd.DataFrame, prev_shap_path: Path) -> str:
        try:
            prev_df = pd.read_csv(prev_shap_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read previous SHAP file %s: %s", prev_shap_path, exc)
            return ""

        merged = current_df.merge(
            prev_df[["feature", "rank", "mean_abs_shap"]].rename(
                columns={"rank": "rank_prev", "mean_abs_shap": "shap_prev"}
            ),
            on="feature",
            how="left",
        )
        merged["rank_delta"] = merged["rank_prev"] - merged["rank"]
        merged["shap_delta"] = merged["mean_abs_shap"] - merged["shap_prev"]

        table = merged[["feature", "rank", "rank_delta", "mean_abs_shap", "shap_delta"]].to_string(index=False)
        return f"## Feature Importance Delta (vs previous fold)\n```\n{table}\n```\n"

    def _format_strategy_yaml(self) -> str:
        try:
            strategy_path = Path("config/strategy.yaml")
            if strategy_path.exists():
                strategy_text = strategy_path.read_text(encoding="utf-8")
            else:
                strategy_text = yaml.dump(self.cfg._raw, default_flow_style=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read strategy.yaml: %s", exc)
            strategy_text = "(unavailable)"
        return f"## Current strategy.yaml\n```yaml\n{strategy_text}\n```\n"

    @staticmethod
    def _task_instructions() -> str:
        return (
            "## Your Task\n\n"
            "You are a quantitative trading research assistant. Based on the fold metrics "
            "and feature importance data above, propose **exactly ONE** concrete improvement "
            "to the trading strategy. Your proposal must be one of:\n\n"
            "1. **A parameter tweak** — a diff to `strategy.yaml` (e.g., adjust `atr_sl_multiple`, "
            "`min_win_prob`, `top_n_candidates`, or similar).\n"
            "2. **A new feature** — a Python function to be added to "
            "`src/trading_bot/features/indicators.py` and wired into `add_all_features()`.\n\n"
            "Format your response as:\n\n"
            "### Proposal\n"
            "<One sentence summary of the change.>\n\n"
            "### Rationale\n"
            "<2–4 sentences explaining why this change addresses the observed metrics.>\n\n"
            "### Expected J Impact\n"
            "<Predict the direction and rough magnitude of change to Objective J.>\n\n"
            "### Implementation\n"
            "```yaml  (or ```python for a new feature)\n"
            "<Minimal, self-contained diff or function.>\n"
            "```\n\n"
            "Be specific and conservative. Do not propose multiple changes at once. "
            "Do not change core risk management parameters (risk_per_trade_pct, "
            "max_gross_exposure_pct) without a compelling quantitative justification."
        )
