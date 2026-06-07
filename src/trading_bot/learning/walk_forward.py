"""Walk-forward splitter and orchestration runner."""

from __future__ import annotations

import csv
import logging
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.backtest.costs import CostModel
from trading_bot.backtest.engine import BacktestEngine
from trading_bot.backtest.metrics import (
    calmar_ratio,
    compute_objective_j,
    expectancy_r,
    max_drawdown,
    sortino_ratio,
    win_rate,
)
from trading_bot.config import Config
from trading_bot.types import FoldMetrics, Horizon, Position, TradeStatus

logger = logging.getLogger(__name__)

_FEATURE_LOOKBACK_DAYS = 400
_TRADING_DAYS_PER_WEEK = 5
_TRADING_DAYS_PER_MONTH = 21
_TRADING_DAYS_PER_YEAR = 252


def _walk_forward_window_days(wf: dict, *, weeks_key: str, months_key: str) -> int:
    """Resolve a walk-forward window from weeks (preferred) or months."""
    if weeks_key in wf and wf[weeks_key] is not None:
        return round(float(wf[weeks_key]) * _TRADING_DAYS_PER_WEEK)
    return round(float(wf[months_key]) * _TRADING_DAYS_PER_MONTH)


# ── Graceful module imports ────────────────────────────────────────────────────

def _try_import(module_path: str) -> object | None:
    """Return the imported module or ``None`` with a logged warning."""
    try:
        import importlib
        return importlib.import_module(module_path)
    except ImportError:
        logger.warning("Optional module '%s' is not available; some features will be skipped.", module_path)
        return None


# ── Walk-forward splitter ──────────────────────────────────────────────────────

class WalkForwardSplitter:
    """Generate non-overlapping (train, validate) date-list pairs.

    Walk-forward parameters (from config):
    - ``train_years``:       Length of the training window in years.
    - ``validate_weeks``:    OOS window in weeks (preferred over ``validate_months``).
    - ``step_weeks``:        Roll-forward step in weeks (preferred over ``step_months``).
    - ``validate_months``: Fallback OOS window when ``validate_weeks`` is omitted.
    - ``step_months``:       Fallback roll-forward step when ``step_weeks`` is omitted.

    Weeks/months/years are approximated using trading-day counts
    (252 days/year, 21 days/month, 5 days/week).
    """

    def __init__(self, cfg: Config) -> None:
        wf = cfg.walk_forward
        self._train_days: int = round(float(wf["train_years"]) * _TRADING_DAYS_PER_YEAR)
        self._validate_days: int = _walk_forward_window_days(
            wf, weeks_key="validate_weeks", months_key="validate_months"
        )
        self._step_days: int = _walk_forward_window_days(
            wf, weeks_key="step_weeks", months_key="step_months"
        )

    def generate_folds(
        self,
        all_dates: list[date],
    ) -> list[tuple[list[date], list[date]]]:
        """Return ``(train_dates, validate_dates)`` pairs.

        Args:
            all_dates: Sorted list of trading dates covering the full period.

        Returns:
            List of ``(train_dates, validate_dates)`` tuples.  The list is
            empty when *all_dates* is shorter than one full train+validate
            window.
        """
        n = len(all_dates)
        min_length = self._train_days + self._validate_days
        if n < min_length:
            logger.warning(
                "WalkForwardSplitter: only %d dates available; need at least %d "
                "(train=%d + validate=%d). No folds generated.",
                n, min_length, self._train_days, self._validate_days,
            )
            return []

        folds: list[tuple[list[date], list[date]]] = []
        start_idx = 0

        while True:
            train_end_idx = start_idx + self._train_days
            val_end_idx = train_end_idx + self._validate_days

            if val_end_idx > n:
                break  # Not enough dates for a complete validate window

            train_dates = all_dates[start_idx:train_end_idx]
            val_dates = all_dates[train_end_idx:val_end_idx]
            folds.append((train_dates, val_dates))

            start_idx += self._step_days

        return folds


# ── Walk-forward runner ────────────────────────────────────────────────────────

class WalkForwardRunner:
    """Orchestrate the full walk-forward training and evaluation pipeline.

    Each fold:
    1. Loads OHLCV data via ``trading_bot.data`` (gracefully skipped if missing).
    2. Builds features via ``trading_bot.features``.
    3. Trains models via ``trading_bot.models``.
    4. Generates signals and runs an OOS backtest.
    5. Computes FoldMetrics and compares against baseline J.
    6. Saves per-fold CSVs to *report_dir*.

    Any step may fail with a logged warning; the fold is skipped but the run
    continues.  The runner returns a summary dict even when zero folds complete.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    @staticmethod
    def _ohlcv_end_for_labels(cfg: Config, end: date) -> date:
        """Extend OHLCV load through *end* so forward labels have future bars."""
        max_hold = max(int(cfg.horizons[h.value]["max_hold_days"]) for h in Horizon)
        calendar_days = max(math.ceil(max_hold * 7 / 5), max_hold) + 5
        return end + timedelta(days=calendar_days)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        start_date: str,
        end_date: str,
        report_dir: Path,
        *,
        model_dir: Path | None = None,
        update_each_fold: bool = False,
    ) -> dict:
        """Orchestrate the full walk-forward.

        Args:
            start_date:  ISO-format start date, e.g. ``"2020-01-01"``.
            end_date:    ISO-format end date, e.g. ``"2024-12-31"``.
            report_dir:  Directory where per-fold CSVs will be written.
            model_dir:   Load this bundle for fold 0 (and keep fixed, or as warm
                         start when ``update_each_fold`` is True).
            update_each_fold: After each fold's OOS validation, retrain on the
                         next fold's train window. Requires ``model_dir``.

        Returns:
            ``dict`` with keys:
            - ``folds_completed`` (int)
            - ``folds_beating_baseline`` (int)
            - ``mean_oos_j`` (float; NaN when no folds completed)
            - ``fixed_model`` (str | None)
            - ``update_each_fold`` (bool)
        """
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        if update_each_fold and model_dir is None:
            raise ValueError("update_each_fold requires model_dir (initial bundle to load).")

        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        current_bundle = None
        if model_dir is not None:
            from trading_bot.models.training import load as load_model_bundle

            current_bundle = load_model_bundle(self._cfg, Path(model_dir))
            if not current_bundle.ranker._fitted:
                raise RuntimeError(f"No fitted ranker found in {model_dir}")
            if update_each_fold:
                logger.info(
                    "Walk-forward warm-start from %s; retrain after each fold",
                    model_dir,
                )
            else:
                logger.info("Walk-forward using fixed model from %s", model_dir)

        # ── Optional module imports ───────────────────────────────────────────
        _data_mod = _try_import("trading_bot.data")
        _features_mod = _try_import("trading_bot.features")
        _models_mod = _try_import("trading_bot.models")
        _risk_mod = _try_import("trading_bot.risk")

        if _data_mod is None:
            logger.warning(
                "trading_bot.data is unavailable — walk-forward cannot load OHLCV data. "
                "Returning empty summary."
            )
            return {"folds_completed": 0, "folds_beating_baseline": 0, "mean_oos_j": float("nan")}

        # ── Build trading calendar ────────────────────────────────────────────
        all_dates = self._generate_trading_dates(start, end)

        # ── Generate folds ────────────────────────────────────────────────────
        splitter = WalkForwardSplitter(self._cfg)
        folds = splitter.generate_folds(all_dates)

        if not folds:
            logger.warning("No walk-forward folds generated for [%s, %s].", start, end)
            return {"folds_completed": 0, "folds_beating_baseline": 0, "mean_oos_j": float("nan")}

        # ── Baseline (buy-and-hold) — computed once over the full period ──────
        baseline_j: float | None = None
        try:
            from trading_bot.backtest.baselines import buy_and_hold_baseline

            index_ohlcv = self._load_index_ohlcv(start, end)
            if index_ohlcv is not None and not index_ohlcv.empty:
                bh = buy_and_hold_baseline(index_ohlcv, start, end, 1_000_000.0)
                baseline_j = bh.objective_j
                logger.info("Buy-and-hold baseline J = %.4f", baseline_j)
        except Exception as exc:
            logger.warning("Could not compute buy-and-hold baseline: %s", exc)

        # ── Fold loop ─────────────────────────────────────────────────────────
        folds_completed = 0
        folds_beating_baseline = 0
        oos_j_values: list[float] = []
        fold_rows: list[dict] = []
        alpha = float(self._cfg.objective.get("alpha", 2.0))
        beta = float(self._cfg.objective.get("beta", 1.0))

        for fold_id, (train_dates, val_dates) in enumerate(folds):
            model_source = "initial" if fold_id == 0 and model_dir else (
                "retrained" if model_dir and update_each_fold and fold_id > 0 else (
                    "retrained" if model_dir is None else "fixed"
                )
            )
            logger.info(
                "Fold %d (%s): train %s → %s | validate %s → %s",
                fold_id,
                model_source,
                train_dates[0],
                train_dates[-1],
                val_dates[0],
                val_dates[-1],
            )

            bundle_for_fold = current_bundle if model_dir is not None else None

            try:
                fold_metrics = self._run_fold(
                    fold_id=fold_id,
                    train_dates=train_dates,
                    val_dates=val_dates,
                    data_mod=_data_mod,
                    features_mod=_features_mod,
                    models_mod=_models_mod,
                    risk_mod=_risk_mod,
                    alpha=alpha,
                    beta=beta,
                    fixed_bundle=bundle_for_fold,
                )
            except Exception as exc:
                logger.error("Fold %d failed unexpectedly: %s", fold_id, exc, exc_info=True)
                continue

            if fold_metrics is None:
                continue

            # Compare to baseline
            if baseline_j is not None and math.isfinite(fold_metrics.objective_j):
                fold_metrics.beats_baseline = fold_metrics.objective_j > baseline_j
            else:
                fold_metrics.beats_baseline = False

            if fold_metrics.beats_baseline:
                folds_beating_baseline += 1

            if math.isfinite(fold_metrics.objective_j):
                oos_j_values.append(fold_metrics.objective_j)

            folds_completed += 1
            self._save_fold_csv(fold_metrics, report_dir, fold_id)
            fold_rows.append(
                {
                    "fold_id": fold_metrics.fold_id,
                    "train_start": fold_metrics.train_start.isoformat(),
                    "train_end": fold_metrics.train_end.isoformat(),
                    "oos_start": fold_metrics.oos_start.isoformat(),
                    "oos_end": fold_metrics.oos_end.isoformat(),
                    "sortino": fold_metrics.sortino,
                    "max_drawdown": fold_metrics.max_drawdown,
                    "calmar": fold_metrics.calmar,
                    "win_rate": fold_metrics.win_rate,
                    "expectancy_r": fold_metrics.expectancy_r,
                    "total_trades": fold_metrics.total_trades,
                    "avg_daily_entries": fold_metrics.avg_daily_entries,
                    "turnover_cost_pct": fold_metrics.turnover_cost_pct,
                    "objective_j": fold_metrics.objective_j,
                    "beats_baseline": fold_metrics.beats_baseline,
                    "initial_model": str(model_dir) if model_dir else "",
                    "model_source": model_source,
                    "update_each_fold": update_each_fold,
                }
            )

            if update_each_fold and fold_id + 1 < len(folds):
                next_train_dates, _ = folds[fold_id + 1]
                logger.info(
                    "Retraining for fold %d on train window %s → %s",
                    fold_id + 1,
                    next_train_dates[0],
                    next_train_dates[-1],
                )
                try:
                    current_bundle = self._retrain_bundle(
                        next_train_dates,
                        data_mod=_data_mod,
                    )
                except Exception as exc:
                    logger.error(
                        "Retrain after fold %d failed: %s — later folds use prior model",
                        fold_id,
                        exc,
                    )

        if update_each_fold and current_bundle is not None:
            updated_dir = report_dir / "updated_model"
            from trading_bot.models.training import save as save_models

            save_models(current_bundle, updated_dir)
            logger.info("Final updated model saved to %s", updated_dir)

        if fold_rows:
            summary_path = report_dir / "fold_summary.csv"
            pd.DataFrame(fold_rows).to_csv(summary_path, index=False)
            logger.info("Fold summary saved to %s", summary_path)

        mean_oos_j = float(np.mean(oos_j_values)) if oos_j_values else float("nan")
        logger.info(
            "Walk-forward complete: %d/%d folds done, %d beat baseline, mean J=%.4f",
            folds_completed, len(folds), folds_beating_baseline, mean_oos_j,
        )

        return {
            "folds_completed": folds_completed,
            "folds_beating_baseline": folds_beating_baseline,
            "mean_oos_j": mean_oos_j,
            "initial_model": str(model_dir) if model_dir else None,
            "update_each_fold": update_each_fold,
            "updated_model_dir": str(report_dir / "updated_model")
            if update_each_fold and current_bundle is not None
            else None,
        }

    def train_final(
        self,
        start_date: str,
        end_date: str,
        model_dir: Path,
    ) -> None:
        """Train on the full date range and save models to *model_dir*.

        Args:
            start_date: ISO-format start date.
            end_date:   ISO-format end date.
            model_dir:  Directory where trained model artefacts will be written.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        _data_mod = _try_import("trading_bot.data")
        _features_mod = _try_import("trading_bot.features")
        _models_mod = _try_import("trading_bot.models")

        if _data_mod is None:
            logger.error("trading_bot.data is unavailable — cannot train final model.")
            return

        all_dates = self._generate_trading_dates(start, end)

        try:
            ohlcv_end = self._ohlcv_end_for_labels(self._cfg, end)
            ohlcv_by_token = self._load_ohlcv(start, ohlcv_end)
        except Exception as exc:
            logger.error("Failed to load OHLCV for final training: %s", exc)
            return

        if not ohlcv_by_token:
            logger.warning("No OHLCV data found for final training [%s, %s].", start, end)
            return

        if _features_mod is None or _models_mod is None:
            logger.warning(
                "trading_bot.features or trading_bot.models unavailable — "
                "cannot complete final training."
            )
            return

        try:
            features = getattr(_features_mod, "build", None)
            if features is not None:
                feature_data = features(self._cfg, ohlcv_by_token, all_dates)
            else:
                logger.warning("trading_bot.features.build not found.")
                return

            trainer = getattr(_models_mod, "train", None)
            if trainer is not None:
                model = trainer(self._cfg, feature_data, all_dates)
            else:
                logger.warning("trading_bot.models.train not found.")
                return

            saver = getattr(_models_mod, "save", None)
            if saver is not None:
                saver(model, model_dir)
                logger.info("Final model saved to %s", model_dir)
            else:
                logger.warning("trading_bot.models.save not found; model not persisted.")

        except Exception as exc:
            logger.error("Final training failed: %s", exc, exc_info=True)

    def _retrain_bundle(
        self,
        train_dates: list[date],
        *,
        data_mod: object,
    ):
        """Retrain ranker/classifiers on *train_dates*."""
        from trading_bot.features.build import build
        from trading_bot.models.training import train as train_models

        fetch_start = train_dates[0] - timedelta(days=_FEATURE_LOOKBACK_DAYS)
        ohlcv_end = self._ohlcv_end_for_labels(self._cfg, train_dates[-1])
        ohlcv_by_token = self._load_ohlcv(fetch_start, ohlcv_end)
        if not ohlcv_by_token:
            raise RuntimeError("No OHLCV available for retrain window.")

        feature_data = build(
            self._cfg, ohlcv_by_token, train_dates, include_labels=True
        )
        if feature_data.empty:
            raise RuntimeError("Empty feature matrix for retrain window.")

        bundle = train_models(self._cfg, feature_data, train_dates)

        fitted = ["ranker"] if bundle.ranker._fitted else []
        fitted.extend(h.value for h, c in bundle.classifiers.items() if c._fitted)
        logger.info(
            "  retrain complete: rows=%d symbols=%d models=%s",
            len(feature_data),
            int(feature_data["symbol"].nunique()),
            ",".join(fitted) or "none",
        )
        return bundle

    # ── Private helpers ────────────────────────────────────────────────────────

    def _run_fold(
        self,
        fold_id: int,
        train_dates: list[date],
        val_dates: list[date],
        data_mod: object,
        features_mod: object | None,
        models_mod: object | None,
        risk_mod: object | None,
        alpha: float,
        beta: float,
        *,
        fixed_bundle: object | None = None,
    ) -> FoldMetrics | None:
        """Execute one walk-forward fold.  Returns None when data is missing."""
        fetch_start = train_dates[0] - timedelta(days=_FEATURE_LOOKBACK_DAYS)
        try:
            ohlcv_by_token = self._load_ohlcv(fetch_start, val_dates[-1])
        except Exception as exc:
            logger.warning("OHLCV load failed for fold %d: %s — skipping", fold_id, exc)
            return None

        if not ohlcv_by_token:
            logger.warning("No OHLCV data for fold %d — skipping", fold_id)
            return None

        signals_by_date: dict[date, list] = {}
        risk_engine_inst = None

        if features_mod is not None and risk_mod is not None:
            try:
                feature_builder = getattr(features_mod, "build", None)
                risk_engine_cls = getattr(risk_mod, "RiskEngine", None)

                if feature_builder and risk_engine_cls:
                    all_fold_dates = train_dates + val_dates
                    include_labels = fixed_bundle is None
                    features = feature_builder(
                        self._cfg, ohlcv_by_token, all_fold_dates, include_labels=include_labels
                    )
                    risk_engine_inst = risk_engine_cls(self._cfg)

                    if fixed_bundle is not None:
                        signal_fn = getattr(risk_mod, "generate_signals", None)

                        if signal_fn:
                            for d in val_dates:
                                try:
                                    sigs = signal_fn(
                                        fixed_bundle, features, d, risk_engine_inst
                                    )
                                    if sigs:
                                        signals_by_date[d] = sigs
                                except Exception as exc:
                                    logger.debug(
                                        "Signal generation failed on %s: %s", d, exc
                                    )
                    elif models_mod is not None:
                        model_trainer = getattr(models_mod, "train", None)
                        signal_gen = getattr(risk_mod, "generate_signals", None)
                        if model_trainer and signal_gen:
                            model = model_trainer(self._cfg, features, train_dates)
                            for d in val_dates:
                                try:
                                    sigs = signal_gen(model, features, d, risk_engine_inst)
                                    if sigs:
                                        signals_by_date[d] = sigs
                                except Exception as exc:
                                    logger.debug(
                                        "Signal generation failed on %s: %s", d, exc
                                    )
            except Exception as exc:
                logger.warning("Feature/model pipeline error in fold %d: %s", fold_id, exc)

        # ── Backtest ─────────────────────────────────────────────────────────
        if risk_engine_inst is None:
            logger.warning(
                "Risk engine unavailable for fold %d; cannot run OOS backtest.", fold_id
            )
            return None

        cost_model = CostModel(self._cfg)
        engine = BacktestEngine(self._cfg, cost_model, risk_engine_inst)

        try:
            closed_positions, equity_curve = engine.run(
                signals_by_date, ohlcv_by_token, initial_equity=1_000_000.0
            )
        except Exception as exc:
            logger.warning("BacktestEngine.run failed for fold %d: %s", fold_id, exc)
            return None

        # ── Compute metrics ───────────────────────────────────────────────────
        daily_returns = equity_curve.pct_change().dropna()
        n_val_days = max(len(val_dates), 1)

        swing_closed = [p for p in closed_positions if p.signal.horizon == Horizon.SWING]
        pos_closed = [p for p in closed_positions if p.signal.horizon == Horizon.POSITIONAL]
        swing_wins = sum(1 for p in swing_closed if p.net_pnl is not None and p.net_pnl > 0)
        pos_wins = sum(1 for p in pos_closed if p.net_pnl is not None and p.net_pnl > 0)

        total_cost_inr = sum(p.cost or 0.0 for p in closed_positions)
        mean_equity = float(equity_curve.mean()) if not equity_curve.empty else 1_000_000.0
        turnover_cost_pct = (
            (total_cost_inr / mean_equity) / n_val_days * 252
            if mean_equity > 0 and len(closed_positions) > 0
            else 0.0
        )

        fm = FoldMetrics(
            fold_id=fold_id,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            oos_start=val_dates[0],
            oos_end=val_dates[-1],
            sortino=sortino_ratio(daily_returns),
            max_drawdown=max_drawdown(equity_curve),
            calmar=calmar_ratio(equity_curve),
            win_rate=win_rate(closed_positions),
            expectancy_r=expectancy_r(closed_positions),
            total_trades=len(closed_positions),
            swing_trades=len(swing_closed),
            positional_trades=len(pos_closed),
            avg_daily_entries=len(closed_positions) / n_val_days,
            turnover_cost_pct=turnover_cost_pct,
            objective_j=0.0,  # set below
            beats_baseline=False,
            swing_win_rate=swing_wins / len(swing_closed) if swing_closed else 0.0,
            positional_win_rate=pos_wins / len(pos_closed) if pos_closed else 0.0,
        )
        fm.objective_j = compute_objective_j(fm, alpha=alpha, beta=beta)
        return fm

    # ── Data loading helpers ───────────────────────────────────────────────────

    def _load_ohlcv(
        self,
        start: date,
        end: date,
    ) -> dict[int, pd.DataFrame]:
        from trading_bot.data.loader import load_ohlcv

        return load_ohlcv(start, end, self._cfg)

    def _load_index_ohlcv(
        self,
        start: date,
        end: date,
    ) -> pd.DataFrame | None:
        from trading_bot.data.loader import load_index_ohlcv

        return load_index_ohlcv(start, end, self._cfg)

    @staticmethod
    def _generate_trading_dates(start: date, end: date) -> list[date]:
        """Generate NSE trading day approximation (Mon–Fri) between start and end."""
        bdays = pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end))
        return [ts.date() for ts in bdays]

    @staticmethod
    def _save_fold_csv(
        fm: FoldMetrics,
        report_dir: Path,
        fold_id: int,
    ) -> None:
        """Persist fold metrics to a CSV file."""
        out_path = report_dir / f"fold_{fold_id:04d}_metrics.csv"
        try:
            row = {
                "fold_id": fm.fold_id,
                "train_start": fm.train_start.isoformat(),
                "train_end": fm.train_end.isoformat(),
                "oos_start": fm.oos_start.isoformat(),
                "oos_end": fm.oos_end.isoformat(),
                "sortino": fm.sortino,
                "max_drawdown": fm.max_drawdown,
                "calmar": fm.calmar,
                "win_rate": fm.win_rate,
                "expectancy_r": fm.expectancy_r,
                "total_trades": fm.total_trades,
                "swing_trades": fm.swing_trades,
                "positional_trades": fm.positional_trades,
                "avg_daily_entries": fm.avg_daily_entries,
                "turnover_cost_pct": fm.turnover_cost_pct,
                "objective_j": fm.objective_j,
                "beats_baseline": fm.beats_baseline,
                "swing_win_rate": fm.swing_win_rate,
                "positional_win_rate": fm.positional_win_rate,
            }
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            logger.info("Fold %d metrics saved to %s", fold_id, out_path)
        except Exception as exc:
            logger.warning("Failed to save fold %d CSV: %s", fold_id, exc)
