"""Train and evaluate models on arbitrary date ranges."""

from __future__ import annotations

import csv
import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from trading_bot.backtest.costs import CostModel
from trading_bot.backtest.engine import BacktestEngine
from trading_bot.backtest.hybrid_engine import HybridBacktestEngine
from trading_bot.backtest.metrics import (
    calmar_ratio,
    compute_objective_j,
    expectancy_r,
    max_drawdown,
    sortino_ratio,
    win_rate,
)
from trading_bot.config import Config
from trading_bot.data.loader import load_index_ohlcv, load_ohlcv
from trading_bot.data.trading_calendar import resolve_period, trading_days_between
from trading_bot.features.build import build
from trading_bot.models.training import load as load_model_bundle
from trading_bot.models.training import save as save_models
from trading_bot.models.training import train as train_models
from trading_bot.models.training import train_intraday_timing
from trading_bot.learning.train_progress import configure_train_logging, train_step
from trading_bot.risk.engine import RiskEngine
from trading_bot.risk.hybrid_signals import generate_hybrid_signals
from trading_bot.risk.signals import generate_signals
from trading_bot.types import FoldMetrics, Horizon

logger = logging.getLogger(__name__)

_FEATURE_LOOKBACK_DAYS = 400


class PeriodRunner:
    """Train on one date range and evaluate saved models on another."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self.last_period_notes: list[str] = []

    @staticmethod
    def _ohlcv_end_for_labels(cfg: Config, end: date) -> date:
        max_hold = max(int(cfg.horizons[h.value]["max_hold_days"]) for h in Horizon)
        calendar_days = max(math.ceil(max_hold * 7 / 5), max_hold) + 5
        return end + timedelta(days=calendar_days)

    @staticmethod
    def trading_dates(
        start: date,
        end: date,
        known_trading_days: set[date] | None = None,
    ) -> list[date]:
        return trading_days_between(start, end, known_trading_days)

    def _load_trading_days(self, start: date, end: date) -> set[date] | None:
        """Derive session dates from index OHLCV when cached."""
        index_df = load_index_ohlcv(start, end, self._cfg)
        if index_df is None or index_df.empty:
            return None
        index = index_df.index
        if hasattr(index, "date"):
            return {ts.date() if hasattr(ts, "date") else ts for ts in index}
        return {date.fromisoformat(str(ts)[:10]) for ts in index}

    def _resolve_period(self, start_date: str, end_date: str) -> tuple[date, date, list[str]]:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        cal_start = start - timedelta(days=30)
        cal_end = end + timedelta(days=30)
        known = self._load_trading_days(cal_start, cal_end)
        resolved_start, resolved_end, notes = resolve_period(start, end, known)
        self.last_period_notes = notes
        for note in notes:
            logger.warning(note)
        return resolved_start, resolved_end, notes

    def resolve_model_dir(self, model_dir: Path, name: str | None) -> Path:
        if name:
            return Path("models") / name
        return Path(model_dir)

    @staticmethod
    def discover_saved_runs(models_root: Path | str = "models") -> list[dict]:
        """Find named runs (subdirs with manifest) and legacy flat ``models/*.lgb`` layout."""
        root = Path(models_root)
        if not root.exists():
            return []

        runs: list[dict] = []
        seen_paths: set[Path] = set()

        for manifest_path in sorted(root.glob("*/run_manifest.json")):
            run_dir = manifest_path.parent.resolve()
            if run_dir in seen_paths:
                continue
            seen_paths.add(run_dir)
            entry = PeriodRunner._manifest_entry(run_dir, manifest_path)
            if entry is not None:
                runs.append(entry)

        legacy_ranker = root / "ranker.lgb"
        if legacy_ranker.exists() and root.resolve() not in seen_paths:
            runs.append(
                {
                    "path": root.resolve(),
                    "name": "(legacy flat)",
                    "train_start": None,
                    "train_end": None,
                    "feature_rows": None,
                    "symbols": None,
                    "has_manifest": False,
                    "model_files": PeriodRunner._model_files(root),
                }
            )

        return runs

    @staticmethod
    def _model_files(run_dir: Path) -> list[str]:
        names = ["ranker.lgb", "classifier_swing.lgb", "classifier_positional.lgb"]
        return [name for name in names if (run_dir / name).exists()]

    @staticmethod
    def _manifest_entry(run_dir: Path, manifest_path: Path) -> dict | None:
        if not (run_dir / "ranker.lgb").exists():
            return None
        data: dict = {}
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                data = {}
        return {
            "path": run_dir,
            "name": run_dir.name,
            "train_start": data.get("train_start"),
            "train_end": data.get("train_end"),
            "feature_rows": data.get("feature_rows"),
            "symbols": data.get("symbols"),
            "hybrid": data.get("hybrid"),
            "has_manifest": manifest_path.exists() and bool(data),
            "model_files": PeriodRunner._model_files(run_dir),
        }

    def train(
        self,
        start_date: str,
        end_date: str,
        model_dir: Path,
        *,
        name: str | None = None,
        hybrid: bool = False,
    ) -> Path:
        """Train on [start, end] and persist models + run_manifest.json."""
        configure_train_logging()
        total_steps = 6 if hybrid else 5
        step = 1

        start, end, _notes = self._resolve_period(start_date, end_date)
        out_dir = self.resolve_model_dir(model_dir, name)
        out_dir.mkdir(parents=True, exist_ok=True)
        train_step(
            step,
            total_steps,
            "Resolved training window",
            start=start.isoformat(),
            end=end.isoformat(),
        )
        step += 1

        cal_start = start - timedelta(days=30)
        cal_end = end + timedelta(days=30)
        known = self._load_trading_days(cal_start, cal_end)
        train_dates = self.trading_dates(start, end, known)
        if not train_dates:
            raise ValueError(
                f"No trading days in [{start.isoformat()}, {end.isoformat()}] "
                f"(requested {start_date} → {end_date})"
            )

        ohlcv_end = self._ohlcv_end_for_labels(self._cfg, end)
        fetch_start = start - timedelta(days=_FEATURE_LOOKBACK_DAYS)
        train_step(
            step,
            total_steps,
            "Loading daily OHLCV",
            fetch_from=fetch_start.isoformat(),
            fetch_to=ohlcv_end.isoformat(),
        )
        ohlcv_by_token = load_ohlcv(fetch_start, ohlcv_end, self._cfg)
        if not ohlcv_by_token:
            raise RuntimeError(
                f"No OHLCV loaded for training [{start_date}, {end_date}]. "
                "Check universe membership and dataset OHLCV under config data.dataset_root."
            )
        train_step(
            step,
            total_steps,
            "Loaded daily OHLCV",
            tokens=len(ohlcv_by_token),
            sessions=len(train_dates),
        )
        step += 1

        train_step(step, total_steps, "Building daily feature matrix")
        feature_data = build(self._cfg, ohlcv_by_token, train_dates, include_labels=True)
        if feature_data.empty:
            raise RuntimeError("Feature matrix is empty after build — check data coverage.")
        train_step(
            step,
            total_steps,
            "Built daily feature matrix",
            rows=len(feature_data),
            symbols=int(feature_data["symbol"].nunique()),
            dates=int(feature_data["date"].nunique()),
        )
        step += 1

        train_step(step, total_steps, "Training ranker and entry classifiers")
        bundle = train_models(self._cfg, feature_data, train_dates)
        step += 1

        intraday_rows = 0
        if hybrid:
            train_step(step, total_steps, "Building 5m intraday matrix and training timing model")
            bundle.intraday_timing, intraday_rows = train_intraday_timing(
                self._cfg, feature_data, train_dates
            )
            if bundle.intraday_timing is None:
                logger.warning(
                    "Hybrid timing model not trained — no 5m rows in range. "
                    "Populate ohlcv/minute/ in the active dataset for hybrid mode."
                )
            else:
                train_step(
                    step,
                    total_steps,
                    "Trained intraday timing model",
                    intraday_rows=intraday_rows,
                )
            step += 1

        train_step(step, total_steps, "Saving models", output=str(out_dir))
        save_models(bundle, out_dir)
        self._write_manifest(
            out_dir,
            train_start=start,
            train_end=end,
            rows=len(feature_data),
            symbols=int(feature_data["symbol"].nunique()),
            hybrid=hybrid and bundle.intraday_timing is not None,
            intraday_rows=intraday_rows,
        )
        logger.info("Training complete → %s", out_dir)
        fitted = ["ranker"] if bundle.ranker._fitted else []
        fitted.extend(h.value for h, c in bundle.classifiers.items() if c._fitted)
        if bundle.intraday_timing is not None and bundle.intraday_timing._fitted:
            fitted.append("intraday_timing")
        logger.info(
            "[train] Complete: models=%s → %s",
            ",".join(fitted) or "none",
            out_dir,
        )
        return out_dir

    def evaluate(
        self,
        model_dir: Path,
        start_date: str,
        end_date: str,
        output_dir: Path,
        *,
        hybrid: bool = False,
    ) -> FoldMetrics:
        """Run a saved model on [start, end] and write metrics + trade log."""
        start, end, _notes = self._resolve_period(start_date, end_date)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cal_start = start - timedelta(days=30)
        cal_end = end + timedelta(days=30)
        known = self._load_trading_days(cal_start, cal_end)
        test_dates = self.trading_dates(start, end, known)
        if not test_dates:
            raise ValueError(
                f"No trading days in [{start.isoformat()}, {end.isoformat()}] "
                f"(requested {start_date} → {end_date})"
            )

        bundle = load_model_bundle(self._cfg, Path(model_dir))
        if not bundle.ranker._fitted:
            raise RuntimeError(f"No fitted ranker found in {model_dir}")
        if hybrid and (bundle.intraday_timing is None or not bundle.intraday_timing._fitted):
            raise RuntimeError(
                f"No intraday_timing.lgb in {model_dir}. Re-train with --hybrid."
            )

        fetch_start = start - timedelta(days=_FEATURE_LOOKBACK_DAYS)
        ohlcv_by_token = load_ohlcv(fetch_start, end, self._cfg)
        if not ohlcv_by_token:
            raise RuntimeError(f"No OHLCV loaded for evaluation [{start_date}, {end_date}]")

        features = build(self._cfg, ohlcv_by_token, test_dates, include_labels=False)
        if features.empty:
            raise RuntimeError("Feature matrix is empty for evaluation period.")

        risk_engine = RiskEngine(self._cfg)
        signal_fn = generate_hybrid_signals if hybrid else generate_signals
        signals_by_date: dict[date, list] = {}
        for d in test_dates:
            sigs = signal_fn(bundle, features, d, risk_engine)
            if sigs:
                signals_by_date[d] = sigs

        cost_model = CostModel(self._cfg)
        engine = (
            HybridBacktestEngine(self._cfg, cost_model, risk_engine)
            if hybrid
            else BacktestEngine(self._cfg, cost_model, risk_engine)
        )
        closed_positions, equity_curve = engine.run(
            signals_by_date,
            ohlcv_by_token,
            initial_equity=1_000_000.0,
        )

        alpha = float(self._cfg.objective.get("alpha", 2.0))
        beta = float(self._cfg.objective.get("beta", 1.0))
        daily_returns = equity_curve.pct_change().dropna()
        n_days = max(len(test_dates), 1)

        swing_closed = [p for p in closed_positions if p.signal.horizon == Horizon.SWING]
        pos_closed = [p for p in closed_positions if p.signal.horizon == Horizon.POSITIONAL]
        swing_wins = sum(1 for p in swing_closed if p.net_pnl is not None and p.net_pnl > 0)
        pos_wins = sum(1 for p in pos_closed if p.net_pnl is not None and p.net_pnl > 0)

        total_cost_inr = sum(p.cost or 0.0 for p in closed_positions)
        mean_equity = float(equity_curve.mean()) if not equity_curve.empty else 1_000_000.0
        turnover_cost_pct = (
            (total_cost_inr / mean_equity) / n_days * 252
            if mean_equity > 0 and closed_positions
            else 0.0
        )

        manifest = self._read_manifest(Path(model_dir))
        metrics = FoldMetrics(
            fold_id=0,
            train_start=manifest.get("train_start", date.min),
            train_end=manifest.get("train_end", date.min),
            oos_start=start,
            oos_end=end,
            sortino=sortino_ratio(daily_returns),
            max_drawdown=max_drawdown(equity_curve),
            calmar=calmar_ratio(equity_curve),
            win_rate=win_rate(closed_positions),
            expectancy_r=expectancy_r(closed_positions),
            total_trades=len(closed_positions),
            swing_trades=len(swing_closed),
            positional_trades=len(pos_closed),
            avg_daily_entries=len(closed_positions) / n_days,
            turnover_cost_pct=turnover_cost_pct,
            objective_j=0.0,
            beats_baseline=False,
            swing_win_rate=swing_wins / len(swing_closed) if swing_closed else 0.0,
            positional_win_rate=pos_wins / len(pos_closed) if pos_closed else 0.0,
        )
        metrics.objective_j = compute_objective_j(metrics, alpha=alpha, beta=beta)

        self._save_evaluation(output_dir, metrics, closed_positions, equity_curve, model_dir)
        return metrics

    @staticmethod
    def _write_manifest(
        out_dir: Path,
        *,
        train_start: date,
        train_end: date,
        rows: int,
        symbols: int,
        hybrid: bool = False,
        intraday_rows: int = 0,
    ) -> None:
        manifest = {
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "feature_rows": rows,
            "symbols": symbols,
            "hybrid": hybrid,
            "intraday_rows": intraday_rows,
            "models": ["ranker.lgb", "classifier_swing.lgb", "classifier_positional.lgb"],
        }
        if hybrid:
            manifest["models"].append("intraday_timing.lgb")
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def _read_manifest(model_dir: Path) -> dict:
        path = Path(model_dir) / "run_manifest.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        out = dict(data)
        for key in ("train_start", "train_end"):
            if key in out and isinstance(out[key], str):
                out[key] = date.fromisoformat(out[key])
        return out

    @staticmethod
    def _save_evaluation(
        output_dir: Path,
        metrics: FoldMetrics,
        closed_positions: list,
        equity_curve: pd.Series,
        model_dir: Path,
    ) -> None:
        row = {
            "model_dir": str(model_dir),
            "train_start": metrics.train_start.isoformat(),
            "train_end": metrics.train_end.isoformat(),
            "eval_start": metrics.oos_start.isoformat(),
            "eval_end": metrics.oos_end.isoformat(),
            "sortino": metrics.sortino,
            "max_drawdown": metrics.max_drawdown,
            "calmar": metrics.calmar,
            "win_rate": metrics.win_rate,
            "expectancy_r": metrics.expectancy_r,
            "total_trades": metrics.total_trades,
            "swing_trades": metrics.swing_trades,
            "positional_trades": metrics.positional_trades,
            "avg_daily_entries": metrics.avg_daily_entries,
            "turnover_cost_pct": metrics.turnover_cost_pct,
            "objective_j": metrics.objective_j,
        }
        out_csv = output_dir / "evaluation_metrics.csv"
        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        if not equity_curve.empty:
            equity_curve.rename("equity").to_csv(output_dir / "equity_curve.csv", header=True)

        if closed_positions:
            trade_rows = []
            for p in closed_positions:
                trade_rows.append(
                    {
                        "symbol": p.signal.instrument.symbol,
                        "horizon": p.signal.horizon.value,
                        "entry_date": p.entry_date.isoformat(),
                        "exit_date": p.exit_date.isoformat() if p.exit_date else "",
                        "entry_price": p.entry_price,
                        "exit_price": p.exit_price,
                        "shares": p.shares,
                        "net_pnl": p.net_pnl,
                        "status": p.status.value,
                    }
                )
            pd.DataFrame(trade_rows).to_csv(output_dir / "trades.csv", index=False)
