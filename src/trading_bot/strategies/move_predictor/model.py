"""LightGBM model: P(large next-day up move) from lagged volume/momentum features."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as _err:  # pragma: no cover
    raise ImportError("lightgbm is required: pip install lightgbm") from _err

from trading_bot.strategies.move_predictor.features import (
    LABEL_BIG_UP_COL,
    LAGGED_FEATURE_COLS,
)

logger = logging.getLogger(__name__)


@dataclass
class MovePredictorModel:
    """Binary classifier for next-day large positive return."""

    _model: lgb.LGBMClassifier | None = None
    feature_cols: list[str] | None = None
    label_col: str = LABEL_BIG_UP_COL

    @property
    def fitted(self) -> bool:
        return self._model is not None

    def train(self, panel: pd.DataFrame, train_dates: list[date]) -> None:
        sub = panel[panel["date"].isin(train_dates)].copy()
        cols = [c for c in LAGGED_FEATURE_COLS if c in sub.columns]
        clean = sub.dropna(subset=cols + [self.label_col])
        if len(clean) < 100:
            raise RuntimeError(f"Too few training rows ({len(clean)}); need at least 100.")

        X = clean[cols]
        y = clean[self.label_col].astype(int)
        split = max(1, int(len(clean) * 0.85))
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_val, y_val = X.iloc[split:], y.iloc[split:]

        self._model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
            scale_pos_weight=max(1.0, (len(y) - y.sum()) / max(y.sum(), 1)),
        )
        fit_kw: dict = {"X": X_train, "y": y_train}
        if len(X_val) > 0:
            fit_kw["eval_set"] = [(X_val, y_val)]
            fit_kw["callbacks"] = [
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=0),
            ]
        self._model.fit(**fit_kw)  # type: ignore[union-attr]
        self.feature_cols = cols
        logger.info(
            "MovePredictor trained on %d rows (%d features), label=%s positive rate %.1f%%",
            len(clean),
            len(cols),
            self.label_col,
            100.0 * y.mean(),
        )

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.feature_cols is None:
            raise RuntimeError("Model not trained.")
        X = panel.reindex(columns=self.feature_cols).copy()
        for col in self.feature_cols:
            med = panel[col].median() if col in panel.columns else 0.0
            X[col] = X[col].fillna(med)
        return self._model.predict_proba(X)[:, 1]  # type: ignore[union-attr]

    def save(self, path: Path) -> None:
        if not self.fitted:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.booster_.save_model(str(path))  # type: ignore[union-attr]

    def load(self, path: Path) -> None:
        path = Path(path)
        booster = lgb.Booster(model_file=str(path))
        clf = lgb.LGBMClassifier()
        clf._Booster = booster  # type: ignore[attr-defined]
        self._model = clf
        self.feature_cols = list(booster.feature_name())
