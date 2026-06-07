"""Dashboard helpers for loading and displaying model run reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPORTS_ROOT = Path(__file__).resolve().parents[1] / "reports"


def list_models(reports_root: Path = REPORTS_ROOT) -> list[str]:
    """Return model names (top-level folders under reports_root)."""
    if not reports_root.is_dir():
        return []
    return sorted(
        p.name for p in reports_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def list_runs_for_model(model_name: str, reports_root: Path = REPORTS_ROOT) -> list[dict]:
    """Return all run metadata for a model, newest first.

    Supports both the new layout  (model/YYYYMMDD_HHMMSS/) and the
    legacy layout (model/metrics.json directly — treated as 'initial_run').
    """
    model_dir = reports_root / model_name
    if not model_dir.is_dir():
        return []

    runs: list[dict] = []
    for folder in model_dir.iterdir():
        if not folder.is_dir():
            continue
        metrics_path = folder / "metrics.json"
        if not metrics_path.is_file():
            continue
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            m = {}
        run_id = m.get("run_id") or folder.name
        runs.append(_run_entry(folder, run_id, m))

    # newest first: YYYYMMDD_HHMMSS sorts lexicographically
    runs.sort(key=lambda r: r["run_id"], reverse=True)
    return runs


def list_report_runs(reports_root: Path = REPORTS_ROOT) -> list[dict]:
    """Return every run across all models, newest first.

    Each entry includes a 'model' key so the dashboard can group by model.
    """
    all_runs: list[dict] = []
    for model_name in list_models(reports_root):
        for run in list_runs_for_model(model_name, reports_root):
            run["model"] = model_name
            all_runs.append(run)
    all_runs.sort(key=lambda r: r["run_id"], reverse=True)
    return all_runs


def _run_entry(folder: Path, run_id: str, m: dict) -> dict:
    return {
        "name": run_id,
        "run_id": run_id,
        "path": folder,
        "metrics": m,
        "has_trades": (folder / "trades.csv").is_file(),
        "has_equity": (folder / "equity_curve.csv").is_file(),
        "has_folds": (folder / "walk_forward_folds.csv").is_file(),
        "has_picks": (folder / "daily_picks.csv").is_file(),
        "has_trade_report": (folder / "trade_report.md").is_file(),
        "has_playbook": (folder / "universe_playbook.md").is_file(),
    }


def load_equity_curve(folder: Path) -> pd.DataFrame:
    p = folder / "equity_curve.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    # Rename the first column to "date" only when it isn't already named and
    # isn't the equity column itself (handles unnamed index exported by pandas).
    if "date" not in df.columns and len(df.columns) >= 1 and df.columns[0] not in ("equity",):
        df = df.rename(columns={df.columns[0]: "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "equity" in df.columns:
        df["equity"] = df["equity"].round(2)
    return df


def load_trades(folder: Path) -> pd.DataFrame:
    p = folder / "trades.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for col in ("entry_price", "exit_price", "net_pnl"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    return df


def load_folds(folder: Path) -> pd.DataFrame:
    p = folder / "walk_forward_folds.csv"
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_picks(folder: Path) -> pd.DataFrame:
    p = folder / "daily_picks.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for col in ("entry", "stop", "target"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    return df


def read_markdown(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""
