#!/usr/bin/env python3
"""Scan universe for significant moves and export factor correlation summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dashboard.data import load_with_indicators
from dashboard.timeline import significant_price_moves
from trading_bot.analysis.move_correlation import build_move_analysis
from trading_bot.data.dataset_store import list_symbols, load_manifest
from trading_bot.data.screener_excel import load_symbol_fundamentals


def analyze_symbol(
    root: Path,
    symbol: str,
    *,
    window: int,
    z_threshold: float,
    min_abs_return: float,
) -> dict:
    bars = load_with_indicators(symbol, "day", root)
    if bars.empty:
        return {"symbol": symbol, "moves": 0, "top_factor": None, "top_corr": None}

    fund = load_symbol_fundamentals(root / "screener_excel", symbol)
    moves = significant_price_moves(
        bars,
        window=window,
        z_threshold=z_threshold,
        min_abs_return=min_abs_return,
    )
    if moves.empty:
        return {"symbol": symbol, "moves": 0, "top_factor": None, "top_corr": None}

    analysis = build_move_analysis(bars, fund, moves)
    corr = analysis["factor_correlations"]
    top = corr.iloc[0] if not corr.empty else None
    return {
        "symbol": symbol,
        "moves": len(moves),
        "top_factor": top["label"] if top is not None else None,
        "top_corr": float(top["corr_z_score_moves"]) if top is not None else None,
        "move_features": analysis["move_features"],
        "factor_correlations": corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", default="dataset_smallcap250")
    parser.add_argument("--symbol", help="Single symbol (default: all with daily OHLCV)")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--min-move-pct", type=float, default=1.5)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("hermes/reports/move_analysis"))
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [args.symbol.upper()] if args.symbol else list_symbols("day", root=root)
    min_abs = args.min_move_pct / 100.0

    summary_rows: list[dict] = []
    all_moves: list[pd.DataFrame] = []

    for sym in symbols:
        result = analyze_symbol(
            root,
            sym,
            window=args.window,
            z_threshold=args.z_threshold,
            min_abs_return=min_abs,
        )
        summary_rows.append(
            {
                "symbol": result["symbol"],
                "moves": result["moves"],
                "top_factor": result["top_factor"],
                "top_corr": result["top_corr"],
            }
        )
        if result["moves"] > 0:
            mf = result["move_features"].copy()
            mf.insert(0, "symbol", sym)
            all_moves.append(mf)
            result["factor_correlations"].to_csv(
                out_dir / f"{sym}_factor_correlations.csv", index=False
            )

    pd.DataFrame(summary_rows).to_csv(out_dir / "universe_summary.csv", index=False)
    if all_moves:
        pooled = pd.concat(all_moves, ignore_index=True)
        pooled.to_csv(out_dir / "all_moves_features.csv", index=False)

        # Pooled correlation across all move days (numeric cols only)
        num_cols = pooled.select_dtypes(include="number").columns
        num_cols = [c for c in num_cols if c not in ("return_pct", "z_score")]
        if len(pooled) >= 5 and num_cols:
            pooled_corr = pooled[num_cols].corrwith(pooled["z_score"], method="spearman")
            pooled_corr.sort_values(key=abs, ascending=False).to_csv(
                out_dir / "pooled_factor_correlations.csv",
                header=["corr_with_z_score"],
            )

    meta = {"dataset": str(root), "symbols_scanned": len(symbols)}
    try:
        meta["date_range"] = load_manifest(root).get("date_range")
    except FileNotFoundError:
        pass
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote reports to {out_dir} ({len(symbols)} symbols)")


if __name__ == "__main__":
    main()
