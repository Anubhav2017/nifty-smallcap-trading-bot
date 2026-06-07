#!/usr/bin/env python3
"""Scan all stocks and write a universe playbook for big price moves."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dashboard.data import load_with_indicators
from dashboard.timeline import significant_price_moves
from trading_bot.analysis.move_correlation import (
    SIMPLE_FACTOR_COLS,
    build_move_analysis,
    build_universe_playbook,
    pool_universe_move_stats,
)
from trading_bot.data.dataset_store import list_symbols, load_manifest
from trading_bot.data.screener_excel import load_symbol_fundamentals

logger = logging.getLogger(__name__)


def scan_universe(
    root: Path,
    *,
    window: int = 60,
    z_threshold: float = 2.0,
    min_abs_return: float = 0.015,
    log_every: int = 10,
    verbose: bool = False,
) -> dict:
    symbols = list_symbols("day", root=root)
    n_symbols = len(symbols)
    logger.info(
        "Starting scan: %d symbols | window=%dd z>=%.2f min_move=%.2f%%",
        n_symbols,
        window,
        z_threshold,
        min_abs_return * 100.0,
    )

    enriched_frames: list[pd.DataFrame] = []
    move_rows: list[dict] = []
    vote_rows: list[dict] = []
    skipped_empty = 0
    skipped_no_moves = 0
    t0 = time.perf_counter()

    for i, sym in enumerate(symbols, start=1):
        sym_t0 = time.perf_counter()
        bars = load_with_indicators(sym, "day", root)
        if bars.empty:
            skipped_empty += 1
            if verbose:
                logger.debug("[%d/%d] %s — skipped (no daily bars)", i, n_symbols, sym)
            elif i % log_every == 0 or i == n_symbols:
                logger.info(
                    "[%d/%d] progress … %d with moves, %d skipped empty, %d no big moves",
                    i,
                    n_symbols,
                    len(enriched_frames),
                    skipped_empty,
                    skipped_no_moves,
                )
            continue

        n_bars = len(bars)
        date_lo = pd.Timestamp(bars["date"].min()).date()
        date_hi = pd.Timestamp(bars["date"].max()).date()

        fund = load_symbol_fundamentals(root / "screener_excel", sym)
        has_screener = not fund.empty

        moves = significant_price_moves(
            bars,
            window=window,
            z_threshold=z_threshold,
            min_abs_return=min_abs_return,
        )
        if moves.empty:
            skipped_no_moves += 1
            if verbose:
                logger.debug(
                    "[%d/%d] %s — no big moves (%d bars, %s → %s)",
                    i,
                    n_symbols,
                    sym,
                    n_bars,
                    date_lo,
                    date_hi,
                )
            elif i % log_every == 0 or i == n_symbols:
                logger.info(
                    "[%d/%d] progress … %d with moves, %d skipped empty, %d no big moves",
                    i,
                    n_symbols,
                    len(enriched_frames),
                    skipped_empty,
                    skipped_no_moves,
                )
            continue

        analysis = build_move_analysis(bars, fund, moves)
        enriched = analysis["enriched"].copy()
        enriched["symbol"] = sym
        enriched_frames.append(enriched)
        n_moves = len(moves)
        up_n = int((moves["direction"] == "up").sum())
        down_n = n_moves - up_n

        for _, mv in moves.iterrows():
            near = enriched.loc[
                pd.to_datetime(enriched["date"]).dt.normalize()
                == pd.Timestamp(mv["date"]).normalize(),
                "filing_within_5d",
            ]
            move_rows.append(
                {
                    "symbol": sym,
                    "date": pd.Timestamp(mv["date"]).strftime("%Y-%m-%d"),
                    "return_pct": float(mv["return"]) * 100.0,
                    "direction": mv["direction"],
                    "z_score": float(mv["z_score"]),
                    "near_filing": bool(near.iloc[0] >= 0.5) if len(near) else False,
                }
            )

        summary = analysis["simple_summary"]
        if not summary.empty:
            for _, row in summary.iterrows():
                vote_rows.append(
                    {
                        "symbol": sym,
                        "indicator": row["indicator"],
                        "pattern": row["pattern"],
                    }
                )

        elapsed = time.perf_counter() - sym_t0
        if verbose:
            logger.info(
                "[%d/%d] %s — %d big moves (up %d / down %d), %d bars, screener=%s, %.2fs",
                i,
                n_symbols,
                sym,
                n_moves,
                up_n,
                down_n,
                n_bars,
                "yes" if has_screener else "no",
                elapsed,
            )
        elif i % log_every == 0 or i == n_symbols:
            logger.info(
                "[%d/%d] progress … %d stocks with moves, %d total big-move days, "
                "skipped empty=%d no_moves=%d (last: %s %d moves in %.2fs)",
                i,
                n_symbols,
                len(enriched_frames),
                len(move_rows),
                skipped_empty,
                skipped_no_moves,
                sym,
                n_moves,
                elapsed,
            )

    scan_elapsed = time.perf_counter() - t0
    logger.info(
        "Scan complete in %.1fs: %d/%d symbols had big moves (%d skipped empty, %d no moves), "
        "%d big-move days collected",
        scan_elapsed,
        len(enriched_frames),
        n_symbols,
        skipped_empty,
        skipped_no_moves,
        len(move_rows),
    )

    logger.info("Pooling factor stats across %d enriched symbol frames …", len(enriched_frames))
    t_pool = time.perf_counter()
    moves_df = pd.DataFrame(move_rows)
    pooled = pool_universe_move_stats(enriched_frames)
    logger.info(
        "Pooled %d indicators in %.2fs",
        len(pooled),
        time.perf_counter() - t_pool,
    )

    per_symbol_votes = pd.DataFrame()
    if vote_rows:
        logger.info("Aggregating per-symbol pattern votes …")
        votes = pd.DataFrame(vote_rows)
        higher = (
            votes[votes["pattern"] == "Higher on big days"]
            .groupby("indicator")["symbol"]
            .nunique()
        )
        eligible = votes.groupby("indicator")["symbol"].nunique()
        per_symbol_votes = (
            pd.DataFrame({"pct_symbols_higher": (higher / eligible * 100).round(0)})
            .reset_index()
            .sort_values("pct_symbols_higher", ascending=False)
        )
        if not pooled.empty:
            pooled = pooled.merge(per_symbol_votes, on="indicator", how="left")
        logger.info("Vote rows: %d across %d indicators", len(votes), len(per_symbol_votes))

    up_moves = int((moves_df["direction"] == "up").sum()) if not moves_df.empty else 0
    down_moves = int((moves_df["direction"] == "down").sum()) if not moves_df.empty else 0
    pct_near = (
        float(moves_df["near_filing"].mean() * 100.0) if not moves_df.empty else 0.0
    )
    median_move = (
        float(moves_df["return_pct"].abs().median()) if not moves_df.empty else 0.0
    )

    if not pooled.empty:
        top = pooled.iloc[0]
        logger.info(
            "Strongest pooled signal: %s (%s) — on big days %s vs usually %s",
            top["indicator"],
            top["pattern"],
            top["on_big_days"],
            top["usually"],
        )

    logger.info("Building playbook markdown …")
    playbook = build_universe_playbook(
        pooled,
        symbols_scanned=n_symbols,
        symbols_with_moves=len(enriched_frames),
        total_moves=len(moves_df),
        up_moves=up_moves,
        down_moves=down_moves,
        pct_near_filing=pct_near,
        median_move_pct=median_move,
        per_symbol_votes=per_symbol_votes,
    )

    return {
        "symbols_scanned": n_symbols,
        "symbols_with_moves": len(enriched_frames),
        "skipped_empty": skipped_empty,
        "skipped_no_moves": skipped_no_moves,
        "total_moves": len(moves_df),
        "moves": moves_df,
        "pooled": pooled,
        "per_symbol_votes": per_symbol_votes,
        "playbook": playbook,
        "scan_seconds": scan_elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", default="dataset_smallcap250")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--min-move-pct", type=float, default=1.5)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("hermes/reports/move_analysis"),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log progress every N symbols (default: 10)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log every symbol (not just progress checkpoints)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only warnings and errors",
    )
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    root = Path(args.dataset_root).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset root: %s", root)
    logger.info("Output dir: %s", out_dir.resolve())
    try:
        manifest = load_manifest(root)
        dr = manifest.get("date_range", {})
        logger.info("Manifest date range: %s → %s", dr.get("from"), dr.get("to"))
    except FileNotFoundError:
        logger.warning("No manifest.json under %s", root)

    result = scan_universe(
        root,
        window=args.window,
        z_threshold=args.z_threshold,
        min_abs_return=args.min_move_pct / 100.0,
        log_every=max(1, args.log_every),
        verbose=args.verbose,
    )

    logger.info("Writing outputs …")
    playbook_path = out_dir / "universe_playbook.md"
    playbook_path.write_text(result["playbook"], encoding="utf-8")
    logger.info("Wrote %s", playbook_path)

    if not result["pooled"].empty:
        pooled_path = out_dir / "universe_pooled_factors.csv"
        result["pooled"].to_csv(pooled_path, index=False)
        logger.info("Wrote %s (%d rows)", pooled_path, len(result["pooled"]))

    if not result["moves"].empty:
        moves_path = out_dir / "universe_all_moves.csv"
        result["moves"].to_csv(moves_path, index=False)
        logger.info("Wrote %s (%d rows)", moves_path, len(result["moves"]))

    if not result["per_symbol_votes"].empty:
        votes_path = out_dir / "universe_factor_votes.csv"
        result["per_symbol_votes"].to_csv(votes_path, index=False)
        logger.info("Wrote %s", votes_path)

    meta = {
        "dataset": str(root),
        "symbols_scanned": result["symbols_scanned"],
        "symbols_with_moves": result["symbols_with_moves"],
        "skipped_empty": result["skipped_empty"],
        "skipped_no_moves": result["skipped_no_moves"],
        "total_moves": result["total_moves"],
        "scan_seconds": round(result["scan_seconds"], 1),
        "factors": SIMPLE_FACTOR_COLS,
        "params": {
            "window": args.window,
            "z_threshold": args.z_threshold,
            "min_move_pct": args.min_move_pct,
        },
    }
    try:
        meta["date_range"] = load_manifest(root).get("date_range")
    except FileNotFoundError:
        pass
    meta_path = out_dir / "universe_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote %s", meta_path)

    logger.info(
        "Done in %.1fs: %s big-move days across %s/%s stocks → %s",
        result["scan_seconds"],
        f"{result['total_moves']:,}",
        result["symbols_with_moves"],
        result["symbols_scanned"],
        playbook_path,
    )


if __name__ == "__main__":
    main()
