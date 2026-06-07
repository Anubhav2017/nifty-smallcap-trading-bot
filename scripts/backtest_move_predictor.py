#!/usr/bin/env python3
"""Train move predictor on 2024, walk-forward backtest on 2025 (no lookahead)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_bot.config import Config
from trading_bot.strategies.move_predictor.runner import MovePredictorBacktest

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=ROOT / "config" / "move_predictor.yaml",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "move_predictor",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = Config(args.config)
    mp = cfg._raw.get("move_predictor", {})
    ef = cfg._raw.get("entry_filters", {})
    logger.info("Config: %s", args.config)
    logger.info(
        "Label: next-day up >= %.1f%% | min volume %.1f× | walk-forward=%s | fundamentals=%s",
        float(mp.get("label_min_move_pct", 1.5)),
        float(mp.get("min_volume_ratio", 1.5)),
        mp.get("walk_forward_quarters", True),
        cfg._raw.get("fundamental_screener", {}).get("enabled", True),
    )
    if ef:
        logger.info(
            "Entry filters: max_20d_ret=%.0f%% | max_ext_200sma=%.0f%% | "
            "sl_cooldown=%d days | multiplier=%.1fx | max_sl_hits=%d",
            float(ef.get("max_20d_return", 1)) * 100,
            float(ef.get("max_ext_from_200sma", 1)) * 100,
            int(ef.get("sl_cooldown_days", 0)),
            float(ef.get("sl_cooldown_multiplier", 2.0)),
            int(ef.get("max_sl_hits", 0)),
        )
    bo = cfg._raw.get("breakout_signal", {})
    if bo.get("enabled", True):
        logger.info(
            "Breakout signal: 52W-ratio≥%.3f | vol≥%.1fx | top_n=%d",
            float(bo.get("min_52w_high_ratio", 0.995)),
            float(bo.get("min_volume_ratio", 2.0)),
            int(bo.get("top_n", 3)),
        )

    result = MovePredictorBacktest(cfg).run(output_dir=args.output_dir)
    m = result["metrics"]
    run_folder = result.get("run_folder") or args.output_dir
    print("\n=== Backtest results (v2) ===")
    print(f"Trades:       {m['total_trades']}")
    print(f"Picks:        {m['total_picks']}")
    print(f"Win rate:     {m['win_rate']:.1%}")
    print(f"Sortino:      {m['sortino']:.3f}")
    print(f"Max drawdown: {m['max_drawdown']:.1%}")
    print(f"Final equity: ₹{m['final_equity']:,.0f}")
    if result.get("folds"):
        print("\nWalk-forward folds:")
        for fold in result["folds"]:
            print(
                f"  {fold['quarter']}: train→{fold['train_end']} "
                f"OOS {fold['oos_start']}–{fold['oos_end']} picks={fold['picks']}"
            )
    print(f"\nRun ID:  {m.get('run_id', '—')}")
    print(f"Reports: {Path(run_folder).resolve()}")


if __name__ == "__main__":
    main()
