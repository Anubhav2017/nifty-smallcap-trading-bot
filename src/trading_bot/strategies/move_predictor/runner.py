"""Train and backtest the volume-momentum move predictor."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from trading_bot.backtest.costs import CostModel
from trading_bot.backtest.engine import BacktestEngine
from trading_bot.backtest.metrics import (
    calmar_ratio,
    max_drawdown,
    sortino_ratio,
    win_rate,
)
from trading_bot.config import Config
from trading_bot.data.loader import instruments_for_range, load_ohlcv
from trading_bot.data.trading_calendar import trading_days_between
from trading_bot.data.universe import Universe
from trading_bot.risk.engine import RiskEngine
from trading_bot.strategies.move_predictor.features import (
    DEFAULT_LABEL_MIN_MOVE_PCT,
    apply_liquidity_filter,
    build_lagged_panel,
)
from trading_bot.strategies.move_predictor.fundamental_screen import (
    FundamentalScreenConfig,
    clear_fundamental_cache,
    enrich_panel_fundamentals,
    merge_fundamentals_into_panel,
)
from trading_bot.strategies.move_predictor.model import MovePredictorModel
from trading_bot.strategies.move_predictor.breakout_signals import generate_breakout_signals
from trading_bot.strategies.move_predictor.signals import generate_move_predictor_signals
from trading_bot.strategies.move_predictor.trade_report import generate_trade_report
from trading_bot.strategies.move_predictor.walk_forward import quarterly_walk_forward_folds

logger = logging.getLogger(__name__)

_FEATURE_WARMUP_DAYS = 90
SELL_PLAN_MARKDOWN = """# Move predictor — sell plan

All exits use **only prices available on or after entry day** (daily OHLCV).

## Entry (day T, market open)

- **Buy filter:** top `top_n` by P(large next-day up move), lagged volume ratio ≥ `min_volume_ratio` (default 1.5×).
- **Fundamental screen (prior close):** ROCE ≥ min, debt/equity ≤ max, YoY profit growth ≥ min, P/E ≤ max, price above DMA — see `fundamental_screener` in config.
- **Fill price:** open on day T (features use data through T−1 close only).
- **Position size:** risk engine sizes off ATR-based stop distance.

## Exit rules (checked each session, priority order)

1. **Stop-loss** — close at stop if low ≤ stop; gap-down fills at open if open < stop.
   - Stop = entry − `atr_sl_multiple` × ATR(14) (default 1.75× for swing).
2. **Take-profit** — close at target if high ≥ target.
   - Target = entry + `reward_risk_ratio` × (entry − stop) (default 2R).
3. **Time stop** — close at session close after `max_hold_days` (default 10 swing / 60 positional).

## Model retraining (walk-forward)

When `walk_forward_quarters: true`, the model **retrains at the start of each calendar quarter**
using all history strictly before that quarter (expanding window from `train_start`).

## Not used (avoid lookahead)

- Same-day close, volume, or return as entry features.
- Forward returns or same-day z-scores during signal generation.
- OOS quarter data in training for that quarter's signals.
"""


class MovePredictorBacktest:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.mp = cfg._raw.get("move_predictor", {})

    def _mp_date(self, key: str) -> date:
        return date.fromisoformat(str(self.mp[key]))

    def _label_min_move_pct(self) -> float:
        return float(self.mp.get("label_min_move_pct", 1.5)) / 100.0

    def _walk_forward_enabled(self) -> bool:
        return bool(self.mp.get("walk_forward_quarters", True))

    def run(self, output_dir: Path | None = None) -> dict:
        train_start = self._mp_date("train_start")
        train_end = self._mp_date("train_end")
        bt_start = self._mp_date("backtest_start")
        bt_end = self._mp_date("backtest_end")
        label_pct = self._label_min_move_pct()

        fetch_start = train_start - timedelta(days=_FEATURE_WARMUP_DAYS)
        ohlcv_by_token = load_ohlcv(fetch_start, bt_end, self.cfg)
        if not ohlcv_by_token:
            raise RuntimeError("No OHLCV loaded.")

        universe = Universe(self.cfg)
        instruments = instruments_for_range(
            universe, fetch_start, bt_end, universe.load_kite_instruments()
        )
        if not instruments:
            raise RuntimeError("No instruments.")

        panel = build_lagged_panel(
            ohlcv_by_token,
            instruments,
            label_min_move_pct=label_pct,
            ann_dir=Path(self.cfg._raw.get("data", {}).get("dataset_root", "dataset_smallcap250"))
                    / "bse_announcements",
        )
        panel = apply_liquidity_filter(
            panel,
            adtv_cr_min=float(self.cfg.universe.get("liquidity_filter_adtv_cr", 2.0)),
            lookback_days=int(self.cfg.universe.get("liquidity_lookback_days", 20)),
            as_of_date=bt_end,   # prevent look-ahead: only use data up to backtest end
        )

        bt_dates = trading_days_between(bt_start, bt_end)

        fund_screen = FundamentalScreenConfig.from_config(self.cfg)
        if fund_screen.enabled:
            clear_fundamental_cache()
            signal_panel = panel[panel["date"].isin(set(bt_dates))].copy()
            enriched = enrich_panel_fundamentals(signal_panel, self.cfg, screen=fund_screen)
            panel = merge_fundamentals_into_panel(panel, enriched)

        instruments_list = list({i.instrument_token: i for i in instruments}.values())

        # ── Regime filter: index + breadth dual condition ─────────────────────
        # When NIFTY_SMALLCAP_250.csv is available, a day qualifies only when:
        #   (a) index close (lagged) > index SMA (lagged)   ← macro regime
        #   (b) stock breadth (% above 20D SMA) ≥ min_breadth_pct ← recovery confirmation
        # Without index data, falls back to breadth-only.
        # All computations use T-1 data — no lookahead.
        regime_cfg = self.cfg._raw.get("regime_filter", {})
        regime_enabled = bool(regime_cfg.get("enabled", False))
        regime_min_breadth = float(regime_cfg.get("min_breadth_pct", 0.35))
        regime_index_sma = int(regime_cfg.get("index_sma_period", 50))

        index_regime_by_date: dict[date, bool] = {}  # True = macro OK
        breadth_by_date: dict[date, float] = {}       # fraction of stocks above 20D SMA
        _has_index = False

        if regime_enabled:
            index_csv = (
                Path(self.cfg._raw.get("data", {}).get("dataset_root", "dataset_smallcap250"))
                / "ohlcv" / "indices" / "NIFTY_SMALLCAP_250.csv"
            )
            use_index = bool(regime_cfg.get("use_index", False))  # opt-in: default OFF
            if use_index and index_csv.exists():
                _has_index = True
                idx = pd.read_csv(index_csv, parse_dates=["date"])
                idx["date"] = pd.to_datetime(idx["date"]).dt.date
                idx = idx.sort_values("date")
                idx["sma"] = idx["close"].rolling(regime_index_sma, min_periods=10).mean()
                idx["sma_lag1"] = idx["sma"].shift(1)
                idx["close_lag1"] = idx["close"].shift(1)
                idx["above_sma"] = idx["close_lag1"] > idx["sma_lag1"]
                for _, row in idx.iterrows():
                    d_val = row["date"]
                    if pd.notna(row["above_sma"]):
                        index_regime_by_date[d_val] = bool(row["above_sma"])

            if "close_sma_20d_lag1" in panel.columns:
                bt_date_set = set(bt_dates)
                bt_panel = panel[panel["date"].isin(bt_date_set)]
                for d_val, grp in bt_panel.groupby("date"):
                    valid = grp["close_sma_20d_lag1"].dropna()
                    breadth_by_date[d_val] = float((valid > 0).sum() / len(valid)) if len(valid) > 0 else 0.5

            if _has_index:
                q_idx = sum(1 for d in bt_dates if index_regime_by_date.get(d, False))
                q_both = sum(
                    1 for d in bt_dates
                    if index_regime_by_date.get(d, False) and breadth_by_date.get(d, 1.0) >= regime_min_breadth
                )
                logger.info(
                    "Regime filter (INDEX+BREADTH, SMA%d, breadth≥%.0f%%): "
                    "index=%d days, both=%d/%d backtest days qualify",
                    regime_index_sma, regime_min_breadth * 100,
                    q_idx, q_both, len(bt_dates),
                )
            elif breadth_by_date:
                above_thresh = sum(1 for v in breadth_by_date.values() if v >= regime_min_breadth)
                logger.info(
                    "Regime filter (BREADTH only, min=%.0f%%): %d/%d days qualify",
                    regime_min_breadth * 100, above_thresh, len(breadth_by_date),
                )
        # ─────────────────────────────────────────────────────────────────────

        signals_by_date: dict[date, list] = {}
        picks_log: list[dict] = []
        fold_rows: list[dict] = []

        if self._walk_forward_enabled():
            folds = quarterly_walk_forward_folds(bt_dates, train_start)
            logger.info(
                "Walk-forward: %d quarterly folds, label=big up >= %.2f%%, min_vol=%.1f×",
                len(folds),
                label_pct * 100,
                float(self.mp.get("min_volume_ratio", 1.5)),
            )
            for fold in folds:
                model = MovePredictorModel()
                model.train(panel, fold["train_dates"])
                fold_picks = 0

                # SL cooldown state: symbol → earliest date allowed to re-enter
                sl_cooldown: dict[str, date] = {}
                # Per-symbol SL hit count this fold (drives progressive cooldown)
                sl_hit_count: dict[str, int] = {}
                # Permanently blocked symbols (hit SL too many times this fold)
                blocked_symbols: set[str] = set()
                # Recent signals: (signal_date, symbol, stop_loss, instrument_token)
                recent_sigs: list[tuple[date, str, float, int]] = []

                ef = self.cfg._raw.get("entry_filters", {})
                sl_cooldown_days         = int(ef.get("sl_cooldown_days", 0))
                sl_cooldown_multiplier   = float(ef.get("sl_cooldown_multiplier", 2.0))
                max_sl_hits              = int(ef.get("max_sl_hits", 0))   # 0 = unlimited

                # ── Portfolio circuit breaker ─────────────────────────────────
                # If N SL hits occur within a rolling window, pause new entries.
                # This prevents "burst of losses" when a correction starts mid-bull.
                cb_cfg              = self.cfg._raw.get("circuit_breaker", {})
                cb_enabled          = bool(cb_cfg.get("enabled", False))
                cb_window_days      = int(cb_cfg.get("window_days", 5))
                cb_sl_trigger       = int(cb_cfg.get("sl_trigger_count", 4))  # SL hits to trigger
                cb_pause_days       = int(cb_cfg.get("pause_days", 5))
                daily_sl_hits: dict[date, int] = {}   # date → SL hits confirmed on that date
                cb_pause_until: date | None = None
                # ─────────────────────────────────────────────────────────────

                for d in fold["oos_dates"]:
                    # ── Update cooldown from prior-day OHLCV ─────────────────
                    still_pending: list[tuple[date, str, float, int]] = []
                    for sig_date, sym, sl_price, token in recent_sigs:
                        ohlcv = ohlcv_by_token.get(token, pd.DataFrame())
                        if ohlcv.empty:
                            continue
                        ohlcv_norm = ohlcv.copy()
                        ohlcv_norm["_date"] = pd.to_datetime(ohlcv_norm["date"]).dt.date
                        after_entry = ohlcv_norm[
                            (ohlcv_norm["_date"] > sig_date) &
                            (ohlcv_norm["_date"] < d)
                        ]
                        if after_entry.empty:
                            still_pending.append((sig_date, sym, sl_price, token))
                            continue
                        # Check if SL was hit
                        sl_hit = after_entry[after_entry["low"] <= sl_price]
                        if not sl_hit.empty:
                            hit_date = pd.to_datetime(sl_hit.iloc[0]["date"]).date()
                            # ── Circuit breaker: count SL hit by date ─────
                            if cb_enabled:
                                daily_sl_hits[hit_date] = daily_sl_hits.get(hit_date, 0) + 1
                            # ── Progressive cooldown ────────────────────────────
                            sl_hit_count[sym] = sl_hit_count.get(sym, 0) + 1
                            if max_sl_hits > 0 and sl_hit_count[sym] >= max_sl_hits:
                                blocked_symbols.add(sym)
                                logger.debug(
                                    "Symbol %s blocked for rest of fold after %d SL hits",
                                    sym, sl_hit_count[sym],
                                )
                            elif sl_cooldown_days > 0:
                                # Each consecutive SL hit multiplies the cooldown
                                effective_days = int(
                                    sl_cooldown_days
                                    * (sl_cooldown_multiplier ** (sl_hit_count[sym] - 1))
                                )
                                cooldown_until = hit_date + timedelta(days=effective_days)
                                existing = sl_cooldown.get(sym)
                                if existing is None or cooldown_until > existing:
                                    sl_cooldown[sym] = cooldown_until
                        else:
                            # SL not yet hit — keep in pending list
                            still_pending.append((sig_date, sym, sl_price, token))
                    recent_sigs = still_pending

                    # Active exclusion: cooldown + permanently blocked
                    cooled = (
                        {sym for sym, exp in sl_cooldown.items() if exp > d} | blocked_symbols
                        if sl_cooldown_days > 0
                        else set()
                    )

                    # ── Regime filter: skip if market is in a downtrend ──────
                    if regime_enabled:
                        if _has_index:
                            # Both must pass: macro (index > SMA) AND micro (breadth)
                            macro_ok = index_regime_by_date.get(d, True)
                            micro_ok = breadth_by_date.get(d, 1.0) >= regime_min_breadth
                            if not (macro_ok and micro_ok):
                                logger.debug(
                                    "Skipping %s — regime: index=%s breadth=%.0f%%",
                                    d, "ok" if macro_ok else "FAIL",
                                    breadth_by_date.get(d, 1.0) * 100,
                                )
                                continue
                        elif breadth_by_date:
                            if breadth_by_date.get(d, 1.0) < regime_min_breadth:
                                logger.debug("Skipping %s — breadth=%.0f%% < %.0f%%",
                                             d, breadth_by_date.get(d, 0) * 100, regime_min_breadth * 100)
                                continue
                    # ─────────────────────────────────────────────────────────

                    # ── Portfolio circuit breaker ─────────────────────────────
                    if cb_enabled:
                        # Count SL hits in the past cb_window_days calendar days
                        window_start = d - timedelta(days=cb_window_days * 2)  # 2x buffer for weekends
                        recent_sl_total = sum(
                            v for k, v in daily_sl_hits.items()
                            if window_start <= k < d
                        )
                        if cb_pause_until is not None and d <= cb_pause_until:
                            logger.debug(
                                "Circuit breaker active — pausing %s (resumes after %s)",
                                d, cb_pause_until,
                            )
                            continue
                        if recent_sl_total >= cb_sl_trigger:
                            cb_pause_until = d + timedelta(days=cb_pause_days)
                            logger.debug(
                                "Circuit breaker TRIGGERED on %s: %d SL hits in window → pause until %s",
                                d, recent_sl_total, cb_pause_until,
                            )
                            continue
                    # ─────────────────────────────────────────────────────────

                    sigs = generate_move_predictor_signals(
                        self.cfg, panel, model, d, instruments_list,
                        excluded_symbols=cooled or None,
                    )
                    # ── Breakout signal (runs alongside momentum model) ───────
                    already_picked = {s.instrument.symbol for s in sigs}
                    bo_sigs = generate_breakout_signals(
                        self.cfg, panel, d, instruments_list,
                        excluded_symbols=cooled or None,
                        already_picked=already_picked,
                        breadth=breadth_by_date.get(d, 1.0),
                    )
                    sigs = sigs + bo_sigs
                    if not sigs:
                        continue
                    signals_by_date[d] = sigs
                    fold_picks += len(sigs)
                    token_map = {i.symbol: i.instrument_token for i in instruments_list}
                    for sig in sigs:
                        picks_log.append(
                            {
                                "date": d.isoformat(),
                                "quarter": fold["quarter"],
                                "symbol": sig.instrument.symbol,
                                "score": round(sig.rank_score, 4),
                                "entry": round(sig.entry_price, 2),
                                "stop": round(sig.stop_loss, 2),
                                "target": round(sig.target, 2),
                                "horizon": sig.horizon.value,
                                "strategy": sig.features.get("strategy", "move_predictor"),
                            }
                        )
                        if sl_cooldown_days > 0 and sig.instrument.symbol not in blocked_symbols:
                            t = token_map.get(sig.instrument.symbol, 0)
                            recent_sigs.append((d, sig.instrument.symbol, sig.stop_loss, t))
                fold_rows.append(
                    {
                        "quarter": fold["quarter"],
                        "train_start": fold["train_start"].isoformat(),
                        "train_end": fold["train_end"].isoformat(),
                        "oos_start": fold["oos_start"].isoformat(),
                        "oos_end": fold["oos_end"].isoformat(),
                        "train_days": len(fold["train_dates"]),
                        "oos_days": len(fold["oos_dates"]),
                        "picks": fold_picks,
                    }
                )
                logger.info(
                    "Fold %s: train through %s (%d days) → OOS %s–%s, %d picks",
                    fold["quarter"],
                    fold["train_end"],
                    len(fold["train_dates"]),
                    fold["oos_start"],
                    fold["oos_end"],
                    fold_picks,
                )
            final_model = MovePredictorModel()
            deploy_train = trading_days_between(train_start, bt_end)
            final_model.train(panel, deploy_train)
        else:
            train_dates = trading_days_between(train_start, train_end)
            final_model = MovePredictorModel()
            final_model.train(panel, train_dates)
            for d in bt_dates:
                sigs = generate_move_predictor_signals(
                    self.cfg, panel, final_model, d, instruments_list
                )
                if sigs:
                    signals_by_date[d] = sigs
                    for sig in sigs:
                        picks_log.append(
                            {
                                "date": d.isoformat(),
                                "quarter": "",
                                "symbol": sig.instrument.symbol,
                                "score": round(sig.rank_score, 4),
                                "entry": sig.entry_price,
                                "stop": sig.stop_loss,
                                "target": sig.target,
                                "horizon": sig.horizon.value,
                            }
                        )

        cost_model = CostModel(self.cfg)
        risk_engine = RiskEngine(self.cfg)
        engine = BacktestEngine(self.cfg, cost_model, risk_engine)
        closed, equity = engine.run(signals_by_date, ohlcv_by_token, initial_equity=1_000_000.0)

        daily_returns = equity.pct_change().dropna()
        metrics = {
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "backtest_start": bt_start.isoformat(),
            "backtest_end": bt_end.isoformat(),
            "label_min_move_pct": label_pct * 100,
            "min_volume_ratio": float(self.mp.get("min_volume_ratio", 1.5)),
            "walk_forward_quarters": self._walk_forward_enabled(),
            "fundamental_screener": fund_screen.__dict__,
            "total_trades": len(closed),
            "win_rate": win_rate(closed),
            "sortino": sortino_ratio(daily_returns),
            "max_drawdown": max_drawdown(equity),
            "calmar": calmar_ratio(equity),
            "final_equity": float(equity.iloc[-1]) if not equity.empty else 1_000_000.0,
            "signals_days": len(signals_by_date),
            "total_picks": len(picks_log),
        }

        if output_dir:
            run_folder = self._write_reports(
                Path(output_dir),
                metrics,
                closed,
                equity,
                picks_log,
                final_model,
                fold_rows,
            )
        else:
            run_folder = None

        return {
            "metrics": metrics,
            "closed": closed,
            "equity": equity,
            "model": final_model,
            "folds": fold_rows,
            "run_folder": run_folder,
        }

    def _write_reports(
        self,
        base_dir: Path,
        metrics: dict,
        closed: list,
        equity: pd.Series,
        picks_log: list[dict],
        model: MovePredictorModel,
        fold_rows: list[dict],
    ) -> Path:
        """Write all reports into a timestamped subfolder of *base_dir*.

        Returns the path of the created run folder.
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = base_dir / run_id
        out.mkdir(parents=True, exist_ok=True)

        # Store run_id so the dashboard can display it
        metrics["run_id"] = run_id

        (out / "sell_plan.md").write_text(SELL_PLAN_MARKDOWN, encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        model.save(out / "move_predictor.lgb")

        if fold_rows:
            pd.DataFrame(fold_rows).to_csv(out / "walk_forward_folds.csv", index=False)

        if picks_log:
            picks_df = pd.DataFrame(picks_log)
            for col in ("entry", "stop", "target"):
                if col in picks_df.columns:
                    picks_df[col] = picks_df[col].round(2)
            picks_df.to_csv(out / "daily_picks.csv", index=False)

        if closed:
            rows = [
                {
                    "symbol": p.signal.instrument.symbol,
                    "entry_date": p.entry_date.isoformat(),
                    "exit_date": p.exit_date.isoformat() if p.exit_date else "",
                    "entry_price": round(p.entry_price, 2),
                    "exit_price": round(p.exit_price, 2) if p.exit_price is not None else None,
                    "shares": p.shares,
                    "net_pnl": round(p.net_pnl, 2) if p.net_pnl is not None else None,
                    "status": p.status.value if p.status else "",
                    "strategy": p.signal.features.get("strategy", "move_predictor"),
                }
                for p in closed
            ]
            pd.DataFrame(rows).to_csv(out / "trades.csv", index=False)
            generate_trade_report(closed, out / "trade_report.md")

        if not equity.empty:
            equity.round(2).rename("equity").to_csv(out / "equity_curve.csv", header=True)

        summary_lines = [
            "# Move predictor backtest (v2)",
            "",
            f"- Run: {run_id}",
            f"- Label: next-day return ≥ {metrics['label_min_move_pct']:.1f}%",
            f"- Min lagged volume: {metrics['min_volume_ratio']:.1f}×",
            f"- Fundamental screen: {metrics.get('fundamental_screener', {})}",
            f"- Walk-forward quarters: {metrics['walk_forward_quarters']}",
            f"- Backtest: {metrics['backtest_start']} → {metrics['backtest_end']}",
            f"- Trades: {metrics['total_trades']}",
            f"- Win rate: {metrics['win_rate']:.1%}",
            f"- Sortino: {metrics['sortino']:.3f}",
            f"- Max drawdown: {metrics['max_drawdown']:.1%}",
            f"- Final equity: ₹{metrics['final_equity']:,.0f}",
            "",
            "See `sell_plan.md` and `walk_forward_folds.csv`.",
        ]
        (out / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")

        # Update the "latest" pointer so external tools can always find the newest run
        latest_ptr = base_dir / "latest.json"
        latest_ptr.write_text(
            json.dumps({"run_id": run_id, "path": str(out)}, indent=2), encoding="utf-8"
        )

        logger.info("Reports written to %s", out)
        return out
