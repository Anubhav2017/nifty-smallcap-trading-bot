"""Paper trading ledger — simulates fills at next open with slippage."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.backtest.costs import CostModel
from trading_bot.backtest.intraday_sim import load_session_bars, parse_entry_datetime, simulate_intraday_session
from trading_bot.config import Config
from trading_bot.data.bars import BarStore
from trading_bot.models.classifier import EntryClassifier
from trading_bot.models.exit_policy import ExitPolicy
from trading_bot.models.ranker import StockRanker
from trading_bot.models.training import load as load_model_bundle
from trading_bot.risk.engine import RiskEngine
from trading_bot.risk.hybrid_signals import generate_hybrid_signals
from trading_bot.types import Horizon, Instrument, Position, Signal, TradeStatus

logger = logging.getLogger(__name__)

LEDGER_COLUMNS = [
    "date",
    "symbol",
    "isin",
    "horizon",
    "action",
    "price",
    "shares",
    "pnl",
    "status",
    "equity",
    "entry_time",
    "stop_loss",
    "target",
]

_INITIAL_EQUITY = 1_000_000.0  # 10 lakh INR — default starting capital


class PaperLedger:
    """Append-only paper trading ledger driven by LightGBM signals.

    Fills are simulated at the next available open price (or today's close
    adjusted for minimum slippage when the next open is unavailable).
    Historical rows in the ledger CSV are never modified or deleted.
    """

    def __init__(
        self,
        cfg: Config,
        model_dir: Path,
        ledger_path: Path,
        *,
        hybrid: bool = False,
    ) -> None:
        self.cfg = cfg
        self.model_dir = Path(model_dir)
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        self.bundle = load_model_bundle(cfg, self.model_dir)
        self.hybrid = hybrid or (
            self.bundle.intraday_timing is not None and self.bundle.intraday_timing._fitted
        )

        self.ranker = self.bundle.ranker
        self.classifiers = self.bundle.classifiers
        self.exit_policy = self.bundle.exit_policy
        self.risk_engine = RiskEngine(cfg)
        self._bar_store = BarStore(cfg=self.cfg)
        self._cost_model = CostModel(cfg)

        self._ledger: pd.DataFrame = self.load_ledger()

    # ── Session execution ──────────────────────────────────────────────────

    def run_session(self, as_of_date: date | None = None) -> dict[str, Any]:
        """Run one paper trading session for *as_of_date*.

        Steps:
        1. Load today's OHLCV from Kite cache.
        2. Build features (no labels).
        3. Generate signals via ranker → classifier → exit_policy.
        4. Gate signals through the risk engine.
        5. Simulate fills at next-open proxy price.
        6. Process exits on open positions.
        7. Append new rows to the ledger CSV.

        Returns a summary dict: new_entries, exits_processed, equity.
        """
        if as_of_date is None:
            as_of_date = date.today()

        if self.hybrid:
            return self._run_hybrid_session(as_of_date)

        logger.info("PaperLedger: running session for %s", as_of_date)

        try:
            ohlcv_by_token, index_df, instruments = self._load_market_data(as_of_date)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load market data for %s: %s", as_of_date, exc)
            return {"new_entries": 0, "exits_processed": 0, "equity": self._current_equity()}

        if not ohlcv_by_token:
            logger.warning("No market data available for %s.", as_of_date)
            return {"new_entries": 0, "exits_processed": 0, "equity": self._current_equity()}

        feature_df = self._build_features(ohlcv_by_token, index_df, instruments)

        new_rows: list[dict] = []
        equity = self._current_equity()

        # ── Process exits first ─────────────────────────────────────────
        open_positions = self._reconstruct_open_positions()
        exits_processed = 0

        for pos in open_positions:
            token = pos.signal.instrument.instrument_token
            ohlcv = ohlcv_by_token.get(token)
            if ohlcv is None or ohlcv.empty:
                continue

            today_row = ohlcv[ohlcv["date"] == as_of_date]
            if today_row.empty:
                continue

            bar_close = float(today_row.iloc[-1]["close"])
            bar_high = float(today_row.iloc[-1]["high"])
            bar_low = float(today_row.iloc[-1]["low"])
            bar_open = float(today_row.iloc[-1]["open"])

            from trading_bot.types import OHLCVBar
            bar = OHLCVBar(
                date=as_of_date,
                open=bar_open,
                high=bar_high,
                low=bar_low,
                close=bar_close,
                volume=float(today_row.iloc[-1]["volume"]),
                instrument_token=token,
            )

            trading_dates = sorted(ohlcv["date"].unique().tolist())
            sessions_held = self.risk_engine.sessions_held(pos, as_of_date, trading_dates)
            should_exit, exit_status, exit_price = self.risk_engine.check_exits(
                pos, bar, sessions_held
            )

            if should_exit:
                gross_pnl = (exit_price - pos.entry_price) * pos.shares
                cost = self._compute_cost(exit_price, pos.shares)
                net_pnl = gross_pnl - cost
                equity += net_pnl

                new_rows.append({
                    "date": as_of_date,
                    "symbol": pos.signal.instrument.symbol,
                    "isin": pos.signal.instrument.isin,
                    "horizon": pos.signal.horizon.value,
                    "action": "exit",
                    "price": exit_price,
                    "shares": pos.shares,
                    "pnl": net_pnl,
                    "status": exit_status.value,
                    "equity": equity,
                    "entry_time": "",
                    "stop_loss": "",
                    "target": "",
                })
                exits_processed += 1
                logger.info(
                    "Exit: %s/%s @ %.2f (%s), net_pnl=%.2f",
                    pos.signal.instrument.symbol,
                    pos.signal.horizon.value,
                    exit_price,
                    exit_status.value,
                    net_pnl,
                )

        # ── Generate new entries ─────────────────────────────────────────
        new_entries = 0
        if not feature_df.empty and self.ranker._fitted:
            signals = self._generate_signals(feature_df, as_of_date, ohlcv_by_token, instruments)
            still_open = [p for p in open_positions
                          if p.signal.instrument.symbol not in
                          {r["symbol"] for r in new_rows if r["action"] == "exit"}]

            daily_entry_count = 0
            for sig in signals:
                approved, shares, reason = self.risk_engine.evaluate_signal(
                    sig, still_open, daily_entry_count, equity
                )
                if not approved:
                    logger.debug("Signal rejected (%s): %s/%s", reason, sig.instrument.symbol, sig.horizon.value)
                    continue

                slippage_pct = self.cfg.costs.get("slippage_min_pct", 0.05)
                token = sig.instrument.instrument_token
                ohlcv = ohlcv_by_token.get(token)
                fill_price = sig.entry_price * (1.0 + slippage_pct / 100.0)
                if ohlcv is not None and not ohlcv.empty:
                    today_close_row = ohlcv[ohlcv["date"] == as_of_date]
                    if not today_close_row.empty:
                        fill_price = float(today_close_row.iloc[-1]["close"]) * (1.0 + slippage_pct / 100.0)

                cost_total = self._compute_cost(fill_price, shares)
                equity -= fill_price * shares

                new_rows.append({
                    "date": as_of_date,
                    "symbol": sig.instrument.symbol,
                    "isin": sig.instrument.isin,
                    "horizon": sig.horizon.value,
                    "action": "entry",
                    "price": fill_price,
                    "shares": shares,
                    "pnl": -cost_total,
                    "status": TradeStatus.OPEN.value,
                    "equity": equity,
                    "entry_time": "",
                    "stop_loss": sig.stop_loss,
                    "target": sig.target,
                })
                new_entries += 1
                daily_entry_count += 1
                logger.info(
                    "Entry: %s/%s @ %.2f × %d shares, EV=%.4f",
                    sig.instrument.symbol,
                    sig.horizon.value,
                    fill_price,
                    shares,
                    sig.expected_value,
                )

        if new_rows:
            new_df = pd.DataFrame(new_rows, columns=LEDGER_COLUMNS)
            self._ledger = pd.concat([self._ledger, new_df], ignore_index=True)
            self.save_ledger(self._ledger)

        logger.info(
            "Session %s complete: entries=%d exits=%d equity=%.2f",
            as_of_date,
            new_entries,
            exits_processed,
            equity,
        )
        return {"new_entries": new_entries, "exits_processed": exits_processed, "equity": equity}

    def _run_hybrid_session(self, as_of_date: date) -> dict[str, Any]:
        """Hybrid paper session: 5m timed entries and bar-level SL/TP/time exits."""
        logger.info("PaperLedger: running hybrid session for %s", as_of_date)

        try:
            ohlcv_by_token, index_df, instruments = self._load_market_data(as_of_date)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load market data for %s: %s", as_of_date, exc)
            return {"new_entries": 0, "exits_processed": 0, "equity": self._current_equity()}

        if not ohlcv_by_token:
            logger.warning("No market data available for %s.", as_of_date)
            return {"new_entries": 0, "exits_processed": 0, "equity": self._current_equity()}

        feature_df = self._build_features(ohlcv_by_token, index_df, instruments)
        if feature_df.empty or not self.ranker._fitted:
            return {"new_entries": 0, "exits_processed": 0, "equity": self._current_equity()}

        token_map = {i.instrument_token: i for i in instruments}
        open_positions = self._reconstruct_open_positions(token_map)
        for pos in open_positions:
            token = pos.signal.instrument.instrument_token
            if token == 0 and pos.signal.instrument.symbol in {i.symbol for i in instruments}:
                for inst in instruments:
                    if inst.symbol == pos.signal.instrument.symbol:
                        pos.signal.instrument = inst
                        break

        signals = generate_hybrid_signals(self.bundle, feature_df, as_of_date, self.risk_engine)
        symbols: set[str] = set()
        for pos in open_positions:
            symbols.add(pos.signal.instrument.symbol.upper())
        for sig in signals:
            symbols.add(sig.instrument.symbol.upper())

        bars_by_symbol = load_session_bars(self._bar_store, symbols, as_of_date)
        trading_dates = sorted(
            {
                d
                for df in ohlcv_by_token.values()
                for d in df["date"].unique().tolist()
            }
        )

        pre_equity = self._current_equity()
        open_before = {
            (p.signal.instrument.symbol, p.signal.horizon.value, p.entry_date)
            for p in open_positions
        }
        session_closed: list[Position] = []

        still_open, session_closed, _post_equity = simulate_intraday_session(
            as_of_date,
            open_positions,
            session_closed,
            signals,
            bars_by_symbol,
            all_trading_dates=trading_dates,
            equity=pre_equity,
            risk_engine=self.risk_engine,
            cost_model=self._cost_model,
            daily_ohlcv_by_token=ohlcv_by_token,
        )

        new_rows: list[dict] = []
        running_equity = pre_equity
        exits_processed = 0
        for pos in session_closed:
            if pos.exit_date != as_of_date:
                continue
            running_equity += pos.net_pnl or 0.0
            new_rows.append(self._exit_row(pos, as_of_date, running_equity))
            exits_processed += 1

        new_entries = 0
        for pos in still_open:
            key = (pos.signal.instrument.symbol, pos.signal.horizon.value, pos.entry_date)
            if key in open_before or pos.entry_date != as_of_date:
                continue
            entry_cost = self._compute_cost(pos.entry_price, pos.shares)
            running_equity -= pos.entry_price * pos.shares
            new_rows.append(self._entry_row(pos, as_of_date, running_equity, entry_cost))
            new_entries += 1

        equity = running_equity

        if new_rows:
            new_df = pd.DataFrame(new_rows, columns=LEDGER_COLUMNS)
            self._ledger = pd.concat([self._ledger, new_df], ignore_index=True)
            self.save_ledger(self._ledger)

        logger.info(
            "Hybrid session %s complete: entries=%d exits=%d equity=%.2f",
            as_of_date,
            new_entries,
            exits_processed,
            equity,
        )
        return {"new_entries": new_entries, "exits_processed": exits_processed, "equity": equity}

    def _entry_row(
        self,
        pos: Position,
        as_of_date: date,
        equity: float,
        entry_cost: float,
    ) -> dict:
        entry_dt = pos.entry_datetime or parse_entry_datetime(pos.signal)
        return {
            "date": as_of_date,
            "symbol": pos.signal.instrument.symbol,
            "isin": pos.signal.instrument.isin,
            "horizon": pos.signal.horizon.value,
            "action": "entry",
            "price": pos.entry_price,
            "shares": pos.shares,
            "pnl": -entry_cost,
            "status": TradeStatus.OPEN.value,
            "equity": equity,
            "entry_time": entry_dt.isoformat() if entry_dt else "",
            "stop_loss": pos.signal.stop_loss,
            "target": pos.signal.target,
        }

    def _exit_row(self, pos: Position, as_of_date: date, equity: float) -> dict:
        return {
            "date": as_of_date,
            "symbol": pos.signal.instrument.symbol,
            "isin": pos.signal.instrument.isin,
            "horizon": pos.signal.horizon.value,
            "action": "exit",
            "price": pos.exit_price,
            "shares": pos.shares,
            "pnl": pos.net_pnl,
            "status": pos.status.value if pos.status else TradeStatus.CLOSED_TIME.value,
            "equity": equity,
            "entry_time": "",
            "stop_loss": "",
            "target": "",
        }

    # ── Signal generation ──────────────────────────────────────────────────

    def _generate_signals(
        self,
        feature_df: pd.DataFrame,
        as_of_date: date,
        ohlcv_by_token: dict[int, pd.DataFrame],
        instruments: list[Instrument],
    ) -> list[Signal]:
        """Score and rank candidates, then build signals for top entries."""
        today_features = feature_df[feature_df["date"] == as_of_date].copy()
        if today_features.empty:
            return []

        ranker_cols = [c for c in StockRanker.FEATURE_COLS if c in today_features.columns]
        if len(ranker_cols) < len(StockRanker.FEATURE_COLS):
            logger.warning("Missing ranker features; cannot generate signals.")
            return []

        scores = self.ranker.predict(today_features)
        today_features = today_features.copy()
        today_features["_rank_score"] = scores

        top_n = self.cfg.entry.get("top_n_candidates", 10)
        candidates = today_features.nlargest(top_n, "_rank_score")

        min_win_prob: float = self.cfg.entry.get("min_win_prob", 0.55)
        token_to_inst = {i.instrument_token: i for i in instruments}
        signals: list[Signal] = []

        clf_cols = [c for c in EntryClassifier.FEATURE_COLS if c in candidates.columns]
        if len(clf_cols) < len(EntryClassifier.FEATURE_COLS):
            logger.warning("Missing classifier features; skipping signal generation.")
            return []

        for _, row in candidates.iterrows():
            token = int(row.get("instrument_token", 0))
            inst = token_to_inst.get(token)
            if inst is None:
                continue

            entry_price = float(row.get("close", 0.0))
            atr = float(row.get("atr_14", 0.0))
            if entry_price <= 0 or atr <= 0:
                continue

            rank_score = float(row["_rank_score"])
            cost_per_share = self._estimate_cost_per_share(entry_price)
            row_df = pd.DataFrame([row])

            for horizon in Horizon:
                clf = self.classifiers[horizon]
                if not clf._fitted:
                    continue
                win_prob = float(clf.predict_proba(row_df)[0])
                if win_prob < min_win_prob:
                    continue

                sig = self.exit_policy.build_signal(
                    instrument=inst,
                    horizon=horizon,
                    entry_price=entry_price,
                    atr=atr,
                    win_prob=win_prob,
                    rank_score=rank_score,
                    signal_date=as_of_date,
                    cost_per_share=cost_per_share,
                )
                if sig is not None:
                    signals.append(sig)

        signals.sort(key=lambda s: s.expected_value, reverse=True)
        return signals

    # ── Market data loading ────────────────────────────────────────────────

    def _load_market_data(
        self, as_of_date: date
    ) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, list[Instrument]]:
        """Load OHLCV from the configured dataset for the universe as of *as_of_date*."""
        from trading_bot.data.loader import load_index_ohlcv, load_ohlcv
        from trading_bot.data.universe import Universe

        universe = Universe(self.cfg)
        kite_instruments = universe.load_kite_instruments()
        instruments = universe.get_instruments(as_of_date, kite_instruments)

        if not instruments:
            return {}, pd.DataFrame(), []

        lookback_start = as_of_date - timedelta(days=400)
        ohlcv_by_token = load_ohlcv(lookback_start, as_of_date, self.cfg)

        index_df = load_index_ohlcv(lookback_start, as_of_date, self.cfg)
        if index_df is None:
            index_df = pd.DataFrame(columns=["date", "close"])

        return ohlcv_by_token, index_df, instruments

    def _build_features(
        self,
        ohlcv_by_token: dict[int, pd.DataFrame],
        index_df: pd.DataFrame,
        instruments: list[Instrument],
    ) -> pd.DataFrame:
        from trading_bot.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(self.cfg)
        return pipeline.build(ohlcv_by_token, index_df, instruments, include_labels=False)

    # ── Position reconstruction ────────────────────────────────────────────

    def _reconstruct_open_positions(
        self,
        token_map: dict[int, Instrument] | None = None,
    ) -> list[Position]:
        """Rebuild open Position objects from the ledger CSV."""
        df = self._ledger
        if df.empty:
            return []

        entries = df[df["action"] == "entry"].copy()
        exits = df[df["action"] == "exit"].copy()
        exited_keys = set(
            zip(exits["symbol"], exits["horizon"], exits["date"].astype(str))
        )

        positions: list[Position] = []
        for _, row in entries.iterrows():
            if row["status"] != TradeStatus.OPEN.value:
                continue
            key = (row["symbol"], row["horizon"], str(row["date"]))
            if key in exited_keys:
                continue

            token = 0
            if token_map:
                for inst in token_map.values():
                    if inst.symbol == row["symbol"]:
                        token = inst.instrument_token
                        break

            inst = Instrument(
                symbol=str(row["symbol"]),
                isin=str(row["isin"]),
                instrument_token=token,
                exchange="NSE",
            )
            try:
                horizon = Horizon(row["horizon"])
            except ValueError:
                continue

            entry_price = float(row["price"])
            stop_loss = float(row["stop_loss"]) if pd.notna(row.get("stop_loss")) and row.get("stop_loss") != "" else entry_price * 0.95
            target = float(row["target"]) if pd.notna(row.get("target")) and row.get("target") != "" else entry_price * 1.05

            entry_dt = None
            raw_time = row.get("entry_time")
            if pd.notna(raw_time) and str(raw_time).strip():
                entry_dt = pd.to_datetime(raw_time).to_pydatetime()

            entry_date = row["date"] if isinstance(row["date"], date) else pd.Timestamp(row["date"]).date()
            sig = Signal(
                instrument=inst,
                horizon=horizon,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target,
                win_prob=0.0,
                expected_value=0.0,
                rank_score=0.0,
                signal_date=entry_date,
            )
            if entry_dt is not None:
                sig.features["entry_datetime"] = entry_dt.isoformat()

            pos = Position(
                signal=sig,
                shares=int(row["shares"]),
                entry_date=entry_date,
                entry_price=entry_price,
                entry_datetime=entry_dt,
                status=TradeStatus.OPEN,
            )
            positions.append(pos)

        return positions

    # ── Ledger I/O ─────────────────────────────────────────────────────────

    def load_ledger(self) -> pd.DataFrame:
        """Read the ledger CSV from disk.

        Returns an empty DataFrame with the expected columns if not found.
        """
        if not self.ledger_path.exists():
            return pd.DataFrame(columns=LEDGER_COLUMNS)
        try:
            df = pd.read_csv(self.ledger_path, parse_dates=["date"])
            df["date"] = pd.to_datetime(df["date"]).dt.date
            for col in LEDGER_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load ledger from %s: %s", self.ledger_path, exc)
            return pd.DataFrame(columns=LEDGER_COLUMNS)

    def save_ledger(self, df: pd.DataFrame) -> None:
        """Persist the full ledger DataFrame to CSV (append-only: df must include all history)."""
        try:
            df.to_csv(self.ledger_path, index=False)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save ledger to %s: %s", self.ledger_path, exc)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _current_equity(self) -> float:
        df = self._ledger
        if df.empty or "equity" not in df.columns:
            return _INITIAL_EQUITY
        last_val = df["equity"].dropna()
        if last_val.empty:
            return _INITIAL_EQUITY
        return float(last_val.iloc[-1])

    def _estimate_cost_per_share(self, price: float) -> float:
        costs = self.cfg.costs
        brokerage = price * costs.get("brokerage_pct", 0.03) / 100.0
        stt = price * costs.get("stt_delivery_pct", 0.1) / 100.0
        stamp = price * costs.get("stamp_duty_pct", 0.015) / 100.0
        exchange = price * costs.get("exchange_txn_charge_pct", 0.00345) / 100.0
        sebi = price * costs.get("sebi_turnover_fee_pct", 0.0001) / 100.0
        gst = (brokerage + exchange) * costs.get("gst_on_charges_pct", 18.0) / 100.0
        slippage = price * costs.get("slippage_min_pct", 0.05) / 100.0
        return brokerage + stt + stamp + exchange + sebi + gst + slippage

    def _compute_cost(self, price: float, shares: int) -> float:
        return self._estimate_cost_per_share(price) * shares
