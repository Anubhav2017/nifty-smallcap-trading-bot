"""Indian cash equity cost model for delivery trades (NSE)."""

from __future__ import annotations

from trading_bot.config import Config
from trading_bot.types import Horizon


class CostModel:
    """Compute round-trip transaction costs for Indian cash equity delivery trades.

    All cost rates are loaded from the ``costs`` section of the YAML config.
    The caller may pass an optional ``atr_proxy`` (e.g. ATR₁₄ in rupees) to
    model spread-based slippage; it falls back to the minimum slippage floor.
    """

    def __init__(self, cfg: Config) -> None:
        c = cfg.costs
        self._brokerage_pct: float = float(c["brokerage_pct"])
        self._stt_delivery_pct: float = float(c["stt_delivery_pct"])
        self._stamp_duty_pct: float = float(c["stamp_duty_pct"])
        self._exchange_txn_charge_pct: float = float(c["exchange_txn_charge_pct"])
        self._sebi_turnover_fee_pct: float = float(c["sebi_turnover_fee_pct"])
        self._gst_on_charges_pct: float = float(c["gst_on_charges_pct"])
        self._slippage_min_pct: float = float(c["slippage_min_pct"])
        self._slippage_spread_factor: float = float(c["slippage_spread_factor"])

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(
        self,
        entry_price: float,
        exit_price: float,
        shares: int,
        horizon: Horizon,
        atr_proxy: float = 0.0,
    ) -> float:
        """Return total round-trip cost in INR for a delivery trade.

        Args:
            entry_price: Buy price per share (INR).
            exit_price:  Sell price per share (INR).
            shares:      Number of shares traded.
            horizon:     Trade horizon (reserved for future per-horizon overrides).
            atr_proxy:   Optional spread proxy in INR (e.g. ATR₁₄). When 0 the
                         minimum slippage floor applies to both legs.

        Returns:
            Total cost in INR (always ≥ 0).
        """
        if shares <= 0:
            return 0.0

        turnover = (entry_price + exit_price) * shares

        brokerage = turnover * self._brokerage_pct / 100.0
        # STT: delivery sell-side only
        stt = exit_price * shares * self._stt_delivery_pct / 100.0
        # Stamp duty: buy-side only
        stamp = entry_price * shares * self._stamp_duty_pct / 100.0
        exchange_charge = turnover * self._exchange_txn_charge_pct / 100.0
        sebi_fee = turnover * self._sebi_turnover_fee_pct / 100.0
        gst = (brokerage + exchange_charge) * self._gst_on_charges_pct / 100.0

        # Slippage on both legs: max(min_pct floor, spread_factor * atr_proxy)
        slip_per_share = max(
            self._slippage_min_pct / 100.0 * entry_price,
            self._slippage_spread_factor * atr_proxy,
        )
        slippage = slip_per_share * shares * 2

        return brokerage + stt + stamp + exchange_charge + sebi_fee + gst + slippage

    def compute_pct(
        self,
        entry_price: float,
        exit_price: float,
        shares: int,
        horizon: Horizon,
        atr_proxy: float = 0.0,
    ) -> float:
        """Return total cost as a percentage of entry position value.

        Returns:
            Cost as % of (entry_price × shares). Returns 0.0 if position value ≤ 0.
        """
        position_value = entry_price * shares
        if position_value <= 0:
            return 0.0
        total_cost = self.compute(entry_price, exit_price, shares, horizon, atr_proxy)
        return (total_cost / position_value) * 100.0
