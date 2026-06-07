"""Generate a human-readable trade report from backtest results."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

from trading_bot.types import Position, TradeStatus


_STATUS_LABEL = {
    TradeStatus.CLOSED_TP:   "Target hit",
    TradeStatus.CLOSED_SL:   "Stop-loss hit",
    TradeStatus.CLOSED_TIME: "Time stop (max hold days reached)",
    TradeStatus.CLOSED_MANUAL: "Manually closed",
}

_STATUS_EMOJI = {
    TradeStatus.CLOSED_TP:   "✅",
    TradeStatus.CLOSED_SL:   "❌",
    TradeStatus.CLOSED_TIME: "⏱",
    TradeStatus.CLOSED_MANUAL: "🔧",
}


def _fmt(v: Any, pct: bool = False, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if pct:
        return f"{float(v) * 100:.1f}%"
    return f"{float(v):.{decimals}f}"


def _buy_reason(f: dict) -> str:
    lines: list[str] = []
    score = f.get("model_score")
    if score is not None:
        lines.append(f"model score {_fmt(score, decimals=3)} (P(next-day ≥ 1.5% up))")

    vol = f.get("volume_ratio_lag1")
    if vol is not None and not math.isnan(float(vol)):
        lines.append(f"volume {_fmt(vol)}× 20-day avg")

    rsi = f.get("rsi_lag1")
    if rsi is not None and not math.isnan(float(rsi)):
        lines.append(f"RSI {_fmt(rsi, decimals=1)}")

    ret20 = f.get("ret_20d_lag1")
    if ret20 is not None and not math.isnan(float(ret20)):
        lines.append(f"20-day return {_fmt(ret20, pct=True)}")

    sma20 = f.get("close_sma20_lag1")
    if sma20 is not None and not math.isnan(float(sma20)):
        direction = "above" if float(sma20) > 0 else "below"
        lines.append(f"price {_fmt(abs(float(sma20)) * 100, decimals=1)}% {direction} SMA-20")

    gap = f.get("gap_risk_lag1")
    if gap is not None and not math.isnan(float(gap)):
        lines.append(f"overnight gap {_fmt(float(gap) * 100, decimals=2)}%")

    roce = f.get("roce")
    if roce is not None and not math.isnan(float(roce)):
        lines.append(f"ROCE {_fmt(float(roce) * 100, decimals=1)}%")

    de = f.get("debt_equity")
    if de is not None and not math.isnan(float(de)):
        lines.append(f"D/E {_fmt(de, decimals=2)}×")

    pg_yoy = f.get("profit_growth_yoy")
    if pg_yoy is not None and not math.isnan(float(pg_yoy)):
        lines.append(f"annual profit growth {_fmt(float(pg_yoy) * 100, decimals=1)}% YoY")

    pg_qtr = f.get("profit_growth_qtr")
    if pg_qtr is not None and not math.isnan(float(pg_qtr)):
        lines.append(f"quarterly profit growth {_fmt(float(pg_qtr) * 100, decimals=1)}% same-qtr YoY")

    pe = f.get("pe")
    if pe is not None and not math.isnan(float(pe)):
        lines.append(f"P/E {_fmt(pe, decimals=1)}×")

    above50 = f.get("above_dma50")
    above200 = f.get("above_dma200")
    dma_parts = []
    if above50 is not None and not math.isnan(float(above50)):
        dma_parts.append(f"50D {'✓' if float(above50) > 0 else '✗'}")
    if above200 is not None and not math.isnan(float(above200)):
        dma_parts.append(f"200D {'✓' if float(above200) > 0 else '✗'}")
    if dma_parts:
        lines.append(f"price above DMA: {', '.join(dma_parts)}")

    return "; ".join(lines) if lines else "see model features"


def _sell_reason(pos: Position) -> str:
    status = pos.status
    label = _STATUS_LABEL.get(status, str(status))
    emoji = _STATUS_EMOJI.get(status, "")
    sl = pos.signal.stop_loss
    tp = pos.signal.target
    entry = pos.entry_price
    exit_p = pos.exit_price or 0.0

    if status == TradeStatus.CLOSED_SL:
        drop_pct = (exit_p - entry) / entry * 100
        return f"{emoji} {label} at ₹{exit_p:.2f} ({drop_pct:+.1f}% from entry). Stop was ₹{sl:.2f} ({(sl - entry) / entry * 100:+.1f}% from entry)."
    elif status == TradeStatus.CLOSED_TP:
        gain_pct = (exit_p - entry) / entry * 100
        return f"{emoji} {label} at ₹{exit_p:.2f} ({gain_pct:+.1f}% from entry). Target was ₹{tp:.2f}."
    elif status == TradeStatus.CLOSED_TIME:
        chg_pct = (exit_p - entry) / entry * 100
        return f"{emoji} {label} at ₹{exit_p:.2f} ({chg_pct:+.1f}% from entry). Neither SL ₹{sl:.2f} nor TP ₹{tp:.2f} reached."
    else:
        return f"{emoji} {label} at ₹{exit_p:.2f}."


def generate_trade_report(
    closed: list[Position],
    output_path: Path,
    *,
    title: str = "Move Predictor — Trade Report",
) -> None:
    """Write a Markdown report with per-trade buy/sell narratives."""
    wins = [p for p in closed if p.status == TradeStatus.CLOSED_TP]
    losses = [p for p in closed if p.status == TradeStatus.CLOSED_SL]
    time_exits = [p for p in closed if p.status == TradeStatus.CLOSED_TIME]
    total_pnl = sum(p.net_pnl or 0.0 for p in closed)
    win_rate = len(wins) / len(closed) if closed else 0.0

    lines: list[str] = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"| | |",
        f"|---|---|",
        f"| Total trades | {len(closed)} |",
        f"| Winners (TP) | {len(wins)} |",
        f"| Losers (SL) | {len(losses)} |",
        f"| Time exits | {len(time_exits)} |",
        f"| Win rate | {win_rate:.1%} |",
        f"| Net P&L | ₹{total_pnl:,.0f} |",
        "",
        "---",
        "",
        "## Trade Log",
        "",
    ]

    for i, pos in enumerate(sorted(closed, key=lambda p: p.entry_date), start=1):
        f = pos.signal.features
        pnl = pos.net_pnl or 0.0
        hold_days = (pos.exit_date - pos.entry_date).days if pos.exit_date else "?"
        r_mult = pos.r_multiple
        r_str = f"{r_mult:+.2f}R" if r_mult is not None else ""

        lines += [
            f"### {i}. {pos.signal.instrument.symbol}  —  {pos.entry_date} → {pos.exit_date}  ({r_str}  ₹{pnl:+,.0f})",
            "",
            f"**Buy** on {pos.entry_date} at ₹{pos.entry_price:.2f} ({hold_days} sessions held)",
            "",
            f"*Why bought:* {_buy_reason(f)}",
            "",
            f"**Sell** on {pos.exit_date}: {_sell_reason(pos)}",
            "",
            "---",
            "",
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
