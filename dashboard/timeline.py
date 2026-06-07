"""Price + fundamentals timeline: events and significant moves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_bot.data.screener_excel import SECTION_LABELS

PERIOD_EVENT_LABELS = {
    "quarterly": "Quarterly results",
    "annual_pl": "Annual P&L",
    "annual_bs": "Balance sheet (annual)",
    "annual_cf": "Cash flow (annual)",
}

METRIC_LABELS = {
    "f_sales": "Sales",
    "f_net_profit": "Net profit",
    "f_operating_profit": "Operating profit",
    "f_profit_margin": "Profit margin",
    "f_sales_growth": "Sales growth",
    "f_roe": "ROE",
    "f_debt_equity": "Debt / equity",
}


def _format_metric(name: str, value: float) -> str:
    label = METRIC_LABELS.get(name, name.replace("f_", "").replace("_", " ").title())
    if name in ("f_profit_margin", "f_roe", "f_sales_growth") and pd.notna(value):
        return f"{label} {float(value) * 100:.1f}%"
    if abs(value) >= 1e7:
        return f"{label} {value / 1e7:.2f} Cr"
    if abs(value) >= 1e5:
        return f"{label} {value / 1e5:.2f} L"
    return f"{label} {value:,.0f}"


def fundamental_events(fund_df: pd.DataFrame) -> pd.DataFrame:
    """One row per Screener report date with a short metrics summary."""
    if fund_df.empty:
        return pd.DataFrame(
            columns=["date", "period_type", "title", "detail", "category"]
        )

    rows: list[dict] = []
    show_cols = [
        c
        for c in (
            "f_sales",
            "f_net_profit",
            "f_operating_profit",
            "f_profit_margin",
            "f_sales_growth",
            "f_roe",
            "f_debt_equity",
        )
        if c in fund_df.columns
    ]

    sub = fund_df.copy()
    sub["report_date"] = pd.to_datetime(sub["report_date"]).dt.normalize()
    sub = sub.sort_values(["report_date", "period_type"])

    for _, row in sub.iterrows():
        period = str(row["period_type"])
        title = PERIOD_EVENT_LABELS.get(
            period, SECTION_LABELS.get(period, period.replace("_", " ").title())
        )
        parts: list[str] = []
        for col in show_cols:
            val = row.get(col)
            if pd.notna(val):
                try:
                    parts.append(_format_metric(col, float(val)))
                except (TypeError, ValueError):
                    continue
        detail = " · ".join(parts[:4]) if parts else "Report filed"
        rows.append(
            {
                "date": row["report_date"],
                "period_type": period,
                "title": title,
                "detail": detail,
                "category": "fundamental",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.drop_duplicates(subset=["date", "period_type"], keep="last").sort_values(
        "date"
    )


def significant_price_moves(
    df: pd.DataFrame,
    *,
    window: int = 60,
    z_threshold: float = 2.0,
    min_abs_return: float = 0.015,
) -> pd.DataFrame:
    """
    Flag days where |return| exceeds rolling mean + z_threshold * rolling std.

    Expects **daily** OHLCV from ``ohlcv/day/``; uses close-to-close returns.
    """
    cols = ["date", "return", "z_score", "direction", "title", "detail", "category"]
    if df.empty or len(df) < window + 2:
        return pd.DataFrame(columns=cols)

    work = df.sort_values("date").copy()
    work["return"] = work["close"].pct_change()
    roll_mean = work["return"].rolling(window, min_periods=max(20, window // 2)).mean()
    roll_std = work["return"].rolling(window, min_periods=max(20, window // 2)).std()
    z = (work["return"] - roll_mean) / roll_std.replace(0, np.nan)

    mask = (
        (z.abs() >= z_threshold)
        & (work["return"].abs() >= min_abs_return)
        & work["return"].notna()
    )
    hits = work.loc[mask].copy()
    if hits.empty:
        return pd.DataFrame(columns=cols)

    hits["z_score"] = z.loc[mask].values
    hits["direction"] = np.where(hits["return"] >= 0, "up", "down")
    hits["title"] = hits.apply(
        lambda r: (
            f"Sharp {'gain' if r['direction'] == 'up' else 'drop'} "
            f"({r['return'] * 100:+.1f}%)"
        ),
        axis=1,
    )
    typical_pct = (roll_std.loc[mask].abs() * 100).values
    hits["detail"] = [
        f"z={z:.1f} vs {window}d typical daily move (~{typ:.2f}%)"
        for z, typ in zip(hits["z_score"], typical_pct)
    ]
    hits["category"] = "price_move"

    return hits[
        ["date", "return", "z_score", "direction", "title", "detail", "category"]
    ].reset_index(drop=True)


def merge_timeline_events(
    fundamental: pd.DataFrame,
    moves: pd.DataFrame,
) -> pd.DataFrame:
    """Single chronological table for the UI."""
    parts: list[pd.DataFrame] = []
    if not fundamental.empty:
        f = fundamental.copy()
        f["return"] = np.nan
        f["z_score"] = np.nan
        f["direction"] = ""
        parts.append(f)
    if not moves.empty:
        parts.append(moves)
    if not parts:
        return pd.DataFrame(
            columns=[
                "date",
                "category",
                "title",
                "detail",
                "return",
                "z_score",
                "direction",
                "period_type",
            ]
        )

    merged = pd.concat(parts, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date", ascending=False)
    display_cols = [
        "date",
        "category",
        "title",
        "detail",
        "return",
        "z_score",
        "direction",
        "period_type",
    ]
    present = [c for c in display_cols if c in merged.columns]
    out = merged[present].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    if "return" in out.columns:
        out["return"] = out["return"].apply(
            lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else ""
        )
    if "z_score" in out.columns:
        out["z_score"] = out["z_score"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else ""
        )
    return out


def build_timeline_chart(
    df: pd.DataFrame,
    fundamental: pd.DataFrame,
    moves: pd.DataFrame,
    *,
    show_fundamentals: bool = True,
    show_moves: bool = True,
) -> go.Figure:
    """Daily price chart with fundamental markers and highlighted significant moves."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data in selected range")
        return fig

    work = df.sort_values("date").copy()
    x = pd.to_datetime(work["date"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=("Price & events", "Daily return (%)"),
    )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=work["open"],
            high=work["high"],
            low=work["low"],
            close=work["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    ret_pct = work["close"].pct_change() * 100.0
    bar_colors = [
        "#26a69a" if (pd.notna(r) and r >= 0) else "#ef5350" for r in ret_pct
    ]
    fig.add_trace(
        go.Bar(x=x, y=ret_pct, name="Return", marker_color=bar_colors, opacity=0.85),
        row=2,
        col=1,
    )

    work_ts = pd.to_datetime(work["date"])

    if show_moves and not moves.empty:
        for direction, color, symbol in (
            ("up", "#66bb6a", "triangle-up"),
            ("down", "#ef5350", "triangle-down"),
        ):
            sub = moves[moves["direction"] == direction]
            if sub.empty:
                continue
            xs = pd.to_datetime(sub["date"])
            ys = []
            for d in xs:
                row = work.loc[work_ts == d]
                if row.empty:
                    idx = work_ts.searchsorted(d, side="right") - 1
                    if idx < 0:
                        ys.append(None)  # date before OHLCV data — keep xs/ys aligned
                        continue
                    row = work.iloc[[idx]]
                hi = float(row["high"].iloc[-1])
                ys.append(hi * (1.02 if direction == "up" else 0.98))
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=f"Significant {direction}",
                    marker=dict(
                        size=14,
                        color=color,
                        symbol=symbol,
                        line=dict(width=1, color="#fff"),
                    ),
                    text=sub["title"],
                    hovertext=sub["detail"],
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=pd.to_numeric(sub["return"], errors="coerce") * 100.0,
                    mode="markers",
                    marker=dict(size=10, color=color, symbol=symbol),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )

    if show_fundamentals and not fundamental.empty:
        fund = fundamental.copy()
        fund["date"] = pd.to_datetime(fund["date"])
        xs = fund["date"]
        ys = []
        for d in xs:
            row = work.loc[work_ts == d]
            if row.empty:
                idx = work_ts.searchsorted(d, side="right") - 1
                if idx < 0:
                    ys.append(None)  # date is before all OHLCV data — skip marker
                    continue
                row = work.iloc[[idx]]
            ys.append(float(row["low"].iloc[-1]) * 0.97)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name="Fundamental filing",
                marker=dict(size=11, color="#42a5f5", symbol="diamond"),
                text=fund["title"].fillna("") + "<br>" + fund["detail"].fillna(""),
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )
        for _, ev in fund.iterrows():
            fig.add_vline(
                x=ev["date"],
                line_width=1,
                line_dash="dot",
                line_color="rgba(66, 165, 245, 0.45)",
                row=1,
                col=1,
            )

    fig.update_layout(
        title="Timeline — daily price & fundamentals",
        height=640,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=1),
        hovermode="x unified",
        margin=dict(l=48, r=24, t=72, b=48),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Return %", row=2, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=2, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    return fig
