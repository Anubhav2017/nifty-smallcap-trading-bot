"""Plotly charts for OHLCV and indicators."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Price-panel overlays (same y-axis as candlesticks)
PRICE_OVERLAY_SPECS: dict[str, tuple[str, str]] = {
    "sma_20": ("SMA 20", "#ff9800"),
    "sma_50": ("SMA 50", "#2196f3"),
    "ema_12": ("EMA 12", "#ab47bc"),
    "ema_26": ("EMA 26", "#7e57c2"),
}

# Separate subplots below price
PANEL_SPECS: dict[str, dict] = {
    "volume": {
        "title": "Volume",
        "kind": "bar",
        "color": "#78909c",
    },
    "rsi_14": {
        "title": "RSI (14)",
        "kind": "line",
        "color": "#fdd835",
        "range": [0, 100],
        "hlines": [70, 30],
    },
    "volume_ratio_20d": {
        "title": "Volume / MA20",
        "kind": "line",
        "color": "#4dd0e1",
    },
    "volatility_20d": {
        "title": "Volatility 20d",
        "kind": "line",
        "color": "#ff7043",
        "pct": True,
    },
    "ret_5d": {
        "title": "Return 5d",
        "kind": "line",
        "color": "#66bb6a",
        "pct": True,
        "zeroline": True,
    },
    "ret_20d": {
        "title": "Return 20d",
        "kind": "line",
        "color": "#42a5f5",
        "pct": True,
        "zeroline": True,
    },
    "close_sma_20d": {
        "title": "Close vs SMA20",
        "kind": "line",
        "color": "#ffa726",
        "pct": True,
        "zeroline": True,
    },
    "close_sma_50d": {
        "title": "Close vs SMA50",
        "kind": "line",
        "color": "#5c6bc0",
        "pct": True,
        "zeroline": True,
    },
    "high_low_range": {
        "title": "High-Low range",
        "kind": "line",
        "color": "#ec407a",
        "pct": True,
    },
}

DEFAULT_PRICE_OVERLAYS = ["sma_20", "sma_50"]
DEFAULT_PANELS = ["volume", "rsi_14"]

# Streamlit → Plotly.js config (scroll wheel zoom, full toolbar)
PLOTLY_CHART_CONFIG: dict = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "doubleClick": "reset+autosize",
}

_RANGE_SELECTOR = dict(
    buttons=[
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(count=3, label="3Y", step="year", stepmode="backward"),
        dict(step="all", label="All"),
    ],
    bgcolor="rgba(40,40,40,0.8)",
    activecolor="#26a69a",
    x=0,
    y=1.01,
    xanchor="left",
    yanchor="bottom",
)


def _y_range_for_x_window(df: pd.DataFrame, x0, x1, pad_frac: float = 0.06) -> list[float] | None:
    """Min/max price (incl. overlays) for the visible x window."""
    if df.empty:
        return None
    ts = pd.to_datetime(df["date"])
    mask = (ts >= pd.Timestamp(x0)) & (ts <= pd.Timestamp(x1))
    window = df.loc[mask]
    if window.empty:
        return None
    lo = float(window["low"].min())
    hi = float(window["high"].max())
    if hi <= lo:
        return None
    pad = (hi - lo) * pad_frac
    return [lo - pad, hi + pad]


def _apply_zoom_layout(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    n_rows: int,
    panel_keys: list[str],
    initial_days: int | None = 180,
) -> None:
    """Enable pan/zoom, range slider, quick ranges, and y-axis fit to visible bars."""
    x = pd.to_datetime(df["date"])
    x_min, x_max = x.min(), x.max()

    if initial_days and len(df) > initial_days:
        view_start = x_max - pd.Timedelta(days=initial_days)
        if view_start < x_min:
            view_start = x_min
    else:
        view_start = x_min

    price_yrange = _y_range_for_x_window(df, view_start, x_max)

    # Price panel: zoom/pan + quick range buttons
    fig.update_xaxes(
        type="date",
        rangeslider=dict(visible=False),
        rangeselector=_RANGE_SELECTOR,
        range=[view_start, x_max],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Price",
        fixedrange=False,
        autorange=price_yrange is None,
        range=price_yrange,
        row=1,
        col=1,
    )

    # Indicator panels
    for row_idx, key in enumerate(panel_keys, start=2):
        spec = PANEL_SPECS[key]
        y_cfg: dict = {"fixedrange": False, "row": row_idx, "col": 1}
        if "range" in spec and not spec.get("pct"):
            # RSI: default 0–100 but allow zoom
            y_cfg["range"] = spec["range"]
            y_cfg["autorange"] = False
        else:
            y_cfg["autorange"] = True
        fig.update_yaxes(**y_cfg)

    # Range slider on bottom row — scroll through full loaded history
    fig.update_xaxes(
        type="date",
        rangeslider=dict(visible=True, thickness=0.05),
        range=[view_start, x_max],
        row=n_rows,
        col=1,
    )

    # Hide rangeslider on upper rows (only bottom thumb strip)
    for row_idx in range(1, n_rows):
        fig.update_xaxes(rangeslider=dict(visible=False), row=row_idx, col=1)

    fig.update_layout(
        dragmode="zoom",
        hovermode="x unified",
        margin=dict(l=48, r=40, t=80, b=60),
    )


def _panel_series(df: pd.DataFrame, key: str, spec: dict) -> pd.Series:
    if key == "volume":
        return df["volume"]
    series = df[key]
    if spec.get("pct"):
        return series * 100.0
    return series


def build_price_chart(
    df: pd.DataFrame,
    *,
    price_overlays: list[str] | None = None,
    panels: list[str] | None = None,
) -> go.Figure:
    """Candlestick chart with optional MA/EMA overlays and indicator subplots."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data in selected range")
        return fig

    overlays = [k for k in (price_overlays or DEFAULT_PRICE_OVERLAYS) if k in PRICE_OVERLAY_SPECS]
    panel_keys = [k for k in (panels or DEFAULT_PANELS) if k in PANEL_SPECS]

    x = df["date"]
    n_rows = 1 + len(panel_keys)
    if n_rows == 1:
        row_heights = [1.0]
    else:
        panel_share = min(0.42, 0.18 * len(panel_keys))
        price_share = 1.0 - panel_share
        each_panel = panel_share / len(panel_keys)
        row_heights = [price_share] + [each_panel] * len(panel_keys)

    subplot_titles = ["Price"] + [PANEL_SPECS[k]["title"] for k in panel_keys]
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04 if n_rows > 2 else 0.06,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    for key in overlays:
        if key not in df.columns:
            continue
        name, color = PRICE_OVERLAY_SPECS[key]
        fig.add_trace(
            go.Scatter(x=x, y=df[key], name=name, line=dict(width=1.2, color=color)),
            row=1,
            col=1,
        )

    for row_idx, key in enumerate(panel_keys, start=2):
        spec = PANEL_SPECS[key]
        y = _panel_series(df, key, spec)

        if spec["kind"] == "bar" and key == "volume":
            colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(df["open"], df["close"])]
            fig.add_trace(
                go.Bar(x=x, y=y, name="Volume", marker_color=colors, opacity=0.7),
                row=row_idx,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=spec["title"],
                    line=dict(color=spec["color"], width=1.5),
                ),
                row=row_idx,
                col=1,
            )

        y_title = spec["title"]
        if spec.get("pct"):
            y_title = f"{spec['title']} (%)"
        fig.update_yaxes(title_text=y_title, row=row_idx, col=1)

        if "range" in spec:
            fig.update_yaxes(range=spec["range"], row=row_idx, col=1)
        for level in spec.get("hlines", []):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="#888",
                opacity=0.6,
                row=row_idx,
                col=1,
            )
        if spec.get("zeroline"):
            fig.add_hline(y=0, line_color="#666", line_width=1, row=row_idx, col=1)

    height = 460 + 150 * len(panel_keys)
    fig.update_layout(
        title="OHLCV (daily)",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
        template="plotly_dark",
    )
    initial_days = 180
    _apply_zoom_layout(fig, df, n_rows=n_rows, panel_keys=panel_keys, initial_days=initial_days)
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def indicator_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [
        "close",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "volatility_20d",
        "volume_ratio_20d",
        "rsi_14",
        "close_sma_20d",
        "close_sma_50d",
    ]
    present = [c for c in cols if c in df.columns]
    row = df[present].iloc[-1]
    labels = {
        "close": "Close",
        "ret_1d": "Return 1d",
        "ret_5d": "Return 5d",
        "ret_20d": "Return 20d",
        "volatility_20d": "Volatility 20d",
        "volume_ratio_20d": "Vol / MA20",
        "rsi_14": "RSI 14",
        "close_sma_20d": "vs SMA20",
        "close_sma_50d": "vs SMA50",
    }
    pct_cols = {"Return 1d", "Return 5d", "Return 20d", "Volatility 20d", "vs SMA20", "vs SMA50"}

    formatted: list[str] = []
    indicator_names: list[str] = []
    for col in present:
        ind = labels.get(col, col)
        val = row[col]
        indicator_names.append(ind)
        if ind in pct_cols and pd.notna(val):
            formatted.append(f"{float(val) * 100:.2f}%")
        elif ind == "RSI 14" and pd.notna(val):
            formatted.append(f"{float(val):.1f}")
        elif ind == "Vol / MA20" and pd.notna(val):
            formatted.append(f"{float(val):.2f}x")
        elif pd.notna(val):
            formatted.append(f"{float(val):,.2f}")
        else:
            formatted.append("—")

    return pd.DataFrame({"Indicator": indicator_names, "Value": formatted})
