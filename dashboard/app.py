#!/usr/bin/env python3
"""Streamlit dashboard: candlesticks, indicators, Screener fundamentals."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dashboard.charts import (
    DEFAULT_PANELS,
    DEFAULT_PRICE_OVERLAYS,
    PANEL_SPECS,
    PLOTLY_CHART_CONFIG,
    PRICE_OVERLAY_SPECS,
    build_price_chart,
    indicator_summary_table,
)
from dashboard.data import (
    bar_date_bounds,
    dataset_summary,
    load_with_indicators,
    screener_path,
    symbol_choices,
)
from trading_bot.data.screener_excel import (
    SECTION_LABELS,
    load_meta,
    load_section_tables,
    load_symbol_fundamentals,
)

CONFIG_PATH = ROOT / "config" / "dashboard.json"
DAY_HISTORY = timedelta(days=3 * 365)
MINUTE_DEFAULT = timedelta(days=30)


@st.cache_data(show_spinner=False)
def _cached_bars(
    symbol: str,
    interval: str,
    root_str: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    root = Path(root_str)
    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    return load_with_indicators(symbol, interval, root, start_ts, end_ts)


@st.cache_data(show_spinner=False)
def _cached_fundamentals(symbol: str, root_str: str) -> pd.DataFrame:
    return load_symbol_fundamentals(Path(root_str) / "screener_excel", symbol)


@st.cache_data(show_spinner=False)
def _cached_bounds(symbol: str, interval: str, root_str: str) -> tuple[str, str]:
    lo, hi = bar_date_bounds(symbol, interval, Path(root_str))
    return str(lo.date()), str(hi.date())


@st.cache_data(show_spinner=False)
def _cached_sections(symbol: str, root_str: str) -> dict:
    path = screener_path(Path(root_str), symbol)
    return {k: v.copy() for k, v in load_section_tables(path).items()}


def _load_config() -> dict:
    if CONFIG_PATH.is_file():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"dataset_root": "dataset_smallcap250"}


def main() -> None:
    st.set_page_config(
        page_title="Smallcap 250 Explorer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cfg = _load_config()
    default_root = ROOT / cfg.get("dataset_root", "dataset_smallcap250")

    st.title("Nifty Smallcap 250 — Market & Fundamentals")
    st.caption("dataset_smallcap250 · OHLCV + Screener Excel")

    with st.sidebar:
        st.header("Dataset")
        root_input = st.text_input("Dataset folder", value=str(default_root))
        root = Path(root_input).expanduser()
        if not root.is_absolute():
            root = (ROOT / root).resolve()
        if not root.is_dir():
            st.error(f"Folder not found: {root}")
            st.stop()

        summary = dataset_summary(root)
        dr = summary.get("date_range") or {}
        st.markdown(
            f"**Day bars:** {summary['day_symbols']} · "
            f"**Minute:** {summary['minute_symbols']} · "
            f"**Screener:** {summary['screener_symbols']}"
        )
        if dr:
            st.caption(f"Dataset range: {dr.get('from', '?')} → {dr.get('to', '?')}")

        st.divider()
        st.header("Symbol")
        choices = symbol_choices(root)
        if not choices:
            st.warning("No symbols found under ohlcv/ or screener_excel/.")
            st.stop()

        labels = [c[0] for c in choices]
        sym_by_label = {c[0]: c[1] for c in choices}
        filter_text = st.text_input("Filter", placeholder="Type to narrow list…")
        filtered = [label for label in labels if filter_text.upper() in label.upper()] if filter_text else labels
        if not filtered:
            filtered = labels
        selected_label = st.selectbox("Stock", filtered, index=0)
        symbol = sym_by_label[selected_label]

        st.divider()
        st.header("Chart")
        interval = st.radio("Interval", ["day", "minute"], horizontal=True)

        try:
            min_s, max_s = _cached_bounds(symbol, interval, str(root))
            min_d = pd.Timestamp(min_s).date()
            max_d = pd.Timestamp(max_s).date()
        except FileNotFoundError:
            st.warning(f"No {interval} OHLCV for {symbol}")
            st.stop()

        lookback = DAY_HISTORY if interval == "day" else MINUTE_DEFAULT
        default_start: date = max(min_d, max_d - lookback)

        range_preset = st.selectbox(
            "History window",
            options=["3 years", "1 year", "6 months", "3 months", "All available", "Custom"],
            index=0 if interval == "day" else 3,
            help="Daily charts load up to 3 years by default. Use the range slider under the chart to pan.",
        )
        preset_days = {
            "3 years": 3 * 365,
            "1 year": 365,
            "6 months": 183,
            "3 months": 92,
        }
        if range_preset in preset_days:
            default_start = max(min_d, max_d - timedelta(days=preset_days[range_preset]))
        elif range_preset == "All available":
            default_start = min_d

        date_range = st.date_input(
            "Date range (data loaded into chart)",
            value=(default_start, max_d),
            min_value=min_d,
            max_value=max_d,
            disabled=range_preset != "Custom",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
        else:
            start_d, end_d = min_d, max_d

        st.divider()
        st.header("Price overlays")
        overlay_labels = {spec[0]: key for key, spec in PRICE_OVERLAY_SPECS.items()}
        selected_overlay_labels = st.multiselect(
            "Lines on price chart",
            options=list(overlay_labels.keys()),
            default=[
                PRICE_OVERLAY_SPECS[k][0]
                for k in DEFAULT_PRICE_OVERLAYS
                if k in PRICE_OVERLAY_SPECS
            ],
        )
        price_overlays = [overlay_labels[label] for label in selected_overlay_labels]

        st.header("Indicator panels")
        panel_labels = {spec["title"]: key for key, spec in PANEL_SPECS.items()}
        default_panel_labels = [
            PANEL_SPECS[k]["title"] for k in DEFAULT_PANELS if k in PANEL_SPECS
        ]
        selected_panel_labels = st.multiselect(
            "Subplots below price",
            options=list(panel_labels.keys()),
            default=default_panel_labels,
        )
        chart_panels = [panel_labels[label] for label in selected_panel_labels]

    df = _cached_bars(symbol, interval, str(root), str(start_d), str(end_d))

    tab_chart, tab_ind, tab_fund = st.tabs(["Charts", "Indicators", "Fundamentals"])

    with tab_chart:
        if df.empty:
            st.info("No bars in this date range.")
        else:
            fig = build_price_chart(
                df,
                price_overlays=price_overlays,
                panels=chart_panels,
                interval=interval,
            )
            st.caption(
                "Scroll to zoom · drag to box-zoom · shift+drag to pan · double-click to reset · "
                "use **1M–3Y / All** buttons or the range slider below the chart to move through history."
            )
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CHART_CONFIG)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bars", f"{len(df):,}")
            c2.metric("Last close", f"{df['close'].iloc[-1]:,.2f}")
            chg = df["close"].pct_change().iloc[-1]
            c3.metric("Last change", f"{chg * 100:.2f}%" if pd.notna(chg) else "—")
            c4.metric("Last volume", f"{df['volume'].iloc[-1]:,.0f}")

    with tab_ind:
        st.subheader("Latest indicator values")
        st.dataframe(indicator_summary_table(df), hide_index=True, width="stretch")
        if not df.empty:
            st.subheader("Indicator history")
            hist_cols = [
                c
                for c in [
                    "date",
                    "close",
                    "rsi_14",
                    "sma_20",
                    "sma_50",
                    "ret_5d",
                    "ret_20d",
                    "volatility_20d",
                    "volume_ratio_20d",
                ]
                if c in df.columns
            ]
            st.dataframe(
                df[hist_cols].tail(120).sort_values("date", ascending=False),
                hide_index=True,
                width="stretch",
            )

    with tab_fund:
        path = screener_path(root, symbol)
        if not path.is_file():
            st.warning(
                f"No Screener export for **{symbol}** (`{path.name}`). "
                "Add files under `dataset_smallcap250/screener_excel/`."
            )
        else:
            meta = load_meta(path)
            if meta:
                st.subheader("Company snapshot (from Screener export)")
                mcols = st.columns(min(len(meta), 4))
                for col, (k, v) in zip(mcols, meta.items()):
                    col.metric(k.title(), v)

            sections = _cached_sections(symbol, str(root))
            derived = _cached_fundamentals(symbol, str(root))

            if not derived.empty:
                st.subheader("Derived ratios (latest filed periods)")
                show = derived.copy()
                show["report_date"] = pd.to_datetime(show["report_date"]).dt.strftime("%Y-%m-%d")
                ratio_cols = [c for c in show.columns if c.startswith("f_")]
                st.dataframe(
                    show[["period_type", "report_date", *ratio_cols]].tail(8),
                    hide_index=True,
                    width="stretch",
                )

            for period_type, table in sections.items():
                title = SECTION_LABELS.get(period_type, period_type)
                with st.expander(title, expanded=(period_type == "quarterly")):
                    st.dataframe(table.reset_index(), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
