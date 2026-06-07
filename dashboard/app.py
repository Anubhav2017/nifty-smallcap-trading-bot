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
from dashboard.move_analysis import plot_move_pattern_summary
from dashboard.reports import (
    REPORTS_ROOT,
    list_models,
    list_runs_for_model,
    load_equity_curve,
    load_folds,
    load_picks,
    load_trades,
    read_markdown,
)
from dashboard.timeline import (
    build_timeline_chart,
    fundamental_events,
    merge_timeline_events,
    significant_price_moves,
)
from trading_bot.analysis.move_correlation import build_move_analysis
from trading_bot.data.screener_excel import (
    SECTION_LABELS,
    load_meta,
    load_section_tables,
    load_symbol_fundamentals,
)

CONFIG_PATH = ROOT / "config" / "dashboard.json"
DAY_HISTORY = timedelta(days=3 * 365)
INTERVAL = "day"


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
    st.caption("Daily OHLCV + Screener Excel")

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
            f"**Daily bars:** {summary['day_symbols']} · "
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

        try:
            min_s, max_s = _cached_bounds(symbol, INTERVAL, str(root))
            min_d = pd.Timestamp(min_s).date()
            max_d = pd.Timestamp(max_s).date()
        except FileNotFoundError:
            st.warning(f"No daily OHLCV for {symbol}")
            st.stop()

        default_start: date = max(min_d, max_d - DAY_HISTORY)

        range_preset = st.selectbox(
            "History window",
            options=["3 years", "1 year", "6 months", "3 months", "All available", "Custom"],
            index=0,
            help="Load up to 3 years by default. Use the range slider under the chart to pan.",
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
        st.header("Big moves")
        z_threshold = st.slider(
            "Move sensitivity",
            min_value=1.5,
            max_value=4.0,
            value=2.0,
            step=0.25,
            help="Higher = only the most unusual daily moves get flagged",
        )
        move_window = st.slider(
            "Compare to last N days",
            min_value=20,
            max_value=252,
            value=60,
            step=10,
            help="How many recent days define a 'normal' move",
        )
        min_move_pct = st.slider(
            "Min move size (%)",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.5,
        ) / 100.0

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

    df = _cached_bars(symbol, INTERVAL, str(root), str(start_d), str(end_d))

    tab_chart, tab_timeline, tab_moves, tab_ind, tab_fund, tab_reports = st.tabs(
        ["Charts", "Timeline", "Big moves", "Indicators", "Fundamentals", "Strategy Reports"]
    )

    with tab_chart:
        if df.empty:
            st.info("No bars in this date range.")
        else:
            fig = build_price_chart(
                df,
                price_overlays=price_overlays,
                panels=chart_panels,
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

    with tab_timeline:
        st.subheader("Price & fundamentals timeline")
        st.caption(
            "Daily bars from `ohlcv/day/`. Screener filing dates on the chart; "
            "green/red markers flag unusually large daily moves."
        )

        show_fund = st.checkbox("Fundamental filings", value=True)
        show_moves = st.checkbox("Significant moves", value=True)

        fund_df = _cached_fundamentals(symbol, str(root))
        fund_events = fundamental_events(fund_df) if show_fund else pd.DataFrame()
        if not fund_events.empty:
            fund_events = fund_events[
                (fund_events["date"] >= pd.Timestamp(start_d))
                & (fund_events["date"] <= pd.Timestamp(end_d) + pd.Timedelta(days=1))
            ]

        moves = (
            significant_price_moves(
                df,
                window=move_window,
                z_threshold=z_threshold,
                min_abs_return=min_move_pct,
            )
            if show_moves and not df.empty
            else pd.DataFrame()
        )
        if not moves.empty:
            moves = moves[
                (moves["date"] >= pd.Timestamp(start_d))
                & (moves["date"] <= pd.Timestamp(end_d) + pd.Timedelta(days=1))
            ]

        if df.empty:
            st.warning(
                f"No daily OHLCV for **{symbol}** in this range. "
                "Add `ohlcv/day/{SYMBOL}.csv` or widen the date window."
            )
        else:
            fig_tl = build_timeline_chart(
                df,
                fund_events,
                moves,
                show_fundamentals=show_fund,
                show_moves=show_moves,
            )
            st.plotly_chart(fig_tl, width="stretch", config=PLOTLY_CHART_CONFIG)

            m1, m2, m3 = st.columns(3)
            m1.metric("Fundamental events", len(fund_events))
            m2.metric("Significant moves", len(moves))
            up_n = int((moves["direction"] == "up").sum()) if not moves.empty else 0
            down_n = int((moves["direction"] == "down").sum()) if not moves.empty else 0
            m3.metric("Moves up / down", f"{up_n} / {down_n}")

            st.subheader("Event log (newest first)")
            event_table = merge_timeline_events(fund_events, moves)
            if event_table.empty:
                st.caption("No events in range — widen the date window or add a Screener export.")
            else:
                st.dataframe(event_table, hide_index=True, width="stretch")

    with tab_moves:
        st.subheader("Big daily moves")
        st.caption(
            "Lists unusually large up/down days, then compares a few indicators "
            "on those days vs normal days. Same detection rules as the Timeline tab."
        )

        if df.empty:
            st.warning("No daily OHLCV in range.")
        else:
            fund_df_moves = _cached_fundamentals(symbol, str(root))
            moves_for_analysis = significant_price_moves(
                df,
                window=move_window,
                z_threshold=z_threshold,
                min_abs_return=min_move_pct,
            )
            if not moves_for_analysis.empty:
                moves_for_analysis = moves_for_analysis[
                    (moves_for_analysis["date"] >= pd.Timestamp(start_d))
                    & (moves_for_analysis["date"] <= pd.Timestamp(end_d) + pd.Timedelta(days=1))
                ]

            analysis = build_move_analysis(df, fund_df_moves, moves_for_analysis)
            move_list = analysis["simple_moves"]
            summary = analysis["simple_summary"]

            st.metric("Big move days found", len(moves_for_analysis))

            if moves_for_analysis.empty:
                st.info(
                    "No big moves in this window. Try a longer date range or lower "
                    "**Move sensitivity** in the sidebar."
                )
            else:
                st.subheader("Move list")
                st.dataframe(
                    move_list.rename(
                        columns={
                            "date": "Date",
                            "move": "Move",
                            "rsi": "RSI",
                            "volume": "Volume",
                            "near_filing": "Results this week?",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )

                st.subheader("What stood out")
                st.plotly_chart(
                    plot_move_pattern_summary(summary),
                    width="stretch",
                    config=PLOTLY_CHART_CONFIG,
                )
                st.dataframe(
                    summary.rename(
                        columns={
                            "indicator": "Indicator",
                            "on_big_days": "On big days (avg)",
                            "usually": "Usually (avg)",
                            "pattern": "Pattern",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Compares averages on big-move days vs all other days. "
                    "This shows association, not proof of cause."
                )

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


    with tab_reports:
        import plotly.express as px
        import plotly.graph_objects as go

        st.subheader("Strategy Reports")
        st.caption(f"Scanning `{REPORTS_ROOT}`")

        models = list_models(REPORTS_ROOT)
        if not models:
            st.info(
                "No reports found yet. Run a backtest first:\n\n"
                "```\npython scripts/backtest_move_predictor.py\n```"
            )
            st.stop()

        # ── Model selector ────────────────────────────────────────────────
        selected_model = st.selectbox("Model", models, index=0)
        runs = list_runs_for_model(selected_model, REPORTS_ROOT)
        if not runs:
            st.warning(f"No completed runs found under **{selected_model}**.")
            st.stop()

        # ── Run selector ─────────────────────────────────────────────────
        run_labels = []
        for r in runs:
            m = r["metrics"]
            label = r["run_id"]
            if r["run_id"] == "initial_run":
                label = "initial_run (legacy)"
            elif len(r["run_id"]) == 15 and r["run_id"][8] == "_":
                # YYYYMMDD_HHMMSS → pretty
                dt_str = r["run_id"]
                label = f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}  {dt_str[9:11]}:{dt_str[11:13]}:{dt_str[13:15]}"
            wr_str = f"  WR {m.get('win_rate', 0):.0%}" if m.get("win_rate") is not None else ""
            eq_str = f"  ₹{m.get('final_equity', 0):,.0f}" if m.get("final_equity") else ""
            run_labels.append(f"{label}{wr_str}{eq_str}")

        col_run, col_compare = st.columns([3, 2])
        with col_run:
            selected_run_label = st.selectbox("Run", run_labels, index=0)
        run_idx = run_labels.index(selected_run_label)
        run = runs[run_idx]
        folder: Path = run["path"]
        m: dict = run["metrics"]

        # ── Equity-curve comparison (all runs) ───────────────────────────
        with col_compare:
            compare_mode = st.checkbox("Overlay all runs", value=len(runs) > 1)

        if compare_mode and len(runs) > 1:
            st.markdown("### All Runs — Equity Curves")
            fig_cmp = go.Figure()
            palette = px.colors.qualitative.Plotly
            for i, r_ in enumerate(runs):
                eq_df_ = load_equity_curve(r_["path"])
                if eq_df_.empty:
                    continue
                date_col = "date" if "date" in eq_df_.columns else eq_df_.columns[0]
                val_col = "equity" if "equity" in eq_df_.columns else eq_df_.columns[-1]
                rid = r_["run_id"]
                run_label_short = (
                    f"{rid[:4]}-{rid[4:6]}-{rid[6:8]} {rid[9:11]}:{rid[11:13]}"
                    if len(rid) == 15 and rid[8] == "_"
                    else rid
                )
                is_selected = i == run_idx
                fig_cmp.add_trace(go.Scatter(
                    x=eq_df_[date_col], y=eq_df_[val_col],
                    mode="lines",
                    name=run_label_short,
                    line=dict(
                        color=palette[i % len(palette)],
                        width=3 if is_selected else 1.5,
                        dash="solid" if is_selected else "dot",
                    ),
                ))
            fig_cmp.add_hline(y=1_000_000, line_dash="dash",
                              line_color="gray", annotation_text="Start ₹10L")
            fig_cmp.update_layout(
                xaxis_title="Date", yaxis_title="Equity (₹)",
                height=340, margin=dict(l=0, r=0, t=10, b=0),
                hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Run-level summary table
            summary_rows = []
            for r_ in runs:
                m_ = r_["metrics"]
                summary_rows.append({
                    "Run": r_["run_id"],
                    "Trades": m_.get("total_trades", "—"),
                    "Win rate": f"{m_.get('win_rate', 0):.1%}" if m_.get("win_rate") is not None else "—",
                    "Sortino": f"{m_.get('sortino', 0):.3f}" if m_.get("sortino") is not None else "—",
                    "Max DD": f"{m_.get('max_drawdown', 0):.1%}" if m_.get("max_drawdown") is not None else "—",
                    "Final equity": f"₹{m_.get('final_equity', 0):,.0f}" if m_.get("final_equity") else "—",
                })
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
            st.divider()

        # ── Single run detail ────────────────────────────────────────────
        run_id_display = m.get("run_id", run["run_id"])
        st.markdown(f"### Run: `{run_id_display}` — {selected_model}")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trades", m.get("total_trades", "—"))
        wr = m.get("win_rate")
        c2.metric("Win rate", f"{wr:.1%}" if wr is not None else "—")
        dd = m.get("max_drawdown")
        c3.metric("Max drawdown", f"{dd:.1%}" if dd is not None else "—")
        sr = m.get("sortino")
        c4.metric("Sortino", f"{sr:.3f}" if sr is not None else "—")
        eq = m.get("final_equity")
        c5.metric(
            "Final equity",
            f"₹{eq:,.0f}" if eq is not None else "—",
            delta=f"₹{eq - 1_000_000:+,.0f}" if eq is not None else None,
        )

        with st.expander("Full metrics JSON", expanded=False):
            st.json(m)

        # ── Equity curve (single run) ────────────────────────────────────
        if run["has_equity"] and not compare_mode:
            st.markdown("### Equity Curve")
            eq_df = load_equity_curve(folder)
            if not eq_df.empty:
                date_col = "date" if "date" in eq_df.columns else eq_df.columns[0]
                val_col = "equity" if "equity" in eq_df.columns else eq_df.columns[-1]
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=eq_df[date_col], y=eq_df[val_col],
                    mode="lines", name="Equity",
                    line=dict(color="#2196f3", width=2),
                    fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
                ))
                fig_eq.add_hline(y=1_000_000, line_dash="dash",
                                 line_color="gray", annotation_text="Start ₹10L")
                fig_eq.update_layout(
                    xaxis_title="Date", yaxis_title="Equity (₹)",
                    height=320, margin=dict(l=0, r=0, t=10, b=0),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_eq, use_container_width=True)

        # ── Walk-forward folds ───────────────────────────────────────────
        if run["has_folds"]:
            st.markdown("### Walk-forward Folds")
            folds_df = load_folds(folder)
            if not folds_df.empty:
                st.dataframe(folds_df, hide_index=True, use_container_width=True)

        # ── Trades table ─────────────────────────────────────────────────
        if run["has_trades"]:
            st.markdown("### Trades")
            trades_df = load_trades(folder)
            if not trades_df.empty:
                status_filter = st.multiselect(
                    "Filter by exit type",
                    options=["closed_tp", "closed_sl", "closed_time"],
                    default=["closed_tp", "closed_sl", "closed_time"],
                    key=f"status_filter_{run_id_display}",
                )
                filtered = trades_df[trades_df["status"].isin(status_filter)] if status_filter else trades_df

                def _color_pnl(val: float) -> str:
                    try:
                        return "color: green" if float(val) > 0 else "color: red"
                    except (ValueError, TypeError):
                        return ""

                st.dataframe(
                    filtered.style.map(_color_pnl, subset=["net_pnl"]),
                    hide_index=True,
                    use_container_width=True,
                )
                sym_pnl = (
                    trades_df.groupby("symbol")["net_pnl"]
                    .sum()
                    .sort_values()
                    .reset_index()
                )
                if not sym_pnl.empty:
                    fig_sym = px.bar(
                        sym_pnl, x="symbol", y="net_pnl",
                        color="net_pnl",
                        color_continuous_scale=["#ef5350", "#e0e0e0", "#66bb6a"],
                        color_continuous_midpoint=0,
                        labels={"net_pnl": "Net P&L (₹)", "symbol": "Symbol"},
                        title="P&L by Symbol",
                        height=300,
                    )
                    fig_sym.update_layout(
                        showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_sym, use_container_width=True)

        # ── Daily picks ──────────────────────────────────────────────────
        if run["has_picks"]:
            with st.expander("Daily picks log", expanded=False):
                picks_df = load_picks(folder)
                if not picks_df.empty:
                    st.dataframe(picks_df, hide_index=True, use_container_width=True)

        # ── Narrative trade report ────────────────────────────────────────
        if run["has_trade_report"]:
            st.markdown("### Trade Report (buy/sell narratives)")
            md = read_markdown(folder / "trade_report.md")
            sections = md.split("\n### ")
            if len(sections) > 1:
                header = sections[0]
                st.markdown(header)
                search = st.text_input(
                    "Search trades (symbol or date)",
                    placeholder="e.g. WELCORP or 2025-04",
                    key=f"trade_search_{run_id_display}",
                )
                entries = sections[1:]
                if search:
                    entries = [e for e in entries if search.upper() in e.upper()]
                st.caption(f"Showing {len(entries)} trade(s)")
                for entry in entries:
                    symbol_line = entry.split("  —  ")[0] if "  —  " in entry else entry[:30]
                    with st.expander(symbol_line.strip(), expanded=False):
                        st.markdown("### " + entry)
            else:
                st.markdown(md)

        # ── Sell plan markdown ────────────────────────────────────────────
        sell_plan_path = folder / "sell_plan.md"
        if sell_plan_path.is_file():
            with st.expander("Sell plan / exit rules", expanded=False):
                st.markdown(read_markdown(sell_plan_path))


if __name__ == "__main__":
    main()
