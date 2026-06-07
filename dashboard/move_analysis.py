"""Simple Plotly views for big-move analysis."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_move_pattern_summary(summary: pd.DataFrame) -> go.Figure:
    """Bar chart: how much each indicator differed on big-move days."""
    fig = go.Figure()
    if summary.empty:
        fig.update_layout(title="Not enough big moves to compare indicators")
        return fig

    # Map pattern text back to a sortable score for the chart.
    score_map = {"Higher on big days": 1.0, "Lower on big days": -1.0, "About the same": 0.0}
    work = summary.copy()
    work["_score"] = work["pattern"].map(score_map).fillna(0.0)
    work = work[work["pattern"] != "About the same"].sort_values("_score", key=abs, ascending=True)

    if work.empty:
        fig.update_layout(title="Indicators looked similar on big days and normal days")
        return fig

    colors = ["#66bb6a" if s > 0 else "#ef5350" for s in work["_score"]]

    fig.add_trace(
        go.Bar(
            x=work["_score"],
            y=work["indicator"],
            orientation="h",
            marker_color=colors,
            text=work["pattern"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="What looked different on big move days?",
        template="plotly_dark",
        height=max(280, 36 * len(work) + 100),
        margin=dict(l=160, r=40, t=60, b=40),
        xaxis=dict(
            title="",
            tickvals=[-1, 0, 1],
            ticktext=["Lower", "", "Higher"],
            range=[-1.3, 1.3],
        ),
        yaxis_title="",
    )
    fig.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.35)")
    return fig
