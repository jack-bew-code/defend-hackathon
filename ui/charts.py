"""Plotly charts. One shared look: thin rounded bars, recessive grid, hover on every mark."""
import pandas as pd
import plotly.graph_objects as go

from ui.style import BORDER, FRAUD, GENUINE, MUTED, TEXT

THIRD = "#6C8EF5"  # third identity colour (model comparison only); validated with FRAUD + GENUINE
FONT = dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=TEXT, size=14)


def _layout(fig: go.Figure, title: str = "", height: int = 340) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT), x=0, xanchor="left"),
        height=height, margin=dict(l=10, r=10, t=50 if title else 10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=FONT, showlegend=False, bargap=0.35,
        hoverlabel=dict(bgcolor="#0F1A30", bordercolor=BORDER, font=dict(color=TEXT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=BORDER, gridwidth=1, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def donut(flagged: int, total: int) -> go.Figure:
    genuine = total - flagged
    fig = go.Figure(go.Pie(
        labels=["Flagged as fraud", "Not flagged"], values=[flagged, genuine], hole=0.68,
        marker=dict(colors=[FRAUD, GENUINE], line=dict(color="#0B1426", width=2)),
        texttemplate="<b>%{label}</b><br>%{value:,} · %{percent}", textposition="outside",
        textfont=dict(color=TEXT, size=14), sort=False, direction="clockwise", rotation=0,
        hovertemplate="%{label}: %{value:,} transactions (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{flagged:,}</b><br><span style='color:{MUTED}'>flagged</span>",
                       showarrow=False, font=dict(size=22, color=TEXT))
    _layout(fig, height=360)
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=50))
    return fig


def flag_rate(df: pd.DataFrame, column: str, title: str, labels: dict | None = None,
              order: list | None = None) -> go.Figure:
    """Share of transactions in each group that the model flagged (single series)."""
    g = df.groupby(column, observed=True)["is_flagged"].agg(["sum", "count"])
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    g["rate"] = g["sum"] / g["count"]
    names = [str(labels.get(i, i)) if labels else str(i) for i in g.index]
    fig = go.Figure(go.Bar(
        x=names, y=g["rate"], marker=dict(color=FRAUD, cornerradius=4),
        text=[f"{r:.0%}" for r in g["rate"]], textposition="outside", textfont=dict(color=TEXT),
        customdata=g[["sum", "count"]].to_numpy(),
        hovertemplate="%{x}<br>%{customdata[0]:,} of %{customdata[1]:,} flagged (%{y:.1%})<extra></extra>",
    ))
    top = max(g["rate"].max() * 1.25, 0.05)
    fig.update_yaxes(tickformat=".0%", range=[0, min(top, 1.12)])
    if top > 1:  # room for the "100%" label without a "120%" tick
        fig.update_yaxes(tickvals=[0, 0.25, 0.5, 0.75, 1])
    return _layout(fig, title)


def flagged_by_hour(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("hour_of_day")["is_flagged"].agg(["sum", "count"]).reindex(range(24), fill_value=0)
    fig = go.Figure(go.Bar(
        x=g.index, y=g["sum"], marker=dict(color=FRAUD, cornerradius=4),
        customdata=g["count"],
        hovertemplate="%{x}:00 - %{x}:59<br>%{y} flagged of %{customdata} transactions<extra></extra>",
    ))
    fig.update_xaxes(tickmode="array", tickvals=list(range(0, 24, 3)),
                     ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)])
    return _layout(fig, "Flagged transactions by hour of day")


def signal_share(df: pd.DataFrame, signals: dict) -> go.Figure:
    """For each red-flag signal: % of flagged vs % of not-flagged transactions that have it."""
    flagged, rest = df[df.is_flagged.eq(1)], df[df.is_flagged.eq(0)]
    names = list(signals.values())
    fig = go.Figure()
    for label, part, color in [("Flagged", flagged, FRAUD), ("Not flagged", rest, GENUINE)]:
        shares = [part[c].eq(1).mean() if len(part) else 0 for c in signals]
        fig.add_trace(go.Bar(
            y=names, x=shares, name=label, orientation="h",
            marker=dict(color=color, cornerradius=4, line=dict(color="#0B1426", width=2)),
            text=[f"{s:.0%}" for s in shares], textposition="outside", textfont=dict(color=TEXT),
            hovertemplate=f"{label}: %{{x:.1%}} have %{{y}}<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_xaxes(tickformat=".0%", range=[0, 1.12], showgrid=True, gridcolor=BORDER)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    _layout(fig, "How often each red flag appears", height=60 + 70 * len(signals))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.02, yanchor="bottom",
                                                   x=1, xanchor="right", font=dict(color=TEXT)))
    return fig


def top_reasons(counts: pd.Series, n: int = 8) -> go.Figure:
    top = counts.head(n)[::-1]
    fig = go.Figure(go.Bar(
        y=top.index, x=top.values, orientation="h", marker=dict(color=FRAUD, cornerradius=4),
        text=top.values, textposition="outside", textfont=dict(color=TEXT),
        hovertemplate="%{y}: in %{x} flagged transactions' reasons<extra></extra>",
    ))
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, range=[0, top.max() * 1.18 if len(top) else 1])
    return _layout(fig, "Most common reasons the model gave", height=60 + 42 * len(top))


def overlap(both: int, only_a: int, only_b: int, name_a: str, name_b: str) -> go.Figure:
    labels = [f"Only {name_a}", "Both models", f"Only {name_b}"]
    fig = go.Figure(go.Bar(
        x=labels, y=[only_a, both, only_b], marker=dict(color=[GENUINE, FRAUD, THIRD], cornerradius=4),
        text=[only_a, both, only_b], textposition="outside", textfont=dict(color=TEXT),
        hovertemplate="%{x}: %{y} transactions flagged<extra></extra>",
    ))
    fig.update_yaxes(range=[0, max(both, only_a, only_b, 1) * 1.2])
    return _layout(fig, "Which transactions each model flags")

