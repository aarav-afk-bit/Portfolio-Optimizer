"""Plotly figures for the report. Shared by the Streamlit app and notebooks.

Palette: validated categorical slots (fixed order, never cycled), one blue
ramp for magnitude, blue↔red with a grey midpoint for polarity.
"""

from __future__ import annotations

import contextvars
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from . import analytics as an
from .config import STRATEGIES, label_for
from .optimizer import efficient_frontier, random_portfolios
from .report import PortfolioReport

# Categorical slots, in order (validated: adjacent CVD ΔE ≥ 9, first three all-pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#d03b3b"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
POS, NEG = "#2a78d6", "#e34948"

INK = "#0f172a"
INK_2 = "#475569"
MUTED = "#94a3b8"
GRID = "#eef0f3"
AXIS = "#cbd5e1"
SURFACE = "#ffffff"
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"

# Dark mode: every light colour above maps to its dark counterpart. Series use the validated dark
# steps of the same hues (all >= 3:1 on the dark surface); the blue magnitude ramp is reversed so low
# values recede into the dark surface, as they recede into white in light mode.
DARK_MAP = {
    "#ffffff": "#161b26", "#0f172a": "#e6edf3", "#475569": "#c3c9d4", "#94a3b8": "#8b95a7",
    "#eef0f3": "#242b38", "#cbd5e1": "#3a4354", "#c3cad5": "#4a5468", "#f0efec": "#383835",
    "#2a78d6": "#3987e5", "#eb6834": "#d95926", "#1baf7a": "#199e70", "#eda100": "#c98500",
    "#e87ba4": "#d55181", "#008300": "#008300", "#4a3aa7": "#9085e9", "#e34948": "#e66767",
    "#d03b3b": "#e66767",
    **dict(zip(BLUE_RAMP, BLUE_RAMP[::-1])),
    "rgba(255,255,255,0.9)": "rgba(22,27,38,0.9)",
}
_COLOR = re.compile(r"#[0-9a-fA-F]{6}|rgba\(255,255,255,0\.9\)")
_DARK = contextvars.ContextVar("portfolio_charts_dark", default=False)


def use_dark(on: bool) -> None:
    """Draw charts for a dark background for the rest of this script run (per session/thread)."""
    _DARK.set(bool(on))


def to_dark(fig: go.Figure) -> go.Figure:
    """Recolour a finished light-theme figure for a dark background in one pass."""
    text = _COLOR.sub(lambda m: DARK_MAP.get(m.group(0).lower(), m.group(0)), fig.to_json())
    return pio.from_json(text)


# Fixed roles for portfolio-level series so the same thing is the same colour on every chart.
ROLE = {
    "Target": SERIES[0],
    "Optimised (walk-forward)": SERIES[0],
    "Current": SERIES[1],
    "Your current mix": SERIES[1],
    "Equal weight": SERIES[2],
    "Benchmark": SERIES[3],
}


def _base(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=16, t=56 if legend else 16, b=12),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_2),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, color=INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=INK_2), bgcolor="rgba(0,0,0,0)", traceorder="normal"),
        showlegend=legend,
        bargap=0.28,
        bargroupgap=0.08,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2, size=12), automargin=True, title_standoff=8)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED),
                     title_font=dict(color=INK_2, size=12), automargin=True, title_standoff=8)
    return to_dark(fig) if _DARK.get() else fig


def _names(tickers) -> list[str]:
    return [label_for(t) for t in tickers]


def _short(tickers) -> list[str]:
    return [t.split(".")[0].lstrip("^") for t in tickers]


def _hbar_room(fig: go.Figure, max_value: float) -> None:
    """Leave space to the right of horizontal bars for their outside labels."""
    fig.update_xaxes(range=[0, max(max_value, 1e-6) * 1.18])


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------

def allocation_bars(r: PortfolioReport) -> go.Figure:
    order = r.target.weights.sort_values().index
    names = _names(order)
    fig = go.Figure()
    # Resampled 10–90% ranges apply to the full move; skip them when rebalancing partially.
    show_range = r.resample is not None and r.settings.rebalance_intensity >= 1
    for name, w, color in (("Current", r.current.weights, ROLE["Current"]),
                           ("Target", r.target.weights, ROLE["Target"])):
        vals = w[order].to_numpy()
        error = None
        hover = "%{y}<br>" + name + ": %{x:.1%}<extra></extra>"
        if name == "Target" and show_range:
            lo, hi = r.resample.low[order].to_numpy(), r.resample.high[order].to_numpy()
            error = dict(type="data", array=np.clip(hi - vals, 0, None), arrayminus=np.clip(vals - lo, 0, None),
                         color=INK_2, thickness=1.2, width=4)
            hover = ("%{y}<br>Target: %{x:.1%}<br>Likely range: %{customdata[0]:.0%}–%{customdata[1]:.0%}"
                     "<extra></extra>")
        label_here = not (name == "Target" and show_range)
        fig.add_bar(y=names, x=vals, name=name, orientation="h", marker=dict(color=color, cornerradius=4),
                    text=[f"{v:.0%}" if v >= 0.01 and label_here else "" for v in vals], textposition="outside",
                    textfont=dict(color=INK_2, size=11), cliponaxis=False, error_x=error,
                    customdata=np.c_[r.resample.low[order], r.resample.high[order]] if show_range else None,
                    hovertemplate=hover)
    fig.update_layout(barmode="group", legend_traceorder="reversed")
    fig.update_xaxes(tickformat=".0%", showgrid=True)
    if show_range:
        # Target bars are drawn second, so they sit in the upper half of each group.
        for nm, v, hi in zip(names, r.target.weights[order], r.resample.high[order]):
            if v >= 0.01:
                fig.add_annotation(x=max(v, hi), y=nm, text=f"{v:.0%}", showarrow=False, xanchor="left", xshift=6,
                                   yshift=9, font=dict(color=INK_2, size=11))
    top = max(r.current.weights.max(), r.target.weights.max(), r.resample.high.max() if show_range else 0)
    _hbar_room(fig, top)
    fig.update_yaxes(showgrid=False)
    return _base(fig, height=max(280, 64 * len(order) + 80))


def contribution_bars(r: PortfolioReport) -> go.Figure:
    """Where the objective's improvement comes from (sums to the total)."""
    recs = sorted(r.recommendations, key=lambda x: x.contribution)
    vol = r.settings.strategy in ("min_volatility", "min_cvar")
    vals = [x.contribution * (100 if vol else 1) for x in recs]
    good = [(v < 0) if vol else (v > 0) for v in vals]
    fmt = (lambda v: f"{v:+.2f} pp") if vol else (lambda v: f"{v:+.3f}")
    fig = go.Figure(go.Bar(
        y=[x.label for x in recs], x=vals, orientation="h",
        marker=dict(color=[POS if g else NEG for g in good], cornerradius=4),
        text=[fmt(v) for v in vals], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK_2, size=11),
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    lo, hi = min(min(vals), 0), max(max(vals), 0)
    pad = (hi - lo) * 0.18 or 0.01
    fig.update_xaxes(zeroline=True, zerolinecolor=AXIS, zerolinewidth=1,
                     range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)])
    fig.update_yaxes(showgrid=False)
    return _base(fig, height=max(220, 52 * len(recs) + 40), legend=False)


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------

def risk_vs_capital(r: PortfolioReport, which: str = "current") -> go.Figure:
    w = r.current.weights if which == "current" else r.target.weights
    rc = r.risk_share_current if which == "current" else r.risk_share_target
    order = rc.sort_values().index
    names = _names(order)
    fig = go.Figure()
    fig.add_bar(y=names, x=w[order], name="Share of capital", orientation="h",
                marker=dict(color=SERIES[0], cornerradius=4),
                hovertemplate="%{y}<br>Capital: %{x:.1%}<extra></extra>")
    fig.add_bar(y=names, x=rc[order], name="Share of risk", orientation="h",
                marker=dict(color=SERIES[1], cornerradius=4),
                text=[f"{v:.0%}" for v in rc[order]], textposition="outside", cliponaxis=False,
                textfont=dict(color=INK_2, size=11),
                hovertemplate="%{y}<br>Risk: %{x:.1%}<extra></extra>")
    fig.update_layout(barmode="group", legend_traceorder="reversed")
    fig.update_xaxes(tickformat=".0%")
    _hbar_room(fig, max(w.max(), rc.max()))
    fig.update_yaxes(showgrid=False)
    return _base(fig, height=max(280, 64 * len(order) + 80))


def correlation_heatmap(r: PortfolioReport) -> go.Figure:
    c = r.correlation
    names = _names(c.index)
    short = _short(c.index)
    z = c.to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=short, y=names, zmin=-1, zmax=1, colorscale=DIVERGING, xgap=2, ygap=2,
        customdata=[[names[j] for j in range(len(names))] for _ in names],
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=MUTED), len=0.9),
        hovertemplate="%{y} × %{customdata}: %{z:.2f}<extra></extra>",
    ))
    for i in range(len(names)):
        for j in range(len(names)):
            fig.add_annotation(x=short[j], y=names[i], text=f"{z[i, j]:.2f}", showarrow=False,
                               font=dict(size=12, color="#ffffff" if abs(z[i, j]) > 0.55 else INK))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom", tickangle=-35)
    return _base(fig, height=max(320, 64 * len(names) + 80), legend=False)


def drawdown_chart(r: PortfolioReport) -> go.Figure:
    daily = pd.DataFrame({
        "Current": r.returns @ r.current.weights.to_numpy(),
        "Target": r.returns @ r.target.weights.to_numpy(),
    })
    fig = go.Figure()
    for name in daily:
        dd = an.drawdown((1 + daily[name]).cumprod())
        fig.add_scatter(x=dd.index, y=dd, name=name, mode="lines",
                        line=dict(color=ROLE[name], width=2),
                        hovertemplate="%{x|%d %b %Y}<br>" + name + ": %{y:.1%}<extra></extra>")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(hovermode="x unified")
    return _base(fig, height=320)


# --------------------------------------------------------------------------
# Efficient frontier
# --------------------------------------------------------------------------

def frontier_chart(r: PortfolioReport, extra: dict[str, pd.Series] | None = None) -> go.Figure:
    """``extra`` adds labelled points (e.g. a what-if mix) on top of the frontier."""
    s = r.settings
    cloud = r.cloud if r.cloud is not None else random_portfolios(r.mu, r.cov_annual, s.risk_free_rate)
    front = r.frontier if r.frontier is not None else efficient_frontier(r.mu, r.cov_annual, s.min_weight,
                                                                          s.max_weight)

    fig = go.Figure()
    fig.add_scatter(
        x=cloud["volatility"], y=cloud["return"], mode="markers", name="Random portfolios",
        marker=dict(size=5, color=cloud["sharpe"], colorscale=BLUE_RAMP, opacity=0.55,
                    colorbar=dict(title=dict(text="Sharpe", font=dict(color=INK_2, size=11)),
                                  thickness=10, outlinewidth=0, tickfont=dict(color=MUTED), len=0.7)),
        hovertemplate="Vol %{x:.1%} · Return %{y:.1%}<extra>Random mix</extra>",
    )
    if not front.empty:
        fig.add_scatter(x=front["volatility"], y=front["return"], mode="lines", name="Efficient frontier",
                        line=dict(color=INK, width=2),
                        hovertemplate="Vol %{x:.1%} · Return %{y:.1%}<extra>Frontier</extra>")
    # Capital market line from the risk-free rate through the max-Sharpe point.
    if s.strategy == "max_sharpe" and r.optimal.volatility > 0:
        x_end = max(cloud["volatility"].max(), r.optimal.volatility) * 1.05
        slope = (r.optimal.expected_return - s.risk_free_rate) / r.optimal.volatility
        fig.add_scatter(x=[0, x_end], y=[s.risk_free_rate, s.risk_free_rate + slope * x_end], mode="lines",
                        name="Capital market line", line=dict(color=MUTED, width=1),
                        hoverinfo="skip")

    vols = np.sqrt(np.diag(r.cov_annual))
    fig.add_scatter(x=vols, y=r.mu, mode="markers", name="Individual assets", customdata=_names(r.tickers),
                    marker=dict(size=9, color="#ffffff", line=dict(color=INK_2, width=1.5)),
                    hovertemplate="%{customdata}<br>Vol %{x:.1%} · Return %{y:.1%}<extra></extra>")
    # Labels on a white backing so they stay legible over the point cloud; alternate
    # above/below by return rank so neighbouring labels don't collide.
    rank = np.argsort(np.argsort(r.mu.to_numpy()))
    for name, x, y, k in zip(_short(r.tickers), vols, r.mu, rank):
        fig.add_annotation(x=x, y=y, text=name, showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=MUTED,
                           ax=0, ay=-24 if k % 2 else 24, bgcolor="rgba(255,255,255,0.9)", borderpad=2,
                           font=dict(size=11, color=INK_2))
    for name, res, sym in (("Current", r.current, "circle"), ("Target", r.target, "diamond")):
        fig.add_scatter(x=[res.volatility], y=[res.expected_return], mode="markers", name=name,
                        marker=dict(size=16, color=ROLE[name], symbol=sym, line=dict(color="#ffffff", width=2)),
                        hovertemplate=f"{name}<br>Vol %{{x:.1%}} · Return %{{y:.1%}}<br>Sharpe {res.sharpe:.2f}"
                                      "<extra></extra>")
    for name, w in (extra or {}).items():
        x = w.reindex(r.mu.index).fillna(0).to_numpy()
        v, ret = an.portfolio_volatility(x, r.cov_annual), an.portfolio_return(x, r.mu)
        fig.add_scatter(x=[v], y=[ret], mode="markers", name=name,
                        marker=dict(size=15, color=SERIES[2], symbol="square", line=dict(color="#ffffff", width=2)),
                        hovertemplate=f"{name}<br>Vol %{{x:.1%}} · Return %{{y:.1%}}<extra></extra>")
    x_max = max(cloud["volatility"].max(), vols.max(), r.current.volatility) * 1.1
    y_all = np.concatenate([cloud["return"], r.mu, [r.current.expected_return]])
    y_pad = (y_all.max() - y_all.min()) * 0.12
    fig.update_xaxes(title="Expected volatility (annual)", tickformat=".0%", range=[0, x_max])
    fig.update_yaxes(title="Expected return (annual)", tickformat=".0%",
                     range=[min(y_all.min(), s.risk_free_rate) - y_pad, y_all.max() + y_pad])
    return _base(fig, height=540)


# --------------------------------------------------------------------------
# Technicals
# --------------------------------------------------------------------------

def technical_chart(r: PortfolioReport, ticker: str, days: int = 260) -> go.Figure:
    from .indicators import compute_indicators

    ind = compute_indicators(r.prices[ticker]).iloc[-days:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.58, 0.21, 0.21],
                        subplot_titles=("", "RSI (14)", "MACD histogram"))
    x = ind.index
    fig.add_scatter(x=x, y=ind["BB_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=x, y=ind["BB_lower"], line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(148,163,184,0.16)", name="Bollinger band", hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=x, y=ind["price"], name="Price", line=dict(color=SERIES[0], width=2),
                    hovertemplate="%{x|%d %b %Y}<br>Price %{y:,.2f}<extra></extra>", row=1, col=1)
    fig.add_scatter(x=x, y=ind["SMA_50"], name="50-day avg", line=dict(color=SERIES[1], width=1.5),
                    hovertemplate="50-DMA %{y:,.2f}<extra></extra>", row=1, col=1)
    fig.add_scatter(x=x, y=ind["SMA_200"], name="200-day avg", line=dict(color=SERIES[2], width=1.5),
                    hovertemplate="200-DMA %{y:,.2f}<extra></extra>", row=1, col=1)

    fig.add_scatter(x=x, y=ind["RSI"], name="RSI", showlegend=False, line=dict(color=SERIES[0], width=1.5),
                    hovertemplate="RSI %{y:.0f}<extra></extra>", row=2, col=1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(color=AXIS, width=1), row=2, col=1)
    fig.add_annotation(x=x[0], y=70, text="overbought", showarrow=False, yshift=8, xanchor="left",
                       font=dict(size=10, color=MUTED), row=2, col=1)
    fig.add_annotation(x=x[0], y=30, text="oversold", showarrow=False, yshift=-8, xanchor="left",
                       font=dict(size=10, color=MUTED), row=2, col=1)

    hist = ind["MACD_hist"]
    fig.add_bar(x=x, y=hist, name="MACD hist", showlegend=False,
                marker=dict(color=np.where(hist >= 0, POS, NEG)),
                hovertemplate="MACD hist %{y:.2f}<extra></extra>", row=3, col=1)

    fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=2, col=1)
    fig.update_layout(hovermode="x unified", bargap=0.1)
    for a in fig.layout.annotations[:2]:
        a.update(font=dict(size=12, color=INK_2), x=0, xanchor="left")
    return _base(fig, height=620)


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------

def backtest_chart(r: PortfolioReport) -> go.Figure:
    g = r.backtest.growth
    fig = go.Figure()
    for name in g.columns:
        fig.add_scatter(x=g.index, y=g[name], name=name, mode="lines",
                        line=dict(color=ROLE.get(name, MUTED), width=2.5 if name.startswith("Optimised") else 1.75),
                        hovertemplate="%{x|%d %b %Y}<br>" + name + ": %{y:.2f}×<extra></extra>")
        fig.add_annotation(x=g.index[-1], y=g[name].iloc[-1], text=f"{g[name].iloc[-1]:.2f}×",
                           showarrow=False, xanchor="left", xshift=6, font=dict(size=11, color=INK_2))
    fig.add_hline(y=1, line=dict(color=AXIS, width=1))
    fig.update_yaxes(title="Growth of 1", tickformat=".2f")
    fig.update_layout(hovermode="x unified")
    fig = _base(fig, height=420)
    fig.update_layout(margin=dict(r=56))
    return fig


def backtest_weights_chart(r: PortfolioReport) -> go.Figure:
    h = r.backtest.weights_history
    fig = go.Figure()
    for i, t in enumerate(h.columns):
        color = SERIES[i] if i < len(SERIES) else MUTED
        fig.add_bar(x=h.index, y=h[t], name=label_for(t), marker=dict(color=color, line=dict(color=SURFACE, width=1)),
                    hovertemplate="%{x|%b %Y}<br>" + label_for(t) + ": %{y:.0%}<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.15)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return _base(fig, height=340)


def strategy_bars(r: PortfolioReport) -> go.Figure:
    """Small multiples: Sharpe, volatility and tail risk per strategy.

    The selected strategy is blue, your current mix orange, the rest grey, so
    the comparison reads at a glance without a colour per strategy.
    """
    c = r.comparison
    chosen = STRATEGIES[r.settings.strategy]
    names = list(c.index)[::-1]
    colors = [ROLE["Current"] if n == "Your current mix" else SERIES[0] if n == chosen else "#c3cad5" for n in names]
    panels = [("Sharpe", "Sharpe ratio ↑", lambda v: f"{v:.2f}"),
              ("Volatility", "Volatility ↓", lambda v: f"{v:.1%}"),
              ("CVaR 95% (daily)", "Worst-5%-day loss ↓", lambda v: f"{v:.2%}")]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=[p[1] for p in panels])
    for k, (col, _, fmt) in enumerate(panels, start=1):
        vals = c.loc[names, col].to_numpy()
        fig.add_bar(y=names, x=vals, orientation="h", marker=dict(color=colors, cornerradius=4), showlegend=False,
                    customdata=[fmt(v) for v in vals], hovertemplate="%{y}: %{customdata}<extra></extra>",
                    row=1, col=k)
        # Labels sit right of the bar end (or of zero, for negatives) so they never hit the axis labels.
        for nm, v in zip(names, vals):
            fig.add_annotation(x=max(v, 0), y=nm, text=fmt(v), showarrow=False, xanchor="left", xshift=4,
                               font=dict(color=INK_2, size=11), row=1, col=k)
        lo, hi = min(vals.min(), 0), max(vals.max(), 0)
        pad = (hi - lo) * 0.3
        fig.update_xaxes(range=[lo - (pad if lo < 0 else 0), hi + pad], showticklabels=False,
                         zeroline=True, zerolinecolor=AXIS, row=1, col=k)
    for a in fig.layout.annotations[:3]:  # the subplot titles
        a.update(font=dict(size=12, color=INK_2))
    fig.update_yaxes(showgrid=False)
    fig = _base(fig, height=max(260, 40 * len(names) + 80), legend=False)
    fig.update_layout(margin=dict(t=40))
    return fig
