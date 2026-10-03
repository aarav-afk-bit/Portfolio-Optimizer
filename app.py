"""Streamlit front end: ``streamlit run app.py``."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from portfolio_optimizer import analytics as an, charts, ui
from portfolio_optimizer.config import (DEFAULT_BENCHMARK, DEFAULT_PORTFOLIO, DEFAULT_RISK_FREE_RATE,
                                        STRATEGIES, label_for)
from portfolio_optimizer.data import DataError, clean_prices, demo_prices, fetch_prices
from portfolio_optimizer.formatting import fmt_compact, fmt_pct
from portfolio_optimizer.report import PortfolioReport, Settings, analyze

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide",
                   initial_sidebar_state="auto")
st.html(ui.CSS)

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
LIVE, DEMO = "Live (Yahoo Finance)", "Demo (offline)"


# --------------------------------------------------------------------------
# Data & analysis (cached)
# --------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers: tuple[str, ...], benchmark: str, start: str, source: str):
    """Returns (prices, benchmark series or None, actual source, warning)."""
    wanted = list(tickers) + ([benchmark] if benchmark else [])
    warning = None
    if source == LIVE:
        try:
            data = fetch_prices(wanted, start)
            return data[list(tickers)], (data[benchmark] if benchmark else None), LIVE, None
        except DataError as e:
            if set(e.missing) == set(wanted):
                warning = ("Couldn't download any prices from Yahoo Finance (offline or rate-limited?). "
                           "Showing synthetic demo data instead. Numbers below are illustrative only.")
            elif e.missing == [benchmark]:
                data = fetch_prices(list(tickers), start)
                return data, None, LIVE, f"Benchmark {benchmark} unavailable; comparisons skip it."
            else:
                raise
        except Exception as e:  # network errors, rate limits
            warning = (f"Live prices unavailable ({type(e).__name__}). Showing synthetic demo data instead. "
                       "Numbers below are illustrative only.")
    data = clean_prices(demo_prices(wanted, start=start))
    return data[list(tickers)], (data[benchmark] if benchmark else None), DEMO, warning


@st.cache_data(show_spinner=False)
def run(holdings_items: tuple, benchmark: str, start: str, source: str, settings: Settings):
    holdings = {t: {"qty": q, "avg_price": a} for t, q, a in holdings_items}
    prices, bench, used, warning = load_prices(tuple(holdings), benchmark, start, source)
    return analyze(holdings, prices, settings, bench), used, warning


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

if "holdings_df" not in st.session_state:
    st.session_state.holdings_df = pd.DataFrame(
        [{"Ticker": t, "Qty": v["qty"], "Avg price": v["avg_price"]} for t, v in DEFAULT_PORTFOLIO.items()]
    )

with st.sidebar:
    st.markdown("### 📈 Portfolio Optimizer")
    st.caption("Enter your holdings, pick a strategy, and get a rebalancing plan with the reasons behind it.")

    with st.form("inputs", border=False):
        st.markdown("**Holdings**")
        edited = st.data_editor(
            st.session_state.holdings_df, num_rows="dynamic", hide_index=True, use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn(required=True, help="Yahoo symbol, e.g. HAL.NS"),
                "Qty": st.column_config.NumberColumn(min_value=0, step=1, format="%d", width="small"),
                "Avg price": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
            },
        )
        st.caption("Use Yahoo symbols (`.NS` for NSE, `.BO` for BSE). Add a ticker with quantity 0 "
                   "to test whether it's worth buying.")

        st.markdown("**Strategy**")
        strategy = st.selectbox("Optimise for", list(STRATEGIES), format_func=STRATEGIES.get,
                                help="Max Sharpe: best return per unit of risk. Min volatility: smoothest ride. "
                                     "Min tail risk: smallest average loss on the worst 5% of days. "
                                     "Risk parity: every holding contributes equal risk. Max diversification: "
                                     "holdings that offset each other.")
        max_w = st.slider("Max position size", 10, 100, 40, step=5, format="%d%%",
                          help="Caps any single holding. Prevents the optimiser going all-in on one asset.")
        intensity = st.slider("Rebalance intensity", 10, 100, 100, step=10, format="%d%%",
                              help="100% moves fully to the optimal weights; 50% goes halfway.")
        source = st.radio("Price data", [LIVE, DEMO], horizontal=True)

        with st.expander("Advanced settings"):
            rf = st.number_input("Risk-free rate (% a year)", 0.0, 15.0, DEFAULT_RISK_FREE_RATE * 100, 0.25,
                                 help="Sharpe ratios measure return above this. Default ≈ Indian 91-day T-bill.")
            min_w = st.slider("Min position size", 0, 20, 0, step=1, format="%d%%")
            start = st.date_input("History from", dt.date(2022, 1, 1), max_value=dt.date.today())
            ret_method = st.selectbox(
                "Expected returns", ["bayes_stein", "historical"],
                format_func={"bayes_stein": "Bayes–Stein shrinkage (robust)",
                             "historical": "Historical average (raw)"}.get,
                help="Raw historical averages make the optimiser chase whatever rallied most. "
                     "Shrinkage pulls extreme estimates toward the group average.")
            benchmark = st.text_input("Benchmark", DEFAULT_BENCHMARK).strip()
            run_bt = st.toggle("Walk-forward backtest", True)
            reb = st.selectbox("Backtest rebalancing", [21, 63, 126], index=1,
                               format_func={21: "Monthly", 63: "Quarterly", 126: "Half-yearly"}.get)
            cost = st.number_input("Trading cost (bps per trade)", 0.0, 100.0, 10.0, 5.0)

        st.form_submit_button("Analyse portfolio", type="primary", use_container_width=True)

# Validate holdings
h = edited.copy()
h["Ticker"] = h["Ticker"].astype(str).str.strip().str.upper()
h = h[h["Ticker"].ne("") & h["Ticker"].ne("NONE") & h["Ticker"].ne("NAN")]
h[["Qty", "Avg price"]] = h[["Qty", "Avg price"]].fillna(0)
if h.empty:
    st.info("Add at least one holding in the sidebar to get started.")
    st.stop()
dupes = h["Ticker"][h["Ticker"].duplicated()].unique()
if len(dupes):
    st.error(f"Duplicate tickers: {', '.join(dupes)}. Combine them into one row.")
    st.stop()
st.session_state.holdings_df = edited

settings = Settings(
    strategy=strategy, risk_free_rate=rf / 100, min_weight=min_w / 100, max_weight=max_w / 100,
    return_method=ret_method, rebalance_intensity=intensity / 100, run_backtest=run_bt,
    backtest_rebalance_every=reb, transaction_cost_bps=cost,
)
items = tuple((t, float(q), float(a)) for t, q, a in h[["Ticker", "Qty", "Avg price"]].to_numpy())

try:
    with st.spinner("Fetching prices and optimising…"):
        report, used_source, warning = run(items, benchmark, str(start), source, settings)
except (DataError, ValueError) as e:
    st.error(str(e))
    st.stop()

r: PortfolioReport = report
sym = r.currency
recs = sorted(r.recommendations, key=lambda x: -abs(x.trade_value))

# --------------------------------------------------------------------------
# Header & KPIs
# --------------------------------------------------------------------------

pnl = r.total_value - r.total_invested
pills = [(STRATEGIES[strategy], ""), (f"Prices to {r.as_of:%d %b %Y}", ""),
         (f"{len(r.tickers)} holdings", ""), (f"Max position {max_w}%", "")]
pills.append(("Demo data — not real prices", "warn") if used_source == DEMO else ("Live prices", ""))
n_trades = sum(x.action != "Hold" for x in recs)
st.html(ui.hero(
    "Your rebalancing plan",
    f"{n_trades} of {len(recs)} holdings need a trade. Every recommendation below lists the reasons behind it: "
    "what the optimiser saw, what the risk numbers say, and whether the chart agrees on timing.",
    pills,
))
if warning:
    st.warning(warning, icon="⚠️")

cur, tgt = r.current, r.target
st.html('<div class="kpis">' + "".join([
    ui.kpi("Portfolio value", fmt_compact(r.total_value, sym),
           f'<span class="{"up" if pnl >= 0 else "down"}">{fmt_compact(pnl, sym, signed=True)}</span> '
           f'unrealised ({fmt_pct(pnl / r.total_invested, 1, signed=True) if r.total_invested else "–"})'),
    ui.kpi("Expected return", ui.change(fmt_pct(cur.expected_return), fmt_pct(tgt.expected_return)),
           ui.delta(tgt.expected_return - cur.expected_return, lambda v: f"{v * 100:.1f} pp") + " a year",
           "Annualised, after Bayes–Stein shrinkage of historical returns."),
    ui.kpi("Volatility", ui.change(fmt_pct(cur.volatility), fmt_pct(tgt.volatility)),
           ui.delta(tgt.volatility - cur.volatility, lambda v: f"{v * 100:.1f} pp", good_if_up=False),
           "Annualised standard deviation of returns. Lower is a smoother ride."),
    ui.kpi("Sharpe ratio", ui.change(f"{cur.sharpe:.2f}", f"{tgt.sharpe:.2f}"),
           ui.delta(tgt.sharpe - cur.sharpe, lambda v: f"{v:.2f}"),
           f"Excess return over the {rf:.2f}% risk-free rate per unit of volatility."),
    ui.kpi("Worst drawdown", ui.change(fmt_pct(r.portfolio_stats.loc['Current', 'max_drawdown'], 0),
                                       fmt_pct(r.portfolio_stats.loc['Target', 'max_drawdown'], 0)),
           ui.delta(r.portfolio_stats.loc['Target', 'max_drawdown'] - r.portfolio_stats.loc['Current', 'max_drawdown'],
                    lambda v: f"{v * 100:.1f} pp") + " over the period",
           "Largest peak-to-trough fall the mix would have suffered over the price history."),
]) + "</div>")

# Verdict
actions = "".join(ui.badge(x.action, x.ticker.split(".")[0]) for x in recs if x.action != "Hold")
headline = (f"Shift toward {', '.join(x.label for x in recs if x.action in ('Buy more', 'Initiate'))[:120] or 'your current mix'}"
            f" to lift the Sharpe ratio from {cur.sharpe:.2f} to {tgt.sharpe:.2f}."
            if strategy == "max_sharpe" else
            f"Rebalance to cut volatility from {fmt_pct(cur.volatility)} to {fmt_pct(tgt.volatility)}."
            if strategy == "min_volatility" else
            "Rebalance so each holding pulls its weight on risk." if strategy == "risk_parity" else
            "Rebalance to shrink your losses on the worst days." if strategy == "min_cvar" else
            "Rebalance toward holdings that offset each other.")
if n_trades == 0:
    headline = "Your portfolio is already close to optimal. No trades needed."
st.html(f'<div class="card verdict"><div><div class="hero-eyebrow" style="color:#2a78d6">The verdict</div>'
        f'<div class="verdict-head">{headline}</div><div class="verdict-actions">{actions}</div></div>'
        f'<div>{ui.reason_list(r.summary)}</div></div>')

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

tabs = st.tabs([":material/checklist: Action plan", ":material/donut_large: Allocation", ":material/shield: Risk",
                ":material/show_chart: Efficient frontier", ":material/candlestick_chart: Technicals",
                ":material/history: Backtest", ":material/info: Method"])

with tabs[0]:
    trades = pd.DataFrame([{
        "Ticker": x.ticker, "Action": x.action, "Trade (units)": x.trade_qty,
        "Trade value": fmt_compact(x.trade_value, sym, signed=True) if x.trade_qty else "–",
        "Price": round(x.price, 2), "Holding now": x.qty,
        "Weight now": x.current_weight, "Target weight": x.target_weight,
        "Timing": x.timing, "Technicals": round(x.technical.score),
    } for x in recs])
    st.html('<div class="section-title">Trade list</div>'
            '<div class="section-sub">Quantities are rounded down to whole units at the latest price.</div>')
    st.dataframe(
        trades, hide_index=True, use_container_width=True,
        column_config={
            "Price": st.column_config.NumberColumn(format=f"{sym}%.2f"),
            "Holding now": st.column_config.NumberColumn(format="%,d"),
            "Trade (units)": st.column_config.NumberColumn(format="%+,d"),
            "Weight now": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Target weight": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Technicals": st.column_config.NumberColumn(help="Score from -100 (strongly bearish) to +100 "
                                                             "(strongly bullish)", format="%+d"),
        },
    )
    csv = trades.assign(**{"Trade value": [round(x.trade_value) for x in recs]})
    st.download_button("Download trade list (CSV)", csv.to_csv(index=False), "rebalance_plan.csv",
                       "text/csv", icon=":material/download:")

    if strategy != "risk_parity":
        obj = {"max_sharpe": "Sharpe ratio", "min_volatility": "volatility (pp)", "min_cvar": "daily CVaR (pp)",
               "max_diversification": "diversification ratio"}[strategy]
        st.html(f'<div class="section-title" style="margin-top:18px">Where the improvement comes from</div>'
                f'<div class="section-sub">Each holding\'s exact share of the change in {obj}. '
                f'Blue bars help, red bars cost a little but free up money for better uses. They add up to the total.</div>')
        st.plotly_chart(charts.contribution_bars(r), config=PLOTLY_CONFIG, theme=None)

    st.html('<div class="section-title" style="margin-top:8px">Holding by holding</div>'
            '<div class="section-sub">Sorted by trade size. Expand “Timing &amp; technicals” for chart signals.</div>')
    scale = max(max(x.current_weight for x in recs), max(x.target_weight for x in recs), 0.01)
    cols = st.columns(2, gap="medium")
    for i, rec in enumerate(recs):
        with cols[i % 2]:
            st.html(ui.rec_card(rec, sym, scale))

with tabs[1]:
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.html('<div class="section-title">Current vs target weights</div>'
                '<div class="section-sub">Share of portfolio value in each holding.</div>')
        st.plotly_chart(charts.allocation_bars(r), config=PLOTLY_CONFIG, theme=None)
    with c2:
        st.html('<div class="section-title">By asset class</div>'
                '<div class="section-sub">How the rebalance changes your exposure.</div>')
        cls = pd.DataFrame([{"Asset class": x.asset_class, "Now": x.current_weight, "Target": x.target_weight}
                            for x in recs])
        cls = cls.groupby("Asset class", as_index=False).sum().sort_values("Target", ascending=False)
        st.dataframe(cls, hide_index=True, use_container_width=True, column_config={
            "Now": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1, width="small"),
            "Target": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1, width="small"),
        })
        st.html('<div class="section-title" style="margin-top:12px">Concentration</div>')
        eff_c, eff_t = an.effective_holdings(cur.weights.to_numpy()), an.effective_holdings(tgt.weights.to_numpy())
        dr_c = an.diversification_ratio(cur.weights.to_numpy(), r.cov_annual)
        dr_t = an.diversification_ratio(tgt.weights.to_numpy(), r.cov_annual)
        st.html('<div class="kpis" style="grid-template-columns:1fr 1fr">'
                + ui.kpi("Effective holdings", ui.change(f"{eff_c:.1f}", f"{eff_t:.1f}"),
                         f"out of {len(recs)} · higher = less concentrated",
                         "Inverse Herfindahl index of the weights.")
                + ui.kpi("Diversification ratio", ui.change(f"{dr_c:.2f}", f"{dr_t:.2f}"),
                         "weighted vol ÷ portfolio vol",
                         "How much volatility cancels out because holdings don't move together.")
                + "</div>")

with tabs[2]:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.html('<div class="section-title">Capital vs risk, today</div>'
                '<div class="section-sub">When the orange bar is longer than the blue one, that holding '
                'drives more of your portfolio\'s swings than its size suggests.</div>')
        st.plotly_chart(charts.risk_vs_capital(r, "current"), config=PLOTLY_CONFIG, theme=None, key="rc_cur")
    with c2:
        st.html('<div class="section-title">Capital vs risk, after rebalancing</div>'
                '<div class="section-sub">Risk is spread more evenly across holdings.</div>')
        st.plotly_chart(charts.risk_vs_capital(r, "target"), config=PLOTLY_CONFIG, theme=None, key="rc_tgt")
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.html('<div class="section-title">Correlation</div>'
                '<div class="section-sub">Blue = moves together, red = moves opposite. Low numbers mean '
                'better diversification. Shrunk with Ledoit–Wolf to damp noise.</div>')
        st.plotly_chart(charts.correlation_heatmap(r), config=PLOTLY_CONFIG, theme=None)
    with c2:
        st.html('<div class="section-title">Drawdowns</div>'
                '<div class="section-sub">How far each mix fell from its previous peak over the price history.</div>')
        st.plotly_chart(charts.drawdown_chart(r), config=PLOTLY_CONFIG, theme=None)

    st.html('<div class="section-title">Risk metrics</div>'
            '<div class="section-sub">Realised over the price history. VaR/CVaR are one-day losses at 95% confidence.</div>')
    stats = pd.concat([r.portfolio_stats.rename(index={"Current": "Portfolio · current", "Target": "Portfolio · target",
                                                       "Benchmark": f"Benchmark · {label_for(benchmark)}"}),
                       r.asset_stats[r.portfolio_stats.columns].rename(index=label_for)])
    st.dataframe(
        stats.rename(columns={"cagr": "CAGR", "volatility": "Volatility", "sharpe": "Sharpe", "sortino": "Sortino",
                              "max_drawdown": "Max drawdown", "calmar": "Calmar", "var_95": "VaR 95%",
                              "cvar_95": "CVaR 95%", "beta": "Beta"}),
        use_container_width=True,
        column_config={c: st.column_config.NumberColumn(format="percent")
                       for c in ["CAGR", "Volatility", "Max drawdown", "VaR 95%", "CVaR 95%"]}
        | {c: st.column_config.NumberColumn(format="%.2f") for c in ["Sharpe", "Sortino", "Calmar", "Beta"]},
    )

with tabs[3]:
    st.html('<div class="section-title">Efficient frontier</div>'
            '<div class="section-sub">Each dot is a random mix of your holdings, shaded by Sharpe ratio. The black curve is '
            'the best return available at each level of risk under your position limits. Moving from the orange circle '
            'toward the blue diamond is what the plan does.</div>')
    st.plotly_chart(charts.frontier_chart(r), config=PLOTLY_CONFIG, theme=None)
    comp = pd.DataFrame({
        "Current": [cur.expected_return, cur.volatility, cur.sharpe],
        "Target": [tgt.expected_return, tgt.volatility, tgt.sharpe],
        "Optimal (100%)": [r.optimal.expected_return, r.optimal.volatility, r.optimal.sharpe],
        "Min volatility": [r.min_vol.expected_return, r.min_vol.volatility, r.min_vol.sharpe],
    }, index=["Expected return", "Volatility", "Sharpe"]).T
    st.dataframe(comp, use_container_width=True, column_config={
        "Expected return": st.column_config.NumberColumn(format="percent"),
        "Volatility": st.column_config.NumberColumn(format="percent"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
    })

with tabs[4]:
    pick = st.segmented_control("Holding", r.tickers, default=recs[0].ticker, format_func=label_for,
                                key="tech_pick") or recs[0].ticker
    snap = r.technicals[pick]
    c1, c2 = st.columns([2.2, 1], gap="large")
    with c1:
        st.plotly_chart(charts.technical_chart(r, pick), config=PLOTLY_CONFIG, theme=None)
    with c2:
        tone = "up" if snap.score >= 25 else "down" if snap.score <= -25 else ""
        st.html(f'<div class="card"><h3>{label_for(pick)}</h3>'
                f'<div class="lede">Composite technical score '
                f'<b class="{tone}">{snap.stance} {snap.score:+.0f}</b> / 100</div>'
                + ui.reason_list([ui.Reason({"bullish": "positive", "bearish": "negative"}.get(s.stance, "neutral"),
                                            s.name, f"{s.value}. {s.detail}") for s in snap.signals])
                + f'<div class="chips"><span class="chip">From 52w high <b>{fmt_pct(snap.from_52w_high)}</b></span>'
                  f'<span class="chip">3M <b>{fmt_pct(snap.momentum_3m, 1, True)}</b></span>'
                  f'<span class="chip">6M <b>{fmt_pct(snap.momentum_6m, 1, True)}</b></span></div></div>')
    st.html('<div class="section-title">All holdings</div>')
    st.dataframe(pd.DataFrame([{
        "Asset": label_for(t), "Price": s.price, "RSI": round(s.rsi, 1),
        **{sig.name: sig.value for sig in s.signals}, "Score": round(s.score), "Stance": s.stance,
    } for t, s in r.technicals.items()]), hide_index=True, use_container_width=True,
        column_config={"Price": st.column_config.NumberColumn(format="%.2f"),
                       "Score": st.column_config.NumberColumn(format="%+d")})
    st.caption("Technicals decide *timing*, not the target weights. The optimiser ignores them; they only "
               "suggest whether to trade now or stagger.")

with tabs[5]:
    if r.backtest is None:
        st.info("Backtest is off or there isn't enough history (needs > 1 year). Turn it on under Advanced settings.")
    else:
        bt = r.backtest
        st.html('<div class="section-title">Walk-forward backtest</div>'
                f'<div class="section-sub">At each {dict([(21, "monthly"), (63, "quarterly"), (126, "half-yearly")])[reb]} '
                'rebalance the optimiser only sees the previous 12 months of prices, so there is no hindsight. '
                f'Trading costs of {cost:.0f} bps are deducted. This is the honest test of whether the strategy works.</div>')
        st.plotly_chart(charts.backtest_chart(r), config=PLOTLY_CONFIG, theme=None)
        st.dataframe(
            bt.stats.rename(columns={"cagr": "CAGR", "volatility": "Volatility", "sharpe": "Sharpe",
                                     "sortino": "Sortino", "max_drawdown": "Max drawdown", "calmar": "Calmar",
                                     "var_95": "VaR 95%", "cvar_95": "CVaR 95%", "beta": "Beta"}),
            use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="percent")
                           for c in ["CAGR", "Volatility", "Max drawdown", "VaR 95%", "CVaR 95%"]}
            | {c: st.column_config.NumberColumn(format="%.2f") for c in ["Sharpe", "Sortino", "Calmar", "Beta"]},
        )
        st.html('<div class="section-title" style="margin-top:12px">How the optimiser\'s weights changed over time</div>'
                '<div class="section-sub">Stable weights mean the signal is robust. Wild swings mean it is chasing noise.</div>')
        st.plotly_chart(charts.backtest_weights_chart(r), config=PLOTLY_CONFIG, theme=None)

with tabs[6]:
    st.markdown(f"""
#### How the plan is built
1. **Prices.** Daily adjusted closes from Yahoo Finance since {start:%b %Y}, aligned across holdings.
2. **Risk model.** Daily covariance with **Ledoit–Wolf shrinkage** of the correlations. Raw sample correlations
   from a few years of data are noisy, and optimisers amplify that noise.
3. **Expected returns.** **Bayes–Stein shrinkage** pulls each holding's historical average toward the
   group mean. The original notebook used raw averages, which let one hot asset dominate.
4. **Optimisation.** Long-only, fully invested, with your min/max position limits, solved with SLSQP from
   several starting points. Sharpe ratios are measured against a **{rf:.2f}% risk-free rate**
   (the original used 0%, which flatters every risky asset).
5. **Reasons.** The headline reason for each trade comes from an exact **attribution** of the objective's
   change: we integrate its gradient along the path from your current to your target weights. Each
   holding's share sums to the total, so the "why" always matches the "what".
6. **Technicals.** RSI (Wilder), MACD, Bollinger bands, 50/200-day averages and 6-month momentum feed a
   −100…+100 score. It only drives *timing* advice, never the weights.
7. **Backtest.** Walk-forward with trailing 12-month estimates, trading costs, and comparison to your
   current mix, equal weight and the benchmark.

#### Caveats
- Expected returns are estimates from the past. They are the least reliable input, even after shrinkage.
- Volatility and correlation assume the future resembles the sample period. Regimes change.
- Taxes, exit loads and lot sizes are not modelled. Check capital-gains impact before selling.
- **Educational tool, not investment advice.**
""")
    if r.notes:
        st.markdown("#### Estimation diagnostics\n" + "\n".join(f"- {n}" for n in r.notes))

st.html('<div class="footnote">Built with Python · SciPy · Plotly · Streamlit. Not investment advice.</div>')
