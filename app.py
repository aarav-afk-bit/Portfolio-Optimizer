"""Streamlit front end: ``streamlit run app.py``."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from portfolio_optimizer import analytics as an, charts, ui
from portfolio_optimizer.config import (DEFAULT_BENCHMARK, DEFAULT_PORTFOLIO, DEFAULT_RISK_FREE_RATE,
                                        STRATEGIES, asset_class_for, label_for)
from portfolio_optimizer.data import DataError, clean_prices, demo_prices, fetch_prices, fetch_prices_nse_github
from portfolio_optimizer.estimators import RETURN_MODELS, RISK_MODELS, View
from portfolio_optimizer.formatting import fmt_compact, fmt_pct
from portfolio_optimizer.orders import build_orders, order_ticket, orders_frame
from portfolio_optimizer.report import PortfolioReport, Settings, analyze

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide",
                   initial_sidebar_state="auto")
st.html(ui.CSS)

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
LIVE, DEMO = "Live (Yahoo Finance)", "Demo (offline)"
GITHUB = "NSE data via GitHub"


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
            if e.missing == [benchmark]:
                data = fetch_prices(list(tickers), start)
                return data, None, LIVE, f"Benchmark {benchmark} unavailable; comparisons skip it."
            if set(e.missing) != set(wanted):
                raise
        except Exception:  # network errors, rate limits
            pass
        # Yahoo is unreachable: real NSE prices from GitHub, if every holding is on NSE.
        try:
            data = fetch_prices_nse_github(wanted, start)
            return (data[list(tickers)], (data[benchmark] if benchmark else None), GITHUB,
                    f"Yahoo Finance was unreachable, so prices come from NSE's daily reports via GitHub "
                    f"(eod2_data, updated weekly; last close {data.index[-1]:%d %b %Y}).")
        except Exception:
            warning = ("Couldn't download prices from Yahoo Finance or GitHub (offline or rate-limited?). "
                       "Showing synthetic demo data instead. Numbers below are illustrative only.")
    data = clean_prices(demo_prices(wanted, start=start))
    return data[list(tickers)], (data[benchmark] if benchmark else None), DEMO, warning


@st.cache_data(show_spinner=False)
def run(holdings_items: tuple, benchmark: str, start: str, source: str, settings: Settings):
    holdings = {t: {"qty": q, "avg_price": a, "class": c} for t, q, a, c in holdings_items}
    prices, bench, used, warning = load_prices(tuple(holdings), benchmark, start, source)
    return analyze(holdings, prices, settings, bench), used, warning


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

HOLDING_COLS = ["Ticker", "Qty", "Avg price", "Class"]

if "holdings_df" not in st.session_state:
    st.session_state.holdings_df = pd.DataFrame(
        [{"Ticker": t, "Qty": v["qty"], "Avg price": v["avg_price"], "Class": asset_class_for(t)}
         for t, v in DEFAULT_PORTFOLIO.items()]
    )
    st.session_state.editor_version = 0


def load_holdings_csv(file) -> pd.DataFrame:
    """Accepts Ticker, Qty/Quantity, Avg price; Class is optional."""
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {"symbol": "Ticker", "ticker": "Ticker", "qty": "Qty", "quantity": "Qty", "avg price": "Avg price",
              "avg_price": "Avg price", "average price": "Avg price", "class": "Class", "asset class": "Class"}
    df = df.rename(columns={c: rename[c] for c in df.columns if c in rename})
    missing = {"Ticker", "Qty", "Avg price"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing column(s): {', '.join(sorted(missing))}")
    if "Class" not in df:
        df["Class"] = ""
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Class"] = [c if isinstance(c, str) and c.strip() else asset_class_for(t) for t, c in zip(df["Ticker"], df["Class"])]
    return df[HOLDING_COLS]


with st.sidebar:
    st.markdown("### 📈 Portfolio Optimizer")
    st.caption("Enter your holdings, pick a strategy, and get a rebalancing plan with the reasons behind it.")

    with st.popover("Import / export holdings", icon=":material/upload_file:", width="stretch"):
        up = st.file_uploader("Upload a CSV with Ticker, Qty, Avg price (Class optional)", type="csv")
        if up is not None and st.session_state.get("uploaded_id") != up.file_id:
            try:
                st.session_state.holdings_df = load_holdings_csv(up)
                st.session_state.uploaded_id = up.file_id
                st.session_state.editor_version += 1
                st.rerun()
            except Exception as e:  # malformed file
                st.error(f"Couldn't read that file: {e}")
        st.download_button("Download current holdings", st.session_state.holdings_df.to_csv(index=False),
                           "holdings.csv", "text/csv", icon=":material/download:", width="stretch")

    with st.form("inputs", border=False):
        st.markdown("**Holdings**")
        edited = st.data_editor(
            st.session_state.holdings_df, num_rows="dynamic", hide_index=True, width="stretch",
            key=f"holdings_editor_{st.session_state.editor_version}",
            column_config={
                "Ticker": st.column_config.TextColumn(required=True, help="Yahoo symbol, e.g. HAL.NS", width=112),
                "Qty": st.column_config.NumberColumn(min_value=0, step=1, format="%d", width=52),
                "Avg price": st.column_config.NumberColumn(min_value=0.0, format="%.2f", width=74),
                "Class": st.column_config.TextColumn(width=104, help="Asset class for class caps, e.g. Indian equity. "
                                                          "Leave blank to auto-fill."),
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

        with st.expander("Limits & robustness", icon=":material/tune:"):
            class_cap = st.slider("Max per asset class", 20, 100, 100, step=5, format="%d%%",
                                  help="Caps the total weight of each asset class (see the Class column). "
                                       "100% = no cap.")
            max_to = st.slider("Max turnover per rebalance", 5, 100, 100, step=5, format="%d%%",
                               help="Limits how much of the portfolio is bought (and sold) in one go. "
                                    "100% = no limit.")
            band = st.slider("No-trade band", 0.0, 10.0, 2.0, step=0.5, format="%.1f pp",
                             help="Skip trades that would move a holding's weight by less than this.")
            robust = st.toggle("Robust (resampled) weights", True,
                               help="Re-optimise on many bootstrapped price histories and average the results. "
                                    "Gives steadier targets and a confidence range for every weight.")
            n_res = st.select_slider("Resamples", [50, 100, 200], 100, disabled=not robust)

        with st.expander("Advanced settings", icon=":material/settings:"):
            rf = st.number_input("Risk-free rate (% a year)", 0.0, 15.0, DEFAULT_RISK_FREE_RATE * 100, 0.25,
                                 help="Sharpe ratios measure return above this. Default ≈ Indian 91-day T-bill.")
            min_w = st.slider("Min position size", 0, 20, 0, step=1, format="%d%%")
            start = st.date_input("History from", dt.date(2022, 1, 1), max_value=dt.date.today())
            ret_method = st.selectbox(
                "Expected returns", list(RETURN_MODELS), format_func=RETURN_MODELS.get,
                help="Black–Litterman starts from the return each holding's risk deserves and lets your price "
                     "history nudge it, so one hot streak can't dominate. Bayes–Stein shrinks raw averages "
                     "toward each other. Raw averages chase whatever rallied most.")
            prior_sharpe = st.slider("Long-run Sharpe for the prior", 0.1, 1.0, 0.4, 0.05,
                                     disabled=ret_method != "black_litterman",
                                     help="Return per unit of risk you expect over the long run. 0.3–0.5 is "
                                          "typical for diversified equity.")
            st.caption("Your return views (Black–Litterman): e.g. HAL.NS, 15% a year, 60% confident.")
            views_df = st.data_editor(
                pd.DataFrame({"Ticker": pd.Series(dtype=str), "Return %": pd.Series(dtype=float),
                              "Confidence %": pd.Series(dtype=float)}),
                num_rows="dynamic", hide_index=True, width="stretch", key="views_editor",
                disabled=ret_method != "black_litterman",
                column_config={"Return %": st.column_config.NumberColumn(min_value=-50.0, max_value=100.0,
                                                                         format="%.1f"),
                               "Confidence %": st.column_config.NumberColumn(min_value=1.0, max_value=99.0,
                                                                             format="%.0f")})
            risk_model = st.selectbox("Risk model", list(RISK_MODELS), format_func=RISK_MODELS.get,
                                      help="EWMA weights recent volatility more, so the plan reacts faster "
                                           "when markets get calmer or rougher.")
            benchmark = st.text_input("Benchmark", DEFAULT_BENCHMARK).strip()
            run_bt = st.toggle("Walk-forward backtest", True)
            reb = st.selectbox("Backtest rebalancing", [21, 63, 126], index=1,
                               format_func={21: "Monthly", 63: "Quarterly", 126: "Half-yearly"}.get)
            cost = st.number_input("Trading cost (bps per trade)", 0.0, 100.0, 10.0, 5.0)

        st.form_submit_button("Analyse portfolio", type="primary", width="stretch")

def parse_views(df: pd.DataFrame) -> tuple[View, ...]:
    """Rows with a ticker and a return become views; missing confidence defaults to 50%."""
    out = []
    for _, row in df.dropna(subset=["Ticker", "Return %"]).iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        if ticker:
            conf = row["Confidence %"] if pd.notna(row["Confidence %"]) else 50.0
            out.append(View(ticker, float(row["Return %"]) / 100, float(conf) / 100))
    return tuple(out)


# Validate holdings
h = edited.copy()
h["Ticker"] = h["Ticker"].astype(str).str.strip().str.upper()
h = h[h["Ticker"].ne("") & h["Ticker"].ne("NONE") & h["Ticker"].ne("NAN")]
h[["Qty", "Avg price"]] = h[["Qty", "Avg price"]].fillna(0)
if "Class" not in h:
    h["Class"] = ""
h["Class"] = [c.strip() if isinstance(c, str) and c.strip() else asset_class_for(t)
              for t, c in zip(h["Ticker"], h["Class"])]
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
    class_cap=None if class_cap >= 100 else class_cap / 100,
    max_turnover=None if max_to >= 100 else max_to / 100,
    no_trade_band=band / 100, robust=robust, n_resamples=n_res,
    risk_model=risk_model, prior_sharpe=prior_sharpe,
    views=parse_views(views_df),
)
items = tuple((t, float(q), float(a), c) for t, q, a, c in h[HOLDING_COLS].to_numpy())

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
if settings.class_cap:
    pills.append((f"Max {class_cap}% per class", ""))
if settings.max_turnover:
    pills.append((f"Turnover ≤ {max_to}%", ""))
if robust:
    pills.append((f"Robust · {len(r.resample.samples)} resamples", ""))
pills.append(("Demo data — not real prices", "warn") if used_source == DEMO else
             ("NSE prices via GitHub", "") if used_source == GITHUB else ("Live prices", ""))
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
            "Spread money across groups of holdings by how risky each group is." if strategy == "hrp" else
            f"Rebalance to cut the average worst-day loss from {fmt_pct(r.comparison.loc['Your current mix', 'CVaR 95% (daily)'], 2)}"
            f" to {fmt_pct(an.historical_cvar(tgt.weights.to_numpy(), r.returns), 2)}." if strategy == "min_cvar" else
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
                ":material/show_chart: Efficient frontier", ":material/compare_arrows: Compare strategies",
                ":material/science: What-if", ":material/candlestick_chart: Technicals",
                ":material/history: Backtest", ":material/info: Method"])
TAB_ACTION, TAB_ALLOC, TAB_RISK, TAB_FRONTIER, TAB_COMPARE, TAB_WHATIF, TAB_TECH, TAB_BT, TAB_METHOD = tabs

with TAB_ACTION:
    trades = pd.DataFrame([{
        "Ticker": x.ticker, "Action": x.action, "Units": x.trade_qty,
        "Trade value": fmt_compact(x.trade_value, sym, signed=True) if x.trade_qty else "–",
        "Weight now": x.current_weight, "Target weight": x.target_weight,
        "Timing": x.timing, "Confidence": x.confidence or "–", "Technicals": round(x.technical.score),
    } for x in recs])
    st.html('<div class="section-title">Trade list</div>'
            '<div class="section-sub">Quantities are rounded down to whole units at the latest price.</div>')
    st.dataframe(
        trades, hide_index=True, width="stretch",
        column_config={
            "Units": st.column_config.NumberColumn(format="%+,d", help="Whole units to buy (+) or sell (−)"),
            "Weight now": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1, width="small"),
            "Target weight": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1,
                                                             width="small"),
            "Technicals": st.column_config.NumberColumn(help="Score from -100 (strongly bearish) to +100 "
                                                             "(strongly bullish)", format="%+d"),
        },
    )
    cash = r.cash
    net = cash["net"]
    net_text = (f"you need to add <b>{fmt_compact(net, sym)}</b>" if net > 0 else
                f"<b>{fmt_compact(-net, sym)}</b> is left over as cash" if net < 0 else "the trades fund each other")
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        st.html(f'<div class="cashline"><span class="chip">Sales raise <b>{fmt_compact(cash["sells"], sym)}</b></span>'
                f'<span class="chip">Purchases cost <b>{fmt_compact(cash["buys"], sym)}</b></span>'
                f'<span class="cashnote">so {net_text}.</span></div>')
    with c2:
        csv = trades.assign(**{"Trade value": [round(x.trade_value) for x in recs],
                               "Price": [round(x.price, 2) for x in recs], "Holding now": [x.qty for x in recs]})
        st.download_button("Trade list CSV", csv.to_csv(index=False), "rebalance_plan.csv",
                           "text/csv", icon=":material/download:", width="stretch")

    if any(x.trade_qty for x in recs):
        st.html('<div class="section-title" style="margin-top:18px">Order sheet</div>'
                '<div class="section-sub">The orders to place, in order. Sells come first because they pay for the '
                'buys. Holdings whose timing says “Stagger” or “Wait for dip” are split into weekly tranches, and '
                'buys are paced alongside staggered sales so you never need much cash up front.</div>')
        spread = st.toggle("Spread staggered trades over 3 weeks", True, key="spread_orders",
                           help="Turn off to place every order now.")
        orders = build_orders(recs, tranches=3 if spread else 1)
        sheet = orders_frame(orders)
        st.dataframe(
            sheet.drop(columns=["Approx. price", "Asset"]), hide_index=True, width="stretch",
            column_config={
                "Date": st.column_config.DateColumn(format="ddd D MMM"),
                "Units": st.column_config.NumberColumn(format="%,d"),
                "Approx. value": st.column_config.NumberColumn(format=f"{sym}%,d"),
                "Cash after": st.column_config.NumberColumn(
                    format=f"{sym}%,d", help="Running cash from the sales so far, minus the buys so far. Negative "
                                             "means new money is needed at that point."),
                "Note": st.column_config.TextColumn(width="large"),
            },
        )
        d1, d2, _ = st.columns([1, 1, 2])
        with d1:
            st.download_button("Order sheet CSV", sheet.to_csv(index=False), "orders.csv", "text/csv",
                               icon=":material/download:", width="stretch")
        with d2:
            st.download_button("Printable order sheet", order_ticket(orders, sym), "orders.txt", "text/plain",
                               icon=":material/print:", width="stretch")

    if strategy not in ("risk_parity", "hrp"):
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

with TAB_ALLOC:
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.html('<div class="section-title">Current vs target weights</div>'
                '<div class="section-sub">Share of portfolio value in each holding.'
                + (' Whiskers show the 10–90% range across resampled histories: long whiskers mean the data '
                   'only loosely pins that weight.' if r.resample is not None and intensity == 100 else '')
                + '</div>')
        st.plotly_chart(charts.allocation_bars(r), config=PLOTLY_CONFIG, theme=None)
    with c2:
        st.html('<div class="section-title">By asset class</div>'
                '<div class="section-sub">How the rebalance changes your exposure.'
                + (f' Each class is capped at {class_cap}%.' if settings.class_cap else '')
                + ' Edit classes in the Class column of your holdings.</div>')
        cls = pd.DataFrame([{"Asset class": x.asset_class, "Now": x.current_weight, "Target": x.target_weight}
                            for x in recs])
        cls = cls.groupby("Asset class", as_index=False).sum().sort_values("Target", ascending=False)
        st.dataframe(cls, hide_index=True, width="stretch", column_config={
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

with TAB_RISK:
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
        width="stretch",
        column_config={c: st.column_config.NumberColumn(format="percent")
                       for c in ["CAGR", "Volatility", "Max drawdown", "VaR 95%", "CVaR 95%"]}
        | {c: st.column_config.NumberColumn(format="%.2f") for c in ["Sharpe", "Sortino", "Calmar", "Beta"]},
    )

with TAB_FRONTIER:
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
    st.dataframe(comp, width="stretch", column_config={
        "Expected return": st.column_config.NumberColumn(format="percent"),
        "Volatility": st.column_config.NumberColumn(format="percent"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
    })

with TAB_COMPARE:
    st.html('<div class="section-title">Every strategy under your limits</div>'
            '<div class="section-sub">Single-estimate optimum of each strategy with the same position, asset-class '
            'and turnover limits, next to your current mix. Use it to pick the strategy that matches what you care '
            'about: return, smoothness, or protection on bad days.</div>')
    comp = r.comparison
    chosen = STRATEGIES[strategy]
    c1, c2 = st.columns([1.9, 1], gap="large")
    with c1:
        st.plotly_chart(charts.strategy_bars(r), config=PLOTLY_CONFIG, theme=None)
    with c2:
        best = {"Sharpe": comp["Sharpe"].idxmax(), "Volatility": comp["Volatility"].idxmin(),
                "CVaR 95% (daily)": comp["CVaR 95% (daily)"].idxmin(), "Max drawdown": comp["Max drawdown"].idxmax()}
        st.html('<div class="card"><h3>Who wins what</h3><div class="lede">Best value of each metric '
                '(expected or in-sample).</div>' + ui.reason_list([
                    ui.Reason("positive", "Sharpe", f"{best['Sharpe']} ({comp.loc[best['Sharpe'], 'Sharpe']:.2f})"),
                    ui.Reason("positive", "Volatility",
                              f"{best['Volatility']} ({fmt_pct(comp.loc[best['Volatility'], 'Volatility'])})"),
                    ui.Reason("positive", "Tail risk", f"{best['CVaR 95% (daily)']} "
                              f"({fmt_pct(comp.loc[best['CVaR 95% (daily)'], 'CVaR 95% (daily)'], 2)} a day)"),
                    ui.Reason("positive", "Drawdown", f"{best['Max drawdown']} "
                              f"({fmt_pct(comp.loc[best['Max drawdown'], 'Max drawdown'])})"),
                    ui.Reason("neutral", "Selected", f"{chosen}" + (" with resampling, so its final weights differ "
                              "slightly from the single-estimate row." if robust else ".")),
                ]) + "</div>")
    st.dataframe(
        comp.rename(index={chosen: f"{chosen}  ← selected"}), width="stretch",
        column_config={c: st.column_config.NumberColumn(format="percent")
                       for c in ["Expected return", "Volatility", "CVaR 95% (daily)", "Max drawdown", "Turnover"]}
        | {"Sharpe": st.column_config.NumberColumn(format="%.2f"),
           "Effective holdings": st.column_config.NumberColumn(format="%.1f")},
    )
    st.html('<div class="section-title" style="margin-top:12px">Weights by strategy</div>')
    wts = r.comparison_weights.rename(columns=label_for, index={chosen: f"{chosen}  ← selected"})
    st.dataframe(wts, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="percent") for c in wts.columns})


@st.fragment
def whatif_tab():
    st.html('<div class="section-title">Try your own mix</div>'
            '<div class="section-sub">Type the weights you have in mind. They are scaled to add up to 100% and '
            'scored with the same risk model as the plan. Only this tab reruns while you edit.</div>')
    base = pd.DataFrame({"Asset": [label_for(t) for t in r.tickers],
                         "Your mix %": (cur.weights * 100).round(1).to_numpy(),
                         "Plan target %": (tgt.weights * 100).round(1).to_numpy()}, index=r.tickers)
    c1, c2 = st.columns([1.15, 1.4], gap="large")
    with c1:
        mix = st.data_editor(base, hide_index=True, width="stretch", disabled=["Asset", "Plan target %"],
                             key="whatif_editor", column_config={
                                 "Asset": st.column_config.TextColumn(width="medium"),
                                 "Your mix %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0,
                                                                             step=0.5, format="%.1f", width="small"),
                                 "Plan target %": st.column_config.NumberColumn(format="%.1f", width="small")})
        raw = mix["Your mix %"].fillna(0).clip(lower=0).to_numpy()
        if raw.sum() <= 0:
            st.warning("Give at least one holding a positive weight.")
            return
        w = pd.Series(raw / raw.sum(), index=r.tickers)
        if abs(raw.sum() - 100) > 0.05:
            st.caption(f"Your weights add up to {raw.sum():.1f}%, so they were scaled to 100%.")
    x = w.to_numpy()

    def metrics(v):
        daily = r.returns @ v
        return {"Expected return": an.portfolio_return(v, r.mu), "Volatility": an.portfolio_volatility(v, r.cov_annual),
                "Sharpe": an.sharpe_ratio(v, r.mu, r.cov_annual, r.settings.risk_free_rate),
                "CVaR 95% (daily)": an.historical_cvar(v, r.returns),
                "Max drawdown": float(an.drawdown((1 + daily).cumprod()).min()),
                "Effective holdings": an.effective_holdings(v)}

    m_cur, m_tgt, m_you = metrics(cur.weights.to_numpy()), metrics(tgt.weights.to_numpy()), metrics(x)
    with c2:
        st.html('<div class="kpis" style="grid-template-columns:repeat(3,minmax(0,1fr))">' + "".join([
            ui.kpi("Expected return", fmt_pct(m_you["Expected return"]),
                   f"plan {fmt_pct(m_tgt['Expected return'])} · now {fmt_pct(m_cur['Expected return'])}"),
            ui.kpi("Volatility", fmt_pct(m_you["Volatility"]),
                   f"plan {fmt_pct(m_tgt['Volatility'])} · now {fmt_pct(m_cur['Volatility'])}"),
            ui.kpi("Sharpe ratio", f"{m_you['Sharpe']:.2f}",
                   f"plan {m_tgt['Sharpe']:.2f} · now {m_cur['Sharpe']:.2f}"),
        ]) + "</div>")
        verdicts = []
        gap = m_you["Sharpe"] - m_tgt["Sharpe"]
        verdicts.append(ui.Reason("positive" if gap >= -0.02 else "negative", "Sharpe",
                                  "Your mix matches or beats the plan's Sharpe ratio." if gap >= -0.02 else
                                  f"The plan earns {-gap:.2f} more Sharpe for the same risk budget."))
        d_cvar = m_you["CVaR 95% (daily)"] - m_cur["CVaR 95% (daily)"]
        verdicts.append(ui.Reason("positive" if d_cvar < -1e-4 else "negative" if d_cvar > 1e-4 else "neutral",
                                  "Tail risk", f"Average loss on the worst 5% of days: "
                                  f"{fmt_pct(m_you['CVaR 95% (daily)'], 2)} (now {fmt_pct(m_cur['CVaR 95% (daily)'], 2)})."))
        breaches = [label_for(t) for t, v in w.items() if v > r.settings.max_weight + 1e-6]
        if breaches:
            verdicts.append(ui.Reason("negative", "Limits", f"Above your {fmt_pct(r.settings.max_weight, 0)} "
                                      f"position cap: {', '.join(breaches)}."))
        if r.settings.class_cap:
            totals = w.groupby(pd.Series(r.asset_classes)).sum()
            over = [f"{c} {fmt_pct(v, 0)}" for c, v in totals.items() if v > r.settings.class_cap + 1e-6]
            if over:
                verdicts.append(ui.Reason("negative", "Limits", f"Over the {fmt_pct(r.settings.class_cap, 0)} "
                                          f"asset-class cap: {', '.join(over)}."))
        st.html(f'<div class="card">{ui.reason_list(verdicts)}</div>')
    st.plotly_chart(charts.frontier_chart(r, extra={"Your mix": w}), config=PLOTLY_CONFIG, theme=None,
                    key="whatif_frontier")


with TAB_WHATIF:
    whatif_tab()


@st.fragment
def technicals_tab():
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
    } for t, s in r.technicals.items()]), hide_index=True, width="stretch",
        column_config={"Price": st.column_config.NumberColumn(format="%.2f"),
                       "Score": st.column_config.NumberColumn(format="%+d")})
    st.caption("Technicals decide *timing*, not the target weights. The optimiser ignores them; they only "
               "suggest whether to trade now or stagger.")


with TAB_TECH:
    technicals_tab()

with TAB_BT:
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
            width="stretch",
            column_config={c: st.column_config.NumberColumn(format="percent")
                           for c in ["CAGR", "Volatility", "Max drawdown", "VaR 95%", "CVaR 95%"]}
            | {c: st.column_config.NumberColumn(format="%.2f") for c in ["Sharpe", "Sortino", "Calmar", "Beta"]},
        )
        st.html('<div class="section-title" style="margin-top:12px">How the optimiser\'s weights changed over time</div>'
                '<div class="section-sub">Stable weights mean the signal is robust. Wild swings mean it is chasing noise.</div>')
        st.plotly_chart(charts.backtest_weights_chart(r), config=PLOTLY_CONFIG, theme=None)

with TAB_METHOD:
    st.markdown(f"""
#### How the plan is built
1. **Prices.** Daily adjusted closes from Yahoo Finance since {start:%b %Y}, aligned across holdings. If Yahoo is
   unreachable, NSE holdings use real prices from NSE's daily reports via GitHub (eod2_data).
2. **Risk model.** Daily covariance with **Ledoit–Wolf shrinkage** of the correlations (or **EWMA** volatilities,
   which weight recent days more). Raw sample correlations are noisy, and optimisers amplify that noise.
3. **Expected returns.** By default **Black–Litterman**: each holding starts at the return its risk deserves at
   a long-run Sharpe ratio, and your price history moves it only as far as its precision justifies. Add your
   own views in Advanced settings. Bayes–Stein shrinkage and raw averages are also available.
4. **Optimisation.** Long-only and fully invested, within your limits. Max Sharpe is solved as a convex problem,
   so the answer is the global optimum (checked against cvxpy on real data). Hierarchical risk parity clusters
   holdings by correlation instead. Sharpe ratios are measured against a **{rf:.2f}% risk-free rate**.
5. **Limits.** Position caps, asset-class caps and a turnover limit are enforced in every strategy,
   the frontier and the backtest. Impossible combinations are caught up front with the exact
   minimum you'd need (e.g. "your limits require trading at least 30%").
6. **Robust weights.** With resampling on, the target averages 100+ optimisations on block-bootstrapped
   price histories (Michaud). Each holding's 10–90% range drives its confidence label.
7. **Tail risk.** The minimum-CVaR strategy solves the Rockafellar–Uryasev linear program on historical
   daily returns: the lowest possible average loss on the worst 5% of days.
8. **Reasons.** The headline reason for each trade comes from an exact **attribution** of the objective's
   change: we integrate its gradient along the path from your current to your target weights. Each
   holding's share sums to the total, so the "why" always matches the "what".
9. **Technicals.** RSI (Wilder), MACD, Bollinger bands, 50/200-day averages and 6-month momentum feed a
   −100…+100 score. It only drives *timing* advice, never the weights.
10. **Backtest.** Walk-forward with trailing 12-month estimates, your limits, trading costs, and
    comparison to your current mix, equal weight and the benchmark. Each rebalance uses a single
    estimate (no resampling) to keep it fast.

#### Caveats
- Expected returns are estimates from the past. They are the least reliable input, even after shrinkage.
- Volatility and correlation assume the future resembles the sample period. Regimes change.
- Taxes, exit loads and lot sizes are not modelled. Check capital-gains impact before selling.
- **Educational tool, not investment advice.**
""")
    if r.notes:
        st.markdown("#### Estimation diagnostics\n" + "\n".join(f"- {n}" for n in r.notes))

st.html('<div class="footnote">Built with Python · SciPy · Plotly · Streamlit. Not investment advice.</div>')
