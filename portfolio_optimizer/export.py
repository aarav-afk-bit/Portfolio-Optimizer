"""A shareable one-page HTML report.

Self-contained: inline CSS, no scripts, no external requests, so it opens
offline, prints cleanly and can be emailed. Charts are drawn with plain
HTML bars.
"""

from __future__ import annotations

from html import escape

from .config import STRATEGIES
from .formatting import fmt_compact, fmt_money, fmt_pct
from .report import PortfolioReport

_CSS = """
:root { --ink:#0f172a; --ink2:#475569; --muted:#94a3b8; --line:#e5e8ef; --card:#fff; --page:#f5f7fb;
  --blue:#2a78d6; --orange:#eb6834; --good:#15803d; --bad:#b91c1c; --warn:#b45309;
  --good-bg:#e9f7ee; --bad-bg:#fdecec; --warn-bg:#fef4e2; --soft:#f1f5f9; --blue-bg:#e8f1fc; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#f1f5f9; --ink2:#cbd5e1; --muted:#94a3b8; --line:#2c3444; --card:#161b26; --page:#0d1117;
    --blue:#3987e5; --orange:#d95926; --good:#4ade80; --bad:#f87171; --warn:#fbbf24;
    --good-bg:#12321f; --bad-bg:#3a1717; --warn-bg:#3a2a0f; --soft:#1f2633; --blue-bg:#162a45; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font:14px/1.5 Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 28px 16px 48px; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.02em; }
h2 { font-size: 17px; margin: 28px 0 10px; letter-spacing: -0.01em; }
.sub { color: var(--ink2); margin: 0 0 18px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
.kpi, .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }
.kpi .l { font-size: 12px; color: var(--ink2); }
.kpi .v { font-size: 20px; font-weight: 700; margin-top: 2px; }
.kpi .v span { color: var(--muted); font-weight: 600; font-size: 15px; }
ul.r { list-style: none; padding: 0; margin: 0; }
ul.r li { display: grid; grid-template-columns: 20px 1fr; gap: 8px; padding: 6px 0; border-top: 1px solid var(--line); }
ul.r li:first-child { border-top: none; }
.i { font-weight: 700; text-align: center; border-radius: 5px; font-size: 12px; height: 20px; line-height: 20px; }
.positive { background: var(--good-bg); color: var(--good); } .negative { background: var(--bad-bg); color: var(--bad); }
.neutral { background: var(--soft); color: var(--ink2); }
table { width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--line);
  border-radius: 12px; overflow: hidden; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--line); font-size: 13px; }
th { color: var(--ink2); font-weight: 600; background: var(--soft); }
td.n, th.n { text-align: right; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.b-buy { background: var(--good-bg); color: var(--good); } .b-trim { background: var(--warn-bg); color: var(--warn); }
.b-exit { background: var(--bad-bg); color: var(--bad); } .b-hold { background: var(--soft); color: var(--ink2); }
.bars { display: grid; grid-template-columns: 170px 1fr; gap: 6px 12px; align-items: center; }
.track { position: relative; height: 22px; }
.bar { height: 9px; border-radius: 0 4px 4px 0; margin: 1px 0; }
.bar.now { background: var(--orange); } .bar.tgt { background: var(--blue); }
.legend { font-size: 12px; color: var(--ink2); margin: 0 0 8px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 12px; }
.holding { margin-bottom: 10px; }
.holding h3 { font-size: 15px; margin: 0; display: flex; justify-content: space-between; gap: 10px; }
.holding .hl { color: var(--ink2); margin: 2px 0 8px; }
.tldr { background: var(--blue-bg); border-radius: 8px; padding: 8px 10px; margin-bottom: 8px; }
ol.steps { margin: 0; padding-left: 20px; } ol.steps li { margin: 4px 0; }
.foot { color: var(--muted); font-size: 12px; margin-top: 28px; }
@media print { body { background: #fff; } .kpi, .card, table { break-inside: avoid; } main { padding: 0; } }
"""

_BADGE = {"Buy more": "b-buy", "Initiate": "b-buy", "Trim": "b-trim", "Exit": "b-exit", "Hold": "b-hold"}
_ICON = {"positive": "✓", "negative": "✕", "neutral": "•"}


def _reasons(items) -> str:
    return '<ul class="r">' + "".join(
        f'<li><span class="i {r.kind}">{_ICON[r.kind]}</span><span>{escape(r.text)}</span></li>' for r in items
    ) + "</ul>"


def _kpi(label: str, before: str, after: str) -> str:
    return f'<div class="kpi"><div class="l">{escape(label)}</div><div class="v"><span>{escape(before)} →</span> {escape(after)}</div></div>'


def report_html(r: PortfolioReport, title: str = "Portfolio rebalancing plan", demo: bool = False) -> str:
    sym = r.currency
    recs = sorted(r.recommendations, key=lambda x: -abs(x.trade_value))
    stats = r.portfolio_stats
    pnl = r.total_value - r.total_invested
    parts = [f"<h1>{escape(title)}</h1>",
             f'<p class="sub">{escape(STRATEGIES[r.settings.strategy])} · prices to {r.as_of:%d %b %Y} · '
             f'portfolio {fmt_money(r.total_value, sym)} ({fmt_compact(pnl, sym, signed=True)} unrealised)'
             + (" · <b>demo data, not real prices</b>" if demo else "") + "</p>"]

    parts.append('<div class="kpis">' + "".join([
        _kpi("Expected return", fmt_pct(r.current.expected_return), fmt_pct(r.target.expected_return)),
        _kpi("Volatility (yearly swings)", fmt_pct(r.current.volatility), fmt_pct(r.target.volatility)),
        _kpi("Sharpe ratio", f"{r.current.sharpe:.2f}", f"{r.target.sharpe:.2f}"),
        _kpi("Worst drawdown", fmt_pct(stats.loc["Current", "max_drawdown"], 0),
             fmt_pct(stats.loc["Target", "max_drawdown"], 0)),
    ]) + "</div>")

    plan = getattr(r, "plan", None)  # plain-English steps, when available
    if plan:
        parts.append('<h2>Your plan</h2><div class="card"><ol class="steps">'
                     + "".join(f"<li>{escape(s)}</li>" for s in plan) + "</ol></div>")
    parts.append(f'<h2>Why this plan</h2><div class="card">{_reasons(r.summary)}</div>')

    top = max(max(x.current_weight for x in recs), max(x.target_weight for x in recs), 0.01)
    rows = "".join(
        f'<div>{escape(x.label)}</div><div class="track">'
        f'<div class="bar now" style="width:{x.current_weight / top * 100:.1f}%" title="Now {x.current_weight:.1%}"></div>'
        f'<div class="bar tgt" style="width:{x.target_weight / top * 100:.1f}%" title="Target {x.target_weight:.1%}"></div>'
        f'</div>' for x in recs)
    parts.append('<h2>Current vs target weights</h2><div class="card"><div class="legend">'
                 '<i style="background:var(--orange)"></i>Now<i style="background:var(--blue)"></i>Target</div>'
                 f'<div class="bars">{rows}</div></div>')

    trade_rows = "".join(
        f'<tr><td>{escape(x.ticker)}</td><td><span class="badge {_BADGE.get(x.action, "b-hold")}">{escape(x.action)}</span></td>'
        f'<td class="n">{x.trade_qty:+,}</td><td class="n">{fmt_money(x.trade_value, sym, signed=True) if x.trade_qty else "–"}</td>'
        f'<td class="n">{fmt_pct(x.current_weight)}</td><td class="n">{fmt_pct(x.target_weight)}</td>'
        f'<td>{escape(x.timing)}</td></tr>' for x in recs)
    cash = r.cash
    net = cash.get("net", 0.0)
    cash_line = (f"Sales raise {fmt_compact(cash.get('sells', 0), sym)} and purchases cost "
                 f"{fmt_compact(cash.get('buys', 0), sym)}: "
                 + (f"add about {fmt_compact(net, sym)} of new money." if net > 0 else
                    f"about {fmt_compact(-net, sym)} is left over as cash." if net < 0 else "they fund each other."))
    parts.append('<h2>Trades</h2><table><tr><th>Ticker</th><th>Action</th><th class="n">Units</th>'
                 '<th class="n">Value</th><th class="n">Now</th><th class="n">Target</th><th>Timing</th></tr>'
                 f'{trade_rows}</table><p class="sub" style="margin-top:8px">{escape(cash_line)}</p>')

    parts.append("<h2>Holding by holding</h2>")
    for x in recs:
        why = [q for q in x.reasons if q.category not in ("Technical",)][:6]
        summary = getattr(x, "summary", "")
        parts.append(
            f'<div class="card holding"><h3><span>{escape(x.label)} <span style="color:var(--muted);font-weight:500">'
            f'{escape(x.ticker)}</span></span><span class="badge {_BADGE.get(x.action, "b-hold")}">{escape(x.action)}</span></h3>'
            f'<div class="hl">{escape(x.headline)}</div>'
            + (f'<div class="tldr">{escape(summary)}</div>' if summary else "")
            + f"{_reasons(why)}</div>")

    if r.backtest is not None:
        bt = r.backtest.stats
        bt_rows = "".join(
            f'<tr><td>{escape(name)}</td><td class="n">{fmt_pct(row["cagr"])}</td><td class="n">{fmt_pct(row["volatility"])}</td>'
            f'<td class="n">{row["sharpe"]:.2f}</td><td class="n">{fmt_pct(row["max_drawdown"])}</td></tr>'
            for name, row in bt.iterrows())
        parts.append('<h2>Backtest (no hindsight)</h2><table><tr><th>Mix</th><th class="n">Yearly return</th>'
                     '<th class="n">Volatility</th><th class="n">Sharpe</th><th class="n">Worst fall</th></tr>'
                     f"{bt_rows}</table>")

    parts.append('<p class="foot">Expected returns are estimates from past prices, the least reliable input. Taxes, '
                 'exit loads and lot sizes are not modelled. Educational tool, not investment advice. '
                 'Generated by Portfolio Optimizer.</p>')
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{escape(title)}</title><style>{_CSS}</style></head><body><main>"
            + "".join(parts) + "</main></body></html>")
