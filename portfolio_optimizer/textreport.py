"""Plain-text rendering of a report, for the CLI and notebooks."""

from __future__ import annotations

import textwrap

from .config import STRATEGIES
from .formatting import fmt_money, fmt_pct
from .plain import history_warning
from .report import PortfolioReport

ICON = {"positive": "+", "negative": "-", "neutral": "·"}


def render_text(r: PortfolioReport, demo: bool = False, width: int = 96) -> str:
    sym = r.currency
    out: list[str] = []

    def wrap(t: str) -> str:
        return textwrap.fill(t, width, initial_indent="    ", subsequent_indent="      ")

    out.append(f"PORTFOLIO REPORT · {STRATEGIES[r.settings.strategy]} · prices to {r.as_of:%d %b %Y}"
               + (" · DEMO DATA" if demo else ""))
    out.append(f"Value {fmt_money(r.total_value, sym)}   "
               f"Unrealised P&L {fmt_money(r.total_value - r.total_invested, sym, signed=True)}")
    short = history_warning(r.history)
    if short:
        out.append(wrap(f"! {short}"))
    out.append(f"{'':12}{'Return':>10}{'Volatility':>12}{'Sharpe':>9}")
    for name, res in (("Current", r.current), ("Target", r.target)):
        out.append(f"{name:12}{fmt_pct(res.expected_return):>10}{fmt_pct(res.volatility):>12}{res.sharpe:>9.2f}")

    out.append("\nYOUR PLAN")
    out.extend(wrap(f"{i}. {step}") for i, step in enumerate(r.plan, 1))

    out.append("\nWHY REBALANCE")
    out.extend(wrap(f"{ICON[x.kind]} {x.text}") for x in r.summary)

    out.append("\nWHAT TO DO")
    for rec in sorted(r.recommendations, key=lambda x: -abs(x.trade_value)):
        out.append(f"\n  {rec.ticker} ({rec.label}) – {rec.action.upper()}: {rec.headline}")
        conf = ""
        if rec.weight_range:
            lo, hi = (fmt_pct(x, 0) for x in rec.weight_range)
            conf = f" · confidence {rec.confidence.lower()} ({lo if lo == hi else f'{lo}–{hi}'})"
        out.append(f"    timing: {rec.timing} · technicals {rec.technical.stance.lower()} "
                   f"({rec.technical.score:+.0f}){conf}")
        out.append(wrap(f"In short: {rec.summary}"))
        out.extend(wrap(f"{ICON[x.kind]} [{x.category}] {x.text}") for x in rec.reasons)

    if r.notes:
        out.append("\nNOTES")
        out.extend(textwrap.fill(n, width, initial_indent="  · ", subsequent_indent="    ") for n in r.notes)
    out.append("\nEducational tool, not investment advice.")
    return "\n".join(out)
