"""Plain-English summaries: one sentence per holding and a three-step plan.

The detailed reasons stay in ``reasons.py``; this module turns the same
numbers into wording a non-specialist can act on without the jargon.
"""

from __future__ import annotations

from .formatting import fmt_compact, fmt_pct

_VERB = {"Buy more": "Add more", "Initiate": "Start a position", "Trim": "Sell some", "Exit": "Sell all",
         "Hold": "Keep as is"}


def holding_summary(action: str, *, current_weight: float, target_weight: float, risk_share: float,
                    expected_return: float, rf: float, avg_corr: float, is_best: bool, is_worst: bool,
                    optimal_zero: bool, confidence: str | None) -> str:
    """One sentence: what to do with a holding and the most important reason why."""
    why: list[str] = []
    cutting = action in ("Trim", "Exit")
    adding = action in ("Buy more", "Initiate")

    if cutting:
        if current_weight > 0.005 and risk_share - current_weight > 0.08:
            why.append(f"{fmt_pct(risk_share, 0)} of your portfolio's ups and downs come from this one holding "
                       f"({fmt_pct(current_weight, 0)} of your money)")
        if expected_return < rf:
            why.append(f"it isn't expected to beat a {fmt_pct(rf)} safe return")
        elif is_worst:
            why.append("it pays the least for the risk it carries of anything you hold")
        if optimal_zero and not why:
            why.append("the rest of your portfolio does its job better")
        if not why:
            why.append("the money earns more for the same risk elsewhere in your portfolio")
    elif adding:
        if avg_corr < 0.2:
            why.append("it tends to move independently of your other holdings, which softens falls")
        if is_best:
            why.append("it pays the most for the risk it carries of anything you hold")
        if current_weight > 0.005 and current_weight - risk_share > 0.08:
            why.append("it steadies the portfolio more than its size suggests")
        if not why:
            why.append("it improves the balance between return and risk")
    else:
        why.append(f"it is already about the right size ({fmt_pct(current_weight, 0)} of your money)")

    sentence = f"{_VERB.get(action, action)}: " + " and ".join(why[:2]) + "."
    if confidence == "Low" and action != "Hold":
        sentence += " The data is unsure about the exact size, so consider moving gradually."
    return sentence


def plan_steps(recs, current, target, currency: str) -> list[str]:
    """Three steps: what to sell, what to buy, and what changes as a result."""
    sells = [r for r in recs if r.trade_qty < 0]
    buys = [r for r in recs if r.trade_qty > 0]

    def names(items):
        items = sorted(items, key=lambda r: -abs(r.trade_value))
        shown = [f"{r.label} ({fmt_compact(abs(r.trade_value), currency)})" for r in items[:3]]
        more = len(items) - 3
        return ", ".join(shown) + (f" and {more} more" if more > 0 else "")

    if not sells and not buys:
        return ["No trades needed: your mix is already close to the target."]

    raised = sum(-r.trade_value for r in sells)
    spent = sum(r.trade_value for r in buys)
    steps = []
    steps.append(f"Sell {fmt_compact(raised, currency)}: {names(sells)}." if sells else
                 "Nothing to sell.")
    gap = spent - raised
    funding = ("" if not sells else
               f" Add about {fmt_compact(gap, currency)} of new money." if gap > 0.01 * max(spent, 1) else
               f" About {fmt_compact(-gap, currency)} is left over as cash." if -gap > 0.01 * max(raised, 1) else
               " The sales pay for it.")
    steps.append((f"Buy {fmt_compact(spent, currency)}: {names(buys)}." if buys else "Nothing to buy.") + funding)

    biggest_now = max(recs, key=lambda r: r.current_weight)
    biggest_after = max(recs, key=lambda r: r.target_weight)
    steps.append(
        f"Result: yearly swings go from {fmt_pct(current.volatility, 0)} to {fmt_pct(target.volatility, 0)}, "
        f"expected return from {fmt_pct(current.expected_return, 0)} to {fmt_pct(target.expected_return, 0)}, "
        f"and your biggest holding goes from {fmt_pct(biggest_now.current_weight, 0)} "
        f"({biggest_now.label}) to {fmt_pct(biggest_after.target_weight, 0)} ({biggest_after.label})."
    )
    return steps
