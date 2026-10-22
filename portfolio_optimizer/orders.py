"""Turn the rebalancing plan into an order sheet you can actually execute.

Sells go first because they raise the money for the buys. Holdings whose
timing signal says "Stagger" or "Wait for dip" are split into weekly
tranches instead of being traded in one go.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .formatting import fmt_money

STAGGER_TIMINGS = ("Stagger", "Wait for dip")


@dataclass
class Order:
    step: int
    when: pd.Timestamp
    side: str  # "Sell" | "Buy"
    ticker: str
    label: str
    units: int
    price: float
    value: float  # positive money amount
    cash_after: float  # running cash balance after this order (negative = new money needed)
    note: str


def _split(units: int, parts: int) -> list[int]:
    """Split whole units into ``parts`` tranches; any remainder goes in the first."""
    parts = max(1, min(parts, units))
    base, rem = divmod(units, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def build_orders(recs, start: pd.Timestamp | None = None, tranches: int = 3, weeks_apart: int = 1) -> list[Order]:
    """Order sheet for a list of ``AssetRecommendation``.

    ``start`` is the first trading day (defaults to the next business day
    after today). Staggered holdings get ``tranches`` orders, ``weeks_apart``
    weeks apart; everything else trades on the first day, sells before buys.
    """
    day0 = pd.Timestamp(start) if start is not None else pd.Timestamp.today().normalize() + pd.offsets.BDay(1)
    planned: list[tuple[pd.Timestamp, int, Order]] = []
    # If a sale is spread out, pace the buys alongside it so the sales keep funding them.
    sells_staggered = any(r.trade_qty < 0 and r.timing in STAGGER_TIMINGS for r in recs)
    for r in recs:
        if r.trade_qty == 0:
            continue
        side = "Sell" if r.trade_qty < 0 else "Buy"
        own_signal = r.timing in STAGGER_TIMINGS
        paced = side == "Buy" and sells_staggered and not own_signal
        chunks = _split(abs(r.trade_qty), tranches if (own_signal or paced) else 1)
        for k, units in enumerate(chunks):
            when = day0 + pd.offsets.Week(k * weeks_apart) if k else day0
            when = when if when.dayofweek < 5 else when + pd.offsets.BDay(1)
            if len(chunks) > 1:
                why = ("paced with the staggered sales so they pay for it" if paced else
                       "momentum is still positive, so sell gradually" if side == "Sell" else
                       "short-term signals are weak, so buy gradually")
                note = f"Tranche {k + 1} of {len(chunks)}: {why}."
            elif r.action == "Exit":
                note = "Sell the whole position."
            else:
                note = ""
            order = Order(0, when, side, r.ticker, r.label, units, r.price, units * r.price, 0.0, note)
            # Sort key: date, then sells before buys, then larger orders first.
            planned.append((when, 0 if side == "Sell" else 1, order))

    planned.sort(key=lambda x: (x[0], x[1], -x[2].value))
    cash = 0.0
    out = []
    for i, (_, _, o) in enumerate(planned, 1):
        cash += o.value if o.side == "Sell" else -o.value
        o.step, o.cash_after = i, cash
        out.append(o)
    return out


def orders_frame(orders: list[Order]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Step": o.step, "Date": o.when.date(), "Side": o.side, "Ticker": o.ticker, "Asset": o.label,
        "Units": o.units, "Approx. price": round(o.price, 2), "Approx. value": round(o.value),
        "Cash after": round(o.cash_after), "Note": o.note,
    } for o in orders])


def order_ticket(orders: list[Order], currency: str, title: str = "Rebalancing orders") -> str:
    """Printable plain-text order sheet."""
    lines = [title, "=" * len(title), ""]
    if not orders:
        return "\n".join(lines + ["No trades needed."])
    current = None
    for o in orders:
        if o.when != current:
            current = o.when
            lines.append(f"{o.when:%a %d %b %Y}")
        lines.append(f"  {o.step:>2}. {o.side.upper():4} {o.units:>7,} × {o.ticker:<15} ≈ {fmt_money(o.value, currency):>14}"
                     f"   cash after {fmt_money(o.cash_after, currency, signed=True)}")
        if o.note:
            lines.append(f"      {o.note}")
    end = orders[-1].cash_after
    lines += ["", ("You need to add about " + fmt_money(-end, currency) + " of new money." if end < 0 else
                   f"About {fmt_money(end, currency)} is left over as cash."),
              "Prices are the latest close; use limit orders near them. Not investment advice."]
    return "\n".join(lines)
