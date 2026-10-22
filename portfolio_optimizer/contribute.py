"""Rebalance with new money instead of selling.

Buying only what is under-weight avoids capital-gains tax and exit costs.
This module works out what to buy with a given amount, how close that gets
you to the target, and how much new money a full rebalance without any
selling would take.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ContributionPlan:
    amount: float  # new money available
    buys: pd.DataFrame  # ticker, label, units, price, value
    spent: float
    left_over: float
    weights_before: pd.Series
    weights_after: pd.Series
    target: pd.Series
    gap_before: float  # share of the portfolio that would have to move to hit the target (one-way)
    gap_after: float
    needed_without_selling: float  # new money that brings every holding up to target (inf if impossible)
    stuck: list[str]  # holdings the target wants to cut to ~0: only selling fixes those


def _gap(weights: pd.Series, target: pd.Series) -> float:
    return float((weights - target).abs().sum() / 2)


def money_to_reach_target(values: pd.Series, target: pd.Series) -> float:
    """Smallest new amount after which no holding is above its target weight.

    Every holding with a positive target must end up at target * (V + A) >= its
    current value, so A = max(value_i / target_i) - V. Holdings the target wants
    to (almost) remove can't be fixed by buying, so they're left out here.
    """
    keep = target > 0.005
    if not keep.any():
        return float("inf")
    total = float(values.sum())
    return max(float((values[keep] / target[keep]).max()) - total, 0.0)


def plan_contribution(recs, target: pd.Series, amount: float) -> ContributionPlan:
    """What to buy with ``amount`` of new money, selling nothing.

    Money goes to holdings below their target value in proportion to how far
    below they are; anything beyond the full shortfall follows the target
    weights. Units are whole, so a little cash is usually left over.
    """
    tickers = [r.ticker for r in recs]
    price = pd.Series({r.ticker: r.price for r in recs})
    label = {r.ticker: r.label for r in recs}
    values = pd.Series({r.ticker: r.current_value for r in recs})
    target = target.reindex(tickers).fillna(0)
    total = float(values.sum())
    new_total = total + amount

    shortfall = (target * new_total - values).clip(lower=0)
    if shortfall.sum() > 0 and amount > 0:
        alloc = amount * shortfall / shortfall.sum() if shortfall.sum() >= amount else shortfall.copy()
        extra = amount - float(alloc.sum())
        if extra > 1e-9:
            alloc += extra * target
    else:
        alloc = amount * target
    units = np.floor(alloc / price).astype(int)

    # Whole units leave cash behind; top up the most under-weight holdings while it lasts.
    left = amount - float((units * price).sum())
    order = (target * new_total - values - units * price).sort_values(ascending=False).index
    for t in order:
        if price[t] <= left and target[t] > 0:
            extra_units = int(left // price[t])
            units[t] += extra_units
            left -= extra_units * price[t]

    bought = units * price
    after_values = values + bought
    w_before = values / total if total else values * 0
    w_after = after_values / after_values.sum()
    buys = pd.DataFrame({"ticker": tickers, "label": [label[t] for t in tickers], "units": units[tickers].values,
                         "price": price[tickers].values, "value": bought[tickers].values})
    buys = buys[buys["units"] > 0].sort_values("value", ascending=False).reset_index(drop=True)
    return ContributionPlan(
        amount=amount, buys=buys, spent=float(bought.sum()), left_over=amount - float(bought.sum()),
        weights_before=w_before, weights_after=w_after, target=target,
        gap_before=_gap(w_before, target), gap_after=_gap(w_after, target),
        needed_without_selling=money_to_reach_target(values, target),
        stuck=[label[t] for t in tickers if target[t] <= 0.005 and w_before[t] > 0.005],
    )
