"""Risk-profile presets, so most people never need to touch the expert settings."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .optimizer import Constraints, check_feasible


@dataclass(frozen=True)
class Preset:
    name: str
    blurb: str
    strategy: str
    max_weight: float
    class_cap: float | None  # None = no asset-class cap
    rebalance_intensity: float
    no_trade_band: float


PRESETS: dict[str, Preset] = {
    "Cautious": Preset(
        "Cautious", "Smoothest ride. Moves halfway toward the target, at most 30% in any one holding and 60% in "
                    "any asset class.",
        strategy="min_volatility", max_weight=0.30, class_cap=0.60, rebalance_intensity=0.5, no_trade_band=0.03),
    "Balanced": Preset(
        "Balanced", "Best return for the risk, with guard rails: at most 35% in one holding and 70% in one asset "
                    "class, moving three-quarters of the way.",
        strategy="max_sharpe", max_weight=0.35, class_cap=0.70, rebalance_intensity=0.75, no_trade_band=0.02),
    "Growth": Preset(
        "Growth", "Best return for the risk with loose limits: up to 45% in one holding, moving all the way to "
                  "the target.",
        strategy="max_sharpe", max_weight=0.45, class_cap=None, rebalance_intensity=1.0, no_trade_band=0.02),
}
DEFAULT_PRESET = "Balanced"


def fit_to_holdings(preset: Preset, tickers: list[str], classes: dict[str, str]) -> tuple[Preset, list[str]]:
    """Loosen a preset's limits just enough to be feasible for these holdings.

    Returns the adjusted preset and plain-English notes on what changed.
    """
    notes: list[str] = []
    n = len(tickers)
    p = preset
    if n * p.max_weight < 1:
        new = math.ceil(100 / n / 5) * 5 / 100  # next 5% step that can hold all the money
        notes.append(f"With {n} holdings no single one can be capped at {p.max_weight:.0%}; using {new:.0%}.")
        p = replace(p, max_weight=new)
    if p.class_cap is not None:
        try:
            check_feasible(tickers, Constraints(max_weight=p.max_weight, asset_classes=classes,
                                                class_cap=p.class_cap))
        except ValueError:
            notes.append(f"Your holdings span too few asset classes for a {p.class_cap:.0%} cap per class; "
                         f"the class cap is off.")
            p = replace(p, class_cap=None)
    return p, notes
