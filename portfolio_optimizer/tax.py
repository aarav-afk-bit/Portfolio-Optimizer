"""Rough Indian capital-gains tax on the plan's sales.

Defaults follow the rules announced in the July 2024 budget (and the 2025
change that put gold/silver and international ETFs under the same 12-month
test): listed equity is taxed 20% short-term and 12.5% long-term above a
yearly ₹1.25 lakh exemption; other listed ETF units at your slab rate
short-term and 12.5% long-term. Everything is editable because rules
change. Surcharge and cess are not included.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

EQUITY = "Listed equity"
OTHER = "Other listed ETF"


@dataclass(frozen=True)
class TaxRates:
    equity_short: float = 0.20
    equity_long: float = 0.125
    equity_exemption: float = 125_000.0  # long-term equity gains tax-free each financial year
    other_short: float = 0.30  # your income-tax slab rate
    other_long: float = 0.125


@dataclass(frozen=True)
class Sale:
    ticker: str
    label: str
    units: int
    price: float
    avg_cost: float
    tax_type: str  # EQUITY or OTHER
    long_term: bool  # held more than 12 months

    @property
    def gain(self) -> float:
        return self.units * (self.price - self.avg_cost)


@dataclass
class TaxEstimate:
    lines: pd.DataFrame  # one row per sale with gain, bucket, rate and tax
    total_gain: float
    total_tax: float
    exemption_used: float
    notes: list[str] = field(default_factory=list)


def default_tax_type(asset_class: str) -> str:
    """Indian equity shares and equity ETFs pay STT and get equity rates; everything else is 'other'."""
    return EQUITY if "indian equity" in (asset_class or "").lower() else OTHER


def _rate(tax_type: str, long_term: bool, r: TaxRates) -> float:
    if tax_type == EQUITY:
        return r.equity_long if long_term else r.equity_short
    return r.other_long if long_term else r.other_short


def estimate_tax(sales: list[Sale], rates: TaxRates = TaxRates()) -> TaxEstimate:
    """Tax on a set of sales, with losses set off the way Indian rules allow.

    Short-term losses can reduce any gain; long-term losses only long-term
    gains. Losses are applied to the highest-taxed gains first. The equity
    long-term exemption is applied last, to what remains of those gains.
    """
    notes: list[str] = []
    rows = []
    for s in sales:
        if s.avg_cost <= 0:
            notes.append(f"{s.label}: no average cost entered, so its gain can't be estimated.")
            continue
        rows.append({"ticker": s.ticker, "label": s.label, "units": s.units, "gain": s.gain, "type": s.tax_type,
                     "long_term": s.long_term, "rate": _rate(s.tax_type, s.long_term, rates)})
    if not rows:
        return TaxEstimate(pd.DataFrame(columns=["ticker", "label", "units", "gain", "type", "long_term", "rate",
                                                 "taxable", "tax"]), 0.0, 0.0, 0.0, notes)
    df = pd.DataFrame(rows)
    df["taxable"] = df["gain"].clip(lower=0)

    # Set off losses: short-term losses against any gain, long-term losses against long-term gains only,
    # highest rate first.
    for long_term_loss in (False, True):
        loss = -df.loc[(df["gain"] < 0) & (df["long_term"] == long_term_loss), "gain"].sum()
        eligible = df[(df["taxable"] > 0) & (df["long_term"] | (not long_term_loss))].sort_values("rate", ascending=False)
        for i in eligible.index:
            if loss <= 0:
                break
            used = min(loss, df.at[i, "taxable"])
            df.at[i, "taxable"] -= used
            loss -= used
        if loss > 0:
            kind = "long-term" if long_term_loss else "short-term"
            notes.append(f"About ₹{loss:,.0f} of {kind} losses is left over; it can be carried forward for 8 years.")

    # Yearly exemption on long-term equity gains.
    exemption_left = rates.equity_exemption
    for i in df[(df["type"] == EQUITY) & df["long_term"] & (df["taxable"] > 0)].index:
        used = min(exemption_left, df.at[i, "taxable"])
        df.at[i, "taxable"] -= used
        exemption_left -= used
    df["tax"] = df["taxable"] * df["rate"]
    return TaxEstimate(df, float(df["gain"].sum()), float(df["tax"].sum()),
                       rates.equity_exemption - exemption_left, notes)
