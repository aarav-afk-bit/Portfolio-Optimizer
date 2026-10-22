"""Plain-English definitions for every term the app shows.

``short`` is used for tooltips; ``plain`` and ``example`` for the Guide tab.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    name: str
    short: str
    plain: str
    example: str = ""
    group: str = "Risk & return"


_TERMS = [
    # Risk & return
    Term("Expected return",
         "The model's estimate of the average yearly return.",
         "What the model thinks a holding or mix will earn in a typical year. It's an estimate built from your "
         "price history and each holding's risk, not a promise.",
         "Expected return 15% on ₹10L means roughly ₹1.5L a year on average, with big swings around it."),
    Term("Volatility",
         "How much the value swings in a typical year. Lower is a smoother ride.",
         "The size of normal ups and downs, measured over a year. A 20% volatility means a typical year can "
         "easily land 20% above or below the expected return.",
         "On ₹10L, 25% volatility means ±₹2.5L in an ordinary year is unremarkable."),
    Term("Sharpe ratio",
         "Return above a safe deposit, per unit of swing. Higher is better.",
         "How much extra return you get, beyond a risk-free option like T-bills, for each unit of volatility you "
         "accept. It's the main 'value for risk' score.",
         "Two mixes both expected to earn 14%: the one with 12% volatility has the higher Sharpe."),
    Term("Risk-free rate",
         "What a safe option (Indian 91-day T-bills, ~6.5%) pays.",
         "The return you could get with almost no risk. A holding has to beat this to be worth its swings."),
    Term("Max drawdown",
         "The deepest fall from a peak before recovering.",
         "The largest peak-to-trough drop over the price history. It shows how painful the worst stretch was.",
         "−40% means at the worst point, ₹10L had fallen to ₹6L."),
    Term("CVaR (worst-day loss)",
         "Average loss on the worst 5% of days.",
         "Look at the worst 1-in-20 trading days and average their losses. It measures how bad the bad days are, "
         "not just how often they come.",
         "CVaR 2% on ₹10L: on a bad day, expect to lose about ₹20,000."),
    Term("VaR",
         "A one-day loss you'd exceed only 1 day in 20.",
         "The loss threshold for the worst 5% of days. CVaR is the average of the losses beyond it."),
    # Diversification
    Term("Correlation",
         "How closely two holdings move together, from −1 to +1.",
         "+1 means they always move together; 0 means unrelated; −1 means opposite. Low correlation is what makes "
         "diversification work.", group="Diversification"),
    Term("Risk share",
         "The share of the portfolio's swings that comes from one holding.",
         "A holding can be 30% of your money but cause 60% of the ups and downs if it's volatile and moves with "
         "the rest. When risk share is far above capital share, that holding dominates.",
         "HAL: 72% of the money, 85% of the risk.", group="Diversification"),
    Term("Effective holdings",
         "How many equal-sized holdings your mix behaves like.",
         "Five holdings where one is 90% of the money behaves like about one holding. Higher means less "
         "concentration.", group="Diversification"),
    Term("Diversification ratio",
         "How much volatility cancels out because holdings don't move together.",
         "Weighted average volatility of the holdings divided by the portfolio's volatility. 1 means no benefit; "
         "higher means more cancelling out.", group="Diversification"),
    Term("Asset class",
         "A group of similar investments, e.g. precious metals or Indian equity.",
         "Used for the per-class caps. Edit it in the Class column of your holdings.", group="Diversification"),
    # The plan
    Term("Rebalance",
         "Buying and selling to move from your current mix to the target.",
         "The plan sells what's over-weight and buys what's under-weight.", group="The plan"),
    Term("Rebalance intensity",
         "How far toward the target to move: 100% all the way, 50% halfway.",
         "Moving partway lowers trading and tax, and hedges against the model being wrong.", group="The plan"),
    Term("Turnover",
         "The share of the portfolio bought (and equally sold) in a rebalance.",
         "50% turnover on ₹10L means selling about ₹5L and buying about ₹5L.", group="The plan"),
    Term("No-trade band",
         "Changes smaller than this are skipped as not worth the cost.",
         "With a 2 pp band, a holding moving from 20% to 21% is left alone.", group="The plan"),
    Term("Confidence",
         "How stable a holding's target is when the model is re-run on reshuffled history.",
         "The model is run on 100 resampled versions of your price history. If the target barely moves, confidence "
         "is high; if it swings from 0% to 40%, it's low and moving gradually is wiser.", group="The plan"),
    Term("Timing",
         "Whether chart signals suggest trading now, gradually (stagger) or waiting for a dip.",
         "Timing never changes how much to buy or sell, only when.", group="The plan"),
    # Strategies & models
    Term("Max Sharpe ratio",
         "Strategy: the best return for the risk taken.",
         "Finds the mix with the highest Sharpe ratio within your limits.", group="Strategies & models"),
    Term("Minimum volatility",
         "Strategy: the smoothest possible ride.",
         "Finds the mix with the smallest swings, whatever the return.", group="Strategies & models"),
    Term("Minimum tail risk",
         "Strategy: the smallest losses on the worst days (lowest CVaR).", "", group="Strategies & models"),
    Term("Risk parity",
         "Strategy: every holding contributes the same share of risk.", "", group="Strategies & models"),
    Term("Hierarchical risk parity (HRP)",
         "Strategy: groups holdings that move together and splits money between groups by risk.",
         "Doesn't rely on return forecasts at all, so it's less sensitive to noisy estimates.",
         group="Strategies & models"),
    Term("Max diversification",
         "Strategy: holdings whose swings cancel each other out the most.", "", group="Strategies & models"),
    Term("Black–Litterman",
         "Return model: start from what each holding's risk deserves, then let your data nudge it.",
         "A few years of returns say little about the future, so the model trusts the 'fair' return for the risk "
         "and only moves it part of the way toward your price history (about 15% for 3–4 years of data). You can "
         "add your own views too.", group="Strategies & models"),
    Term("Efficient frontier",
         "The best return available at each level of risk.",
         "Any mix below the curve could earn more for the same risk.", group="Strategies & models"),
    Term("Walk-forward backtest",
         "A test that only uses data available at the time, with no hindsight.",
         "At each rebalance the model sees only the past year, then holds until the next one. It's the honest "
         "check on whether a strategy would have worked.", group="Strategies & models"),
    # Technicals
    Term("RSI", "Momentum gauge from 0 to 100: above 70 overbought, below 30 oversold.", "", group="Chart signals"),
    Term("MACD", "Short-term momentum: positive when the recent trend is up.", "", group="Chart signals"),
    Term("200-day average", "The average price over about the last 10 months; above it means a long-term uptrend.",
         "", group="Chart signals"),
    Term("Bollinger bands", "A band around the recent average; closes outside it are unusually stretched.", "",
         group="Chart signals"),
]

GLOSSARY: dict[str, Term] = {t.name: t for t in _TERMS}
GROUPS = list(dict.fromkeys(t.group for t in _TERMS))


def search(query: str) -> list[Term]:
    q = query.strip().lower()
    return [t for t in _TERMS if not q or q in t.name.lower() or q in t.short.lower() or q in t.plain.lower()]


def tip(name: str) -> str:
    """Tooltip text for a term."""
    return GLOSSARY[name].short
