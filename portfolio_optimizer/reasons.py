"""Turn the optimiser's output into actions and plain-English reasons.

Every reason is tied to a number the user can check. The headline "why" for
each weight change comes from the first-order condition of the chosen
strategy, so it explains what the optimiser actually responded to rather
than offering a generic story.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import STRATEGIES, label_for
from .formatting import fmt_compact, fmt_money, fmt_pct
from .indicators import TechnicalSnapshot

HOLD_BAND = 0.02  # weight changes under 2 percentage points are not worth trading


@dataclass
class Reason:
    kind: str  # "positive" | "negative" | "neutral"
    category: str  # "Optimiser" | "Risk" | "Diversification" | "Return" | "Technical" | "Position"
    text: str


@dataclass
class AssetRecommendation:
    ticker: str
    label: str
    asset_class: str
    action: str  # Buy more | Initiate | Trim | Exit | Hold
    price: float
    qty: float
    avg_price: float
    current_value: float
    target_value: float
    trade_value: float
    trade_qty: int
    current_weight: float
    target_weight: float
    pnl: float
    pnl_pct: float
    expected_return: float
    volatility: float
    sharpe: float
    risk_share: float
    technical: TechnicalSnapshot
    timing: str
    headline: str
    contribution: float  # this asset's share of the objective's improvement
    reasons: list[Reason] = field(default_factory=list)
    confidence: str | None = None  # High | Medium | Low (resampling on) or None
    weight_range: tuple[float, float] | None = None  # 10th–90th percentile of resampled weights

    @property
    def delta_weight(self) -> float:
        return self.target_weight - self.current_weight


def decide_action(cur: float, tgt: float, band: float = HOLD_BAND) -> str:
    if tgt < 0.005 and cur >= 0.005:
        return "Exit"
    if cur < 0.005 and tgt >= 0.005:
        return "Initiate"
    if abs(tgt - cur) < band:
        return "Hold"
    return "Buy more" if tgt > cur else "Trim"


_OBJECTIVE_NAMES = {
    "max_sharpe": "Sharpe ratio",
    "min_volatility": "volatility",
    "min_cvar": "tail risk (daily CVaR)",
    "max_diversification": "diversification ratio",
}
_LOWER_IS_BETTER = {"min_volatility", "min_cvar"}


def _fmt_obj(strategy: str, x: float) -> str:
    return f"{abs(x) * 100:.2f} pp" if strategy in _LOWER_IS_BETTER else f"{abs(x):.3f}"


_WHY_ADD = {
    "max_sharpe": "it earns more return per unit of risk it adds than the rest of the portfolio",
    "min_volatility": "it adds less risk at the margin than the holdings it replaces",
    "min_cvar": "it holds up better than the holdings it replaces on the worst days",
    "max_diversification": "its swings largely offset those of your other holdings",
}
_WHY_CUT = {
    "max_sharpe": "the money earns more per unit of risk elsewhere",
    "min_volatility": "it adds more risk at the margin than the holdings that replace it",
    "min_cvar": "it contributes heavily to losses on the worst days",
    "max_diversification": "it moves too much in step with your other holdings",
}


def _driver_reason(strategy: str, t: str, action: str, ctx: dict) -> Reason:
    """Why the optimiser moved this weight, from the exact attribution of the
    objective's improvement to each asset (see analytics.attribute_change)."""
    d = float(ctx["target_w"][t] - ctx["current_w"][t])

    if strategy == "hrp":
        split = ctx["hrp_split"]
        mine, other = (split["left"], split["right"]) if t in split["left"] else (split["right"], split["left"])
        share = split["left_share"] if t in split["left"] else 1 - split["left_share"]
        peers = [label_for(x) for x in mine if x != t]
        group = f"with {', '.join(peers)}" if peers else "on its own"
        low_risk = share >= 0.5
        return Reason("positive" if (d > 0) == low_risk else "neutral", "Optimiser",
                      f"HRP groups holdings that move together and splits money between groups by their risk. "
                      f"It sits {group} in the {'lower' if low_risk else 'higher'}-risk group, which gets "
                      f"{fmt_pct(share, 0)} of the portfolio; within a group, steadier holdings get more.")

    if strategy == "risk_parity":
        rc, fair = float(ctx["risk_share"][t]), 1 / len(ctx["tickers"])
        rc_tgt = float(ctx["risk_share_target"][t])
        if action == "Hold":
            return Reason("neutral", "Optimiser",
                          f"Its weight is already about right. Its share of portfolio risk moves from "
                          f"{fmt_pct(rc, 0)} to {fmt_pct(rc_tgt, 0)} as the other positions are rebalanced "
                          f"around it.")
        kind = "negative" if rc > fair else "positive"
        verb = "more" if rc > fair else "less"
        return Reason(kind, "Optimiser",
                      f"Carries {fmt_pct(rc, 0)} of your portfolio's risk today, {verb} than the equal "
                      f"{fmt_pct(fair, 0)} share risk parity aims for. After rebalancing: {fmt_pct(rc_tgt, 0)}.")

    obj = _OBJECTIVE_NAMES[strategy]
    if action == "Hold":
        return Reason("neutral", "Optimiser",
                      f"Already within {ctx['band'] * 100:.0f} pp of its optimal weight, so trading it would barely move "
                      f"the {obj}.")

    c, total = float(ctx["contrib"][t]), float(ctx["objective_change"])
    helps = c > 0 if ctx["higher_better"] else c < 0
    verb = "raises" if c > 0 else "lowers"
    gain = "reduction" if strategy in _LOWER_IS_BETTER else "improvement"
    move = "Adding" if d > 0 else "Reducing"
    if helps:
        why = (_WHY_ADD if d > 0 else _WHY_CUT)[strategy]
        return Reason("positive", "Optimiser",
                      f"{move} it {verb} the {obj} by {_fmt_obj(strategy, c)} (of the "
                      f"{_fmt_obj(strategy, total)} total {gain}): {why}.")
    role = ("absorbs money freed from positions that hurt more" if d > 0
            else "frees money for positions that help more")
    return Reason("neutral", "Optimiser",
                  f"On its own, {move.lower()} it {verb} the {obj} by {_fmt_obj(strategy, c)}, but it {role}. "
                  f"Net {gain} across the portfolio: {_fmt_obj(strategy, total)}.")


def _technical_reasons(snap: TechnicalSnapshot) -> list[Reason]:
    kind = {"bullish": "positive", "bearish": "negative", "neutral": "neutral"}
    out = []
    for s in snap.signals:
        if s.name == "Bollinger" and s.stance == "neutral":
            continue
        out.append(Reason(kind[s.stance], "Technical", f"{s.name}: {s.value}. {s.detail}"))
    if snap.from_52w_high > -0.03:
        out.append(Reason("neutral", "Technical", "Trading within 3% of its 52-week high."))
    elif snap.from_52w_high < -0.25:
        out.append(Reason("negative", "Technical",
                          f"{fmt_pct(-snap.from_52w_high, 0)} below its 52-week high."))
    return out


def _timing(action: str, snap: TechnicalSnapshot) -> tuple[str, Reason | None]:
    stance = snap.stance
    if action in ("Buy more", "Initiate"):
        if stance == "Bearish":
            return "Stagger", Reason("neutral", "Technical",
                                     "Short-term momentum is negative. Consider buying in 2–3 tranches over "
                                     "a few weeks instead of all at once.")
        if snap.rsi >= 70:
            return "Wait for dip", Reason("neutral", "Technical",
                                          "The optimiser wants more, but RSI says it is overbought. A pullback "
                                          "may give a better entry.")
        if stance == "Bullish":
            return "Good", Reason("positive", "Technical", "The technical picture supports adding now.")
        return "Good", Reason("neutral", "Technical", "No strong technical signal either way. Adding now is fine.")
    if action in ("Trim", "Exit"):
        if stance == "Bullish":
            return "Stagger", Reason("neutral", "Technical",
                                     "Momentum is still positive. Trimming gradually lets you capture some "
                                     "of the trend.")
        if stance == "Bearish":
            return "Good", Reason("positive", "Technical",
                                  "Momentum is against it, so reducing now is reasonable.")
        return "Good", Reason("neutral", "Technical", "No strong technical signal either way. Reducing now is fine.")
    return "–", None


def build_recommendations(ctx: dict) -> list[AssetRecommendation]:
    """``ctx`` is assembled in report.analyze; see there for the keys."""
    recs = []
    tickers = ctx["tickers"]
    sym = ctx["currency"]
    strategy = ctx["strategy"]
    sharpes = ctx["asset_sharpe"]
    best, worst = sharpes.idxmax(), sharpes.idxmin()

    for t in tickers:
        h = ctx["holdings"][t]
        price = float(ctx["prices"][t])
        cur_w, tgt_w = float(ctx["current_w"][t]), float(ctx["target_w"][t])
        cur_val = h["qty"] * price
        tgt_val = tgt_w * ctx["total_value"]
        trade_val = tgt_val - cur_val
        action = decide_action(cur_w, tgt_w, ctx["band"])
        trade_qty = int(np.sign(trade_val) * np.floor(abs(trade_val) / price)) if action != "Hold" else 0
        if action == "Exit":
            trade_qty = -int(h["qty"])
        trade_val = trade_qty * price  # what the whole-unit order actually costs or raises
        invested = h["qty"] * h["avg_price"]
        pnl = cur_val - invested
        pnl_pct = pnl / invested if invested > 0 else float("nan")
        snap: TechnicalSnapshot = ctx["technicals"][t]

        reasons: list[Reason] = [_driver_reason(strategy, t, action, ctx)]

        # Return quality
        mu, vol, sh = ctx["mu"][t], ctx["vols"][t], sharpes[t]
        if mu < ctx["rf"]:
            reasons.append(Reason("negative", "Return",
                                  f"Expected return ({fmt_pct(mu)}) is below the {fmt_pct(ctx['rf'])} risk-free "
                                  f"rate, so on these estimates you are not paid for the risk."))
        elif t == best and len(tickers) > 1:
            reasons.append(Reason("positive", "Return",
                                  f"Best risk-adjusted asset you hold: Sharpe {sh:.2f} on an expected "
                                  f"{fmt_pct(mu)} return with {fmt_pct(vol)} volatility."))
        elif t == worst and len(tickers) > 1:
            reasons.append(Reason("negative", "Return",
                                  f"Weakest risk-adjusted asset you hold: Sharpe {sh:.2f} "
                                  f"(expected {fmt_pct(mu)}, volatility {fmt_pct(vol)})."))
        else:
            reasons.append(Reason("neutral", "Return",
                                  f"Sharpe {sh:.2f}: expected {fmt_pct(mu)} a year with {fmt_pct(vol)} volatility."))
        bl = ctx.get("bl")
        if bl is not None:
            prior, sample = float(bl["prior"][t]), float(bl["sample"][t])
            view = next((v for v in bl["views"] if v.ticker == t), None)
            text = (f"Where the {fmt_pct(mu)} comes from: its risk alone justifies {fmt_pct(prior)} at a long-run "
                    f"Sharpe of {bl['prior_sharpe']:.2f}. Its {fmt_pct(sample)} a year in your price history pulls "
                    f"that only {fmt_pct(bl['data_weight'], 0)} of the way up or down, because a few years of "
                    f"returns say little about the next few.")
            if view is not None:
                text += (f" Your view of {fmt_pct(view.expected)} ({fmt_pct(view.confidence, 0)} confidence) is "
                         f"blended in too.")
            reasons.append(Reason("neutral", "Return", text))

        # Risk concentration in the current portfolio
        rc = float(ctx["risk_share"][t])
        if cur_w > 0.005:
            if rc - cur_w > 0.08:
                reasons.append(Reason("negative", "Risk",
                                      f"Punches above its weight on risk: {fmt_pct(cur_w, 0)} of your money but "
                                      f"{fmt_pct(rc, 0)} of your portfolio's volatility."))
            elif cur_w - rc > 0.08:
                reasons.append(Reason("positive", "Risk",
                                      f"Stabiliser: {fmt_pct(cur_w, 0)} of your money but only {fmt_pct(rc, 0)} "
                                      f"of your portfolio's volatility."))

        # Diversification
        rho = float(ctx["avg_corr"][t])
        if rho < 0.2:
            reasons.append(Reason("positive", "Diversification",
                                  f"Low correlation ({rho:.2f}) with the rest of your holdings. It tends to move "
                                  f"independently, which cushions drawdowns."))
        elif rho > 0.6:
            reasons.append(Reason("negative", "Diversification",
                                  f"Moves closely with your other holdings (avg. correlation {rho:.2f}), so it "
                                  f"adds little diversification."))
        else:
            reasons.append(Reason("neutral", "Diversification",
                                  f"Moderate correlation ({rho:.2f}) with the rest of your holdings."))

        mdd = float(ctx["asset_mdd"][t])
        if mdd < -0.30:
            reasons.append(Reason("negative", "Risk",
                                  f"Fell {fmt_pct(-mdd, 0)} peak-to-trough in this period. Expect deep swings."))

        # Constraint effects
        cls = ctx["asset_classes"][t]
        opt_w = float(ctx["optimal_w"][t])
        if opt_w >= ctx["max_weight"] - 0.005 and ctx["max_weight"] < 1:
            reasons.append(Reason("neutral", "Optimiser",
                                  f"Held at your {fmt_pct(ctx['max_weight'], 0)} position cap. Without the cap "
                                  f"the optimiser would hold more."))
        elif cls in ctx["capped_classes"] and opt_w > 0.005:
            reasons.append(Reason("neutral", "Optimiser",
                                  f"Its asset class ({cls}) is at your {fmt_pct(ctx['class_cap'], 0)} cap, which "
                                  f"limits how much it can hold."))
        elif opt_w < 0.005:
            reasons.append(Reason("negative", "Optimiser",
                                  "The optimiser gives it zero weight: on these estimates it does not improve "
                                  "the portfolio at any size."))

        # Robustness: how stable is this weight across resampled histories?
        confidence, w_range = None, None
        rs = ctx.get("resample")
        if rs is not None:
            lo, hi, held = float(rs.low[t]), float(rs.high[t]), float(rs.held[t])
            w_range = (lo, hi)
            spread = hi - lo
            n_s = len(rs.samples)
            span = f"{fmt_pct(lo, 0)}–{fmt_pct(hi, 0)}"
            if spread < 0.005:
                confidence = "High"
                reasons.append(Reason("neutral", "Confidence",
                                      f"Every one of {n_s} resampled optimisations lands on {fmt_pct(lo, 0)}: "
                                      f"your limits set this weight, not the noisy estimates."))
            elif spread <= 0.10:
                confidence = "High"
                reasons.append(Reason("positive", "Confidence",
                                      f"High confidence: across {n_s} resampled price histories the optimal weight "
                                      f"stayed within {span}."))
            elif spread <= 0.25:
                confidence = "Medium"
                reasons.append(Reason("neutral", "Confidence",
                                      f"Medium confidence: the optimal weight ranged {span} across {n_s} resampled "
                                      f"histories. The target uses their average."))
            else:
                confidence = "Low"
                reasons.append(Reason("neutral", "Confidence",
                                      f"Low confidence: the optimal weight swung between {span} across {n_s} "
                                      f"resampled histories (held in {fmt_pct(held, 0)} of them), so the data does "
                                      f"not pin it down. Consider moving gradually."))

        # Technicals and timing
        reasons.extend(_technical_reasons(snap))
        timing, timing_reason = _timing(action, snap)
        if timing_reason:
            reasons.append(timing_reason)

        # Position / P&L
        if invested > 0:
            reasons.append(Reason("positive" if pnl >= 0 else "negative", "Position",
                                  f"Unrealised {'gain' if pnl >= 0 else 'loss'} of {fmt_money(abs(pnl), sym)} "
                                  f"({fmt_pct(pnl_pct, 1, signed=True)}) on your average cost of "
                                  f"{fmt_money(h['avg_price'], sym, 2)}."))
            if action in ("Trim", "Exit") and pnl > 0:
                reasons.append(Reason("neutral", "Position",
                                      "Selling realises part of this gain. Check short- vs long-term capital "
                                      "gains tax before you trade."))
            if action in ("Trim", "Exit") and pnl < 0:
                reasons.append(Reason("neutral", "Position",
                                      "Selling books a capital loss, which can offset gains elsewhere."))

        recs.append(AssetRecommendation(
            ticker=t, label=label_for(t), asset_class=cls, action=action, price=price,
            qty=h["qty"], avg_price=h["avg_price"], current_value=cur_val, target_value=tgt_val,
            trade_value=trade_val, trade_qty=trade_qty,
            current_weight=cur_w, target_weight=tgt_w, pnl=pnl, pnl_pct=pnl_pct,
            expected_return=mu, volatility=vol, sharpe=sh, risk_share=rc, technical=snap,
            timing=timing, headline=_headline(action, cur_w, tgt_w, trade_qty, trade_val, sym),
            contribution=float(ctx["contrib"][t]), confidence=confidence, weight_range=w_range,
            reasons=reasons,
        ))
    return recs


def cash_flow(recs: list[AssetRecommendation]) -> dict[str, float]:
    """Money raised by sells and spent on buys, at whole units and the latest price."""
    buys = sum(r.trade_qty * r.price for r in recs if r.trade_qty > 0)
    sells = sum(-r.trade_qty * r.price for r in recs if r.trade_qty < 0)
    return {"buys": buys, "sells": sells, "net": buys - sells}


def _headline(action, cur, tgt, qty, val, sym) -> str:
    move = f"{fmt_pct(cur)} → {fmt_pct(tgt)}"
    if action == "Hold":
        return f"Keep as is ({move})"
    verb = "Buy" if qty > 0 else "Sell"
    return f"{verb} {abs(qty):,} units (≈{fmt_compact(abs(val), sym)}), {move}"


def portfolio_reasons(ctx: dict, current, target, recs: list[AssetRecommendation],
                      backtest=None) -> list[Reason]:
    """Portfolio-level case for (or against) rebalancing."""
    out: list[Reason] = []
    sym = ctx["currency"]
    name = STRATEGIES[ctx["strategy"]]

    d_sh = target.sharpe - current.sharpe
    out.append(Reason("positive" if d_sh > 0.02 else "neutral", "Optimiser",
                      f"{name}: expected Sharpe ratio {current.sharpe:.2f} → {target.sharpe:.2f}, "
                      f"return {fmt_pct(current.expected_return)} → {fmt_pct(target.expected_return)}, "
                      f"volatility {fmt_pct(current.volatility)} → {fmt_pct(target.volatility)}."))

    top = max(recs, key=lambda r: r.risk_share)
    if top.risk_share > 0.35:
        out.append(Reason("negative", "Risk",
                          f"{top.label} drives {fmt_pct(top.risk_share, 0)} of your current portfolio risk "
                          f"on {fmt_pct(top.current_weight, 0)} of the capital."))

    eff_cur, eff_tgt = ctx["eff_current"], ctx["eff_target"]
    out.append(Reason("positive" if eff_tgt > eff_cur + 0.2 else "neutral", "Diversification",
                      f"Effective number of holdings {eff_cur:.1f} → {eff_tgt:.1f} "
                      f"(of {len(recs)}). Higher means less concentration."))

    classes: dict[str, list[float]] = {}
    for r in recs:
        c = classes.setdefault(r.asset_class, [0.0, 0.0])
        c[0] += r.current_weight
        c[1] += r.target_weight
    mix = ", ".join(f"{k} {fmt_pct(a, 0)} → {fmt_pct(b, 0)}" for k, (a, b) in classes.items())
    out.append(Reason("neutral", "Diversification", f"Asset-class mix: {mix}."))

    cv_cur, cv_tgt = ctx["cvar_current"], ctx["cvar_target"]
    out.append(Reason("positive" if cv_tgt < cv_cur - 0.0005 else "negative" if cv_tgt > cv_cur + 0.0005 else "neutral",
                      "Risk", f"Average loss on the worst 5% of days {fmt_pct(cv_cur, 2)} → {fmt_pct(cv_tgt, 2)} "
                              f"(historical CVaR)."))

    turnover = sum(abs(r.trade_value) for r in recs) / 2
    out.append(Reason("neutral", "Position",
                      f"Rebalancing means moving about {fmt_compact(turnover, sym)} "
                      f"({fmt_pct(turnover / ctx['total_value'], 0)} of the portfolio)."))
    cash = cash_flow(recs)
    if cash["buys"] or cash["sells"]:
        if cash["net"] > 0:
            tail = f"you need to add about {fmt_compact(cash['net'], sym)} of fresh money"
        elif cash["net"] < 0:
            tail = f"about {fmt_compact(-cash['net'], sym)} is left over as cash"
        else:
            tail = "the trades fund each other exactly"
        out.append(Reason("neutral", "Position",
                          f"Sales raise {fmt_compact(cash['sells'], sym)} and purchases cost "
                          f"{fmt_compact(cash['buys'], sym)}, so {tail} (whole units, no-trade band applied)."))

    max_to = ctx.get("max_turnover")
    if max_to is not None and max_to < 1 and ctx["optimal_turnover"] >= max_to - 0.005:
        out.append(Reason("neutral", "Optimiser",
                          f"Your {fmt_pct(max_to, 0)} turnover limit is binding: without it the optimiser would "
                          f"trade more to reach a better mix."))

    rs = ctx.get("resample")
    if rs is not None:
        low = [r.label for r in recs if r.confidence == "Low"]
        out.append(Reason("neutral" if low else "positive", "Confidence",
                          f"Target weights average {len(rs.samples)} optimisations on resampled price histories, "
                          f"which damps estimation noise."
                          + (f" Low-confidence weights: {', '.join(low)}." if low else
                             " Every weight is reasonably stable.")))

    if backtest is not None:
        s = backtest.stats
        opt, cur = s.loc["Optimised (walk-forward)"], s.loc["Your current mix"]
        better = opt["sharpe"] > cur["sharpe"]
        out.append(Reason("positive" if better else "negative", "Return",
                          f"Walk-forward test (no hindsight): the strategy delivered Sharpe {opt['sharpe']:.2f} "
                          f"and {fmt_pct(opt['cagr'])} a year vs {cur['sharpe']:.2f} and {fmt_pct(cur['cagr'])} "
                          f"for your current mix. Max drawdown {fmt_pct(opt['max_drawdown'])} vs "
                          f"{fmt_pct(cur['max_drawdown'])}."))
        if not better:
            out.append(Reason("neutral", "Optimiser",
                              "Out of sample, the strategy has not beaten simply holding your mix. Treat these "
                              "weights as a direction to lean, not a precise target."))

    if abs(d_sh) <= 0.02 and all(r.action == "Hold" for r in recs):
        out.append(Reason("positive", "Optimiser", "Your portfolio is already close to optimal. No trades needed."))
    return out
