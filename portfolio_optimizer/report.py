"""One call that runs the whole analysis and returns everything the UI needs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import analytics as an
from .backtest import BacktestResult, walk_forward
from .config import (DEFAULT_MAX_WEIGHT, DEFAULT_MIN_WEIGHT, DEFAULT_RISK_FREE_RATE, STRATEGIES, TRADING_DAYS,
                     asset_class_for)
from .formatting import currency_symbol
from .indicators import TechnicalSnapshot, technical_snapshot
from .optimizer import (Constraints, OptimizationResult, ResampleResult, efficient_frontier, evaluate,
                        hrp_top_split, optimize, random_portfolios, resample)
from .reasons import AssetRecommendation, Reason, build_recommendations, cash_flow, portfolio_reasons


@dataclass
class Settings:
    strategy: str = "max_sharpe"
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    min_weight: float = DEFAULT_MIN_WEIGHT
    max_weight: float = DEFAULT_MAX_WEIGHT
    return_method: str = "bayes_stein"
    rebalance_intensity: float = 1.0  # 1 = go all the way to the optimal weights
    run_backtest: bool = True
    backtest_lookback: int = TRADING_DAYS
    backtest_rebalance_every: int = 63
    transaction_cost_bps: float = 10.0
    class_cap: float | None = None  # max total weight per asset class (None = no cap)
    max_turnover: float | None = None  # max one-way turnover per rebalance (None = no limit)
    no_trade_band: float = 0.02  # skip trades that move a weight by less than this
    robust: bool = True  # average weights over resampled histories (Michaud)
    n_resamples: int = 100


@dataclass
class PortfolioReport:
    settings: Settings
    tickers: list[str]
    currency: str
    as_of: pd.Timestamp
    prices: pd.DataFrame
    returns: pd.DataFrame
    mu: pd.Series
    cov_annual: pd.DataFrame
    correlation: pd.DataFrame
    total_value: float
    total_invested: float
    current: OptimizationResult
    optimal: OptimizationResult
    target: OptimizationResult  # optimal blended with current by rebalance_intensity
    min_vol: OptimizationResult
    point: OptimizationResult  # single-estimate optimum (equals ``optimal`` when robust is off)
    risk_share_current: pd.Series
    risk_share_target: pd.Series
    asset_stats: pd.DataFrame
    portfolio_stats: pd.DataFrame
    technicals: dict[str, TechnicalSnapshot]
    recommendations: list[AssetRecommendation]
    summary: list[Reason]
    asset_classes: dict[str, str] = field(default_factory=dict)
    constraints: Constraints | None = None
    resample: ResampleResult | None = None
    comparison: pd.DataFrame | None = None  # metrics per strategy (point estimates)
    comparison_weights: pd.DataFrame | None = None  # weights per strategy
    cash: dict[str, float] = field(default_factory=dict)
    frontier: pd.DataFrame | None = None
    cloud: pd.DataFrame | None = None
    backtest: BacktestResult | None = None
    benchmark: pd.Series | None = None
    notes: list[str] = field(default_factory=list)


def analyze(
    holdings: dict[str, dict[str, float]],
    prices: pd.DataFrame,
    settings: Settings | None = None,
    benchmark: pd.Series | None = None,
) -> PortfolioReport:
    s = settings or Settings()
    tickers = list(holdings)
    prices = prices[tickers].dropna(how="any")
    rf = s.risk_free_rate
    notes: list[str] = []

    returns = an.daily_returns(prices)
    cov_daily = an.ledoit_wolf_covariance(returns)
    cov = cov_daily * TRADING_DAYS
    mu = an.expected_returns(returns, cov_daily, s.return_method)
    vols = pd.Series(np.sqrt(np.diag(cov)), index=tickers)
    corr = cov / np.outer(vols, vols)

    if "shrinkage" in cov_daily.attrs:
        notes.append(f"Correlations shrunk {cov_daily.attrs['shrinkage']:.0%} toward zero (Ledoit–Wolf).")
    if "shrinkage" in mu.attrs:
        notes.append(f"Expected returns shrunk {mu.attrs['shrinkage']:.0%} toward the common mean (Bayes–Stein).")

    last = prices.iloc[-1]
    values = pd.Series({t: holdings[t]["qty"] * last[t] for t in tickers})
    total_value = float(values.sum())
    if total_value <= 0:
        raise ValueError("Portfolio value is zero. Enter a quantity for at least one holding.")
    total_invested = float(sum(h["qty"] * h["avg_price"] for h in holdings.values()))
    w_cur = values / total_value

    classes = {t: (holdings[t].get("class") or asset_class_for(t)) for t in tickers}
    cons = Constraints(min_weight=s.min_weight, max_weight=s.max_weight, asset_classes=classes,
                       class_cap=s.class_cap, current=w_cur, max_turnover=s.max_turnover)

    current = evaluate(w_cur, mu, cov, rf, "current")
    point = optimize(mu, cov, s.strategy, rf, constraints=cons, returns=returns)
    rs = None
    if s.robust:
        rs = resample(returns, s.strategy, rf, cons, s.n_resamples, s.return_method)
        optimal = evaluate(rs.mean, mu, cov, rf, s.strategy)
        notes.append(f"Target = average of {len(rs.samples)} optimisations on block-bootstrapped histories "
                     f"(Michaud resampling).")
    else:
        optimal = point
    k = float(np.clip(s.rebalance_intensity, 0, 1))
    target = evaluate(w_cur + k * (optimal.weights - w_cur), mu, cov, rf, "target")
    min_vol = optimize(mu, cov, "min_volatility", rf, constraints=cons)

    comparison, comparison_weights = compare_strategies(mu, cov, rf, cons, returns, w_cur)

    technicals = {t: technical_snapshot(t, prices[t]) for t in tickers}

    # Correlation of each asset with the rest of the portfolio (weighted by
    # the other holdings' current weights; equal weights if it's the only one).
    avg_corr = {}
    for t in tickers:
        others = [o for o in tickers if o != t]
        if not others:
            avg_corr[t] = 0.0
            continue
        wo = w_cur[others]
        wo = wo / wo.sum() if wo.sum() > 0 else pd.Series(1 / len(others), index=others)
        avg_corr[t] = float((corr.loc[t, others] * wo).sum())

    bench_ret = benchmark.pct_change(fill_method=None).reindex(returns.index) if benchmark is not None else None
    asset_stats = pd.DataFrame({t: an.series_stats(returns[t], rf, bench_ret).as_dict() for t in tickers}).T
    asset_stats["expected_return"] = mu
    asset_stats["exp_volatility"] = vols
    asset_stats["exp_sharpe"] = (mu - rf) / vols

    port_daily = pd.DataFrame({
        "Current": returns @ w_cur.to_numpy(),
        "Target": returns @ target.weights.to_numpy(),
    })
    if bench_ret is not None and bench_ret.notna().sum() > 20:
        port_daily["Benchmark"] = bench_ret
    portfolio_stats = pd.DataFrame({c: an.series_stats(port_daily[c], rf, bench_ret).as_dict()
                                    for c in port_daily}).T

    risk_cur = an.risk_contributions(w_cur.to_numpy(), cov)
    risk_tgt = an.risk_contributions(target.weights.to_numpy(), cov)

    f, higher_better = an.strategy_objective(s.strategy, mu, cov, rf, returns)
    contrib = an.attribute_change(f, w_cur.to_numpy(), target.weights.to_numpy())

    ctx = dict(
        tickers=tickers, holdings=holdings, prices=last, currency=currency_symbol(tickers),
        strategy=s.strategy, rf=rf, mu=mu, vols=vols, asset_sharpe=asset_stats["exp_sharpe"],
        risk_share=risk_cur, risk_share_target=risk_tgt,
        contrib=pd.Series(contrib, index=tickers), higher_better=higher_better,
        objective_change=f(target.weights.to_numpy()) - f(w_cur.to_numpy()),
        avg_corr=pd.Series(avg_corr), asset_mdd=asset_stats["max_drawdown"],
        current_w=w_cur, target_w=target.weights, optimal_w=optimal.weights,
        total_value=total_value, max_weight=s.max_weight, technicals=technicals,
        band=s.no_trade_band, asset_classes=classes, class_cap=s.class_cap,
        capped_classes=_capped_classes(optimal.weights, classes, s.class_cap),
        max_turnover=s.max_turnover, optimal_turnover=float((optimal.weights - w_cur).abs().sum() / 2),
        resample=rs,
        hrp_split=_hrp_split(cov, tickers) if s.strategy == "hrp" else None,
        cvar_current=an.historical_cvar(w_cur.to_numpy(), returns),
        cvar_target=an.historical_cvar(target.weights.to_numpy(), returns),
        eff_current=an.effective_holdings(w_cur.to_numpy()),
        eff_target=an.effective_holdings(target.weights.to_numpy()),
    )
    recs = build_recommendations(ctx)

    bt = None
    if s.run_backtest:
        try:
            bt = walk_forward(prices, w_cur, s.strategy, rf, s.min_weight, s.max_weight,
                              s.backtest_lookback, s.backtest_rebalance_every, s.transaction_cost_bps,
                              benchmark, s.return_method,
                              constraints=Constraints(s.min_weight, s.max_weight, classes, s.class_cap,
                                                      max_turnover=s.max_turnover))
        except ValueError as e:
            notes.append(f"Backtest skipped: {e}")

    summary = portfolio_reasons(ctx, current, target, recs, bt)

    return PortfolioReport(
        settings=s, tickers=tickers, currency=ctx["currency"], as_of=prices.index[-1], prices=prices,
        returns=returns, mu=mu, cov_annual=cov, correlation=corr, total_value=total_value,
        total_invested=total_invested, current=current, optimal=optimal, target=target, min_vol=min_vol,
        point=point, asset_classes=classes, constraints=cons, resample=rs, comparison=comparison,
        comparison_weights=comparison_weights, cash=cash_flow(recs),
        frontier=efficient_frontier(mu, cov, constraints=cons), cloud=random_portfolios(mu, cov, rf),
        risk_share_current=risk_cur, risk_share_target=risk_tgt, asset_stats=asset_stats,
        portfolio_stats=portfolio_stats, technicals=technicals, recommendations=recs, summary=summary,
        backtest=bt, benchmark=benchmark, notes=notes,
    )


def _capped_classes(weights: pd.Series, classes: dict[str, str], cap: float | None) -> set[str]:
    if cap is None or cap >= 1:
        return set()
    totals: dict[str, float] = {}
    for t, w in weights.items():
        totals[classes[t]] = totals.get(classes[t], 0.0) + float(w)
    return {c for c, v in totals.items() if v >= cap - 0.005}


def compare_strategies(mu: pd.Series, cov: pd.DataFrame, rf: float, cons: Constraints,
                       returns: pd.DataFrame, w_cur: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point-estimate optimum of every strategy under the same limits, plus your current mix."""
    weights = {"Your current mix": w_cur}
    for key, name in STRATEGIES.items():
        try:
            weights[name] = optimize(mu, cov, key, rf, constraints=cons, returns=returns, n_starts=4).weights
        except ValueError:
            continue
    rows = {}
    for name, w in weights.items():
        x = w.to_numpy()
        daily = returns @ x
        rows[name] = {
            "Expected return": an.portfolio_return(x, mu),
            "Volatility": an.portfolio_volatility(x, cov),
            "Sharpe": an.sharpe_ratio(x, mu, cov, rf),
            "CVaR 95% (daily)": an.historical_cvar(x, returns),
            "Max drawdown": float(an.drawdown((1 + daily).cumprod()).min()),
            "Effective holdings": an.effective_holdings(x),
            "Turnover": float((w - w_cur).abs().sum() / 2),
        }
    return pd.DataFrame(rows).T, pd.DataFrame(weights).T


def _hrp_split(cov: pd.DataFrame, tickers: list[str]) -> dict:
    left, right, share = hrp_top_split(cov.to_numpy())
    return {"left": [tickers[i] for i in left], "right": [tickers[i] for i in right], "left_share": share}
