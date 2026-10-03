"""Walk-forward backtest: re-optimise on trailing data only, then hold.

An in-sample "optimised" portfolio always looks great because it was chosen
with hindsight. This module re-estimates the weights at each rebalance date
using only the data available at that time, which is what you would actually
have experienced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import analytics as an
from .config import TRADING_DAYS
from .estimators import Estimator
from .optimizer import Constraints, optimize, resample


@dataclass
class BacktestResult:
    daily: pd.DataFrame  # daily returns per strategy
    growth: pd.DataFrame  # growth of 1 unit
    stats: pd.DataFrame  # one row per strategy
    weights_history: pd.DataFrame  # optimiser weights at each rebalance


def _run_schedule(returns: pd.DataFrame, start: int, step: int, weight_fn, cost_bps: float) -> pd.Series:
    r = returns.to_numpy()
    out = np.full(len(r), np.nan)
    w = None
    for i in range(start, len(r)):
        if (i - start) % step == 0:
            target = weight_fn(i, w)
            turnover = np.abs(target - w).sum() if w is not None else 0.0
            w = target.copy()
            cost = turnover * cost_bps / 1e4
        else:
            cost = 0.0
        day = float(w @ r[i])
        out[i] = day - cost
        w = w * (1 + r[i]) / (1 + day)  # weights drift with prices between rebalances
    return pd.Series(out, index=returns.index).iloc[start:]


def walk_forward(
    prices: pd.DataFrame,
    current_weights: pd.Series,
    strategy: str = "max_sharpe",
    rf: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    lookback: int = TRADING_DAYS,
    rebalance_every: int = 63,
    cost_bps: float = 10.0,
    benchmark: pd.Series | None = None,
    mu_method: str = "bayes_stein",
    constraints: Constraints | None = None,
    estimator: Estimator | None = None,
    n_resamples: int = 0,
) -> BacktestResult:
    """``constraints`` (asset-class caps, turnover limit) apply at every rebalance;
    the turnover limit is measured from the weights held just before it. The
    first rebalance starts from ``current_weights``."""
    returns = an.daily_returns(prices)
    if len(returns) <= lookback + 20:
        raise ValueError("Not enough history for a walk-forward test; use an earlier start date.")
    n = returns.shape[1]
    history: dict[pd.Timestamp, pd.Series] = {}

    cur = current_weights.reindex(returns.columns).fillna(0).to_numpy()
    base = constraints or Constraints(min_weight=min_weight, max_weight=max_weight)
    est = estimator or Estimator(return_model=mu_method, rf=rf)

    def optimised(i: int, held: np.ndarray | None) -> np.ndarray:
        window = returns.iloc[i - lookback:i]
        mu, cov = est.fit(window)
        start_w = cur if held is None else held
        c = replace(base, current=pd.Series(start_w, index=returns.columns))
        if n_resamples:
            w = resample(window, strategy, rf, c, n_resamples, estimator=est, seed=i).mean
        else:
            w = optimize(mu, cov * TRADING_DAYS, strategy, rf, constraints=c, returns=window, n_starts=4).weights
        history[returns.index[i]] = w
        return w.to_numpy()

    eq = np.full(n, 1 / n)

    daily = pd.DataFrame({
        "Optimised (walk-forward)": _run_schedule(returns, lookback, rebalance_every, optimised, cost_bps),
        "Your current mix": _run_schedule(returns, lookback, rebalance_every, lambda i, w: cur, cost_bps),
        "Equal weight": _run_schedule(returns, lookback, rebalance_every, lambda i, w: eq, cost_bps),
    })
    bench_daily = None
    if benchmark is not None:
        bench_daily = benchmark.pct_change(fill_method=None).reindex(daily.index)
        if bench_daily.notna().sum() > 20:
            daily["Benchmark"] = bench_daily

    growth = (1 + daily.fillna(0)).cumprod()
    stats = pd.DataFrame({
        name: an.series_stats(daily[name], rf, bench_daily).as_dict() for name in daily.columns
    }).T
    return BacktestResult(daily, growth, stats, pd.DataFrame(history).T)
