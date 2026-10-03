"""Long-only portfolio optimisation with position limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import analytics as an


@dataclass
class OptimizationResult:
    strategy: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


def _bounds_ok(n: int, lo: float, hi: float) -> None:
    if not 0 <= lo <= hi <= 1:
        raise ValueError("Weight limits must satisfy 0 ≤ min ≤ max ≤ 1.")
    if n * hi < 1 - 1e-9:
        raise ValueError(
            f"A {hi:.0%} cap on {n} assets can only invest {n * hi:.0%} of the money. "
            f"Raise the max weight to at least {1 / n:.0%}."
        )
    if n * lo > 1 + 1e-9:
        raise ValueError(f"A {lo:.0%} floor on {n} assets needs more than 100% of the money.")


def _solve(objective, n: int, lo: float, hi: float, starts: list[np.ndarray]) -> np.ndarray:
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(lo, hi)] * n
    best, best_val = None, np.inf
    for x0 in starts:
        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        if res.success or best is None:
            val = objective(res.x)
            if val < best_val:
                best, best_val = res.x, val
    w = np.clip(best, lo, hi)
    return w / w.sum()


def _starts(n: int, lo: float, hi: float, seed: int = 0, k: int = 6) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = [np.full(n, 1 / n)]
    for _ in range(k):
        w = rng.dirichlet(np.ones(n))
        starts.append(np.clip(w, lo, hi) / np.clip(w, lo, hi).sum())
    return starts


def optimize(
    mu: pd.Series,
    cov_annual: pd.DataFrame,
    strategy: str = "max_sharpe",
    rf: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> OptimizationResult:
    n = len(mu)
    _bounds_ok(n, min_weight, max_weight)
    sigma = cov_annual.to_numpy()
    mu_v = mu.to_numpy()
    vols = np.sqrt(np.diag(sigma))
    starts = _starts(n, min_weight, max_weight)

    if strategy == "max_sharpe":
        def obj(w):
            vol = np.sqrt(w @ sigma @ w)
            return -(w @ mu_v - rf) / vol
    elif strategy == "min_volatility":
        def obj(w):
            return w @ sigma @ w * 100
    elif strategy == "risk_parity":
        def obj(w):
            rc = w * (sigma @ w)
            return np.sum((rc / rc.sum() - 1 / n) ** 2) * 1e3
        # Inverse-vol is the closed-form answer when correlations are equal.
        inv = (1 / vols) / (1 / vols).sum()
        starts = [np.clip(inv, min_weight, max_weight) / np.clip(inv, min_weight, max_weight).sum()] + starts
    elif strategy == "max_diversification":
        def obj(w):
            return -(w @ vols) / np.sqrt(w @ sigma @ w)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    w = _solve(obj, n, min_weight, max_weight, starts)
    weights = pd.Series(w, index=mu.index)
    weights[weights.abs() < 1e-6] = 0.0
    weights /= weights.sum()
    return evaluate(weights, mu, cov_annual, rf, strategy)


def evaluate(weights: pd.Series, mu: pd.Series, cov_annual: pd.DataFrame, rf: float,
             strategy: str = "current") -> OptimizationResult:
    w = weights.reindex(mu.index).fillna(0).to_numpy()
    return OptimizationResult(
        strategy=strategy,
        weights=pd.Series(w, index=mu.index),
        expected_return=an.portfolio_return(w, mu),
        volatility=an.portfolio_volatility(w, cov_annual),
        sharpe=an.sharpe_ratio(w, mu, cov_annual, rf),
    )


def efficient_frontier(
    mu: pd.Series, cov_annual: pd.DataFrame, min_weight: float = 0.0, max_weight: float = 1.0,
    points: int = 40,
) -> pd.DataFrame:
    """Minimum-volatility portfolio for a grid of target returns."""
    n = len(mu)
    _bounds_ok(n, min_weight, max_weight)
    sigma = cov_annual.to_numpy()
    mu_v = mu.to_numpy()

    w_min = optimize(mu, cov_annual, "min_volatility", 0, min_weight, max_weight).weights.to_numpy()
    # Highest achievable return under the caps: fill the best assets first.
    order = np.argsort(-mu_v)
    w_max = np.full(n, min_weight)
    room = 1 - w_max.sum()
    for i in order:
        add = min(max_weight - w_max[i], room)
        w_max[i] += add
        room -= add
    lo_ret, hi_ret = float(w_min @ mu_v), float(w_max @ mu_v)

    rows = []
    prev = w_min
    for target in np.linspace(lo_ret, hi_ret, points):
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu_v - t},
        )
        res = minimize(lambda w: w @ sigma @ w * 100, prev, method="SLSQP",
                       bounds=[(min_weight, max_weight)] * n, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        if not res.success:
            continue
        prev = res.x
        rows.append({"return": float(res.x @ mu_v), "volatility": float(np.sqrt(res.x @ sigma @ res.x)),
                     **{t: float(x) for t, x in zip(mu.index, res.x)}})
    return pd.DataFrame(rows)


def random_portfolios(mu: pd.Series, cov_annual: pd.DataFrame, rf: float, n: int = 3000,
                      seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.full(len(mu), 0.7), size=n)
    sigma = cov_annual.to_numpy()
    ret = w @ mu.to_numpy()
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, sigma, w))
    return pd.DataFrame({"return": ret, "volatility": vol, "sharpe": (ret - rf) / vol})
