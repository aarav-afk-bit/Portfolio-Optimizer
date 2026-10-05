"""Long-only portfolio optimisation with position, asset-class and turnover limits."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

from . import analytics as an
from .config import TRADING_DAYS
from .estimators import Estimator


@dataclass
class OptimizationResult:
    strategy: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


@dataclass
class Constraints:
    """Everything the optimiser must respect besides "long-only, fully invested".

    ``asset_classes`` maps ticker → class; ``class_cap`` limits each class's
    total weight. ``max_turnover`` is one-way turnover (share of the portfolio
    bought, equal to the share sold) measured from ``current``.
    """

    min_weight: float = 0.0
    max_weight: float = 1.0
    asset_classes: dict[str, str] = field(default_factory=dict)
    class_cap: float | None = None
    current: pd.Series | None = None
    max_turnover: float | None = None

    def class_groups(self, tickers) -> dict[str, list[int]]:
        if self.class_cap is None or self.class_cap >= 1:
            return {}
        groups: dict[str, list[int]] = {}
        for i, t in enumerate(tickers):
            groups.setdefault(self.asset_classes.get(t, "Other"), []).append(i)
        return groups

    def turnover_active(self) -> bool:
        return self.current is not None and self.max_turnover is not None and self.max_turnover < 1


# --------------------------------------------------------------------------
# Feasibility
# --------------------------------------------------------------------------

def check_feasible(tickers, c: Constraints) -> None:
    """Raise a ValueError that says exactly which limit to loosen."""
    n = len(tickers)
    lo, hi = c.min_weight, c.max_weight
    if not 0 <= lo <= hi <= 1:
        raise ValueError("Weight limits must satisfy 0 ≤ min ≤ max ≤ 1.")
    if n * hi < 1 - 1e-9:
        raise ValueError(
            f"A {hi:.0%} cap on {n} assets can only invest {n * hi:.0%} of the money. "
            f"Raise the max weight to at least {1 / n:.0%}."
        )
    if n * lo > 1 + 1e-9:
        raise ValueError(f"A {lo:.0%} floor on {n} assets needs more than 100% of the money.")

    groups = c.class_groups(tickers)
    if groups:
        capacity = sum(min(c.class_cap, len(ix) * hi) for ix in groups.values())
        if capacity < 1 - 1e-9:
            raise ValueError(
                f"With a {c.class_cap:.0%} cap on each of {len(groups)} asset classes you can only invest "
                f"{capacity:.0%}. Raise the asset-class cap or add holdings from another class."
            )
        for name, ix in groups.items():
            if len(ix) * lo > c.class_cap + 1e-9:
                raise ValueError(f"The {lo:.0%} minimum per holding already exceeds the {c.class_cap:.0%} "
                                 f"cap for {name}.")

    if c.turnover_active():
        need = min_feasible_turnover(tickers, c)
        if need > c.max_turnover + 1e-6:
            raise ValueError(
                f"Your position and asset-class limits alone require trading at least {need:.0%} of the "
                f"portfolio, above the {c.max_turnover:.0%} turnover limit. Raise the turnover limit."
            )


def min_feasible_turnover(tickers, c: Constraints) -> float:
    """Smallest one-way turnover that brings ``c.current`` inside the limits (an LP)."""
    n = len(tickers)
    cur = c.current.reindex(tickers).fillna(0).to_numpy()
    # variables: w (n), d (n) with d >= |w - cur|; minimise sum(d) / 2
    cost = np.r_[np.zeros(n), np.full(n, 0.5)]
    a_ub, b_ub = _linear_ub(tickers, c, n_extra=n)
    eye = np.eye(n)
    a_ub += [np.r_[eye[i], -eye[i]] for i in range(n)] + [np.r_[-eye[i], -eye[i]] for i in range(n)]
    b_ub += list(cur) + list(-cur)
    res = linprog(cost, A_ub=np.array(a_ub), b_ub=np.array(b_ub),
                  A_eq=np.r_[np.ones(n), np.zeros(n)][None, :], b_eq=[1.0],
                  bounds=[(c.min_weight, c.max_weight)] * n + [(0, None)] * n, method="highs")
    return float(res.fun) if res.success else np.inf


def _linear_ub(tickers, c: Constraints, n_extra: int = 0) -> tuple[list, list]:
    """Asset-class caps as A_ub rows (padded with zeros for extra LP variables)."""
    n = len(tickers)
    rows, rhs = [], []
    for ix in c.class_groups(tickers).values():
        row = np.zeros(n + n_extra)
        row[ix] = 1
        rows.append(row)
        rhs.append(c.class_cap)
    return rows, rhs


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------

def _slsqp_constraints(tickers, c: Constraints) -> list[dict]:
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}]
    for ix in c.class_groups(tickers).values():
        cons.append({"type": "ineq", "fun": lambda w, ix=ix: c.class_cap - np.sum(w[ix])})
    if c.turnover_active():
        cur = c.current.reindex(tickers).fillna(0).to_numpy()
        # Smooth |x| so SLSQP gets a usable gradient; error is < 1e-5 per asset.
        cons.append({"type": "ineq",
                     "fun": lambda w: 2 * c.max_turnover - np.sum(np.sqrt((w - cur) ** 2 + 1e-10))})
    return cons


def _starts(tickers, c: Constraints, k: int, seed: int = 0) -> list[np.ndarray]:
    n = len(tickers)
    lo, hi = c.min_weight, c.max_weight

    def fix(w):
        w = np.clip(w, lo, hi)
        return w / w.sum()

    starts = []
    if c.current is not None:
        starts.append(fix(c.current.reindex(tickers).fillna(0).to_numpy() + 1e-9))
    starts.append(np.full(n, 1 / n))
    rng = np.random.default_rng(seed)
    starts += [fix(rng.dirichlet(np.ones(n))) for _ in range(max(k - len(starts), 0))]
    return starts[:max(k, 1)]


def _solve(objective, tickers, c: Constraints, starts: list[np.ndarray]) -> np.ndarray:
    n = len(tickers)
    cons = _slsqp_constraints(tickers, c)
    bounds = [(c.min_weight, c.max_weight)] * n
    best, best_val = None, np.inf
    for x0 in starts:
        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        feasible = all((con["fun"](res.x) >= -1e-6) if con["type"] == "ineq" else abs(con["fun"](res.x)) < 1e-6
                       for con in cons)
        val = objective(res.x)
        if feasible and val < best_val:
            best, best_val = res.x, val
    if best is None:
        raise ValueError("The optimiser could not satisfy all limits. Loosen the turnover or asset-class caps.")
    w = np.clip(best, c.min_weight, c.max_weight)
    return w / w.sum()


def _min_cvar(returns: pd.DataFrame, tickers, c: Constraints, level: float = 0.95) -> np.ndarray:
    """Rockafellar–Uryasev linear program for minimum historical CVaR."""
    r = returns[tickers].to_numpy()
    t, n = r.shape
    use_to = c.turnover_active()
    n_d = n if use_to else 0
    # variables: w (n), alpha (1), z (t), d (n_d)
    nv = n + 1 + t + n_d
    cost = np.zeros(nv)
    cost[n] = 1.0
    cost[n + 1:n + 1 + t] = 1 / ((1 - level) * t)

    # z_t >= -r_t.w - alpha   ->   -r_t.w - alpha - z_t <= 0
    a_loss = np.zeros((t, nv))
    a_loss[:, :n] = -r
    a_loss[:, n] = -1
    a_loss[np.arange(t), n + 1 + np.arange(t)] = -1
    rows, rhs = _linear_ub(tickers, c, n_extra=nv - n)
    a_ub = [a_loss] + ([np.array(rows)] if rows else [])
    b_ub = [np.zeros(t)] + ([np.array(rhs)] if rhs else [])
    if use_to:
        cur = c.current.reindex(tickers).fillna(0).to_numpy()
        a_d = np.zeros((2 * n + 1, nv))
        for i in range(n):
            a_d[i, i], a_d[i, n + 1 + t + i] = 1, -1          # w - d <= cur
            a_d[n + i, i], a_d[n + i, n + 1 + t + i] = -1, -1  # -w - d <= -cur
        a_d[2 * n, n + 1 + t:] = 1                              # sum d <= 2 * turnover
        a_ub.append(a_d)
        b_ub.append(np.r_[cur, -cur, 2 * c.max_turnover])
    a_eq = np.zeros((1, nv))
    a_eq[0, :n] = 1
    bounds = [(c.min_weight, c.max_weight)] * n + [(None, None)] + [(0, None)] * (t + n_d)
    res = linprog(cost, A_ub=np.vstack(a_ub), b_ub=np.concatenate(b_ub), A_eq=a_eq, b_eq=[1.0],
                  bounds=bounds, method="highs")
    if not res.success:
        raise ValueError("The tail-risk optimiser could not satisfy all limits. Loosen the turnover or caps.")
    w = np.clip(res.x[:n], c.min_weight, c.max_weight)
    return w / w.sum()


def _max_sharpe_exact(mu: pd.Series, sigma: np.ndarray, rf: float, tickers, c: Constraints) -> np.ndarray | None:
    """Global max-Sharpe portfolio via the Cornuejols–Tütüncü convex reformulation.

    Substituting y = kappa * w turns "maximise (mu - rf)'w / sqrt(w'Sw)" into
    the convex QP "minimise y'Sy s.t. (mu - rf)'y = 1" with every linear limit
    scaled by kappa, so any local solution is the global one. Returns None when
    no feasible portfolio beats the risk-free rate (the reformulation needs a
    positive excess return); the caller then falls back to direct search.
    """
    n = len(tickers)
    ex = mu.to_numpy() - rf
    # Best achievable excess return under the limits (LP) – must be positive.
    rows, rhs = _linear_ub(tickers, c)
    lp = linprog(-ex, A_ub=np.array(rows) if rows else None, b_ub=np.array(rhs) if rhs else None,
                 A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(c.min_weight, c.max_weight)] * n, method="highs")
    if not lp.success or -lp.fun <= 1e-9:
        return None
    w0 = lp.x
    k0 = 1 / float(ex @ w0)
    x0 = np.r_[w0 * k0, k0]

    groups = list(c.class_groups(tickers).values())
    cons = [
        {"type": "eq", "fun": lambda x: ex @ x[:n] - 1.0, "jac": lambda x: np.r_[ex, 0.0]},
        {"type": "eq", "fun": lambda x: x[:n].sum() - x[n], "jac": lambda x: np.r_[np.ones(n), -1.0]},
        # lo*kappa <= y_i <= hi*kappa
        {"type": "ineq", "fun": lambda x: c.max_weight * x[n] - x[:n],
         "jac": lambda x: np.c_[-np.eye(n), np.full(n, c.max_weight)]},
        {"type": "ineq", "fun": lambda x: x[:n] - c.min_weight * x[n],
         "jac": lambda x: np.c_[np.eye(n), np.full(n, -c.min_weight)]},
    ]
    for ix in groups:
        mask = np.zeros(n)
        mask[ix] = 1
        cons.append({"type": "ineq", "fun": lambda x, m=mask: c.class_cap * x[n] - m @ x[:n],
                     "jac": lambda x, m=mask: np.r_[-m, c.class_cap]})
    res = minimize(lambda x: x[:n] @ sigma @ x[:n], x0, jac=lambda x: np.r_[2 * sigma @ x[:n], 0.0],
                   method="SLSQP", constraints=cons, bounds=[(0, None)] * (n + 1),
                   options={"maxiter": 1000, "ftol": 1e-15})
    if not res.success or res.x[n] <= 0:
        return None
    w = np.clip(res.x[:n] / res.x[n], c.min_weight, c.max_weight)
    return w / w.sum()


def _cluster_order(cov: np.ndarray, linkage_method: str = "single") -> list[int]:
    """Assets ordered so that highly correlated ones sit next to each other."""
    std = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(std, std), -1, 1)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0)
    return [int(i) for i in leaves_list(linkage(squareform(dist, checks=False), method=linkage_method))]


def _cluster_var(cov: np.ndarray, ix: list[int]) -> float:
    """Variance of the inverse-variance portfolio of a cluster."""
    sub = cov[np.ix_(ix, ix)]
    ivp = 1 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(cov: np.ndarray, linkage_method: str = "single") -> np.ndarray:
    """Hierarchical Risk Parity (López de Prado, 2016).

    Clusters assets by correlation distance, orders them so similar assets
    sit together, then splits capital top-down between clusters in inverse
    proportion to each cluster's variance. It never inverts the covariance
    matrix, so it is far less sensitive to estimation noise than
    mean-variance optimisation.
    """
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    w = np.ones(n)
    clusters = [_cluster_order(cov, linkage_method)]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) < 2:
                continue
            left, right = cl[: len(cl) // 2], cl[len(cl) // 2:]
            v_l, v_r = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= 1 - alpha
            nxt += [left, right]
        clusters = nxt
    return w / w.sum()


def hrp_top_split(cov: np.ndarray, linkage_method: str = "single") -> tuple[list[int], list[int], float]:
    """The first split HRP makes: (left cluster, right cluster, share given to left)."""
    n = cov.shape[0]
    if n < 2:
        return list(range(n)), [], 1.0
    order = _cluster_order(cov, linkage_method)
    left, right = order[: n // 2], order[n // 2:]
    v_l, v_r = _cluster_var(cov, left), _cluster_var(cov, right)
    return left, right, 1 - v_l / (v_l + v_r)


def project(target: np.ndarray, tickers, c: Constraints) -> np.ndarray:
    """Closest portfolio to ``target`` (least squares) that satisfies every limit."""
    if _satisfies(target, tickers, c):
        return target
    return _solve(lambda w: float(np.sum((w - target) ** 2)) * 1e3, tickers, c,
                  [np.clip(target, c.min_weight, c.max_weight) / np.clip(target, c.min_weight, c.max_weight).sum()]
                  + _starts(tickers, c, 2))


def _satisfies(w: np.ndarray, tickers, c: Constraints, tol: float = 1e-8) -> bool:
    return all((con["fun"](w) >= -tol) if con["type"] == "ineq" else abs(con["fun"](w)) <= tol
               for con in _slsqp_constraints(tickers, c)) and bool(
        np.all(w >= c.min_weight - tol) and np.all(w <= c.max_weight + tol))


def optimize(
    mu: pd.Series,
    cov_annual: pd.DataFrame,
    strategy: str = "max_sharpe",
    rf: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    *,
    constraints: Constraints | None = None,
    returns: pd.DataFrame | None = None,
    n_starts: int = 7,
    check: bool = True,
) -> OptimizationResult:
    """Optimise ``strategy`` under the limits.

    ``constraints`` overrides ``min_weight``/``max_weight`` when given.
    ``returns`` (daily) is required for ``min_cvar``.
    """
    tickers = list(mu.index)
    c = constraints or Constraints(min_weight=min_weight, max_weight=max_weight)
    if check:
        check_feasible(tickers, c)
    n = len(tickers)
    sigma = cov_annual.to_numpy()
    mu_v = mu.to_numpy()
    vols = np.sqrt(np.diag(sigma))
    starts = _starts(tickers, c, n_starts)

    exact = None
    if strategy == "max_sharpe" and not c.turnover_active():
        exact = _max_sharpe_exact(mu, sigma, rf, tickers, c)

    if strategy == "min_cvar":
        if returns is None:
            raise ValueError("Minimum CVaR needs daily returns.")
        w = _min_cvar(returns, tickers, c)
    elif exact is not None:
        w = exact
    elif strategy == "hrp":
        w = project(hrp_weights(sigma), tickers, c)
    else:
        if strategy == "max_sharpe":
            def obj(w):
                return -(w @ mu_v - rf) / np.sqrt(w @ sigma @ w)
        elif strategy == "min_volatility":
            def obj(w):
                return w @ sigma @ w * 100
        elif strategy == "risk_parity":
            def obj(w):
                rc = w * (sigma @ w)
                return np.sum((rc / rc.sum() - 1 / n) ** 2) * 1e3
            # Inverse-vol is the closed-form answer when correlations are equal.
            inv = (1 / vols) / (1 / vols).sum()
            starts = [np.clip(inv, c.min_weight, c.max_weight) / np.clip(inv, c.min_weight, c.max_weight).sum()] + starts
        elif strategy == "max_diversification":
            def obj(w):
                return -(w @ vols) / np.sqrt(w @ sigma @ w)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        w = _solve(obj, tickers, c, starts)

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


# --------------------------------------------------------------------------
# Robustness: resampled optimisation
# --------------------------------------------------------------------------

@dataclass
class ResampleResult:
    samples: pd.DataFrame  # one row of optimal weights per resampled history
    mean: pd.Series  # Michaud-style averaged weights (feasible: the limits are convex)
    low: pd.Series  # 10th percentile
    high: pd.Series  # 90th percentile
    held: pd.Series  # share of samples where the asset gets more than 1%

    @property
    def spread(self) -> pd.Series:
        return self.high - self.low


def block_bootstrap(returns: pd.DataFrame, rng: np.random.Generator, block: int = 21) -> pd.DataFrame:
    """Resample whole blocks of consecutive days so volatility clusters survive."""
    t = len(returns)
    n_blocks = int(np.ceil(t / block))
    starts = rng.integers(0, max(t - block, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:t]
    return returns.iloc[np.clip(idx, 0, t - 1)]


def resample(
    returns: pd.DataFrame,
    strategy: str,
    rf: float,
    constraints: Constraints,
    n_samples: int = 100,
    mu_method: str = "bayes_stein",
    seed: int = 7,
    estimator: Estimator | None = None,
) -> ResampleResult:
    """Re-run the optimiser on ``n_samples`` bootstrapped histories.

    A weight that swings wildly between samples is driven by noise in the
    estimates; averaging the samples (Michaud resampling) gives a target
    that is far less sensitive to that noise.
    """
    rng = np.random.default_rng(seed)
    est = estimator or Estimator(return_model=mu_method, rf=rf)
    tickers = list(returns.columns)
    check_feasible(tickers, constraints)
    rows = []
    for _ in range(n_samples):
        sample = block_bootstrap(returns, rng)
        mu, cov = est.fit(sample)
        try:
            res = optimize(mu, cov * TRADING_DAYS, strategy, rf, constraints=constraints, returns=sample,
                           n_starts=2, check=False)
        except ValueError:
            continue
        rows.append(res.weights)
    if not rows:
        raise ValueError("Resampling failed: no bootstrapped history produced a feasible portfolio.")
    samples = pd.DataFrame(rows).reset_index(drop=True)
    mean = samples.mean()
    return ResampleResult(samples, mean / mean.sum(), samples.quantile(0.10), samples.quantile(0.90),
                          (samples > 0.01).mean())


# --------------------------------------------------------------------------
# Frontier & cloud
# --------------------------------------------------------------------------

def efficient_frontier(
    mu: pd.Series, cov_annual: pd.DataFrame, min_weight: float = 0.0, max_weight: float = 1.0,
    points: int = 40, constraints: Constraints | None = None,
) -> pd.DataFrame:
    """Minimum-volatility portfolio for a grid of target returns.

    Turnover limits are ignored here: the frontier shows what the limits on
    the *portfolio* allow, wherever you start from.
    """
    tickers = list(mu.index)
    c = constraints or Constraints(min_weight=min_weight, max_weight=max_weight)
    c = Constraints(c.min_weight, c.max_weight, c.asset_classes, c.class_cap)  # drop turnover
    check_feasible(tickers, c)
    n = len(tickers)
    sigma = cov_annual.to_numpy()
    mu_v = mu.to_numpy()

    w_min = optimize(mu, cov_annual, "min_volatility", 0, constraints=c).weights.to_numpy()
    # Highest achievable return under the limits (an LP).
    rows, rhs = _linear_ub(tickers, c)
    lp = linprog(-mu_v, A_ub=np.array(rows) if rows else None, b_ub=np.array(rhs) if rhs else None,
                 A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(c.min_weight, c.max_weight)] * n, method="highs")
    lo_ret = float(w_min @ mu_v)
    hi_ret = float(-lp.fun) if lp.success else lo_ret

    base = _slsqp_constraints(tickers, c)
    out = []
    prev = w_min
    for target in np.linspace(lo_ret, hi_ret, points):
        cons = base + [{"type": "eq", "fun": lambda w, t=target: w @ mu_v - t}]
        res = minimize(lambda w: w @ sigma @ w * 100, prev, method="SLSQP",
                       bounds=[(c.min_weight, c.max_weight)] * n, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        if not res.success:
            continue
        prev = res.x
        out.append({"return": float(res.x @ mu_v), "volatility": float(np.sqrt(res.x @ sigma @ res.x)),
                    **{t: float(x) for t, x in zip(tickers, res.x)}})
    return pd.DataFrame(out)


def random_portfolios(mu: pd.Series, cov_annual: pd.DataFrame, rf: float, n: int = 3000,
                      seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.full(len(mu), 0.7), size=n)
    sigma = cov_annual.to_numpy()
    ret = w @ mu.to_numpy()
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, sigma, w))
    return pd.DataFrame({"return": ret, "volatility": vol, "sharpe": (ret - rf) / vol})
