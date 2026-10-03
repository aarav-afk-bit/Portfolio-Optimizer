"""Return / risk estimation and portfolio statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna(how="any")


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Daily covariance with Ledoit–Wolf shrinkage applied to the correlations.

    Correlations estimated from a few years of data are noisy, and a
    mean-variance optimizer leans hardest on exactly those errors. We shrink
    the sample correlation matrix toward the identity with the Ledoit–Wolf
    (2004) optimal intensity and keep each asset's own volatility intact.
    """
    x = returns.to_numpy()
    t, n = x.shape
    std = x.std(axis=0, ddof=0)
    z = (x - x.mean(axis=0)) / std
    s = z.T @ z / t  # sample correlation matrix

    m = np.trace(s) / n
    target = m * np.eye(n)
    d2 = np.sum((s - target) ** 2) / n
    # sum_t ||z_t z_t' - S||^2 simplifies to sum_t ||z_t||^4 - t * ||S||^2 (no per-row loop).
    b2_bar = (np.sum(np.sum(z**2, axis=1) ** 2) - t * np.sum(s**2)) / n / t**2
    b2 = min(b2_bar, d2)
    shrink = b2 / d2 if d2 > 0 else 1.0
    corr = shrink * target + (1 - shrink) * s
    np.fill_diagonal(corr, 1.0)

    cov = corr * np.outer(std, std) * t / (t - 1)
    out = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    out.attrs["shrinkage"] = float(shrink)
    return out


def expected_returns(
    returns: pd.DataFrame, cov: pd.DataFrame | None = None, method: str = "bayes_stein"
) -> pd.Series:
    """Annualised expected returns.

    * ``historical`` – arithmetic mean × 252 (what the original notebook used).
    * ``bayes_stein`` – Jorion (1986) shrinkage of each mean toward the
      minimum-variance portfolio's mean. Historical averages over 2–4 years
      are extremely noisy; shrinking them stops the optimizer from going
      all-in on whatever happened to rally most in the sample.
    """
    mu = returns.mean()
    if method == "historical":
        return mu * TRADING_DAYS
    if method != "bayes_stein":
        raise ValueError(f"Unknown expected-return method: {method}")

    t, n = returns.shape
    sigma = (cov if cov is not None else returns.cov()).to_numpy()
    inv = np.linalg.pinv(sigma)
    ones = np.ones(n)
    mu0 = float(ones @ inv @ mu.to_numpy() / (ones @ inv @ ones))
    diff = mu.to_numpy() - mu0
    denom = float(diff @ inv @ diff)
    lam = (n + 2) / denom if denom > 0 else np.inf
    w = lam / (lam + t) if np.isfinite(lam) else 1.0
    shrunk = (1 - w) * mu + w * mu0
    out = shrunk * TRADING_DAYS
    out.attrs["shrinkage"] = float(w)
    return out


# --------------------------------------------------------------------------
# Portfolio statistics
# --------------------------------------------------------------------------

def portfolio_return(w: np.ndarray, mu: pd.Series) -> float:
    return float(np.dot(w, mu))


def portfolio_volatility(w: np.ndarray, cov_annual: pd.DataFrame) -> float:
    return float(np.sqrt(max(w @ cov_annual.to_numpy() @ w, 0.0)))


def sharpe_ratio(w: np.ndarray, mu: pd.Series, cov_annual: pd.DataFrame, rf: float) -> float:
    vol = portfolio_volatility(w, cov_annual)
    return (portfolio_return(w, mu) - rf) / vol if vol > 0 else 0.0


def risk_contributions(w: np.ndarray, cov_annual: pd.DataFrame) -> pd.Series:
    """Share of total portfolio variance coming from each asset (sums to 1)."""
    sigma = cov_annual.to_numpy()
    var = float(w @ sigma @ w)
    rc = w * (sigma @ w) / var if var > 0 else np.zeros_like(w)
    return pd.Series(rc, index=cov_annual.index)


def historical_cvar(w: np.ndarray, returns: pd.DataFrame | np.ndarray, level: float = 0.95) -> float:
    """Average daily loss on the worst (1 - level) share of days (positive = loss)."""
    r = np.asarray(returns) @ w
    k = max(int(np.ceil(len(r) * (1 - level))), 1)
    return float(-np.sort(r)[:k].mean())


def diversification_ratio(w: np.ndarray, cov_annual: pd.DataFrame) -> float:
    vols = np.sqrt(np.diag(cov_annual.to_numpy()))
    pvol = portfolio_volatility(w, cov_annual)
    return float(w @ vols / pvol) if pvol > 0 else 1.0


def effective_holdings(w: np.ndarray) -> float:
    """Inverse Herfindahl index: 1 = everything in one asset, N = equal weights."""
    return float(1 / np.sum(np.square(w))) if np.sum(w) > 0 else 0.0


@dataclass
class SeriesStats:
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    var_95: float  # one-day historical Value-at-Risk (positive number = loss)
    cvar_95: float  # average loss on the worst 5% of days
    beta: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return self.__dict__.copy()


def drawdown(values: pd.Series) -> pd.Series:
    return values / values.cummax() - 1


def series_stats(
    daily: pd.Series, rf: float, benchmark_daily: pd.Series | None = None
) -> SeriesStats:
    """Realised statistics of a daily-return series."""
    daily = daily.dropna()
    growth = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS
    cagr = float(growth.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
    vol = float(daily.std() * np.sqrt(TRADING_DAYS))
    rf_daily = (1 + rf) ** (1 / TRADING_DAYS) - 1
    excess = daily - rf_daily
    sharpe = float(excess.mean() / daily.std() * np.sqrt(TRADING_DAYS)) if daily.std() > 0 else 0.0
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    sortino = float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else 0.0
    mdd = float(drawdown(growth).min())
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    q = daily.quantile(0.05)
    var_95 = float(-q)
    cvar_95 = float(-daily[daily <= q].mean())

    beta = None
    if benchmark_daily is not None:
        aligned = pd.concat([daily, benchmark_daily], axis=1, join="inner").dropna()
        if len(aligned) > 20 and aligned.iloc[:, 1].var() > 0:
            beta = float(aligned.cov().iloc[0, 1] / aligned.iloc[:, 1].var())

    return SeriesStats(cagr, vol, sharpe, sortino, mdd, calmar, var_95, cvar_95, beta)


# --------------------------------------------------------------------------
# Explaining a rebalance
# --------------------------------------------------------------------------

def strategy_objective(strategy: str, mu: pd.Series, cov_annual: pd.DataFrame, rf: float,
                       returns: pd.DataFrame | None = None):
    """The quantity each strategy improves, as a function of weights.

    Returns ``(f, higher_is_better)``. ``returns`` (daily) is needed for min_cvar.
    """
    sigma = cov_annual.to_numpy()
    mu_v = mu.to_numpy()
    vols = np.sqrt(np.diag(sigma))

    def vol(w):
        return float(np.sqrt(max(w @ sigma @ w, 1e-18)))

    if strategy == "max_sharpe":
        return (lambda w: (w @ mu_v - rf) / vol(w)), True
    if strategy == "min_volatility":
        return vol, False
    if strategy == "max_diversification":
        return (lambda w: (w @ vols) / vol(w)), True
    if strategy == "min_cvar":
        if returns is None:
            raise ValueError("min_cvar needs daily returns")
        r = returns.to_numpy()
        return (lambda w: historical_cvar(w, r)), False
    if strategy == "risk_parity":
        def concentration(w):
            rc = w * (sigma @ w)
            return float(np.sum((rc / rc.sum() - 1 / len(w)) ** 2))
        return concentration, False
    raise ValueError(f"Unknown strategy: {strategy}")


def attribute_change(f, w0: np.ndarray, w1: np.ndarray, steps: int = 24, h: float = 1e-6) -> np.ndarray:
    """Split f(w1) - f(w0) into one additive contribution per asset.

    Integrates the gradient along the straight line from ``w0`` to ``w1``
    (path-integrated gradients), so the contributions sum exactly to the
    total change and each one reflects the whole move, not just the first
    step of it.
    """
    d = w1 - w0
    n = len(w0)
    acc = np.zeros(n)
    s_grid = (np.arange(steps) + 0.5) / steps  # midpoint rule
    for s in s_grid:
        w = w0 + s * d
        for i in range(n):
            e = np.zeros(n)
            e[i] = h
            acc[i] += (f(w + e) - f(w - e)) / (2 * h)
    contrib = d * acc / steps
    # Remove the tiny numerical residual so the parts add up exactly.
    total = f(w1) - f(w0)
    if np.abs(d).sum() > 0:
        contrib += (total - contrib.sum()) * np.abs(d) / np.abs(d).sum()
    return contrib
