"""Return and risk models, bundled into one ``Estimator`` used everywhere.

The report, the resampler and the walk-forward backtest all call
``Estimator.fit`` so a single setting changes the model consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import analytics as an
from .config import TRADING_DAYS

RETURN_MODELS = {
    "black_litterman": "Black–Litterman (equilibrium + your data)",
    "bayes_stein": "Bayes–Stein shrinkage",
    "historical": "Historical average (raw)",
}
RISK_MODELS = {
    "ledoit_wolf": "Ledoit–Wolf shrinkage",
    "ewma": "EWMA (recent volatility weighted more)",
}


@dataclass(frozen=True)
class View:
    """An absolute view: ``ticker`` returns ``expected`` a year, held with ``confidence`` in (0, 1]."""

    ticker: str
    expected: float
    confidence: float = 0.5


# --------------------------------------------------------------------------
# Risk models
# --------------------------------------------------------------------------

def ewma_covariance(returns: pd.DataFrame, halflife: float = 126) -> pd.DataFrame:
    """Daily covariance with exponentially weighted volatilities.

    Recent volatility counts more (half-life in trading days), while the
    correlations come from the Ledoit–Wolf-shrunk full sample: correlations
    need far more data than volatilities to estimate reliably.
    """
    lw = an.ledoit_wolf_covariance(returns)
    std_lw = np.sqrt(np.diag(lw))
    corr = lw.to_numpy() / np.outer(std_lw, std_lw)

    x = returns.to_numpy()
    t = len(x)
    decay = 0.5 ** (1 / halflife)
    w = decay ** np.arange(t - 1, -1, -1)
    w /= w.sum()
    mean = w @ x
    var = w @ (x - mean) ** 2
    std = np.sqrt(var)
    out = pd.DataFrame(corr * np.outer(std, std), index=returns.columns, columns=returns.columns)
    out.attrs["shrinkage"] = lw.attrs.get("shrinkage")
    out.attrs["halflife"] = halflife
    return out


# --------------------------------------------------------------------------
# Black–Litterman
# --------------------------------------------------------------------------

def black_litterman(
    returns: pd.DataFrame,
    cov_daily: pd.DataFrame,
    rf: float,
    views: tuple[View, ...] = (),
    prior_sharpe: float = 0.4,
    tau: float = 0.05,
    prior_weights: pd.Series | None = None,
    use_sample: bool = True,
) -> pd.Series:
    """Annual expected returns from a Black–Litterman posterior.

    Prior (equilibrium): the returns that would make ``prior_weights``
    (equal weights by default) the max-Sharpe portfolio with a long-run
    Sharpe ratio of ``prior_sharpe``: ``pi = delta * Sigma @ w``.

    Views: when ``use_sample`` is on, the sample mean of every asset is a
    view whose uncertainty is its sampling error (``Sigma / years``), so
    more history earns more trust. User ``views`` are added on top with
    Idzorek-style confidence. With ``tau`` = 0.05 and 3–4 years of data the
    sample gets roughly 15% weight against the prior.

    ``attrs`` carries ``prior``, ``sample`` and ``data_weight`` per asset so
    the reasons can explain where each estimate came from.
    """
    tickers = list(returns.columns)
    n = len(tickers)
    sigma = cov_daily.to_numpy() * TRADING_DAYS
    w_p = (prior_weights.reindex(tickers).fillna(0).to_numpy() if prior_weights is not None
           else np.full(n, 1 / n))
    w_p = w_p / w_p.sum()
    vol_p = float(np.sqrt(w_p @ sigma @ w_p))
    delta = prior_sharpe / vol_p if vol_p > 0 else 0.0
    pi = delta * sigma @ w_p  # equilibrium excess returns

    years = len(returns) / TRADING_DAYS
    sample_excess = returns.mean().to_numpy() * TRADING_DAYS - rf

    p_rows, q, omega_blocks = [], [], []
    if use_sample and years > 0:
        p_rows.append(np.eye(n))
        q.append(sample_excess)
        omega_blocks.append(sigma / years)
    for v in views:
        if v.ticker not in tickers:
            continue
        row = np.zeros((1, n))
        row[0, tickers.index(v.ticker)] = 1
        c = float(np.clip(v.confidence, 1e-3, 0.999))
        p_rows.append(row)
        q.append(np.array([v.expected - rf]))
        omega_blocks.append(np.array([[(1 - c) / c * tau * (row @ sigma @ row.T).item()]]))

    if p_rows:
        p_mat = np.vstack(p_rows)
        q_vec = np.concatenate(q)
        k = sum(b.shape[0] for b in omega_blocks)
        omega = np.zeros((k, k))
        i = 0
        for b in omega_blocks:
            omega[i:i + b.shape[0], i:i + b.shape[0]] = b
            i += b.shape[0]
        ts_inv = np.linalg.inv(tau * sigma)
        om_inv = np.linalg.inv(omega)
        post = np.linalg.solve(ts_inv + p_mat.T @ om_inv @ p_mat, ts_inv @ pi + p_mat.T @ om_inv @ q_vec)
    else:
        post = pi

    out = pd.Series(rf + post, index=tickers)
    out.attrs.update(
        prior=pd.Series(rf + pi, index=tickers),
        sample=pd.Series(rf + sample_excess, index=tickers),
        data_weight=(years * tau) / (1 + years * tau) if use_sample else 0.0,
        views=tuple(v for v in views if v.ticker in tickers),
        prior_sharpe=prior_sharpe,
    )
    return out


# --------------------------------------------------------------------------
# One object for the whole pipeline
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Estimator:
    return_model: str = "black_litterman"
    risk_model: str = "ledoit_wolf"
    rf: float = 0.065
    ewma_halflife: float = 126
    prior_sharpe: float = 0.4
    tau: float = 0.05
    views: tuple[View, ...] = field(default_factory=tuple)

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        if self.risk_model == "ledoit_wolf":
            return an.ledoit_wolf_covariance(returns)
        if self.risk_model == "ewma":
            return ewma_covariance(returns, self.ewma_halflife)
        raise ValueError(f"Unknown risk model: {self.risk_model}")

    def fit(self, returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        """Returns (annual expected returns, daily covariance)."""
        cov = self.covariance(returns)
        if self.return_model == "black_litterman":
            mu = black_litterman(returns, cov, self.rf, self.views, self.prior_sharpe, self.tau)
        else:
            mu = an.expected_returns(returns, cov, self.return_model)
        return mu, cov
