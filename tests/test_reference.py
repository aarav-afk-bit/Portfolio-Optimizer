"""Cross-checks against independent implementations (skipped when not installed).

    pip install cvxpy scikit-learn pyportfolioopt
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer import analytics as an
from portfolio_optimizer.config import DEFAULT_PORTFOLIO, TRADING_DAYS, asset_class_for
from portfolio_optimizer.data import demo_prices
from portfolio_optimizer.optimizer import Constraints, hrp_weights, optimize

TICKERS = list(DEFAULT_PORTFOLIO)
CLASSES = {t: asset_class_for(t) for t in TICKERS}


@pytest.fixture(scope="module")
def data():
    rets = an.daily_returns(demo_prices(TICKERS, start="2022-01-01", end="2026-09-30")[TICKERS])
    cov = an.ledoit_wolf_covariance(rets)
    return rets, an.expected_returns(rets, cov), cov * TRADING_DAYS


def test_ledoit_wolf_matches_sklearn(data):
    sk = pytest.importorskip("sklearn.covariance")
    rets, _, _ = data
    x = rets.to_numpy()
    z = (x - x.mean(0)) / x.std(0)
    ref = sk.LedoitWolf().fit(z).covariance_
    ours = an.ledoit_wolf_covariance(rets).to_numpy()
    std = np.sqrt(np.diag(ours))
    np.testing.assert_allclose(ours / np.outer(std, std), ref, atol=1e-10)


def _cvx_problem(n, lo, hi, groups, cap):
    cp = pytest.importorskip("cvxpy")
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= lo, w <= hi]
    cons += [cp.sum(w[ix]) <= cap for ix in groups]
    return cp, w, cons


@pytest.mark.parametrize("cap", [None, 0.5])
def test_max_sharpe_is_global_optimum(data, cap):
    _, mu, cov = data
    rf, lo, hi = 0.065, 0.0, 0.45
    c = Constraints(min_weight=lo, max_weight=hi, asset_classes=CLASSES, class_cap=cap)
    ours = optimize(mu, cov, "max_sharpe", rf, constraints=c)

    cp = pytest.importorskip("cvxpy")
    n = len(TICKERS)
    groups = list(c.class_groups(TICKERS).values())
    y, k = cp.Variable(n), cp.Variable()
    cons = [(mu.to_numpy() - rf) @ y == 1, cp.sum(y) == k, y >= lo * k, y <= hi * k, k >= 0]
    cons += [cp.sum(y[ix]) <= cap * k for ix in groups]
    cp.Problem(cp.Minimize(cp.quad_form(y, cov.to_numpy())), cons).solve()
    ref = y.value / k.value
    ref_sharpe = an.sharpe_ratio(ref, mu, cov, rf)
    assert ours.sharpe >= ref_sharpe - 1e-6
    np.testing.assert_allclose(ours.weights.to_numpy(), ref, atol=2e-3)


def test_min_volatility_matches_cvxpy(data):
    _, mu, cov = data
    c = Constraints(max_weight=0.4, asset_classes=CLASSES, class_cap=0.6)
    ours = optimize(mu, cov, "min_volatility", constraints=c)
    cp, w, cons = _cvx_problem(len(TICKERS), 0, 0.4, list(c.class_groups(TICKERS).values()), 0.6)
    cp.Problem(cp.Minimize(cp.quad_form(w, cov.to_numpy())), cons).solve()
    assert ours.volatility <= an.portfolio_volatility(w.value, cov) + 1e-7


def test_min_cvar_matches_cvxpy(data):
    rets, mu, cov = data
    ours = optimize(mu, cov, "min_cvar", constraints=Constraints(max_weight=0.5), returns=rets)
    cp, w, cons = _cvx_problem(len(TICKERS), 0, 0.5, [], None)
    r = rets.to_numpy()
    alpha = cp.Variable()
    cvar = alpha + cp.sum(cp.pos(-r @ w - alpha)) / (0.05 * len(r))
    cp.Problem(cp.Minimize(cvar), cons).solve()
    assert an.historical_cvar(ours.weights.to_numpy(), rets) <= an.historical_cvar(w.value, rets) + 1e-7


def test_hrp_matches_pyportfolioopt(data):
    hrp_mod = pytest.importorskip("pypfopt.hierarchical_portfolio")
    rets, _, cov = data
    # Pass only the covariance: given returns, HRPOpt would cluster on the sample covariance instead.
    ref = hrp_mod.HRPOpt(cov_matrix=cov).optimize(linkage_method="single")
    ours = hrp_weights(cov.to_numpy(), "single")
    np.testing.assert_allclose(ours, pd.Series(ref)[TICKERS].to_numpy(), atol=1e-8)
