"""Tests on real NSE prices (downloaded from GitHub; skipped when offline)."""

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer import analytics as an
from portfolio_optimizer.config import DEFAULT_PORTFOLIO, TRADING_DAYS
from portfolio_optimizer.data import fetch_prices_nse_github
from portfolio_optimizer.estimators import Estimator
from portfolio_optimizer.optimizer import Constraints, _solve, _starts, optimize
from portfolio_optimizer.report import Settings, analyze

TICKERS = list(DEFAULT_PORTFOLIO)


@pytest.fixture(scope="module")
def real():
    try:
        return fetch_prices_nse_github(TICKERS + ["^NSEI"], "2023-01-01")
    except Exception as e:  # offline or GitHub blocked
        pytest.skip(f"real NSE data unavailable: {e}")


def test_real_data_is_sane(real):
    assert (real > 0).all().all()
    assert len(real) > 500
    assert real.index.is_monotonic_increasing and not real.index.duplicated().any()
    assert an.daily_returns(real).abs().max().max() < 0.35  # no unadjusted splits


def test_exact_max_sharpe_beats_multistart_on_real_data(real):
    rets = an.daily_returns(real[TICKERS])
    mu, cov_d = Estimator().fit(rets)
    cov = cov_d * TRADING_DAYS
    c = Constraints(max_weight=0.4)
    exact = optimize(mu, cov, "max_sharpe", 0.065, constraints=c)
    sigma, m = cov.to_numpy(), mu.to_numpy()
    w = _solve(lambda w: -(w @ m - 0.065) / np.sqrt(w @ sigma @ w), TICKERS, c, _starts(TICKERS, c, 7))
    assert exact.sharpe >= an.sharpe_ratio(w, mu, cov, 0.065) - 1e-9


def test_black_litterman_estimates_are_plausible(real):
    mu, _ = Estimator(rf=0.065).fit(an.daily_returns(real[TICKERS]))
    raw = an.expected_returns(an.daily_returns(real[TICKERS]), method="historical")
    assert (mu > 0.065).all() and (mu < 0.35).all()
    assert mu.std() < raw.std()  # far less dispersed than raw sample means


@pytest.mark.parametrize("strategy", ["max_sharpe", "hrp", "min_cvar"])
def test_full_report_on_real_portfolio(real, strategy):
    r = analyze(DEFAULT_PORTFOLIO, real[TICKERS], Settings(strategy=strategy, n_resamples=30), real["^NSEI"])
    assert r.target.weights.sum() == pytest.approx(1)
    assert (r.optimal.weights <= 0.4 + 1e-6).all()
    hal = next(x for x in r.recommendations if x.ticker == "HAL.NS")
    # HAL is ~70% of this portfolio and ~88% of its risk: every strategy should cut it, and the
    # headline reason must not argue against the cut.
    assert hal.action in ("Trim", "Exit")
    assert not hal.reasons[0].text.startswith("Adding")
    assert r.backtest is not None and np.isfinite(r.backtest.stats.loc["Optimised (walk-forward)", "sharpe"])
    assert isinstance(r.as_of, pd.Timestamp)
