import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer import analytics as an
from portfolio_optimizer.backtest import walk_forward
from portfolio_optimizer.config import DEFAULT_PORTFOLIO, TRADING_DAYS
from portfolio_optimizer.data import DataError, clean_prices, demo_prices
from portfolio_optimizer.formatting import fmt_compact, fmt_money
from portfolio_optimizer.indicators import rsi, technical_snapshot
from portfolio_optimizer.optimizer import efficient_frontier, optimize
from portfolio_optimizer.reasons import decide_action
from portfolio_optimizer.report import Settings, analyze

TICKERS = list(DEFAULT_PORTFOLIO)


@pytest.fixture(scope="module")
def prices():
    return demo_prices(TICKERS + ["^NSEI"], start="2022-01-01", end="2026-09-30")


@pytest.fixture(scope="module")
def estimates(prices):
    rets = an.daily_returns(prices[TICKERS])
    cov = an.ledoit_wolf_covariance(rets)
    return an.expected_returns(rets, cov), cov * TRADING_DAYS


# --- indicators -------------------------------------------------------------

def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 100, dtype=float))
    assert rsi(up).dropna().eq(100).all()
    noisy = pd.Series(100 + np.random.default_rng(0).standard_normal(500).cumsum())
    r = rsi(noisy).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_technical_score_range(prices):
    for t in TICKERS:
        snap = technical_snapshot(t, prices[t])
        assert -100 <= snap.score <= 100
        assert snap.signals


# --- estimators -------------------------------------------------------------

def test_ledoit_wolf_is_psd_and_keeps_variances(prices):
    rets = an.daily_returns(prices[TICKERS])
    cov = an.ledoit_wolf_covariance(rets)
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > 0
    np.testing.assert_allclose(np.diag(cov), rets.var().to_numpy(), rtol=1e-10)
    assert 0 <= cov.attrs["shrinkage"] <= 1


def test_bayes_stein_shrinks_dispersion(prices):
    rets = an.daily_returns(prices[TICKERS])
    raw = an.expected_returns(rets, method="historical")
    shrunk = an.expected_returns(rets, an.ledoit_wolf_covariance(rets), "bayes_stein")
    assert shrunk.std() < raw.std()


# --- optimiser --------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["max_sharpe", "min_volatility", "risk_parity", "max_diversification"])
def test_weights_respect_constraints(estimates, strategy):
    mu, cov = estimates
    res = optimize(mu, cov, strategy, rf=0.065, min_weight=0.02, max_weight=0.4)
    assert res.weights.sum() == pytest.approx(1, abs=1e-8)
    assert (res.weights >= 0.02 - 1e-6).all() and (res.weights <= 0.4 + 1e-6).all()


def test_max_sharpe_beats_equal_weight(estimates):
    mu, cov = estimates
    best = optimize(mu, cov, "max_sharpe", rf=0.065)
    eq = np.full(len(mu), 1 / len(mu))
    assert best.sharpe >= an.sharpe_ratio(eq, mu, cov, 0.065) - 1e-9


def test_min_vol_beats_random_portfolios(estimates):
    mu, cov = estimates
    best = optimize(mu, cov, "min_volatility")
    w = np.random.default_rng(1).dirichlet(np.ones(len(mu)), 200)
    assert best.volatility <= min(an.portfolio_volatility(x, cov) for x in w) + 1e-6


def test_risk_parity_equalises_contributions(estimates):
    mu, cov = estimates
    res = optimize(mu, cov, "risk_parity")
    rc = an.risk_contributions(res.weights.to_numpy(), cov)
    assert rc.max() - rc.min() < 0.01


def test_infeasible_cap_raises(estimates):
    mu, cov = estimates
    with pytest.raises(ValueError, match="Raise the max weight"):
        optimize(mu, cov, max_weight=0.1)


def test_frontier_is_monotone(estimates):
    mu, cov = estimates
    f = efficient_frontier(mu, cov, max_weight=0.5, points=15)
    assert len(f) > 5
    assert f["return"].is_monotonic_increasing
    assert (np.diff(f["volatility"]) > -1e-6).all()


# --- explanations -----------------------------------------------------------

@pytest.mark.parametrize("strategy", ["max_sharpe", "min_volatility", "max_diversification"])
def test_attribution_sums_to_total_change(estimates, strategy):
    mu, cov = estimates
    f, _ = an.strategy_objective(strategy, mu, cov, 0.065)
    w0 = np.array([0.1, 0.1, 0.05, 0.7, 0.05])
    w1 = optimize(mu, cov, strategy, 0.065, 0, 0.4).weights.to_numpy()
    parts = an.attribute_change(f, w0, w1)
    assert parts.sum() == pytest.approx(f(w1) - f(w0), abs=1e-9)


def test_decide_action():
    assert decide_action(0.30, 0.31) == "Hold"
    assert decide_action(0.30, 0.40) == "Buy more"
    assert decide_action(0.30, 0.10) == "Trim"
    assert decide_action(0.30, 0.0) == "Exit"
    assert decide_action(0.0, 0.10) == "Initiate"


@pytest.mark.parametrize("strategy", ["max_sharpe", "min_volatility", "risk_parity", "max_diversification"])
def test_report_is_consistent(prices, strategy):
    r = analyze(DEFAULT_PORTFOLIO, prices[TICKERS], Settings(strategy=strategy, run_backtest=False),
                prices["^NSEI"])
    assert r.target.weights.sum() == pytest.approx(1)
    assert sum(x.target_value for x in r.recommendations) == pytest.approx(r.total_value)
    for rec in r.recommendations:
        assert rec.reasons and rec.reasons[0].category == "Optimiser"
        if rec.action in ("Buy more", "Initiate"):
            assert rec.trade_qty >= 0 and rec.trade_value > 0
        if rec.action in ("Trim", "Exit"):
            assert rec.trade_qty <= 0 and rec.trade_value < 0
        if rec.action == "Hold":
            assert rec.trade_qty == 0
        # The headline reason must never argue against the action it explains.
        text = rec.reasons[0].text
        if rec.action in ("Buy more", "Initiate"):
            assert not text.startswith("Reducing")
        if rec.action in ("Trim", "Exit"):
            assert not text.startswith("Adding")
    assert r.summary


def test_rebalance_intensity_blends(prices):
    full = analyze(DEFAULT_PORTFOLIO, prices[TICKERS], Settings(run_backtest=False))
    half = analyze(DEFAULT_PORTFOLIO, prices[TICKERS], Settings(run_backtest=False, rebalance_intensity=0.5))
    expected = (full.current.weights + full.optimal.weights) / 2
    np.testing.assert_allclose(half.target.weights, expected, atol=1e-9)


def test_zero_quantity_ticker_can_be_initiated(prices):
    holdings = {**DEFAULT_PORTFOLIO, "GOLDETF.NS": {"qty": 0, "avg_price": 0}}
    r = analyze(holdings, prices[TICKERS], Settings(run_backtest=False))
    gold = next(x for x in r.recommendations if x.ticker == "GOLDETF.NS")
    assert gold.action in ("Initiate", "Hold")


# --- backtest ---------------------------------------------------------------

def test_walk_forward_has_no_lookahead(prices):
    p = prices[TICKERS]
    w = pd.Series(1 / len(TICKERS), index=TICKERS)
    base = walk_forward(p, w, rf=0.065, max_weight=0.5)
    shocked = p.copy()
    shocked.iloc[400:] *= np.linspace(1, 3, len(shocked) - 400)[:, None]  # change only the future
    alt = walk_forward(shocked, w, rf=0.065, max_weight=0.5)
    first = base.weights_history.index[0]
    pd.testing.assert_series_equal(base.weights_history.loc[first], alt.weights_history.loc[first])
    assert set(base.stats.index) >= {"Optimised (walk-forward)", "Your current mix", "Equal weight"}


# --- data & formatting ------------------------------------------------------

def test_clean_prices_requires_overlap():
    short = pd.DataFrame({"A": range(10)}, index=pd.bdate_range("2024-01-01", periods=10), dtype=float)
    with pytest.raises(DataError):
        clean_prices(short)


def test_indian_number_formatting():
    assert fmt_money(1234567, "₹") == "₹12,34,567"
    assert fmt_money(-950, "₹") == "-₹950"
    assert fmt_compact(2_310_000, "₹") == "₹23.10L"
    assert fmt_compact(12_500_000, "₹") == "₹1.25Cr"
    assert fmt_compact(1_500_000, "$") == "$1.50M"
