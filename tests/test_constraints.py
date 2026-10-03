import pandas as pd
import pytest

from portfolio_optimizer import analytics as an
from portfolio_optimizer.backtest import walk_forward
from portfolio_optimizer.config import DEFAULT_PORTFOLIO, STRATEGIES, TRADING_DAYS, asset_class_for
from portfolio_optimizer.data import demo_prices
from portfolio_optimizer.optimizer import Constraints, min_feasible_turnover, optimize, resample
from portfolio_optimizer.reasons import cash_flow
from portfolio_optimizer.report import Settings, analyze

TICKERS = list(DEFAULT_PORTFOLIO)
CLASSES = {t: asset_class_for(t) for t in TICKERS}
# HAL-heavy, like the real portfolio: 70% in one stock.
CURRENT = pd.Series([0.15, 0.13, 0.01, 0.70, 0.01], index=TICKERS)


@pytest.fixture(scope="module")
def data():
    p = demo_prices(TICKERS, start="2022-01-01", end="2026-09-30")[TICKERS]
    rets = an.daily_returns(p)
    cov = an.ledoit_wolf_covariance(rets)
    return p, rets, an.expected_returns(rets, cov), cov * TRADING_DAYS


def class_totals(w: pd.Series) -> pd.Series:
    return w.groupby(pd.Series(CLASSES)).sum()


@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_class_cap_is_respected(data, strategy):
    _, rets, mu, cov = data
    c = Constraints(max_weight=0.5, asset_classes=CLASSES, class_cap=0.45)
    w = optimize(mu, cov, strategy, 0.065, constraints=c, returns=rets).weights
    assert w.sum() == pytest.approx(1)
    assert (class_totals(w) <= 0.45 + 1e-6).all()


@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_turnover_limit_is_respected(data, strategy):
    _, rets, mu, cov = data
    c = Constraints(max_weight=0.75, current=CURRENT, max_turnover=0.25)
    w = optimize(mu, cov, strategy, 0.065, constraints=c, returns=rets).weights
    assert (w - CURRENT).abs().sum() / 2 <= 0.25 + 1e-4


def test_too_tight_turnover_explains_minimum(data):
    _, rets, mu, cov = data
    c = Constraints(max_weight=0.40, current=CURRENT, max_turnover=0.10)
    assert min_feasible_turnover(TICKERS, c) == pytest.approx(0.30, abs=1e-6)  # HAL 70% → 40%
    with pytest.raises(ValueError, match="at least 30%"):
        optimize(mu, cov, "max_sharpe", constraints=c)


def test_infeasible_class_cap_raises(data):
    _, _, mu, cov = data
    with pytest.raises(ValueError, match="asset-class cap"):
        optimize(mu, cov, constraints=Constraints(asset_classes=CLASSES, class_cap=0.30))


def test_min_cvar_has_lowest_tail_risk(data):
    _, rets, mu, cov = data
    c = Constraints(max_weight=0.6)
    best = an.historical_cvar(optimize(mu, cov, "min_cvar", constraints=c, returns=rets).weights.to_numpy(), rets)
    for s in ("max_sharpe", "min_volatility", "risk_parity"):
        other = optimize(mu, cov, s, 0.065, constraints=c).weights.to_numpy()
        assert best <= an.historical_cvar(other, rets) + 1e-9


def test_resampled_weights_stay_inside_limits(data):
    _, rets, _, _ = data
    c = Constraints(max_weight=0.4, asset_classes=CLASSES, class_cap=0.6)
    rs = resample(rets, "max_sharpe", 0.065, c, n_samples=25)
    assert rs.mean.sum() == pytest.approx(1)
    assert (rs.mean <= 0.4 + 1e-6).all()
    assert (class_totals(rs.mean) <= 0.6 + 1e-6).all()
    assert (rs.low <= rs.high).all() and ((rs.held >= 0) & (rs.held <= 1)).all()


def test_cvar_attribution_sums_to_total(data):
    _, rets, mu, cov = data
    f, higher = an.strategy_objective("min_cvar", mu, cov, 0.065, rets)
    assert not higher
    w1 = optimize(mu, cov, "min_cvar", constraints=Constraints(max_weight=0.5), returns=rets).weights.to_numpy()
    parts = an.attribute_change(f, CURRENT.to_numpy(), w1)
    assert parts.sum() == pytest.approx(f(w1) - f(CURRENT.to_numpy()), abs=1e-9)


def test_backtest_first_rebalance_respects_turnover(data):
    p, *_ = data
    bt = walk_forward(p, CURRENT, "max_sharpe", 0.065, max_weight=0.75,
                      constraints=Constraints(max_weight=0.75, max_turnover=0.2))
    first = bt.weights_history.iloc[0]
    assert (first - CURRENT).abs().sum() / 2 <= 0.2 + 1e-4


def test_report_with_all_limits(data):
    p, *_ = data
    s = Settings(strategy="max_sharpe", max_weight=0.5, class_cap=0.6, max_turnover=0.5,
                 no_trade_band=0.05, n_resamples=20, run_backtest=False)
    r = analyze(DEFAULT_PORTFOLIO, p, s)
    assert (class_totals(r.optimal.weights) <= 0.6 + 1e-6).all()
    assert (r.optimal.weights - r.current.weights).abs().sum() / 2 <= 0.5 + 1e-4
    for rec in r.recommendations:
        assert rec.confidence in ("High", "Medium", "Low")
        if rec.action == "Hold":
            assert abs(rec.delta_weight) < 0.05
    cash = cash_flow(r.recommendations)
    assert cash["net"] == pytest.approx(sum(x.trade_value for x in r.recommendations))
    assert set(r.comparison.index) >= {"Your current mix", *STRATEGIES.values()}


def test_holding_class_override(data):
    p, *_ = data
    holdings = {t: dict(v) for t, v in DEFAULT_PORTFOLIO.items()}
    holdings["MAFANG.NS"]["class"] = "Indian equity"
    r = analyze(holdings, p, Settings(robust=False, run_backtest=False))
    assert r.asset_classes["MAFANG.NS"] == "Indian equity"
    assert next(x for x in r.recommendations if x.ticker == "MAFANG.NS").asset_class == "Indian equity"
