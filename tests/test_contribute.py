import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.config import DEFAULT_PORTFOLIO
from portfolio_optimizer.contribute import money_to_reach_target, plan_contribution
from portfolio_optimizer.data import demo_prices
from portfolio_optimizer.report import Settings, analyze


@pytest.fixture(scope="module")
def report():
    prices = demo_prices(list(DEFAULT_PORTFOLIO), start="2022-01-01", end="2026-09-30")
    return analyze(DEFAULT_PORTFOLIO, prices, Settings(run_backtest=False, n_resamples=20))


@pytest.mark.parametrize("amount", [50_000, 500_000, 5_000_000])
def test_buys_only_and_gets_closer(report, amount):
    plan = plan_contribution(report.recommendations, report.target.weights, amount)
    assert (plan.buys["units"] > 0).all()  # never sells
    assert plan.spent <= amount + 1e-6 and plan.left_over >= -1e-6
    assert plan.left_over < plan.buys["price"].max() if not plan.buys.empty else True
    assert plan.gap_after <= plan.gap_before + 1e-9
    assert plan.weights_after.sum() == pytest.approx(1)


def test_money_goes_to_underweight_holdings_first(report):
    plan = plan_contribution(report.recommendations, report.target.weights, 100_000)
    under = {r.ticker for r in report.recommendations if r.target_weight > r.current_weight + 1e-9}
    assert set(plan.buys["ticker"]) <= under


def test_money_to_reach_target_formula():
    values = pd.Series({"A": 70.0, "B": 30.0})
    target = pd.Series({"A": 0.5, "B": 0.5})
    need = money_to_reach_target(values, target)  # A must be <= 50%: 70 / 0.5 - 100 = 40
    assert need == pytest.approx(40)
    after = values + pd.Series({"A": 0.0, "B": need})
    assert np.allclose(after / after.sum(), target)


def test_exits_are_reported_as_stuck(report):
    target = report.target.weights.copy()
    hal = "HAL.NS"
    target[hal] = 0.0
    target /= target.sum()
    plan = plan_contribution(report.recommendations, target, 100_000)
    assert "Hindustan Aeronautics" in plan.stuck and hal not in set(plan.buys["ticker"])
