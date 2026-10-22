import pandas as pd
import pytest

from portfolio_optimizer.config import DEFAULT_PORTFOLIO
from portfolio_optimizer.data import demo_prices
from portfolio_optimizer.orders import _split, build_orders, order_ticket, orders_frame
from portfolio_optimizer.report import Settings, analyze

MONDAY = pd.Timestamp("2026-10-26")


@pytest.fixture(scope="module")
def recs():
    prices = demo_prices(list(DEFAULT_PORTFOLIO), start="2022-01-01", end="2026-09-30")
    return analyze(DEFAULT_PORTFOLIO, prices, Settings(run_backtest=False, n_resamples=20)).recommendations


def test_split_keeps_every_unit():
    assert _split(10, 3) == [4, 3, 3] and sum(_split(433, 3)) == 433
    assert _split(2, 3) == [1, 1]  # never more tranches than units


def test_orders_cover_the_whole_plan(recs):
    orders = build_orders(recs, MONDAY)
    for r in recs:
        traded = sum(o.units for o in orders if o.ticker == r.ticker)
        assert traded == abs(r.trade_qty)
        assert all((o.side == "Sell") == (r.trade_qty < 0) for o in orders if o.ticker == r.ticker)
    final_cash = sum(o.value if o.side == "Sell" else -o.value for o in orders)
    assert orders[-1].cash_after == pytest.approx(final_cash)


def test_sells_come_first_each_day(recs):
    orders = build_orders(recs, MONDAY)
    for day in {o.when for o in orders}:
        sides = [o.side for o in orders if o.when == day]
        assert sides == sorted(sides, key=lambda s: s != "Sell")
    assert all(o.when.dayofweek < 5 for o in orders)


def test_no_stagger_puts_everything_on_day_one(recs):
    orders = build_orders(recs, MONDAY, tranches=1)
    assert {o.when for o in orders} == {MONDAY}
    assert len(orders) == sum(1 for r in recs if r.trade_qty)


def test_exports(recs):
    orders = build_orders(recs, MONDAY)
    df = orders_frame(orders)
    assert list(df["Step"]) == list(range(1, len(orders) + 1))
    ticket = order_ticket(orders, "₹")
    assert "Mon 26 Oct 2026" in ticket and "Not investment advice" in ticket
    assert order_ticket([], "₹").endswith("No trades needed.")
