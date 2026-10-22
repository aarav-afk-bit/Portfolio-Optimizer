import pytest

from portfolio_optimizer.config import DEFAULT_PORTFOLIO, asset_class_for
from portfolio_optimizer.data import demo_prices
from portfolio_optimizer.presets import PRESETS, fit_to_holdings
from portfolio_optimizer.report import Settings, analyze

TICKERS = list(DEFAULT_PORTFOLIO)
CLASSES = {t: asset_class_for(t) for t in TICKERS}


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_run_on_default_portfolio(name):
    preset, notes = fit_to_holdings(PRESETS[name], TICKERS, CLASSES)
    assert preset == PRESETS[name] and notes == []  # no loosening needed for 5 holdings in 3 classes
    prices = demo_prices(TICKERS, start="2022-01-01", end="2026-09-30")
    s = Settings(strategy=preset.strategy, max_weight=preset.max_weight, class_cap=preset.class_cap,
                 rebalance_intensity=preset.rebalance_intensity, no_trade_band=preset.no_trade_band,
                 run_backtest=False, n_resamples=20)
    r = analyze(DEFAULT_PORTFOLIO, prices, s)
    assert (r.optimal.weights <= preset.max_weight + 1e-6).all()


def test_presets_get_riskier_in_order():
    c, b, g = PRESETS["Cautious"], PRESETS["Balanced"], PRESETS["Growth"]
    assert c.max_weight < b.max_weight < g.max_weight
    assert c.rebalance_intensity < b.rebalance_intensity <= g.rebalance_intensity


def test_fit_raises_position_cap_for_few_holdings():
    tickers = ["HAL.NS", "GOLDETF.NS"]  # two classes: a 60% class cap still works with 50% positions
    preset, notes = fit_to_holdings(PRESETS["Cautious"], tickers, {t: asset_class_for(t) for t in tickers})
    assert preset.max_weight == 0.5 and preset.class_cap == 0.6
    assert len(notes) == 1 and "30%" in notes[0]


def test_fit_drops_class_cap_when_one_class():
    tickers = ["HAL.NS", "HDFCSML250.NS", "MAFANG.NS"]
    classes = {t: "Equity" for t in tickers}  # everything in one class: a 60% cap can't hold 100%
    preset, notes = fit_to_holdings(PRESETS["Cautious"], tickers, classes)
    assert preset.class_cap is None and any("class cap is off" in n for n in notes)
