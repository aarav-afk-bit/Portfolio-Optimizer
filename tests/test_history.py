import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.config import DEFAULT_PORTFOLIO
from portfolio_optimizer.data import clean_prices, demo_prices, history_limit
from portfolio_optimizer.plain import history_warning
from portfolio_optimizer.report import Settings, analyze

TICKERS = list(DEFAULT_PORTFOLIO)
LATE = pd.Timestamp("2024-07-08")


@pytest.fixture(scope="module")
def raw():
    """Demo prices where Silver only starts on 8 Jul 2024, like the real NSE data in Oct 2026."""
    px = demo_prices(TICKERS, start="2023-02-01", end="2026-10-13")
    px.loc[px.index < LATE, "SILVERBETA.NS"] = np.nan
    return px


def test_clean_prices_remembers_first_dates(raw):
    px = clean_prices(raw)
    assert px.index[0] == LATE
    assert px.attrs["first_valid"]["SILVERBETA.NS"] == LATE
    assert px.attrs["first_valid"]["HAL.NS"] == raw.index[0]
    assert px[TICKERS[:2]].attrs["first_valid"]  # survives column selection


def test_history_limit_names_the_late_holding(raw):
    px = clean_prices(raw)
    h = history_limit(px.attrs["first_valid"], px.index[-1])
    assert h.ticker == "SILVERBETA.NS" and h.start == LATE and h.start_without == raw.index[0]
    assert h.days_used == len(px) and 0.3 < h.share_lost < 0.5


def test_no_limit_when_starts_are_shared():
    d = pd.Timestamp("2024-01-01")
    assert history_limit({"A": d, "B": d}, pd.Timestamp("2025-01-01")) is None
    assert history_limit({"A": d}, pd.Timestamp("2025-01-01")) is None


@pytest.mark.parametrize("cleaned", [True, False])
def test_report_warns_about_short_history(raw, cleaned):
    prices = clean_prices(raw) if cleaned else raw  # also works when the caller passes uncleaned prices
    r = analyze(DEFAULT_PORTFOLIO, prices, Settings(run_backtest=False, n_resamples=20))
    assert r.history is not None and r.history.ticker == "SILVERBETA.NS"
    assert any(n.startswith("History starts 08 Jul 2024 because Silver ETF (SILVERBETA.NS)") for n in r.notes)
    warning = history_warning(r.history)
    assert warning and "Silver ETF" in warning and "years" in warning


def test_no_warning_with_full_history():
    px = demo_prices(TICKERS, start="2023-02-01", end="2026-10-13")
    r = analyze(DEFAULT_PORTFOLIO, clean_prices(px), Settings(run_backtest=False, n_resamples=20))
    assert history_warning(r.history) is None
    assert r.notes[0].startswith("Uses ")
