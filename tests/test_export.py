import re

from portfolio_optimizer.config import DEFAULT_PORTFOLIO
from portfolio_optimizer.data import demo_prices
from portfolio_optimizer.export import report_html
from portfolio_optimizer.report import Settings, analyze


def test_report_is_self_contained_and_complete():
    prices = demo_prices(list(DEFAULT_PORTFOLIO), start="2022-01-01", end="2026-09-30")
    r = analyze(DEFAULT_PORTFOLIO, prices, Settings(n_resamples=20))
    html = report_html(r, demo=True)
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    assert "<script" not in html and not re.search(r'(src|href)="https?://', html)  # works offline
    for rec in r.recommendations:
        assert rec.ticker in html and rec.label in html
    assert "Backtest (no hindsight)" in html and "demo data" in html
    assert "not investment advice" in html
