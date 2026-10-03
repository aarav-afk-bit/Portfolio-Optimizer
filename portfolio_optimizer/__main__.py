"""Command-line report: ``python -m portfolio_optimizer [--source yahoo|github|demo]``."""

from __future__ import annotations

import argparse

from .config import DEFAULT_BENCHMARK, DEFAULT_PORTFOLIO, DEFAULT_START, STRATEGIES
from .data import demo_prices, fetch_prices, fetch_prices_nse_github
from .report import Settings, analyze
from .textreport import render_text


def main() -> None:
    p = argparse.ArgumentParser(description="Optimise the default portfolio and explain every change.")
    p.add_argument("--strategy", choices=list(STRATEGIES), default="max_sharpe")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--rf", type=float, default=None, help="risk-free rate, e.g. 0.065")
    p.add_argument("--max-weight", type=float, default=None, help="position cap, e.g. 0.4")
    p.add_argument("--class-cap", type=float, default=None, help="max weight per asset class, e.g. 0.5")
    p.add_argument("--max-turnover", type=float, default=None, help="max one-way turnover, e.g. 0.3")
    p.add_argument("--no-trade-band", type=float, default=None, help="skip weight changes below this, e.g. 0.02")
    p.add_argument("--no-robust", action="store_true", help="single-estimate weights (no resampling)")
    p.add_argument("--source", choices=["yahoo", "github", "demo"], default="yahoo",
                   help="price source: Yahoo Finance, NSE data on GitHub (eod2_data), or synthetic")
    p.add_argument("--demo", action="store_true", help="same as --source demo")
    args = p.parse_args()

    tickers = list(DEFAULT_PORTFOLIO)
    source = "demo" if args.demo else args.source
    load = {"yahoo": fetch_prices, "github": fetch_prices_nse_github, "demo": demo_prices}[source]
    data = load(tickers + [DEFAULT_BENCHMARK], start=args.start)

    s = Settings(strategy=args.strategy)
    if args.rf is not None:
        s.risk_free_rate = args.rf
    if args.max_weight is not None:
        s.max_weight = args.max_weight
    s.class_cap, s.max_turnover = args.class_cap, args.max_turnover
    if args.no_trade_band is not None:
        s.no_trade_band = args.no_trade_band
    s.robust = not args.no_robust

    report = analyze(DEFAULT_PORTFOLIO, data[tickers], s, data[DEFAULT_BENCHMARK])
    print(render_text(report, demo=source == "demo"))


if __name__ == "__main__":
    main()
