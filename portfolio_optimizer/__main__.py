"""Command-line report: ``python -m portfolio_optimizer [--demo]``."""

from __future__ import annotations

import argparse

from .config import DEFAULT_BENCHMARK, DEFAULT_PORTFOLIO, DEFAULT_START, STRATEGIES
from .data import demo_prices, fetch_prices
from .report import Settings, analyze
from .textreport import render_text


def main() -> None:
    p = argparse.ArgumentParser(description="Optimise the default portfolio and explain every change.")
    p.add_argument("--strategy", choices=list(STRATEGIES), default="max_sharpe")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--rf", type=float, default=None, help="risk-free rate, e.g. 0.065")
    p.add_argument("--max-weight", type=float, default=None, help="position cap, e.g. 0.4")
    p.add_argument("--demo", action="store_true", help="use synthetic prices (no internet needed)")
    args = p.parse_args()

    tickers = list(DEFAULT_PORTFOLIO)
    load = demo_prices if args.demo else fetch_prices
    data = load(tickers + [DEFAULT_BENCHMARK], start=args.start)

    s = Settings(strategy=args.strategy)
    if args.rf is not None:
        s.risk_free_rate = args.rf
    if args.max_weight is not None:
        s.max_weight = args.max_weight

    report = analyze(DEFAULT_PORTFOLIO, data[tickers], s, data[DEFAULT_BENCHMARK])
    print(render_text(report, demo=args.demo))


if __name__ == "__main__":
    main()
