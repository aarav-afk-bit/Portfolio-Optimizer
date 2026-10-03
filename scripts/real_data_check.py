"""Validate the engine on real market data.

    python scripts/real_data_check.py            # writes docs/real_data_results.md
    python scripts/real_data_check.py --quick    # fewer configurations

Datasets (downloaded at run time, nothing is stored in the repo):
  * Your holdings: NSE daily closes from the eod2_data project on GitHub.
  * Indian ETFs + HAL since 2018: same source.
  * 15 US stocks, Aug 2004 – Apr 2018 (incl. 2008 and Sears' collapse): the
    real price file shipped with PyPortfolioOpt's test suite.

For each dataset every strategy is run walk-forward (quarterly rebalancing,
trailing 1-year estimates, 10 bps costs, 40% position cap) with two return
models, and compared with equal weight. Solver results on the real data are
also cross-checked against cvxpy when it is installed.
"""

from __future__ import annotations

import argparse
import sys
import time
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio_optimizer import analytics as an  # noqa: E402
from portfolio_optimizer.backtest import walk_forward  # noqa: E402
from portfolio_optimizer.config import DEFAULT_PORTFOLIO, STRATEGIES, TRADING_DAYS  # noqa: E402
from portfolio_optimizer.data import clean_prices, fetch_prices_nse_github  # noqa: E402
from portfolio_optimizer.estimators import Estimator  # noqa: E402
from portfolio_optimizer.optimizer import Constraints, optimize  # noqa: E402

US_URL = "https://raw.githubusercontent.com/robertmartin8/PyPortfolioOpt/master/tests/resources/stock_prices.csv"
US_TICKERS = ["AAPL", "AMZN", "BAC", "BBY", "GE", "AMD", "GOOG", "JPM", "PFE", "RRC", "SBUX", "SHLD", "T",
              "WMT", "XOM"]
INDIA_LONG = ["NIFTYBEES.NS", "JUNIORBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "ITBEES.NS", "HAL.NS"]
RF = {"india": 0.065, "us": 0.02}


def load_datasets() -> dict[str, tuple[pd.DataFrame, float]]:
    holdings = fetch_prices_nse_github(list(DEFAULT_PORTFOLIO), "2000-01-01")
    india = fetch_prices_nse_github(INDIA_LONG, "2018-01-01")
    with urlopen(US_URL, timeout=30) as resp:
        us = pd.read_csv(StringIO(resp.read().decode()), parse_dates=["date"], index_col="date")
    us = clean_prices(us[US_TICKERS].loc["2004-08-19":])
    return {
        "Your holdings (NSE)": (holdings, RF["india"]),
        "Indian ETFs + HAL": (india, RF["india"]),
        "15 US stocks": (us, RF["us"]),
    }


def solver_checks(prices: pd.DataFrame, rf: float) -> list[str]:
    try:
        import cvxpy as cp
    except ImportError:
        return ["cvxpy not installed: solver cross-check skipped."]
    rets = an.daily_returns(prices)
    mu, cov_d = Estimator(rf=rf).fit(rets)
    cov = cov_d * TRADING_DAYS
    n = len(mu)
    hi = max(0.4, 1 / n + 0.05)
    out = []
    ours = optimize(mu, cov, "max_sharpe", rf, constraints=Constraints(max_weight=hi))
    y, k = cp.Variable(n), cp.Variable()
    cp.Problem(cp.Minimize(cp.quad_form(y, cov.to_numpy())),
               [(mu.to_numpy() - rf) @ y == 1, cp.sum(y) == k, y >= 0, y <= hi * k, k >= 0]).solve()
    ref = an.sharpe_ratio(y.value / k.value, mu, cov, rf)
    out.append(f"max Sharpe: ours {ours.sharpe:.4f} vs cvxpy {ref:.4f}")
    ours = optimize(mu, cov, "min_volatility", constraints=Constraints(max_weight=hi))
    w = cp.Variable(n)
    cp.Problem(cp.Minimize(cp.quad_form(w, cov.to_numpy())), [cp.sum(w) == 1, w >= 0, w <= hi]).solve()
    out.append(f"min volatility: ours {ours.volatility:.4%} vs cvxpy {an.portfolio_volatility(w.value, cov):.4%}")
    return out


def evaluate(prices: pd.DataFrame, rf: float, quick: bool) -> pd.DataFrame:
    n = prices.shape[1]
    eq = pd.Series(1 / n, index=prices.columns)
    cons = Constraints(max_weight=max(0.4, 1 / n + 0.05))
    strategies = ["max_sharpe", "min_volatility", "hrp"] if quick else list(STRATEGIES)
    configs = [(s, "black_litterman", 0) for s in strategies]
    configs += [("max_sharpe", "bayes_stein", 0), ("max_sharpe", "historical", 0)]
    if not quick:
        configs += [("max_sharpe", "black_litterman", 30)]
    rows = {}
    for strategy, model, n_res in configs:
        t0 = time.time()
        bt = walk_forward(prices, eq, strategy, rf, constraints=cons, estimator=Estimator(model, rf=rf),
                          rebalance_every=63, cost_bps=10, n_resamples=n_res)
        s = bt.stats.loc["Optimised (walk-forward)"]
        w = bt.weights_history
        label = f"{STRATEGIES[strategy]} · {model.replace('_', '-')}" + (f" · resampled×{n_res}" if n_res else "")
        rows[label] = {"CAGR": s["cagr"], "Volatility": s["volatility"], "Sharpe": s["sharpe"],
                       "Max drawdown": s["max_drawdown"], "Avg turnover": float(w.diff().abs().sum(axis=1).iloc[1:].mean() / 2),
                       "seconds": time.time() - t0}
        if "Equal weight" not in rows:
            e = bt.stats.loc["Equal weight"]
            rows["Equal weight (rebalanced)"] = {"CAGR": e["cagr"], "Volatility": e["volatility"],
                                                 "Sharpe": e["sharpe"], "Max drawdown": e["max_drawdown"],
                                                 "Avg turnover": np.nan, "seconds": np.nan}
    return pd.DataFrame(rows).T


def to_markdown(df: pd.DataFrame) -> str:
    def fmt(col, v):
        if pd.isna(v):
            return "–"
        if col in ("Sharpe",):
            return f"{v:.2f}"
        if col == "seconds":
            return f"{v:.1f}"
        return f"{v:.1%}"
    head = "| Configuration | " + " | ".join(df.columns) + " |\n|---|" + "---:|" * len(df.columns) + "\n"
    return head + "".join(f"| {i} | " + " | ".join(fmt(c, r[c]) for c in df.columns) + " |\n" for i, r in df.iterrows())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "real_data_results.md"))
    args = p.parse_args()

    parts = ["# Real-data validation\n",
             "Generated by `scripts/real_data_check.py`. Walk-forward: quarterly rebalancing, trailing 1-year "
             "estimates, 10 bps trading costs, 40% position cap (higher only when needed to stay feasible), "
             "starting from equal weights. Past performance on a few datasets is evidence, not a guarantee.\n"]
    for name, (prices, rf) in load_datasets().items():
        print(f"== {name}: {prices.shape[1]} assets, {prices.index[0]:%d %b %Y} – {prices.index[-1]:%d %b %Y}")
        checks = solver_checks(prices, rf)
        res = evaluate(prices, rf, args.quick)
        print(res.round(3).to_string())
        parts.append(f"\n## {name}\n\n{prices.shape[1]} assets: {', '.join(prices.columns)}. "
                     f"{prices.index[0]:%d %b %Y} – {prices.index[-1]:%d %b %Y} ({len(prices)} trading days), "
                     f"risk-free rate {rf:.1%}.\n\nSolver cross-check on the full sample: "
                     + "; ".join(checks) + ".\n\n" + to_markdown(res))
    Path(args.out).write_text("".join(parts))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
