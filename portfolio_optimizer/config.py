"""Default settings and asset metadata."""

from __future__ import annotations

TRADING_DAYS = 252

# Approximate yield of Indian 91-day T-bills. Sharpe ratios are measured
# against this, so a holding has to beat "risk-free" cash to look attractive.
DEFAULT_RISK_FREE_RATE = 0.065
DEFAULT_BENCHMARK = "^NSEI"  # Nifty 50
DEFAULT_START = "2022-01-01"
DEFAULT_MAX_WEIGHT = 0.40
DEFAULT_MIN_WEIGHT = 0.0

DEFAULT_PORTFOLIO: dict[str, dict[str, float]] = {
    "SILVERBETA.NS": {"qty": 2121, "avg_price": 198.64},
    "GOLDETF.NS": {"qty": 2950, "avg_price": 135.57},
    "HDFCSML250.NS": {"qty": 250, "avg_price": 173.13},
    "HAL.NS": {"qty": 500, "avg_price": 4278.30},
    "MAFANG.NS": {"qty": 100, "avg_price": 176.16},
}

# Friendly labels shown in the UI. Unknown tickers fall back to the symbol.
ASSET_LABELS: dict[str, str] = {
    "SILVERBETA.NS": "Silver ETF",
    "GOLDETF.NS": "Gold ETF",
    "HDFCSML250.NS": "Smallcap 250 ETF",
    "HAL.NS": "Hindustan Aeronautics",
    "MAFANG.NS": "NYSE FANG+ ETF",
    "^NSEI": "Nifty 50",
}

ASSET_CLASSES: dict[str, str] = {
    "SILVERBETA.NS": "Precious metals",
    "GOLDETF.NS": "Precious metals",
    "HDFCSML250.NS": "Indian equity",
    "HAL.NS": "Indian equity",
    "MAFANG.NS": "US equity",
}

STRATEGIES: dict[str, str] = {
    "max_sharpe": "Max Sharpe ratio",
    "min_volatility": "Minimum volatility",
    "min_cvar": "Minimum tail risk (CVaR)",
    "risk_parity": "Risk parity",
    "hrp": "Hierarchical risk parity",
    "max_diversification": "Max diversification",
}


def label_for(ticker: str) -> str:
    return ASSET_LABELS.get(ticker, ticker.replace(".NS", "").replace(".BO", ""))


def asset_class_for(ticker: str) -> str:
    return ASSET_CLASSES.get(ticker, "Equity")
