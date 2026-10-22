"""Price loading: live data from Yahoo Finance, or a synthetic demo set."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


class DataError(RuntimeError):
    """Raised when prices cannot be loaded for the requested tickers.

    ``missing`` lists the tickers that came back empty. When it equals every
    requested ticker the cause is usually connectivity, not bad symbols.
    """

    def __init__(self, message: str, missing: list[str] | None = None):
        super().__init__(message)
        self.missing = missing or []


def fetch_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted daily closes for ``tickers`` from Yahoo Finance.

    Returns a DataFrame indexed by date with one column per ticker. Tickers
    with no data are reported in a ``DataError`` instead of silently becoming
    an all-NaN column.
    """
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise DataError("Yahoo Finance returned no data. Check the tickers and your connection.", tickers)

    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    if missing:
        raise DataError(f"No price history found for: {', '.join(missing)}. Check the symbols "
                        f"(NSE tickers end in .NS, BSE in .BO).", missing)

    return clean_prices(close[tickers])


EOD2_URL = "https://raw.githubusercontent.com/BennyThadikaran/eod2_data/main/daily/{}.csv"
_EOD2_INDEX_FILES = {"^NSEI": "nifty 50", "^NSEBANK": "nifty bank"}


def fetch_prices_nse_github(tickers: list[str], start: str, end: str | None = None,
                            timeout: float = 20) -> pd.DataFrame:
    """NSE daily closes from the community eod2_data repository on GitHub.

    A fallback for when Yahoo Finance is blocked or rate-limited. The data
    comes from NSE's own daily reports and is adjusted for splits and
    bonuses (not dividends). It is updated weekly, so the last few days may
    be missing. Only NSE tickers (``.NS``) and the Nifty 50 / Bank indices work.
    """
    from io import StringIO
    from urllib.parse import quote
    from urllib.request import urlopen

    closes, missing = {}, []
    for t in dict.fromkeys(tickers):
        name = _EOD2_INDEX_FILES.get(t) or (t[:-3].lower() if t.endswith(".NS") else None)
        if name is None:
            missing.append(t)
            continue
        try:
            with urlopen(EOD2_URL.format(quote(name)), timeout=timeout) as resp:
                df = pd.read_csv(StringIO(resp.read().decode()), parse_dates=["Date"])
            closes[t] = df.set_index("Date")["Close"].sort_index()
        except Exception:  # 404 for unknown symbols, network errors
            missing.append(t)
    if missing:
        raise DataError(f"No NSE data on GitHub for: {', '.join(missing)}", missing)
    prices = pd.DataFrame(closes)
    prices = prices.loc[pd.Timestamp(start):(pd.Timestamp(end) if end else None)]
    return clean_prices(prices[list(dict.fromkeys(tickers))])


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Align calendars: forward-fill short gaps (holidays), then drop the
    leading rows before every asset has started trading.

    Each column's first available date is kept in ``attrs["first_valid"]``
    so the report can say which holding cut the history short.
    """
    prices = prices.sort_index().ffill(limit=5)
    first_valid = {c: prices[c].first_valid_index() for c in prices.columns}
    prices = prices.dropna(how="any")
    if len(prices) < 60:
        raise DataError(
            "Fewer than 60 overlapping trading days. Use an earlier start date "
            "or drop recently listed tickers."
        )
    prices.attrs["first_valid"] = {c: d for c, d in first_valid.items() if d is not None}
    return prices


@dataclass(frozen=True)
class HistoryLimit:
    """The holding whose late start sets the common start of the price history."""

    ticker: str
    start: pd.Timestamp  # first date the analysis uses
    start_without: pd.Timestamp  # first date it could use without that holding
    end: pd.Timestamp
    days_used: int  # trading days used
    days_without: int  # trading days available without that holding

    @property
    def share_lost(self) -> float:
        return 1 - self.days_used / self.days_without if self.days_without else 0.0

    @property
    def years_used(self) -> float:
        return self.days_used / TRADING_DAYS


def history_limit(first_valid: dict, end: pd.Timestamp) -> HistoryLimit | None:
    """Which holding starts latest, and how much shared history it costs.

    Returns None when fewer than two holdings have dates or when the latest
    start is shared (no single holding is to blame).
    """
    dates = sorted((pd.Timestamp(d), t) for t, d in first_valid.items() if d is not None)
    if len(dates) < 2 or dates[-1][0] <= dates[-2][0]:
        return None
    (start_without, _), (start, ticker) = dates[-2], dates[-1]
    end = pd.Timestamp(end)
    return HistoryLimit(ticker, start, start_without, end,
                        days_used=len(pd.bdate_range(start, end)),
                        days_without=len(pd.bdate_range(start_without, end)))


# Rough annual drift / volatility plus factor loadings used to make the demo
# data behave like the real assets (metals co-move, equities co-move).
_DEMO_PROFILES = {
    "SILVERBETA.NS": dict(end=232.0, mu=0.22, vol=0.27, eq=0.05, metal=0.85),
    "GOLDETF.NS": dict(end=158.0, mu=0.20, vol=0.14, eq=0.02, metal=0.80),
    "HDFCSML250.NS": dict(end=171.0, mu=0.18, vol=0.20, eq=0.85, metal=0.05),
    "HAL.NS": dict(end=4620.0, mu=0.48, vol=0.36, eq=0.55, metal=0.0),
    "MAFANG.NS": dict(end=183.0, mu=0.24, vol=0.30, eq=0.35, metal=0.0),
    "^NSEI": dict(end=25400.0, mu=0.11, vol=0.13, eq=0.95, metal=0.0),
}


def demo_prices(
    tickers: list[str], start: str = "2022-01-01", end: str | None = None, seed: int = 7
) -> pd.DataFrame:
    """Generate reproducible, correlated price paths for offline use.

    Uses a two-factor model (equity, precious metals) with regime-shifting
    volatility, so indicators and the optimizer have realistic inputs. Paths
    are scaled to end near recent price levels so P&L figures look sensible.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end or pd.Timestamp.today().normalize())
    n = len(dates)
    dt = 1 / TRADING_DAYS

    equity = rng.standard_normal(n)
    metal = rng.standard_normal(n)
    regime = 1 + 0.6 * np.sin(np.linspace(0, 6 * np.pi, n)) ** 2

    out = {}
    for i, t in enumerate(dict.fromkeys(tickers)):
        p = _DEMO_PROFILES.get(t)
        if p is None:
            p = dict(end=100.0, mu=0.12, vol=0.25, eq=0.6, metal=0.0)
        idio_w = np.sqrt(max(1 - p["eq"] ** 2 - p["metal"] ** 2, 0.05))
        shock = p["eq"] * equity + p["metal"] * metal + idio_w * rng.standard_normal(n)
        shock /= np.sqrt(p["eq"] ** 2 + p["metal"] ** 2 + idio_w ** 2)
        sigma = p["vol"] * regime / regime.mean()
        log_ret = (p["mu"] - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock
        path = np.exp(np.cumsum(log_ret))
        out[t] = p["end"] * path / path[-1]

    return pd.DataFrame(out, index=dates)
