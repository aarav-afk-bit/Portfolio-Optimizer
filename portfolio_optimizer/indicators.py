"""Technical indicators and a combined signal score per asset."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def rsi(price: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing (the standard definition).

    The original notebook used a simple rolling mean, which reacts differently
    from the RSI shown on trading platforms.
    """
    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger(price: pd.Series, window: int = 20, k: float = 2.0):
    mid = price.rolling(window).mean()
    std = price.rolling(window).std()
    return mid + k * std, mid, mid - k * std


def compute_indicators(price: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"price": price})
    df["RSI"] = rsi(price)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(price)
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = bollinger(price)
    df["SMA_50"] = price.rolling(50).mean()
    df["SMA_200"] = price.rolling(200).mean()
    return df


@dataclass
class Signal:
    name: str
    value: str
    stance: str  # "bullish" | "bearish" | "neutral"
    detail: str


@dataclass
class TechnicalSnapshot:
    ticker: str
    price: float
    rsi: float
    macd_hist: float
    pct_b: float
    sma_50: float
    sma_200: float
    from_52w_high: float
    momentum_3m: float
    momentum_6m: float
    score: float  # -100 (strongly bearish) .. +100 (strongly bullish)
    signals: list[Signal] = field(default_factory=list)

    @property
    def stance(self) -> str:
        if self.score >= 25:
            return "Bullish"
        if self.score <= -25:
            return "Bearish"
        return "Neutral"


def _pct_change_over(price: pd.Series, days: int) -> float:
    if len(price) <= days:
        return float("nan")
    return float(price.iloc[-1] / price.iloc[-days - 1] - 1)


def technical_snapshot(ticker: str, price: pd.Series) -> TechnicalSnapshot:
    """Summarise the latest indicator readings into labelled signals and a score."""
    ind = compute_indicators(price)
    last = ind.iloc[-1]
    prev_hist = ind["MACD_hist"].iloc[-2] if len(ind) > 1 else last["MACD_hist"]

    p = float(last["price"])
    r = float(last["RSI"])
    hist = float(last["MACD_hist"])
    band = float(last["BB_upper"] - last["BB_lower"])
    pct_b = float((p - last["BB_lower"]) / band) if band > 0 else 0.5
    sma50, sma200 = float(last["SMA_50"]), float(last["SMA_200"])
    high_52w = float(price.iloc[-TRADING_DAYS:].max())
    from_high = p / high_52w - 1
    mom3, mom6 = _pct_change_over(price, 63), _pct_change_over(price, 126)

    signals: list[Signal] = []
    score_parts: list[tuple[float, float]] = []  # (value in [-1, 1], weight)

    # Trend: moving-average alignment
    if np.isnan(sma200):
        signals.append(Signal("Trend", "n/a", "neutral", "Not enough history for a 200-day average."))
    elif p > sma50 > sma200:
        signals.append(Signal("Trend", "Strong uptrend", "bullish",
                              "Price is above the 50-day, which is above the 200-day average."))
        score_parts.append((1.0, 0.30))
    elif p < sma50 < sma200:
        signals.append(Signal("Trend", "Downtrend", "bearish",
                              "Price is below the 50-day, which is below the 200-day average."))
        score_parts.append((-1.0, 0.30))
    elif p > sma200:
        signals.append(Signal("Trend", "Above 200-DMA", "bullish",
                              "Long-term trend is up but short-term averages are mixed."))
        score_parts.append((0.4, 0.30))
    else:
        signals.append(Signal("Trend", "Below 200-DMA", "bearish",
                              "Price is under its 200-day average: long-term trend is weak."))
        score_parts.append((-0.4, 0.30))

    # Momentum: MACD histogram and its direction
    rising = hist > prev_hist
    if hist > 0:
        signals.append(Signal("MACD", "Bullish" + (", strengthening" if rising else ", fading"), "bullish",
                              "MACD is above its signal line: short-term momentum is positive."))
        score_parts.append((1.0 if rising else 0.5, 0.25))
    else:
        signals.append(Signal("MACD", "Bearish" + (", easing" if rising else ", deepening"), "bearish",
                              "MACD is below its signal line: short-term momentum is negative."))
        score_parts.append((-0.5 if rising else -1.0, 0.25))

    # RSI: contrarian at the extremes
    if r >= 70:
        signals.append(Signal("RSI", f"{r:.0f} – overbought", "bearish",
                              "RSI above 70: the rally is stretched and prone to a pullback."))
        score_parts.append((-1.0, 0.20))
    elif r <= 30:
        signals.append(Signal("RSI", f"{r:.0f} – oversold", "bullish",
                              "RSI below 30: selling looks exhausted; bounces are common from here."))
        score_parts.append((1.0, 0.20))
    elif r >= 55:
        signals.append(Signal("RSI", f"{r:.0f} – firm", "bullish",
                              "Buyers have the upper hand without being overbought."))
        score_parts.append(((r - 50) / 40, 0.20))
    elif r <= 45:
        signals.append(Signal("RSI", f"{r:.0f} – soft", "bearish",
                              "Sellers have the edge, but it is not yet oversold."))
        score_parts.append(((r - 50) / 40, 0.20))
    else:
        signals.append(Signal("RSI", f"{r:.0f} – neutral", "neutral",
                              "No overbought or oversold extreme."))
        score_parts.append(((r - 50) / 40, 0.20))

    # Bollinger %B
    if pct_b > 1:
        signals.append(Signal("Bollinger", "Above upper band", "bearish",
                              "Price closed above the upper Bollinger band: short-term overextended."))
        score_parts.append((-0.7, 0.10))
    elif pct_b < 0:
        signals.append(Signal("Bollinger", "Below lower band", "bullish",
                              "Price closed below the lower Bollinger band: short-term washed out."))
        score_parts.append((0.7, 0.10))
    else:
        signals.append(Signal("Bollinger", f"%B {pct_b:.2f}", "neutral",
                              "Price is inside its Bollinger bands."))
        score_parts.append((0.0, 0.10))

    # 6-month momentum
    if not np.isnan(mom6):
        stance = "bullish" if mom6 > 0.05 else "bearish" if mom6 < -0.05 else "neutral"
        detail = {"bullish": "Medium-term momentum is positive.",
                  "bearish": "Medium-term momentum is negative.",
                  "neutral": "Roughly flat over six months."}[stance]
        signals.append(Signal("6M momentum", f"{mom6:+.1%}", stance, detail))
        score_parts.append((float(np.clip(mom6 / 0.25, -1, 1)), 0.15))

    total_w = sum(w for _, w in score_parts) or 1.0
    score = 100 * sum(v * w for v, w in score_parts) / total_w

    return TechnicalSnapshot(
        ticker=ticker, price=p, rsi=r, macd_hist=hist, pct_b=pct_b, sma_50=sma50, sma_200=sma200,
        from_52w_high=from_high, momentum_3m=mom3, momentum_6m=mom6, score=float(score), signals=signals,
    )
