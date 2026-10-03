"""Number formatting helpers shared by the CLI and the web app."""

from __future__ import annotations

import math


def currency_symbol(tickers) -> str:
    tickers = list(tickers)
    if tickers and all(t.endswith((".NS", ".BO")) for t in tickers):
        return "₹"
    if tickers and not any("." in t for t in tickers):
        return "$"
    return ""


def _indian_group(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def fmt_money(x: float, symbol: str = "₹", decimals: int = 0, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    sign = "-" if x < 0 else ("+" if signed and x > 0 else "")
    v = abs(x)
    if symbol == "₹":
        whole = int(v)
        body = _indian_group(whole)
        if decimals:
            body += f"{v - whole:.{decimals}f}"[1:]
    else:
        body = f"{v:,.{decimals}f}"
    return f"{sign}{symbol}{body}"


def fmt_compact(x: float, symbol: str = "₹", signed: bool = False) -> str:
    """₹12.4L / ₹1.2Cr for rupees, $12.4K / $1.2M otherwise."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    sign = "-" if x < 0 else ("+" if signed and x > 0 else "")
    v = abs(x)
    if symbol == "₹":
        if v >= 1e7:
            return f"{sign}₹{v / 1e7:.2f}Cr"
        if v >= 1e5:
            return f"{sign}₹{v / 1e5:.2f}L"
        return f"{sign}{fmt_money(v, symbol)}"
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= div:
            return f"{sign}{symbol}{v / div:.2f}{suffix}"
    return f"{sign}{symbol}{v:,.0f}"


def fmt_pct(x: float, decimals: int = 1, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{decimals}%}" if signed else f"{x:.{decimals}%}"

