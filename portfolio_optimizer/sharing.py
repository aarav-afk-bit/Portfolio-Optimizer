"""Put a portfolio in a link, and read it back.

The holdings go in a single ``h`` query parameter:
``TICKER:QTY:AVG[:CLASS]`` entries separated by ``,``. Classes are only
written when they differ from the default for that ticker, which keeps
links short.
"""

from __future__ import annotations

from urllib.parse import urlencode, urlsplit, urlunsplit

import pandas as pd

from .config import asset_class_for

PARAM = "h"
COLUMNS = ["Ticker", "Qty", "Avg price", "Class"]
MAX_HOLDINGS = 50


def _num(x: float) -> str:
    return f"{float(x):.10g}"


def encode_holdings(df: pd.DataFrame) -> str:
    parts = []
    for _, row in df.iterrows():
        t = str(row["Ticker"]).strip().upper()
        if not t or t in ("NAN", "NONE"):
            continue
        qty = row.get("Qty", 0) if pd.notna(row.get("Qty", 0)) else 0
        avg = row.get("Avg price", 0) if pd.notna(row.get("Avg price", 0)) else 0
        entry = [t.replace(":", "").replace(",", ""), _num(qty), _num(avg)]
        cls = str(row.get("Class") or "").strip()
        if cls and cls != asset_class_for(t):
            entry.append(cls.replace(":", " ").replace(",", " "))
        parts.append(":".join(entry))
    return ",".join(parts)


def decode_holdings(text: str) -> pd.DataFrame:
    """Parse a link's holdings; raises ValueError with a readable message on bad input."""
    rows = []
    for i, entry in enumerate(e for e in str(text).split(",") if e.strip()):
        if i >= MAX_HOLDINGS:
            raise ValueError(f"The link has more than {MAX_HOLDINGS} holdings.")
        bits = entry.split(":")
        if len(bits) < 3:
            raise ValueError(f"Couldn't read '{entry}' in the link (expected TICKER:QTY:AVG).")
        ticker = bits[0].strip().upper()
        try:
            qty, avg = float(bits[1]), float(bits[2])
        except ValueError:
            raise ValueError(f"Couldn't read the numbers for {ticker} in the link.") from None
        if not ticker or qty < 0 or avg < 0:
            raise ValueError(f"Invalid holding '{entry}' in the link.")
        cls = ":".join(bits[3:]).strip() or asset_class_for(ticker)
        rows.append({"Ticker": ticker, "Qty": qty, "Avg price": avg, "Class": cls})
    if not rows:
        raise ValueError("The link doesn't contain any holdings.")
    return pd.DataFrame(rows, columns=COLUMNS)


def share_url(base_url: str, df: pd.DataFrame) -> str:
    """``base_url`` with the holdings in its query string (other parameters are dropped)."""
    parts = urlsplit(base_url or "")
    query = urlencode({PARAM: encode_holdings(df)}, safe=":,")  # keep ":" and "," readable
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", query, ""))
