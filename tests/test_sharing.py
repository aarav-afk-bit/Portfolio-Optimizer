from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from portfolio_optimizer.config import DEFAULT_PORTFOLIO, asset_class_for
from portfolio_optimizer.sharing import PARAM, decode_holdings, encode_holdings, share_url


def default_df():
    return pd.DataFrame([{"Ticker": t, "Qty": v["qty"], "Avg price": v["avg_price"], "Class": asset_class_for(t)}
                         for t, v in DEFAULT_PORTFOLIO.items()])


def test_round_trip_keeps_holdings():
    df = default_df()
    back = decode_holdings(encode_holdings(df))
    pd.testing.assert_frame_equal(back, df, check_dtype=False)


def test_default_classes_are_not_written():
    text = encode_holdings(default_df())
    assert text.startswith("SILVERBETA.NS:2121:198.64,") and "Precious" not in text


def test_custom_class_survives():
    df = default_df()
    df.loc[df["Ticker"] == "MAFANG.NS", "Class"] = "Global equity"
    back = decode_holdings(encode_holdings(df))
    assert back.set_index("Ticker").at["MAFANG.NS", "Class"] == "Global equity"


def test_share_url_replaces_the_query():
    url = share_url("https://example.app/?old=1", default_df())
    q = parse_qs(urlsplit(url).query)
    assert list(q) == [PARAM] and decode_holdings(q[PARAM][0]).shape == (5, 4)


@pytest.mark.parametrize("bad", ["", "HAL.NS", "HAL.NS:abc:1", "HAL.NS:-5:100", ",".join(["A.NS:1:1"] * 51)])
def test_bad_links_raise_readable_errors(bad):
    with pytest.raises(ValueError):
        decode_holdings(bad)
