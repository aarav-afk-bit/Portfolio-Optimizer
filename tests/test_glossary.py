import re
from pathlib import Path

from portfolio_optimizer.glossary import GLOSSARY, GROUPS, search, tip


def test_every_term_is_defined():
    for t in GLOSSARY.values():
        assert t.short.endswith(".") and len(t.short) < 120
        assert t.group in GROUPS


def test_search_matches_names_and_text():
    assert {t.name for t in search("sharpe")} >= {"Sharpe ratio", "Max Sharpe ratio"}
    assert any(t.name == "Max drawdown" for t in search("peak"))
    assert len(search("")) == len(GLOSSARY)
    assert search("zzzz") == []


def test_app_tooltips_use_known_terms():
    app = Path(__file__).resolve().parents[1].joinpath("app.py").read_text()
    for name in re.findall(r'tip\("([^"]+)"\)', app):
        assert tip(name)
