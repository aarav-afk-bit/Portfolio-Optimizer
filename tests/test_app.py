"""Run the whole Streamlit app headlessly and make sure nothing on the page crashes."""

from pathlib import Path

import pytest

testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run_app():
    at = testing.AppTest.from_file(APP, default_timeout=300)
    at.run()  # first run uses the default (live) price source, falling back as needed
    return at


def test_app_renders_every_tab_without_errors():
    at = run_app()
    assert not at.exception, [e.value for e in at.exception]
    # Switch to the offline demo data and re-run through the form's submit button.
    next(r for r in at.radio if r.label == "Price data").set_value("Demo (offline)")
    next(b for b in at.button if b.label == "Analyse portfolio").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    # Every tab's content runs on each rerun, so reaching the end proves all of them rendered.
    assert len(at.tabs) >= 8
