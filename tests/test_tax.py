import pytest

from portfolio_optimizer.tax import EQUITY, OTHER, Sale, TaxRates, default_tax_type, estimate_tax


def sale(label, gain, tax_type=EQUITY, long_term=True):
    # 100 units bought at 100; price chosen to give the requested gain
    return Sale(label, label, 100, 100 + gain / 100, 100.0, tax_type, long_term)


def test_long_term_equity_uses_the_exemption():
    est = estimate_tax([sale("HAL", 300_000)])
    # (3,00,000 - 1,25,000) * 12.5%
    assert est.total_tax == pytest.approx(21_875)
    assert est.exemption_used == pytest.approx(125_000)


def test_short_term_rates():
    est = estimate_tax([sale("A", 100_000, EQUITY, long_term=False), sale("B", 100_000, OTHER, long_term=False)])
    assert est.total_tax == pytest.approx(20_000 + 30_000)  # 20% equity, 30% slab for other ETFs


def test_short_term_loss_offsets_the_highest_taxed_gain_first():
    est = estimate_tax([sale("Gold", 100_000, OTHER, long_term=False),  # taxed at 30%
                        sale("HAL", 100_000, EQUITY, long_term=False),  # taxed at 20%
                        sale("Loss", -60_000, EQUITY, long_term=False)])
    assert est.total_tax == pytest.approx(40_000 * 0.30 + 100_000 * 0.20)


def test_long_term_loss_cannot_offset_short_term_gain():
    est = estimate_tax([sale("A", 100_000, EQUITY, long_term=False), sale("L", -50_000, EQUITY, long_term=True)])
    assert est.total_tax == pytest.approx(20_000)
    assert any("left over" in n for n in est.notes)


def test_missing_cost_is_skipped_with_a_note():
    est = estimate_tax([Sale("X", "X", 10, 100.0, 0.0, EQUITY, True)])
    assert est.total_tax == 0 and "no average cost" in est.notes[0]


def test_custom_rates_and_default_types():
    est = estimate_tax([sale("A", 200_000)], TaxRates(equity_long=0.10, equity_exemption=100_000))
    assert est.total_tax == pytest.approx(10_000)
    assert default_tax_type("Indian equity") == EQUITY
    assert default_tax_type("Precious metals") == OTHER and default_tax_type("US equity") == OTHER
