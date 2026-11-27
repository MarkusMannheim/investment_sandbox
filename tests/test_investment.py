"""The decision rule: caution, exposure and the forecast rebuilt after each block."""

import numpy as np
import pytest

from investment_sandbox.config import load_settings
from investment_sandbox.forward import forward_view
from investment_sandbox.investment import (block_mw, certainty_equivalent, coefficient, decide,
                                           exposure, appraise)

W = np.array([0.25, 0.25, 0.25, 0.25])
Y = np.array([0.0, 100_000.0, 200_000.0, 300_000.0])


def test_the_certainty_equivalent_is_the_mean_without_caution():
    assert certainty_equivalent(Y, W, 0.0) == pytest.approx(150_000.0)


def test_caution_lowers_the_certainty_equivalent_below_the_mean_and_never_below_the_worst():
    ce = certainty_equivalent(Y, W, coefficient(1.0))
    assert Y.min() < ce < Y @ W
    assert certainty_equivalent(Y, W, coefficient(3.0)) < ce


def test_a_certain_income_is_worth_itself_whatever_the_caution():
    assert certainty_equivalent(np.full(4, 90_000.0), W, coefficient(2.0)) == pytest.approx(90_000.0)


def test_a_future_at_the_price_cap_cannot_overflow():
    huge = np.array([0.0, 0.0, 0.0, 2e7])
    assert np.isfinite(certainty_equivalent(huge, W, coefficient(3.0)))


def test_weights_must_be_a_distribution():
    with pytest.raises(ValueError):
        certainty_equivalent(Y, W * 0.5, coefficient(1.0))


def test_exposure_falls_with_the_contract_and_vanishes_when_it_runs_for_the_whole_life():
    assert exposure(0, 25) == 1.0
    assert exposure(12, 25) == pytest.approx(1.0 - 12 / 25)
    assert exposure(25, 25) == 0.0
    assert exposure(20, 15) == 0.0


@pytest.fixture(scope="module")
def view(settings, weather):
    return forward_view(settings.fleet, settings.years[1], float(settings.demand["average_mw"]),
                        settings, weather)


def test_without_caution_the_test_is_expected_income_against_cost(settings, view):
    for tech in settings.tech_costs:
        t = appraise(tech, view, 0.0, 1.0)
        assert t.certainty_equivalent == pytest.approx(t.expected)
        assert t.caution_cost == pytest.approx(0.0)
        assert t.passes == (t.expected >= t.fixed_cost)


def test_caution_adds_to_the_bar_and_a_full_contract_removes_it(settings, view):
    ocgt = settings.tech("ocgt")
    exposed = appraise(ocgt, view, 1.0, 1.0)
    covered = appraise(ocgt, view, 1.0, 0.0)
    assert exposed.caution_cost > 0
    assert exposed.hurdle == pytest.approx(exposed.fixed_cost + exposed.caution_cost)
    assert covered.caution_cost == pytest.approx(0.0)


def test_a_block_is_whole_units_and_at_least_one(settings):
    wind = settings.tech("wind")
    assert block_mw(wind, 12_000.0, settings) == 600.0
    assert block_mw(wind, 100.0, settings) == wind.unit_size_mw


def test_the_forecast_after_the_blocks_pays_them_less_than_the_one_before(settings, weather):
    d = decide(settings.fleet, settings.years[0], 7000.0, 12_400.0, settings, weather, caution=1.0)
    assert d.builds and not d.stopped
    for b in d.builds:
        assert d.last_view.expected(b.unit.technology) <= d.first_view.expected(b.unit.technology) + 1e-6
    assert len(d.first_tests) == len(settings.tech_costs), "one first-round test a technology"
    assert not any(appraise(t, d.last_view, 1.0).passes for t in settings.tech_costs)


def test_the_year_ends_when_nothing_passes(settings, weather):
    from investment_sandbox.config import Unit
    glut = settings.fleet + (Unit("coal_glut", "coal", 20_000.0, 0.9, 40.0, 2020, 2060, 65.0, None),)
    d = decide(glut, settings.years[0], 7000.0, 12_400.0, settings, weather, caution=0.0)
    assert d.builds == [] and not d.stopped
    assert not any(t.passes for t in d.first_tests)


def test_the_safety_stop_is_reported_when_it_binds(weather):
    tight = load_settings({"investment": {"max_blocks_per_year": 1}})
    d = decide(tight.fleet, tight.years[0], 7000.0, 12_400.0, tight, weather, caution=0.0)
    assert len(d.builds) == 1 and d.stopped
