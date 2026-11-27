"""The tender buys plant down to the reliability standard, and its contracts are counted apart from the bill."""

import pytest

from investment_sandbox.config import Unit, load_settings
from investment_sandbox.forward import forward_view
from investment_sandbox.investment import certainty_equivalent, coefficient, exposure
from investment_sandbox.simulate import opening_pace, pace, run
from investment_sandbox.tender import fixed_sum, hold

AVERAGE_MW, PEAK_MW = 7000.0, 12_400.0


@pytest.fixture(scope="module")
def with_tender(settings):
    return run(settings, seed=1, years=10)


@pytest.fixture(scope="module")
def first_year(settings, weather):
    """A fleet that the forecast finds short, with that forecast: the starting fleet less its two largest coal units."""
    recent = pace(opening_pace(settings))
    year = settings.years[0]
    units = tuple(u for u in settings.fleet if u.unit not in ("coal_a", "coal_b"))
    return units, year, forward_view(units, year, AVERAGE_MW, settings, weather, recent), recent


def test_the_tender_buys_until_the_forecast_is_back_at_the_standard(settings, weather, first_year):
    units, year, view, recent = first_year
    standard = settings.tender["reliability_standard"]
    assert view.unserved_share > standard
    contracts, stopped = hold(units, year, AVERAGE_MW, PEAK_MW, view, settings, weather, 1.0, recent)
    assert contracts and not stopped
    bought = tuple(c.unit for c in contracts)
    after = forward_view(units + bought, year, AVERAGE_MW, settings, weather, recent)
    assert after.unserved_share <= standard
    one_less = forward_view(units + bought[:-1], year, AVERAGE_MW, settings, weather, recent)
    assert one_less.unserved_share > standard, "it bought no more than it needed"


def test_the_tender_buys_nothing_when_the_forecast_meets_the_standard(settings, weather, first_year):
    units, year, _, recent = first_year
    glut = units + (Unit("coal_glut", "coal", 20_000.0, 0.9, 40.0, 2020, 2060, 65.0, None),)
    view = forward_view(glut, year, AVERAGE_MW, settings, weather, recent)
    assert hold(glut, year, AVERAGE_MW, PEAK_MW, view, settings, weather, 1.0, recent) == ([], False)


def test_only_the_technologies_named_can_win(weather, first_year):
    units, year, view, recent = first_year
    narrow = load_settings({"tender": {"technologies": ["battery_8h"]}})
    contracts, _ = hold(units, year, AVERAGE_MW, PEAK_MW, view, narrow, weather, 1.0, recent)
    assert contracts and {c.unit.technology for c in contracts} == {"battery_8h"}


def test_a_contract_for_the_whole_life_pays_the_yearly_cost(settings, first_year):
    view = first_year[2]
    for tech in settings.tech_costs:
        assert fixed_sum(tech, view, 1.0, tech.life_years) == pytest.approx(tech.fixed_cost_per_mw_year)


def test_the_fixed_sum_is_the_one_at_which_a_firm_would_only_just_build(settings, first_year):
    view = first_year[2]
    for tech in settings.tech_costs:
        exposed = exposure(12, tech.life_years)
        later = certainty_equivalent(exposed * view.rents[tech.technology], view.weights, coefficient(1.0))
        worth = (1.0 - exposed) * fixed_sum(tech, view, 1.0, 12) + later
        assert worth == pytest.approx(tech.fixed_cost_per_mw_year)


def test_no_tender_means_no_contracts(settings):
    for r in (run(settings, seed=1, years=4, tender=False),
              run(load_settings({"tender": {"technologies": []}}), seed=1, years=4)):
        assert not r.tender and r.contracts == [] and r.underwriting_cost == 0.0
        assert not any(y.awards for y in r.years)


def test_underwriting_is_the_fixed_sums_less_market_income_and_sits_outside_the_bill(with_tender):
    r = with_tender
    assert r.contracts
    for y in r.years:
        expected = sum(c.fixed_per_mw_year * c.unit.capacity_mw - y.income[c.unit.unit]
                       for c in r.contracts if c.live(y.year))
        assert y.underwriting_cost == pytest.approx(expected)
        assert y.bill == pytest.approx(y.energy_cost + y.unserved_cost)
    assert any(y.underwriting_cost != 0 for y in r.years)


def test_a_contract_runs_from_the_first_year_of_the_plant_for_its_term(settings, with_tender):
    years = settings.tender["contract_years"]
    for c in with_tender.contracts:
        assert c.start_year == c.unit.commissioned_year
        assert c.end_year == c.start_year + min(years, settings.tech(c.unit.technology).life_years)


def test_plant_bought_by_tender_joins_the_fleet_and_the_megawatts_committed(with_tender):
    r = with_tender
    names = {u.unit for u in r.units}
    assert all(c.unit.unit in names for c in r.contracts)
    for y in r.years:
        bought = sum(c.unit.capacity_mw for c in y.awards)
        assert sum(y.committed_mw.values()) == pytest.approx(sum(y.built_mw.values()) + bought)


def test_a_contract_s_expected_cost_covers_its_whole_term_wherever_the_run_ends(settings):
    short, long = run(settings, seed=1, years=6), run(settings, seed=1, years=10)
    early = long.contracts[:len(short.contracts)]
    assert short.contracts and [c.expected_cost for c in short.contracts] == [c.expected_cost for c in early]
    for c in long.contracts:
        years = c.end_year - c.start_year
        assert c.expected_cost == pytest.approx(
            (c.fixed_per_mw_year - c.expected_income_per_mw_year) * c.unit.capacity_mw * years)
    assert long.expected_underwriting_cost == pytest.approx(sum(c.expected_cost for c in long.contracts))
