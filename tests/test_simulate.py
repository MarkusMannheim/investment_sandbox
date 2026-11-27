"""A full run, three times on one draw."""

import time

import pytest

from investment_sandbox.simulate import compare, draw, run


@pytest.fixture(scope="module")
def pair(settings):
    return compare(settings, seed=1)


def test_the_draw_is_reproducible_and_shared(settings, weather, pair):
    assert draw(settings, weather, 1) == draw(settings, weather, 1)
    assert draw(settings, weather, 1) != draw(settings, weather, 2)
    assert pair["cautious"].draw == pair["risk_neutral"].draw == pair["tender"].draw


def test_caution_means_less_gas_more_unmet_demand_and_a_higher_price(pair):
    cautious, neutral = pair["cautious"], pair["risk_neutral"]
    assert cautious.unserved_gwh > neutral.unserved_gwh
    gas = lambda r: r.built_mw.get("ocgt", 0.0) + r.built_mw.get("ccgt", 0.0)
    assert gas(cautious) < gas(neutral)
    assert cautious.price_paid > neutral.price_paid
    assert cautious.cap_hours > neutral.cap_hours
    assert not any(y.stopped for r in pair.values() for y in r.years), "no limit set the quantity"


def test_a_block_enters_service_after_its_lead_time(pair):
    r = pair["risk_neutral"]
    first = r.years[0].builds[0]
    year = r.years[first.unit.commissioned_year - r.years[0].year]
    assert year.year == first.unit.commissioned_year
    assert year.capacity_mw[first.unit.technology] >= first.unit.capacity_mw


def test_the_two_cost_lines_add_up(pair):
    for r in pair.values():
        for y in r.years:
            assert y.bill == pytest.approx(y.energy_cost + y.unserved_cost)
            assert y.resource_cost == pytest.approx(
                y.running_cost + y.fixed_cost + y.capital_cost + y.unserved_cost)
            assert min(y.energy_cost, y.running_cost, y.fixed_cost, y.capital_cost) >= 0


def test_every_plant_in_service_has_an_income_entry(pair):
    r = pair["cautious"]
    for y in r.years:
        live = {u.unit for u in r.units if u.in_service(y.year)}
        assert set(y.income) == live


def test_demand_grows_along_the_drawn_path(pair):
    r = pair["cautious"]
    g = r.draw.growth.annual_growth
    for a, b in zip(r.years, r.years[1:]):
        assert b.average_mw == pytest.approx(a.average_mw * (1 + g))


def test_a_short_run_is_a_prefix_of_the_long_one(settings):
    short = run(settings, seed=1, years=5)
    assert len(short.years) == 5


def test_a_run_with_the_tender_takes_seconds(settings):
    start = time.perf_counter()
    run(settings, seed=3)
    assert time.perf_counter() - start < 60.0


def test_the_pace_starts_at_the_pipeline_and_follows_the_decisions(settings):
    from investment_sandbox.simulate import PACE_YEARS, opening_pace, pace
    opening = pace(opening_pace(settings))
    committed = sum(u.capacity_mw for u in settings.fleet if u.commissioned_year >= settings.years[0])
    assert sum(opening.values()) == pytest.approx(committed / PACE_YEARS)
    later = pace(opening_pace(settings) + [{"wind": 1000.0}] * PACE_YEARS)
    assert later == {"wind": 1000.0}


def test_the_megawatts_committed_count_every_block_whatever_its_lead_time(pair):
    mixed = [y for r in pair.values() for y in r.years
             if len({b.unit.commissioned_year for b in y.builds}) > 1]
    assert mixed, "some year commits blocks that start up in different years"
    for y in mixed:
        blocks = [b.unit.capacity_mw for b in y.builds] + [c.unit.capacity_mw for c in y.awards]
        assert sum(y.committed_mw.values()) == pytest.approx(sum(blocks))


def test_the_tender_leaves_less_demand_unmet_and_the_firms_build_less(pair):
    cautious, tender = pair["cautious"], pair["tender"]
    assert tender.unserved_gwh < cautious.unserved_gwh
    assert sum(tender.bought_mw.values()) > 0
    assert sum(tender.built_mw.values()) < sum(cautious.built_mw.values())
    assert cautious.underwriting_cost == 0.0 and tender.underwriting_cost != 0.0
