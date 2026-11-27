"""The settings loader is strict and the data tables carry their derivations."""

import pytest

from investment_sandbox.config import SECTIONS, load_settings, read_csv


def test_an_unknown_key_is_refused():
    with pytest.raises(ValueError, match="unknown key"):
        load_settings({"investment": {"caution_level": 1.0}})


def test_an_unknown_section_is_refused():
    with pytest.raises(ValueError, match="unknown settings section"):
        load_settings({"weather": {"seed": 1}})


def test_an_override_replaces_one_value_and_nothing_else(settings):
    changed = load_settings({"investment": {"caution": 0.0}})
    assert changed.investment["caution"] == 0.0
    assert changed.investment["block_fraction_of_peak"] == settings.investment["block_fraction_of_peak"]
    assert changed.tender == settings.tender


def test_a_tender_for_a_technology_with_no_cost_row_is_refused():
    with pytest.raises(ValueError, match="no cost row"):
        load_settings({"tender": {"technologies": ["nuclear"]}})


def test_a_tender_needs_a_contract_of_at_least_a_year():
    with pytest.raises(ValueError, match="contract"):
        load_settings({"tender": {"contract_years": 0}})


def test_the_forecast_must_look_far_enough_ahead_to_judge_every_technology():
    with pytest.raises(ValueError, match="too near"):
        load_settings({"forward": {"years_ahead": [2]}})
    with pytest.raises(ValueError, match="rising order"):
        load_settings({"forward": {"years_ahead": [8, 4]}})


def test_every_section_in_the_file_is_declared(settings):
    for section in SECTIONS:
        assert getattr(settings, section)


@pytest.mark.parametrize("table", ["fleet.csv", "tech_costs.csv", "growth.csv", "dsr.csv"])
def test_every_data_row_says_where_it_came_from(table):
    for row in read_csv(table):
        assert len(row["derivation"].strip()) > 20, f"{table}: {row}"


def test_growth_weights_are_a_distribution(settings):
    assert abs(sum(g.weight for g in settings.growth) - 1.0) < 1e-3


def test_fixed_cost_is_annualised_capital_plus_fixed_operating_cost(settings):
    ocgt = settings.tech("ocgt")
    crf = 0.07 / (1 - 1.07 ** -25)
    assert ocgt.fixed_cost_per_mw_year == pytest.approx(1850 * 1000 * crf + 25 * 1000)


def test_a_technology_with_no_cost_row_is_an_error(settings):
    with pytest.raises(KeyError):
        settings.tech("nuclear")


def test_every_candidate_technology_has_a_unit_size_that_divides_a_block(settings):
    for tech in settings.tech_costs:
        assert tech.unit_size_mw > 0
        assert tech.lead_years >= 1


def test_every_plant_but_coal_retires_at_the_life_a_new_plant_of_its_kind_gets(settings):
    for unit in settings.fleet:
        if unit.technology != "coal":
            life = settings.tech(unit.technology).life_years
            assert unit.retirement_year == unit.commissioned_year + life, unit.unit
