"""Every chart draws from real output, carries no title and uses the house colours."""

import pytest

from investment_sandbox import explore, plots
from investment_sandbox.simulate import compare

HOUSE = {"#333333", "#708090", "#375e97", "#6992cf", "#3bb16a", "#fb6542", "#fda58f", "#ffbb00"}


@pytest.fixture(scope="module")
def runs(settings):
    return compare(settings, seed=1, years=6)


@pytest.fixture(scope="module")
def figures(runs):
    year = runs["cautious"].years[-1].year
    return {
        "committed": plots.committed(runs), "capacity": plots.capacity(runs),
        "generation": plots.generation(runs), "prices": plots.prices(runs),
        "reliability": plots.reliability(runs), "costs": plots.costs(runs),
        "hurdle": plots.hurdle(runs["cautious"], year),
    }


def test_every_chart_has_data_and_no_title(figures):
    for name, fig in figures.items():
        assert fig.data, name
        assert fig.layout.title.text is None, f"{name} carries a title"
        assert fig.to_json()


def test_every_colour_is_a_house_colour(figures):
    assert set(plots.COLOUR.values()) | set(plots.CASE_COLOUR.values()) <= HOUSE
    assert len(set(plots.CASE_COLOUR.values())) == len(plots.CASE_COLOUR)
    for name, fig in figures.items():
        for trace in fig.data:
            colour = trace.marker.color if trace.type == "bar" else trace.line.color or trace.marker.line.color
            assert colour in HOUSE, f"{name}: {trace.name} is drawn in {colour}"


def test_every_technology_and_run_has_a_colour_and_a_label(settings):
    techs = {u.technology for u in settings.fleet} | {t.technology for t in settings.tech_costs}
    assert techs | {"demand_response"} == set(plots.ORDER) == set(plots.COLOUR) == set(explore.TECHNOLOGY)
    assert set(plots.CASE_COLOUR) == set(explore.CASE)


def test_plant_bought_by_tender_is_hatched(runs, figures):
    assert runs["tender"].contracts
    hatched = [t for t in figures["committed"].data if t.marker.pattern.shape == "/"]
    assert hatched


def test_generation_by_technology_adds_up_to_demand_met_and_battery_charging(settings, weather):
    from investment_sandbox.dispatch import build_stack, dispatch_year, generation_by_technology
    from investment_sandbox.weather import demand_mw
    stack = build_stack(settings.fleet, settings.years[0], settings)
    demand = demand_mw(weather.demand_shape[0], float(settings.demand["average_mw"]))
    d = dispatch_year(stack, demand, weather.wind_cf[0], weather.solar_cf[0], settings)
    supplied = sum(generation_by_technology(d, stack).values())
    assert supplied == pytest.approx(float((demand - d.unserved + d.charge).sum()))


def test_the_headline_table_has_a_column_a_run_and_the_same_rows_in_each(runs):
    table = explore.headline(runs)
    assert list(table) == [explore.CASE[name] for name in runs]
    rows = [list(column) for column in table.values()]
    assert rows[0] == rows[1] == rows[2]
    assert table["Caution"]["Unmet demand, GWh"] == pytest.approx(runs["cautious"].unserved_gwh)
