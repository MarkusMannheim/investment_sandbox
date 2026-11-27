"""The forecast: nine outlooks at fixed odds for each year it looks at, priced by the same dispatch."""

import numpy as np
import pytest

from investment_sandbox.forward import forward_view


def test_nine_futures_with_weights_that_sum_to_one(settings, weather):
    view = forward_view(settings.fleet, settings.years[0], 7000.0, settings, weather)
    assert len(view.futures) == 9
    assert np.isclose(view.weights.sum(), 1.0)
    assert view.target_year == settings.years[0] + settings.forward["years_ahead"][0]
    for path in settings.growth:
        share = sum(w for f, w in zip(view.futures, view.weights) if f.growth_path == path.path)
        assert np.isclose(share, path.weight, atol=1e-3)


def test_rents_are_never_negative(settings, weather):
    view = forward_view(settings.fleet, settings.years[0], 7000.0, settings, weather)
    for tech, rents in view.rents.items():
        assert rents.shape == (9,) and rents.min() >= 0.0, tech


def test_more_plant_in_the_fleet_means_lower_rent_for_every_technology(settings, weather):
    from investment_sandbox.config import Unit
    extra = Unit("ocgt_x", "ocgt", 1000.0, 0.9, 181.0, 2020, 2060, 16.0, None)
    thin = forward_view(settings.fleet, settings.years[0], 7000.0, settings, weather)
    thick = forward_view(settings.fleet + (extra,), settings.years[0], 7000.0, settings, weather)
    for tech in thin.rents:
        assert thick.expected(tech) <= thin.expected(tech) + 1e-6
    assert thick.expected_unserved_mwh <= thin.expected_unserved_mwh


def test_anticipated_building_by_others_lowers_every_rent(settings, weather):
    plain = forward_view(settings.fleet, settings.years[0], 7000.0, settings, weather)
    busy = forward_view(settings.fleet, settings.years[0], 7000.0, settings, weather,
                        pace_mw={"ccgt": 500.0, "ocgt": 300.0, "solar": 400.0})
    for tech in plain.rents:
        assert busy.expected(tech) <= plain.expected(tech) + 1e-6
    assert busy.expected_unserved_mwh < plain.expected_unserved_mwh


def test_a_technology_whose_lead_time_fills_the_horizon_is_never_anticipated(settings):
    from investment_sandbox.forward import anticipated_entry
    slow = anticipated_entry({"ccgt": 1000.0, "ocgt": 1000.0}, 2030, 4, settings)
    by_tech = {u.technology: u.capacity_mw for u in slow}
    assert by_tech == {"ccgt": 1000.0, "ocgt": 2000.0}, "one year of ccgt decisions arrive in time, two of ocgt"


def test_a_plant_is_judged_on_the_average_of_the_years_the_forecast_looks_at(settings, weather):
    from investment_sandbox.forward import one_year
    year = settings.years[0]
    view = forward_view(settings.fleet, year, 7000.0, settings, weather)
    each = [one_year(settings.fleet, year, k, 7000.0, settings, weather) for k in settings.forward["years_ahead"]]
    for tech in settings.tech_costs:
        assert np.allclose(view.rents[tech.technology], np.mean([v.rents[tech.technology] for v in each], axis=0))
    assert np.allclose(view.unserved_mwh, each[0].unserved_mwh), "unmet demand is read from the nearest year"


def test_a_plant_too_slow_for_the_nearest_year_is_judged_on_the_later_one(settings, weather):
    import dataclasses
    from investment_sandbox.forward import one_year
    slow = dataclasses.replace(settings, tech_costs=tuple(
        dataclasses.replace(t, lead_years=6) if t.technology == "ccgt" else t for t in settings.tech_costs))
    year = slow.years[0]
    view = forward_view(slow.fleet, year, 7000.0, slow, weather)
    later = one_year(slow.fleet, year, slow.forward["years_ahead"][-1], 7000.0, slow, weather)
    assert np.allclose(view.rents["ccgt"], later.rents["ccgt"])


def test_demand_in_the_forecast_grows_for_as_many_years_as_it_looks_ahead(settings, weather):
    from investment_sandbox.forward import one_year
    last = settings.years[-1]
    for k in (4, 8):
        view = one_year(settings.fleet, last, k, 7000.0, settings, weather)
        assert view.target_year == last + k, "a forecast made in the last year looks past the end of the run"
        for outlook, energy in zip(view.futures, view.demand_mwh):
            assert energy == pytest.approx(7000.0 * 8760 * (1 + outlook.annual_growth) ** k, rel=1e-6)
