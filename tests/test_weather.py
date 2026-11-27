"""The weather table is three whole years of shapes in a sensible range."""

import numpy as np
import pytest

from investment_sandbox.weather import HOURS, demand_mw


def test_three_years_of_8760_hours(weather):
    assert len(weather) == 3
    assert weather.demand_shape.shape == (3, HOURS)
    assert weather.wind_cf.shape == (3, HOURS)
    assert weather.solar_cf.shape == (3, HOURS)


def test_each_demand_shape_has_a_mean_of_one(weather):
    assert np.allclose(weather.demand_shape.mean(axis=1), 1.0, atol=1e-4)
    assert weather.demand_shape.min() > 0


def test_capacity_factors_lie_between_zero_and_one(weather):
    for cf in (weather.wind_cf, weather.solar_cf):
        assert cf.min() >= 0.0 and cf.max() <= 1.0
    assert (weather.solar_cf[:, 0::24] == 0).mean() > 0.9, "solar at midnight should be zero"


def test_demand_scales_with_average(weather):
    demand = demand_mw(weather.demand_shape[0], 7000.0)
    assert demand.mean() == pytest.approx(7000.0, rel=1e-4)


def test_years_sit_in_descending_stress_order(weather):
    # The notebook says index zero is the most stressed year and two the least. Stress is
    # the derivation script's measure: the mean of the 50 highest hours of residual demand
    # after wind and solar fleets each sized at three tenths of average demand
    residual = weather.demand_shape - 0.3 * (weather.wind_cf + weather.solar_cf)
    stress = np.sort(residual, axis=1)[:, -50:].mean(axis=1)
    assert stress[0] > stress[1] > stress[2]
