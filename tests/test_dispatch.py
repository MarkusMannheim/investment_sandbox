"""Properties every dispatched year must have."""

import time

import numpy as np
import pytest

from investment_sandbox.config import Unit
from investment_sandbox.dispatch import (Stack, battery_dispatch, build_stack, dispatch_year,
                                         rent_per_mw, unit_income)
from investment_sandbox.weather import demand_mw


@pytest.fixture(scope="module")
def years(settings, weather):
    stack = build_stack(settings.fleet, settings.years[0], settings)
    out = []
    for i in range(len(weather)):
        demand = demand_mw(weather.demand_shape[i], float(settings.demand["average_mw"]))
        out.append((stack, dispatch_year(stack, demand, weather.wind_cf[i], weather.solar_cf[i], settings), i))
    return out


def test_energy_balances_in_every_hour(years):
    for stack, d, _ in years:
        supplied = (d.wind_used + d.solar_used + d.generation.sum(axis=0)
                    + d.discharge - d.charge + d.unserved)
        assert np.allclose(supplied, d.demand, atol=1e-6)


def test_the_price_is_the_offer_of_the_last_unit_needed(years):
    for stack, d, _ in years:
        for row in range(stack.available_mw.size):
            running = d.generation[row] > 1e-9
            assert np.all(d.price[running] >= stack.offer[row] - 1e-9)
            full = d.generation[row] >= stack.available_mw[row] - 1e-9
            partial = running & ~full
            assert np.allclose(d.price[partial], stack.offer[row])


def test_unserved_energy_only_when_the_stack_is_exhausted(years, settings):
    cap = settings.market["price_cap_per_mwh"]
    for stack, d, _ in years:
        short = d.unserved > 0
        assert np.all(d.price[short] == cap)
        assert np.allclose(d.generation[:, short], stack.available_mw[:, None])


def test_curtailment_only_in_surplus_hours(years):
    for stack, d, _ in years:
        surplus = d.curtailed > 0
        assert np.all(d.price[surplus] == 0.0)
        assert np.all(d.generation[:, surplus] == 0.0)
        assert np.all(d.wind_used >= -1e-9) and np.all(d.solar_used >= -1e-9)


def battery_days(hourly, settings):
    """Hourly values as rows of the battery's own days, which start at the hour it is taken to be empty."""
    return np.roll(hourly, -int(settings.storage["day_start_hour"])).reshape(-1, 24)


def by_day(hourly, settings):
    return battery_days(hourly, settings).sum(axis=1)


def test_the_battery_keeps_its_rules(years, settings):
    efficiency = settings.storage["round_trip_efficiency"]
    for stack, d, _ in years:
        (hours, mw), = stack.battery_groups
        spare = stack.available_mw[:stack.thermal_rows].sum() - d.residual
        assert np.all(d.charge <= np.maximum(spare, 0) + 1e-6), "charges only from plant that is spare"
        assert not np.any((d.charge > 1e-9) & (d.discharge > 1e-9))
        assert d.charge.max() <= mw + 1e-9 and d.discharge.max() <= mw + 1e-9
        assert by_day(d.discharge, settings).max() <= hours * mw + 1e-6, "no more than one full cycle a day"
        assert np.allclose(by_day(d.discharge, settings), efficiency * by_day(d.charge, settings), atol=1e-6),             "each day delivers what it stored, less the losses"
        assert d.discharge.sum() > 0


def stack_of(*rows):
    """A stack of plant alone: each row is megawatts available and the price it offers at."""
    available, offer = zip(*rows)
    return Stack(thermal=(), available_mw=np.array(available, dtype=float), offer=np.array(offer, dtype=float),
                 thermal_rows=len(rows), wind_mw=0.0, solar_mw=0.0, battery_mw=0.0, battery_groups=(),
                 wind_units=(), solar_units=(), battery_units=())


def test_the_battery_moves_energy_from_the_cheapest_hours_to_the_dearest(settings):
    net = np.tile(np.r_[np.full(6, 3000.0), np.full(6, 500.0), np.full(6, 2000.0), np.full(6, 6000.0)], 365)
    stack = stack_of((1000.0, 10.0), (2000.0, 40.0), (4000.0, 100.0))
    charge, discharge = battery_dispatch(net, 100.0, 4.0, stack, settings)
    hour = np.arange(8760) % 24
    assert np.all(charge[(hour < 6) | (hour >= 12)] == 0), "charges in the six cheapest hours, 6 am to noon here"
    assert np.all(discharge[hour < 18] == 0), "delivers in the six dearest hours, 6 pm to midnight here"
    assert discharge.sum() == pytest.approx(365 * 400.0, rel=1e-3), "four hours at full power, each day"
    assert charge.sum() == pytest.approx(discharge.sum() / settings.storage["round_trip_efficiency"])
    assert np.ptp(discharge[hour >= 18]) < 1e-6, "the peak is shaved to one level, not to a notch"


def test_the_battery_rests_when_the_price_gap_does_not_cover_its_losses(settings):
    net = np.tile(np.r_[np.full(12, 1500.0), np.full(12, 4000.0)], 365)
    one_price = stack_of((8000.0, 50.0))
    close_prices = stack_of((2000.0, 46.0), (6000.0, 50.0))
    wide_prices = stack_of((2000.0, 40.0), (6000.0, 50.0))
    for stack, cycles in ((one_price, False), (close_prices, False), (wide_prices, True)):
        charge, discharge = battery_dispatch(net, 100.0, 4.0, stack, settings)
        assert (discharge.sum() > 0) == cycles and (charge.sum() > 0) == cycles


def test_the_battery_cannot_charge_when_no_plant_is_spare(settings):
    net = np.tile(np.r_[np.full(18, 3000.0), np.full(6, 6000.0)], 365)
    charge, discharge = battery_dispatch(net, 100.0, 4.0, stack_of((3000.0, 40.0)), settings)
    assert charge.sum() == 0.0 and discharge.sum() == 0.0


def test_the_battery_seldom_delivers_energy_it_has_not_yet_stored(years, settings):
    efficiency = settings.storage["round_trip_efficiency"]
    for stack, d, _ in years:
        (hours, mw), = stack.battery_groups
        moved = battery_days(d.discharge - d.charge, settings)
        held = np.cumsum(np.where(moved < 0, -efficiency * moved, -moved), axis=1)
        borrowed = np.maximum(-held.min(axis=1), 0.0) / (hours * mw)
        assert borrowed.mean() < 0.10, "from an empty start, delivery runs ahead of charging by under a tenth of a store"
        assert held.max() <= hours * mw + 1e-6, "and the store is never over-full"


def test_the_battery_never_loses_money_over_a_day(years, settings):
    for stack, d, _ in years:
        assert by_day((d.discharge - d.charge) * d.price, settings).min() >= -1e-6


def test_more_batteries_never_mean_more_unmet_demand(settings, weather):
    late = settings.years[8]
    base = {}
    for extra_mw in (0.0, 2000.0, 6000.0, 20000.0):
        more = settings.fleet + (Unit("more", "battery_4h", extra_mw, 1.0, 0.0, 2000, 2100, 19.0, 4.0),)
        stack = build_stack(more, late, settings)
        for i in range(len(weather)):
            demand = demand_mw(weather.demand_shape[i], float(settings.demand["average_mw"]))
            d = dispatch_year(stack, demand, weather.wind_cf[i], weather.solar_cf[i], settings)
            assert not np.any((d.unserved > 0) & (d.charge > 0)), "never charges in an hour that is short"
            base.setdefault(i, d.unserved_mwh)
            assert d.unserved_mwh <= base[i] + 1e-6,                 f"{extra_mw:.0f} MW more battery raised unmet demand in weather year {i}"
        if extra_mw == 20000.0:
            assert d.unserved_mwh < base[i], "and enough of them lower it"


def test_batteries_of_different_lengths_each_keep_their_own_books(settings, weather):
    long = Unit("long", "battery_8h", 1000.0, 1.0, 0.0, 2000, 2100, 19.0, 8.0)
    stack = build_stack(settings.fleet + (long,), settings.years[0], settings)
    demand = demand_mw(weather.demand_shape[0], float(settings.demand["average_mw"]))
    d = dispatch_year(stack, demand, weather.wind_cf[0], weather.solar_cf[0], settings)
    assert [hours for hours, _ in stack.battery_groups] == [4.0, 8.0]
    assert by_day(np.maximum(d.stored[8.0], 0), settings).max() <= 8000.0 + 1e-6
    assert np.allclose(d.stored[4.0] + d.stored[8.0], d.discharge - d.charge)
    assert "long" in unit_income(d, stack)


def test_income_adds_up_to_the_market_paid_out(years):
    for stack, d, _ in years:
        income = unit_income(d, stack)
        running_cost = sum((d.generation[row] * unit.srmc_per_mwh).sum()
                           for row, unit in enumerate(stack.thermal))
        dsr_cost = (d.generation[stack.thermal_rows:] * stack.offer[stack.thermal_rows:, None]).sum()
        paid = (d.price * (d.demand - d.unserved)).sum()
        dsr_margin = ((d.price - stack.offer[stack.thermal_rows:, None]) * d.generation[stack.thermal_rows:]).sum()
        assert sum(income.values()) == pytest.approx(paid - running_cost - dsr_cost - dsr_margin, rel=1e-9)


def test_a_price_taker_earns_about_what_a_marginal_unit_in_the_fleet_earned(years, settings, weather):
    for stack, d, i in years:
        ocgt = settings.tech("ocgt")
        per_mw = rent_per_mw(d.price, ocgt, weather.wind_cf[i], weather.solar_cf[i])
        fleet_ocgt = next(u for u in stack.thermal if u.unit == "ocgt_a")
        realised = unit_income(d, stack)["ocgt_a"] / fleet_ocgt.capacity_mw
        assert per_mw == pytest.approx(realised, rel=1e-6)


def test_the_starting_system_sits_inside_the_reliability_standard(years):
    for _, d, _ in years:
        assert d.unserved_mwh / d.demand.sum() <= 0.00002, "0.002% of energy, the standard in force"


def test_a_dispatched_year_stays_inside_the_budget(years, settings, weather):
    stack, _, _ = years[2]
    demand = demand_mw(weather.demand_shape[2], float(settings.demand["average_mw"]))
    for _ in range(3):
        dispatch_year(stack, demand, weather.wind_cf[2], weather.solar_cf[2], settings)
    start = time.perf_counter()
    for _ in range(20):
        dispatch_year(stack, demand, weather.wind_cf[2], weather.solar_cf[2], settings)
    per_year_ms = (time.perf_counter() - start) / 20 * 1000
    assert per_year_ms < 30.0, f"{per_year_ms:.2f} ms a year against a 30 ms budget"
