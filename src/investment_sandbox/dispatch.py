"""Dispatch one year hour by hour: plant stacked cheapest first, batteries moving energy within each day, and the price set by the last unit needed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings, TechCost, Unit


@dataclass(frozen=True)
class Stack:
    """The merit order for one fleet state: thermal rows, then demand response."""

    thermal: tuple[Unit, ...]
    available_mw: np.ndarray        # thermal rows then demand-response rungs
    offer: np.ndarray               # the price each row offers at
    thermal_rows: int
    wind_mw: float
    solar_mw: float
    battery_mw: float
    battery_groups: tuple[tuple[float, float], ...]     # (hours of storage, MW), shortest first
    wind_units: tuple[Unit, ...]
    solar_units: tuple[Unit, ...]
    battery_units: tuple[Unit, ...]


def build_stack(units: tuple[Unit, ...], year: int, settings: Settings) -> Stack:
    live = [u for u in units if u.in_service(year)]
    thermal = sorted((u for u in live if not u.is_storage and not u.is_variable),
                     key=lambda u: u.srmc_per_mwh)
    wind = tuple(u for u in live if u.technology == "wind")
    solar = tuple(u for u in live if u.technology == "solar")
    stores = tuple(u for u in live if u.is_storage)
    groups = tuple((hours, sum(u.capacity_mw for u in stores if u.duration_h == hours))
                   for hours in sorted({u.duration_h for u in stores}))
    available = [u.available_mw for u in thermal] + [t.capacity_mw for t in settings.dsr]
    offer = [u.srmc_per_mwh for u in thermal] + [t.price_per_mwh for t in settings.dsr]
    return Stack(
        thermal=tuple(thermal), available_mw=np.array(available, dtype=float),
        offer=np.array(offer, dtype=float), thermal_rows=len(thermal),
        wind_mw=sum(u.capacity_mw for u in wind), solar_mw=sum(u.capacity_mw for u in solar),
        battery_mw=sum(mw for _, mw in groups), battery_groups=groups,
        wind_units=wind, solar_units=solar, battery_units=stores,
    )


@dataclass(frozen=True)
class Dispatch:
    """One dispatched year. Every array has one entry an hour."""

    demand: np.ndarray
    price: np.ndarray
    residual: np.ndarray            # demand less wind and solar, before the battery
    generation: np.ndarray          # (stack rows, hours): thermal then demand response
    wind_used: np.ndarray
    solar_used: np.ndarray
    curtailed: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    unserved: np.ndarray
    stored: dict[float, np.ndarray]     # hours of storage -> discharge less charge, MW

    @property
    def unserved_mwh(self) -> float:
        return float(self.unserved.sum())


ROUNDING_MW = 1e-6


def shaving_level(values: np.ndarray, power_mw: float, energy: np.ndarray) -> np.ndarray:
    """The level that each row of values must be shaved down to, by no more than the power in any hour, to remove the energy."""
    # An hour starts to give below its own value and is at full power once the level is the power below that.
    # Sorting the two kinds of point together, highest first, carries each point's kind in the imaginary part
    points = np.sort(np.concatenate((-values + 1j, power_mw - values - 1j), axis=1), axis=1)
    giving = np.cumsum(points.imag, axis=1)
    points = -points.real
    removed = np.zeros_like(points)
    removed[:, 1:] = np.cumsum(giving[:, :-1] * -np.diff(points, axis=1), axis=1)
    row = np.arange(len(values))
    at = (removed <= energy[:, None]).sum(axis=1) - 1
    left, hours = energy - removed[row, at], giving[row, at]
    return points[row, at] - np.divide(left, hours, out=np.zeros_like(left), where=hours > 0)


def battery_dispatch(net: np.ndarray, power_mw: float, hours: float, stack: Stack,
                     settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Charge and discharge, in MW an hour, that move each day's energy from its cheapest hours to its dearest wherever the price gap covers the losses."""
    if power_mw <= 0 or hours <= 0:
        return np.zeros_like(net), np.zeros_like(net)
    efficiency = float(settings.storage["round_trip_efficiency"])
    # The battery's day starts at the hour it is taken to be empty
    start = int(settings.storage["day_start_hour"])
    days = np.roll(net, -start).reshape(-1, 24)
    # Plant offering at one price is one step; demand response follows as the stack has it
    first = np.flatnonzero(np.diff(stack.offer[:stack.thermal_rows], prepend=-1.0) > 0)
    plant = first.size
    rest = slice(stack.thermal_rows, None)
    steps = np.concatenate((np.add.reduceat(stack.available_mw[:stack.thermal_rows], first) if plant else [],
                            stack.available_mw[rest]))
    # Net demand at which each step starts, and the price below, on and above the steps
    edges = np.concatenate(([0.0], np.cumsum(steps)))
    prices = np.concatenate(([0.0], stack.offer[first], stack.offer[rest],
                             [float(settings.market["price_cap_per_mwh"])]))

    # Energy sold by shaving the day down to the foot of each step, and bought by filling it to the top of each step of plant
    sold = np.clip(days[:, None, :] - edges[None, :, None], 0.0, power_mw).sum(axis=2)
    bought = np.clip(edges[None, :plant + 1, None] - days[:, None, :], 0.0, power_mw).sum(axis=2)
    # Selling on a step pays only against energy bought on steps cheap enough to cover the losses
    cheap_enough = np.searchsorted(prices[:plant + 1], efficiency * prices[1:], side="left") - 1
    worth = np.minimum(sold, efficiency * bought[:, cheap_enough.clip(min=0)] * (cheap_enough >= 0))
    energy = np.minimum(worth.max(axis=1), power_mw * hours)

    discharge = np.clip(days - shaving_level(days, power_mw, energy)[:, None], 0.0, power_mw)
    charge = np.clip(-shaving_level(-days, power_mw, energy / efficiency)[:, None] - days, 0.0, power_mw)
    return np.roll(charge.reshape(-1), start), np.roll(discharge.reshape(-1), start)


def dispatch_year(stack: Stack, demand: np.ndarray, wind_cf: np.ndarray,
                  solar_cf: np.ndarray, settings: Settings) -> Dispatch:
    """Price every hour of a year and say what each row generated."""
    wind_avail = wind_cf * stack.wind_mw
    solar_avail = solar_cf * stack.solar_mw
    residual = demand - wind_avail - solar_avail
    net = residual
    stored = {}
    for hours, mw in stack.battery_groups:
        charge, discharge = battery_dispatch(net, mw, hours, stack, settings)
        stored[hours] = discharge - charge
        net = net - stored[hours]
    moved = residual - net
    charge, discharge = np.maximum(-moved, 0.0), np.maximum(moved, 0.0)
    curtailed = np.maximum(-net, 0.0)
    net = np.maximum(net, 0.0)

    cumulative = np.cumsum(stack.available_mw)
    # A battery that has just displaced a plant sells at that plant's offer
    nudge = np.where(discharge > 0, ROUNDING_MW, -ROUNDING_MW)
    marginal = np.searchsorted(cumulative, net + nudge, side="left")
    rows = stack.available_mw.size
    price = np.where(marginal < rows, stack.offer[np.minimum(marginal, rows - 1)],
                     float(settings.market["price_cap_per_mwh"]))
    price = np.where(net > ROUNDING_MW, price, 0.0)
    before = np.concatenate(([0.0], cumulative[:-1]))
    generation = np.clip(net[None, :] - before[:, None], 0.0, stack.available_mw[:, None])
    unserved = np.maximum(net - cumulative[-1], 0.0)

    vre_avail = wind_avail + solar_avail
    share = np.divide(wind_avail, vre_avail, out=np.zeros_like(wind_avail), where=vre_avail > 0)
    wind_used = wind_avail - curtailed * share
    solar_used = solar_avail - curtailed * (1.0 - share)
    return Dispatch(demand=demand, price=price, residual=residual, generation=generation,
                    wind_used=wind_used, solar_used=solar_used, curtailed=curtailed,
                    charge=charge, discharge=discharge, unserved=unserved, stored=stored)


def rent_per_mw(price: np.ndarray, tech: TechCost | Unit, wind_cf: np.ndarray, solar_cf: np.ndarray) -> float:
    """What one MW of plant, wind or solar would earn above its running cost at these prices, as a price-taker, per year."""
    if tech.technology == "wind":
        return float((wind_cf * price).sum())
    if tech.technology == "solar":
        return float((solar_cf * price).sum())
    return float(tech.availability * np.maximum(price - tech.srmc_per_mwh, 0.0).sum())


def battery_income_per_mw(dispatch: Dispatch, stack: Stack) -> dict[float, float]:
    """What a megawatt of the batteries of each length earned in the year."""
    return {hours: float((dispatch.stored[hours] * dispatch.price).sum()) / mw for hours, mw in stack.battery_groups}


def generation_by_technology(dispatch: Dispatch, stack: Stack) -> dict[str, float]:
    """The year's energy supplied by each technology, in MWh, with demand response and battery discharge counted as supply."""
    out: dict[str, float] = {}
    for row, unit in enumerate(stack.thermal):
        out[unit.technology] = out.get(unit.technology, 0.0) + float(dispatch.generation[row].sum())
    out["wind"] = float(dispatch.wind_used.sum())
    out["solar"] = float(dispatch.solar_used.sum())
    for unit in stack.battery_units:
        out[unit.technology] = float(np.maximum(dispatch.stored[unit.duration_h], 0.0).sum())
    out["demand_response"] = float(dispatch.generation[stack.thermal_rows:].sum())
    return out


def unit_income(dispatch: Dispatch, stack: Stack) -> dict[str, float]:
    """Each plant's realised income above its running cost in the year, by unit name."""
    income: dict[str, float] = {}
    for row, unit in enumerate(stack.thermal):
        income[unit.unit] = float((dispatch.generation[row] * (dispatch.price - unit.srmc_per_mwh)).sum())
    for units, used in ((stack.wind_units, dispatch.wind_used), (stack.solar_units, dispatch.solar_used)):
        total = sum(u.capacity_mw for u in units)
        for unit in units:
            income[unit.unit] = float((used * dispatch.price).sum() * unit.capacity_mw / total)
    per_mw = battery_income_per_mw(dispatch, stack)
    for unit in stack.battery_units:
        income[unit.unit] = per_mw[unit.duration_h] * unit.capacity_mw
    return income
