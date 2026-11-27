"""The forecast: what a megawatt of each technology would earn in each outlook, over the years ahead that it looks at."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings, Unit
from .dispatch import battery_income_per_mw, build_stack, dispatch_year, rent_per_mw
from .weather import Weather, demand_mw


@dataclass(frozen=True)
class Future:
    growth_path: str
    annual_growth: float
    weather_year: int
    weight: float


@dataclass(frozen=True)
class View:
    """A forecast made in one year, with one entry an outlook in every array."""

    year: int
    target_year: int                    # the nearest year the forecast looks at
    futures: tuple[Future, ...]
    weights: np.ndarray
    rents: dict[str, np.ndarray]        # technology -> $ per MW-year above running cost
    unserved_mwh: np.ndarray            # in the target year
    demand_mwh: np.ndarray              # in the target year
    mean_price: np.ndarray              # time-weighted $/MWh in the target year

    def expected(self, technology: str) -> float:
        return float(self.rents[technology] @ self.weights)

    @property
    def expected_unserved_mwh(self) -> float:
        return float(self.unserved_mwh @ self.weights)

    @property
    def unserved_share(self) -> float:
        """Expected unmet demand as a share of expected demand."""
        return self.expected_unserved_mwh / float(self.demand_mwh @ self.weights)


def anticipated_entry(pace_mw: dict[str, float], year: int, ahead: int, settings: Settings) -> tuple[Unit, ...]:
    """The plant assumed to be added before a year ahead if building continues at its recent pace."""
    out = []
    for tech in settings.tech_costs:
        years = max(0, ahead - tech.lead_years)
        mw = pace_mw.get(tech.technology, 0.0) * years
        if mw > 0:
            out.append(Unit(f"anticipated_{tech.technology}", tech.technology, mw, tech.availability,
                            tech.srmc_per_mwh, year + ahead, year + ahead + 1, tech.fom_per_kw_year,
                            tech.duration_h))
    return tuple(out)


def one_year(units: tuple[Unit, ...], year: int, ahead: int, average_mw: float, settings: Settings,
             weather: Weather, pace_mw: dict[str, float] | None = None) -> View:
    """Dispatch one year ahead in every outlook and work out what a megawatt of each technology would earn in it."""
    target = year + ahead
    fleet = units + anticipated_entry(pace_mw or {}, year, ahead, settings)
    # A battery is priced on what batteries of its length earn, so a length not yet in the fleet gets one megawatt
    lengths = {u.duration_h for u in fleet if u.is_storage and u.in_service(target)}
    fleet += tuple(Unit(f"first_{t.technology}", t.technology, 1.0, t.availability, t.srmc_per_mwh, target, target + 1,
                        t.fom_per_kw_year, t.duration_h) for t in settings.tech_costs
                   if t.duration_h and t.duration_h not in lengths)
    stack = build_stack(fleet, target, settings)
    futures, rents = [], {t.technology: [] for t in settings.tech_costs}
    unserved, energy, mean_price = [], [], []
    for path in settings.growth:
        average = average_mw * (1.0 + path.annual_growth) ** ahead
        for w, weather_year in enumerate(weather.years):
            futures.append(Future(path.path, path.annual_growth, weather_year, path.weight / len(weather)))
            demand = demand_mw(weather.demand_shape[w], average)
            result = dispatch_year(stack, demand, weather.wind_cf[w], weather.solar_cf[w], settings)
            stored = battery_income_per_mw(result, stack)
            for tech in settings.tech_costs:
                rents[tech.technology].append(
                    stored[tech.duration_h] if tech.duration_h
                    else rent_per_mw(result.price, tech, weather.wind_cf[w], weather.solar_cf[w]))
            unserved.append(result.unserved_mwh)
            energy.append(float(demand.sum()))
            mean_price.append(float(result.price.mean()))
    weights = np.array([f.weight for f in futures])
    weights = weights / weights.sum()
    return View(year=year, target_year=target, futures=tuple(futures), weights=weights,
                rents={k: np.array(v) for k, v in rents.items()},
                unserved_mwh=np.array(unserved), demand_mwh=np.array(energy),
                mean_price=np.array(mean_price))


def forward_view(units: tuple[Unit, ...], year: int, average_mw: float, settings: Settings, weather: Weather,
                 pace_mw: dict[str, float] | None = None) -> View:
    """Forecast each year ahead that the settings name and average what each technology would earn across them."""
    distances = [int(k) for k in settings.forward["years_ahead"]]
    views = [one_year(units, year, k, average_mw, settings, weather, pace_mw) for k in distances]
    rents = {}
    for tech in settings.tech_costs:
        # A plant is judged on the years by which it could be running
        running = [v.rents[tech.technology] for k, v in zip(distances, views) if k >= tech.lead_years]
        rents[tech.technology] = np.mean(running, axis=0)
    nearest = views[0]
    return View(year=year, target_year=nearest.target_year, futures=nearest.futures, weights=nearest.weights,
                rents=rents, unserved_mwh=nearest.unserved_mwh, demand_mwh=nearest.demand_mwh,
                mean_price=nearest.mean_price)
