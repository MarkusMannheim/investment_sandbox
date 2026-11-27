"""One market run forward a year at a time: dispatch, forecast, the firms' decisions, the tender, then growth in demand."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GrowthPath, Settings, Unit
from .dispatch import build_stack, dispatch_year, generation_by_technology, unit_income
from .forward import View
from .investment import Build, Test, decide
from .tender import Contract, hold
from .weather import Weather, demand_mw, load_weather


@dataclass(frozen=True)
class Draw:
    """The future a run faces: one demand path and a weather year for each year."""

    growth: GrowthPath
    weather_index: tuple[int, ...]


def draw(settings: Settings, weather: Weather, seed: int, path: str | None = None) -> Draw:
    """Draw a run's weather years from the seed, and its demand path too unless one is named."""
    rng = np.random.default_rng(seed)
    weights = np.array([g.weight for g in settings.growth])
    growth = settings.growth[int(rng.choice(len(settings.growth), p=weights / weights.sum()))]
    if path is not None:
        named = [g for g in settings.growth if g.path == path]
        if not named:
            raise ValueError(f"no demand path called {path!r}")
        growth = named[0]
    years = int(settings.time["years"])
    return Draw(growth, tuple(int(i) for i in rng.integers(0, len(weather), size=years)))


def by_technology(units) -> dict[str, float]:
    """Megawatts by technology."""
    out: dict[str, float] = {}
    for u in units:
        out[u.technology] = out.get(u.technology, 0.0) + u.capacity_mw
    return out


@dataclass
class Year:
    """What happened in one year."""

    year: int
    weather_year: int
    average_mw: float
    peak_mw: float
    mean_price: float               # time-weighted $/MWh
    cap_hours: int                  # hours priced at the price cap
    served_mwh: float
    unserved_mwh: float
    capacity_mw: dict[str, float]
    generation_mwh: dict[str, float]
    income: dict[str, float]        # each plant's income above its running cost
    view: View                      # the forecast before anything was committed
    first_tests: list[Test]
    builds: list[Build]             # committed by the firms
    awards: list[Contract]          # bought by the tender
    stopped: bool
    energy_cost: float              # energy supplied, at the spot price
    unserved_cost: float            # unmet demand at the price cap
    running_cost: float             # fuel, and demand response at its price
    fixed_cost: float               # fixed operating cost of plant in service
    capital_cost: float             # annualised capital of plant committed during the run
    underwriting_cost: float        # what the contracts cost whoever stands behind them, net

    @property
    def bill(self) -> float:
        """What consumers pay for energy: supply at the spot price and unmet demand at the price cap."""
        return self.energy_cost + self.unserved_cost

    @property
    def resource_cost(self) -> float:
        """What supplying the year's electricity used up: fuel, upkeep, capital and unmet demand at the price cap."""
        return self.running_cost + self.fixed_cost + self.capital_cost + self.unserved_cost

    @property
    def price_paid(self) -> float:
        """The average price of a megawatt hour supplied."""
        return self.energy_cost / self.served_mwh

    @property
    def built_mw(self) -> dict[str, float]:
        return by_technology(b.unit for b in self.builds)

    @property
    def bought_mw(self) -> dict[str, float]:
        return by_technology(c.unit for c in self.awards)

    @property
    def committed_mw(self) -> dict[str, float]:
        """Megawatts committed in the year by the firms and the tender together."""
        return by_technology([b.unit for b in self.builds] + [c.unit for c in self.awards])


@dataclass
class Run:
    settings: Settings
    seed: int
    caution: float
    tender: bool
    draw: Draw
    years: list[Year] = field(default_factory=list)
    units: tuple[Unit, ...] = ()
    contracts: list[Contract] = field(default_factory=list)

    def _total(self, pick) -> dict[str, float]:
        out: dict[str, float] = {}
        for y in self.years:
            for technology, mw in pick(y).items():
                out[technology] = out.get(technology, 0.0) + mw
        return out

    @property
    def built_mw(self) -> dict[str, float]:
        """Megawatts the firms committed over the run, by technology."""
        return self._total(lambda y: y.built_mw)

    @property
    def bought_mw(self) -> dict[str, float]:
        """Megawatts the tender bought over the run, by technology."""
        return self._total(lambda y: y.bought_mw)

    @property
    def committed_mw(self) -> dict[str, float]:
        return self._total(lambda y: y.committed_mw)

    @property
    def unserved_gwh(self) -> float:
        return sum(y.unserved_mwh for y in self.years) / 1e3

    @property
    def bill(self) -> float:
        return sum(y.bill for y in self.years)

    @property
    def underwriting_cost(self) -> float:
        """What the contracts cost their underwriter in the years of the run."""
        return sum(y.underwriting_cost for y in self.years)

    @property
    def expected_underwriting_cost(self) -> float:
        """What the contracts were expected to cost over their whole terms when they were signed."""
        return sum(c.expected_cost for c in self.contracts)

    @property
    def resource_cost(self) -> float:
        return sum(y.resource_cost for y in self.years)

    @property
    def price_paid(self) -> float:
        return sum(y.energy_cost for y in self.years) / sum(y.served_mwh for y in self.years)

    @property
    def cap_hours(self) -> int:
        return sum(y.cap_hours for y in self.years)


PACE_YEARS = 5


def pace(decided: list[dict[str, float]]) -> dict[str, float]:
    """The megawatts committed a year, by technology, on average over the last five years."""
    recent = decided[-PACE_YEARS:]
    out: dict[str, float] = {}
    for year in recent:
        for tech, mw in year.items():
            out[tech] = out.get(tech, 0.0) + mw / len(recent)
    return out


def opening_pace(settings: Settings) -> list[dict[str, float]]:
    """The pace the run starts with: the plant under construction, as though committed evenly over the five years before."""
    start = settings.years[0]
    pipeline = by_technology(u for u in settings.fleet if u.commissioned_year >= start)
    return [{tech: mw / PACE_YEARS for tech, mw in pipeline.items()} for _ in range(PACE_YEARS)]


def capacity_by_technology(units: tuple[Unit, ...], year: int) -> dict[str, float]:
    return by_technology(u for u in units if u.in_service(year))


def run(settings: Settings, seed: int = 1, caution: float | None = None, tender: bool = True,
        the_draw: Draw | None = None, years: int | None = None, weather: Weather | None = None) -> Run:
    """Run the market forward from the start year."""
    weather = weather or load_weather()
    the_draw = the_draw or draw(settings, weather, seed)
    caution = float(settings.investment["caution"]) if caution is None else float(caution)
    tender = tender and bool(settings.tender["technologies"])
    horizon = list(settings.years)[: years or len(settings.years)]
    cap = float(settings.market["price_cap_per_mwh"])
    units = tuple(settings.fleet)
    average = float(settings.demand["average_mw"])
    result = Run(settings, seed, caution, tender, the_draw)
    decided = opening_pace(settings)

    for i, year in enumerate(horizon):
        w = the_draw.weather_index[i]
        stack = build_stack(units, year, settings)
        demand = demand_mw(weather.demand_shape[w], average)
        dispatched = dispatch_year(stack, demand, weather.wind_cf[w], weather.solar_cf[w], settings)
        live = [u for u in units if u.in_service(year)]
        income = unit_income(dispatched, stack)
        underwriting = sum(c.cost(income[c.unit.unit]) for c in result.contracts if c.live(year))

        peak = float(demand.max())
        recent = pace(decided)
        decision = decide(units, year, average, peak, settings, weather, caution, recent)
        units = units + tuple(b.unit for b in decision.builds)
        awards, tender_stopped = [], False
        if tender:
            # The tender is held after the firms decide, so it buys only what they leave short
            awards, tender_stopped = hold(units, year, average, peak, decision.last_view, settings,
                                          weather, caution, recent)
            units = units + tuple(c.unit for c in awards)
            result.contracts += awards

        served = demand - dispatched.unserved
        result.years.append(Year(
            year=year, weather_year=weather.years[w], average_mw=average, peak_mw=peak,
            mean_price=float(dispatched.price.mean()),
            cap_hours=int((dispatched.price >= cap).sum()),
            served_mwh=float(served.sum()),
            unserved_mwh=dispatched.unserved_mwh,
            capacity_mw=capacity_by_technology(units, year),
            generation_mwh=generation_by_technology(dispatched, stack),
            income=income, view=decision.first_view, first_tests=decision.first_tests,
            builds=decision.builds, awards=awards, stopped=decision.stopped or tender_stopped,
            energy_cost=float((dispatched.price * served).sum()),
            unserved_cost=dispatched.unserved_mwh * cap,
            running_cost=float((dispatched.generation * stack.offer[:, None]).sum()),
            fixed_cost=sum(u.fom_per_kw_year * 1000.0 * u.capacity_mw for u in live),
            capital_cost=sum(u.annual_capital_per_mw * u.capacity_mw for u in live),
            underwriting_cost=underwriting,
        ))
        decided.append(result.years[-1].committed_mw)
        average *= 1.0 + the_draw.growth.annual_growth
    result.units = units
    return result


def compare(settings: Settings, seed: int = 1, caution: float | None = None,
            years: int | None = None, path: str | None = None) -> dict[str, Run]:
    """The same market three times on one draw: firms with no caution, cautious firms, and cautious firms with the tender."""
    weather = load_weather()
    the_draw = draw(settings, weather, seed, path)
    caution = float(settings.investment["caution"]) if caution is None else float(caution)
    return {
        "risk_neutral": run(settings, seed, 0.0, False, the_draw, years, weather),
        "cautious": run(settings, seed, caution, False, the_draw, years, weather),
        "tender": run(settings, seed, caution, True, the_draw, years, weather),
    }
