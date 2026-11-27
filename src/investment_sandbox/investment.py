"""The investment decision: a firm builds when the certain sum it would accept for a plant's uncertain income covers the plant's cost."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings, TechCost, Unit
from .forward import View, forward_view
from .weather import Weather

# A caution setting of 1 is a coefficient of 1 per $100,000 a megawatt a year
CAUTION_UNIT = 100_000.0


def coefficient(caution: float) -> float:
    return float(caution) / CAUTION_UNIT


def certainty_equivalent(payoffs: np.ndarray, weights: np.ndarray, a: float) -> float:
    """The certain sum worth as much as the uncertain payoffs to a firm with risk aversion a."""
    if not np.isclose(float(weights.sum()), 1.0, atol=1e-9):
        raise ValueError("weights must be a probability distribution")
    if a <= 0:
        return float(payoffs @ weights)
    # Measuring from the worst payoff keeps every exponent at or below zero, so a year at the price cap cannot overflow
    worst = float(payoffs.min())
    return worst - np.log(float(weights @ np.exp(-a * (payoffs - worst)))) / a


def exposure(contract_years: int, life_years: int) -> float:
    """The share of a plant's life that a contract of this length leaves to the market."""
    return max(0.0, 1.0 - contract_years / life_years)


@dataclass(frozen=True)
class Test:
    """One technology against its hurdle in one forecast."""

    technology: str
    fixed_cost: float
    expected: float
    certainty_equivalent: float
    exposure: float

    @property
    def caution_cost(self) -> float:
        return self.expected - self.certainty_equivalent

    @property
    def hurdle(self) -> float:
        """The expected income a plant needs before a firm builds it."""
        return self.fixed_cost + self.caution_cost

    @property
    def passes(self) -> bool:
        return self.certainty_equivalent >= self.fixed_cost

    @property
    def surplus(self) -> float:
        return self.certainty_equivalent - self.fixed_cost


def appraise(tech: TechCost, view: View, caution: float, exposed: float = 1.0) -> Test:
    """Test a technology against its hurdle, with caution charged on the exposed share of its income."""
    rents = view.rents[tech.technology]
    expected = float(rents @ view.weights)
    income = (1.0 - exposed) * expected + exposed * rents
    ce = certainty_equivalent(income, view.weights, coefficient(caution))
    return Test(tech.technology, tech.fixed_cost_per_mw_year, expected, ce, exposed)


def block_mw(tech: TechCost, peak_mw: float, settings: Settings) -> float:
    """One block of a technology, in MW: a share of peak demand in whole units."""
    wanted = float(settings.investment["block_fraction_of_peak"]) * peak_mw
    return max(1, round(wanted / tech.unit_size_mw)) * tech.unit_size_mw


def new_unit(tech: TechCost, name: str, mw: float, year: int) -> Unit:
    """A plant committed in a year, which starts up after the technology's lead time."""
    start = year + tech.lead_years
    return Unit(unit=name, technology=tech.technology, capacity_mw=mw, availability=tech.availability,
                srmc_per_mwh=tech.srmc_per_mwh, commissioned_year=start,
                retirement_year=start + tech.life_years, fom_per_kw_year=tech.fom_per_kw_year,
                duration_h=tech.duration_h, annual_capital_per_mw=tech.capex_per_kw * 1000.0 * tech.crf)


@dataclass(frozen=True)
class Build:
    """A block a firm committed, with the test it passed."""

    unit: Unit
    test: Test
    decided_year: int


@dataclass(frozen=True)
class Decision:
    """A year's building by the firms, with the forecast before any of it and after all of it."""

    builds: list[Build]
    first_view: View
    last_view: View
    first_tests: list[Test]
    stopped: bool


def decide(units: tuple[Unit, ...], year: int, average_mw: float, peak_mw: float, settings: Settings,
           weather: Weather, caution: float | None = None,
           pace_mw: dict[str, float] | None = None) -> Decision:
    """Commit one block at a time of the technology furthest past its hurdle, rebuilding the forecast after each, until nothing clears."""
    caution = float(settings.investment["caution"]) if caution is None else float(caution)
    limit = int(settings.investment["max_blocks_per_year"])
    first_view = view = forward_view(units, year, average_mw, settings, weather, pace_mw)
    first_tests: list[Test] = []
    builds: list[Build] = []
    stopped = False
    while True:
        tests = [appraise(tech, view, caution) for tech in settings.tech_costs]
        first_tests = first_tests or tests
        passing = [t for t in tests if t.passes]
        if not passing:
            break
        if len(builds) >= limit:
            stopped = True
            break
        best = max(passing, key=lambda t: t.surplus)
        tech = settings.tech(best.technology)
        unit = new_unit(tech, f"{tech.technology}_{year}_{len(builds) + 1}", block_mw(tech, peak_mw, settings), year)
        builds.append(Build(unit, best, year))
        units = units + (unit,)
        view = forward_view(units, year, average_mw, settings, weather, pace_mw)
    return Decision(builds, first_view, view, first_tests, stopped)
