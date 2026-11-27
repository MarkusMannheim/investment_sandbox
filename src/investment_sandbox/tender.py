"""The government's tender: firm plant bought under long contracts when the forecast shows more unmet demand than the reliability standard allows."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, TechCost, Unit
from .forward import View, forward_view
from .investment import block_mw, certainty_equivalent, coefficient, exposure, new_unit
from .weather import Weather


@dataclass(frozen=True)
class Contract:
    """A contract that pays a plant a fixed sum each year and takes its market income in return."""

    unit: Unit
    fixed_per_mw_year: float
    expected_income_per_mw_year: float      # what the forecast expected the plant to earn when the contract was signed
    start_year: int
    end_year: int       # the first year the plant is back on the market

    def live(self, year: int) -> bool:
        return self.start_year <= year < self.end_year

    @property
    def expected_cost(self) -> float:
        """What the contract was expected to cost its underwriter over its whole term when it was signed."""
        years = self.end_year - self.start_year
        return (self.fixed_per_mw_year - self.expected_income_per_mw_year) * self.unit.capacity_mw * years

    def cost(self, market_income: float) -> float:
        """What the contract costs its underwriter in a year, negative when the plant earned more than the fixed sum."""
        return self.fixed_per_mw_year * self.unit.capacity_mw - market_income


def fixed_sum(tech: TechCost, view: View, caution: float, contract_years: int) -> float:
    """The lowest yearly sum per megawatt at which a firm would build under a contract of this length."""
    exposed = exposure(contract_years, tech.life_years)
    if exposed == 0:
        return tech.fixed_cost_per_mw_year
    # The firm counts the years after the contract at their certain worth, and the fixed sums make up the rest of the plant's cost
    later = certainty_equivalent(exposed * view.rents[tech.technology], view.weights, coefficient(caution))
    return (tech.fixed_cost_per_mw_year - later) / (1.0 - exposed)


def hold(units: tuple[Unit, ...], year: int, average_mw: float, peak_mw: float, view: View,
         settings: Settings, weather: Weather, caution: float,
         pace_mw: dict[str, float] | None = None) -> tuple[list[Contract], bool]:
    """Buy blocks of plant, cheapest for consumers first, until the forecast's unmet demand is back at the reliability standard."""
    standard = float(settings.tender["reliability_standard"])
    limit = int(settings.investment["max_blocks_per_year"])
    bidders = [settings.tech(name) for name in settings.tender["technologies"]]
    term = {tech.technology: min(int(settings.tender["contract_years"]), tech.life_years) for tech in bidders}
    bought: list[Unit] = []
    stopped = False
    while bidders and view.unserved_share > standard:
        if len(bought) >= limit:
            stopped = True
            break
        best = None
        for tech in bidders:
            name = f"tender_{tech.technology}_{year}_{len(bought) + 1}"
            unit = new_unit(tech, name, block_mw(tech, peak_mw, settings), year)
            after = forward_view(units + (unit,), year, average_mw, settings, weather, pace_mw)
            removed = (view.expected_unserved_mwh - after.expected_unserved_mwh) / unit.capacity_mw
            if removed <= 0:
                continue
            net_cost = fixed_sum(tech, after, caution, term[tech.technology]) - after.expected(tech.technology)
            # A contract expected to cost consumers nothing is ranked on the unmet demand it removes
            score = max(net_cost, 1.0) / removed
            if best is None or score < best[0]:
                best = (score, unit, after)
        if best is None:
            break
        _, unit, view = best
        units = units + (unit,)
        bought.append(unit)
    # Every contract of the year is priced on the forecast with all of the year's plant in it
    contracts = []
    for unit in bought:
        years = term[unit.technology]
        price = fixed_sum(settings.tech(unit.technology), view, caution, years)
        contracts.append(Contract(unit, price, view.expected(unit.technology), unit.commissioned_year,
                                  unit.commissioned_year + years))
    return contracts, stopped
