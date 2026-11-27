"""Settings and the packaged data tables, loaded strictly so that an unknown or missing key raises."""

from __future__ import annotations

import csv
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

SECTIONS = {
    "market": {"price_cap_per_mwh"},
    "time": {"start_year", "years"},
    "demand": {"average_mw"},
    "storage": {"round_trip_efficiency", "day_start_hour"},
    "forward": {"years_ahead"},
    "investment": {"caution", "block_fraction_of_peak", "max_blocks_per_year"},
    "tender": {"technologies", "reliability_standard", "contract_years"},
}


def _data_dir():
    return resources.files("investment_sandbox") / "data"


def read_csv(name: str) -> list[dict[str, str]]:
    """Read a packaged CSV into a list of row dicts."""
    with (_data_dir() / name).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(row: dict[str, str], key: str, default: float | None = None) -> float | None:
    raw = (row.get(key) or "").strip()
    return default if raw == "" else float(raw)


@dataclass(frozen=True)
class Unit:
    """One plant: a row of the fleet, or a block the model has built."""

    unit: str
    technology: str
    capacity_mw: float
    availability: float
    srmc_per_mwh: float
    commissioned_year: int
    retirement_year: int
    fom_per_kw_year: float
    duration_h: float | None
    # Zero for plant standing at the start, whose capital is already spent
    annual_capital_per_mw: float = 0.0

    @property
    def available_mw(self) -> float:
        return self.capacity_mw * self.availability

    @property
    def is_storage(self) -> bool:
        return self.duration_h is not None

    @property
    def is_variable(self) -> bool:
        return self.technology in ("wind", "solar")

    def in_service(self, year: int) -> bool:
        return self.commissioned_year <= year < self.retirement_year


@dataclass(frozen=True)
class TechCost:
    """What it costs to build and keep one MW of a technology that can be built."""

    technology: str
    capex_per_kw: float
    fom_per_kw_year: float
    srmc_per_mwh: float
    availability: float
    wacc: float
    lead_years: int
    life_years: int
    unit_size_mw: float
    duration_h: float | None

    @property
    def crf(self) -> float:
        """The annuity that repays one dollar over the plant's life at its cost of capital."""
        r, n = self.wacc, self.life_years
        return 1.0 / n if r <= 0 else r / (1.0 - (1.0 + r) ** -n)

    @property
    def fixed_cost_per_mw_year(self) -> float:
        """What a megawatt costs to own each year: annualised capital plus fixed operating cost."""
        return self.capex_per_kw * 1000.0 * self.crf + self.fom_per_kw_year * 1000.0


@dataclass(frozen=True)
class GrowthPath:
    path: str
    weight: float
    annual_growth: float


@dataclass(frozen=True)
class DsrTier:
    tier: str
    price_per_mwh: float
    capacity_mw: float


@dataclass(frozen=True)
class Settings:
    market: dict[str, Any]
    time: dict[str, Any]
    demand: dict[str, Any]
    storage: dict[str, Any]
    forward: dict[str, Any]
    investment: dict[str, Any]
    tender: dict[str, Any]
    fleet: tuple[Unit, ...] = field(repr=False, default=())
    dsr: tuple[DsrTier, ...] = field(repr=False, default=())
    tech_costs: tuple[TechCost, ...] = field(repr=False, default=())
    growth: tuple[GrowthPath, ...] = field(repr=False, default=())

    def tech(self, technology: str) -> TechCost:
        for t in self.tech_costs:
            if t.technology == technology:
                return t
        raise KeyError(f"no cost row for technology {technology!r}")

    @property
    def years(self) -> range:
        start = int(self.time["start_year"])
        return range(start, start + int(self.time["years"]))


def load_settings(overrides: dict[str, dict[str, Any]] | None = None) -> Settings:
    """Load settings.toml and the data tables, with any overrides checked the same way as the file."""
    with (_data_dir() / "settings.toml").open("rb") as fh:
        raw = tomllib.load(fh)

    for section, keys in SECTIONS.items():
        if section not in raw:
            raise ValueError(f"settings.toml is missing the [{section}] section")
        unknown = set(raw[section]) - keys
        if unknown:
            raise ValueError(f"unknown key(s) in [{section}]: {', '.join(sorted(unknown))}")
        missing = keys - set(raw[section])
        if missing:
            raise ValueError(f"missing key(s) in [{section}]: {', '.join(sorted(missing))}")
    unknown_sections = set(raw) - set(SECTIONS)
    if unknown_sections:
        raise ValueError(f"unknown section(s): {', '.join(sorted(unknown_sections))}")

    for section, values in (overrides or {}).items():
        if section not in SECTIONS:
            raise ValueError(f"unknown settings section: {section}")
        bad = set(values) - SECTIONS[section]
        if bad:
            raise ValueError(f"unknown key(s) in [{section}]: {', '.join(sorted(bad))}")
        raw[section] = {**raw[section], **values}

    fleet = tuple(
        Unit(
            unit=r["unit"],
            technology=r["technology"],
            capacity_mw=_num(r, "capacity_mw", 0.0),
            availability=_num(r, "availability", 1.0),
            srmc_per_mwh=_num(r, "srmc_per_mwh", 0.0),
            commissioned_year=int(_num(r, "commissioned_year", 0)),
            retirement_year=int(_num(r, "retirement_year", 9999)),
            fom_per_kw_year=_num(r, "fom_per_kw_year", 0.0),
            duration_h=_num(r, "duration_h"),
        )
        for r in read_csv("fleet.csv")
    )
    dsr = tuple(
        DsrTier(tier=r["tier"], price_per_mwh=float(r["price_per_mwh"]),
                capacity_mw=float(r["capacity_mw"]))
        for r in read_csv("dsr.csv")
    )
    tech_costs = tuple(
        TechCost(
            technology=r["technology"],
            capex_per_kw=_num(r, "capex_per_kw", 0.0),
            fom_per_kw_year=_num(r, "fom_per_kw_year", 0.0),
            srmc_per_mwh=_num(r, "srmc_per_mwh", 0.0),
            availability=_num(r, "availability", 1.0),
            wacc=_num(r, "wacc", 0.07),
            lead_years=int(_num(r, "lead_years", 2)),
            life_years=int(_num(r, "life_years", 25)),
            unit_size_mw=_num(r, "unit_size_mw", 100.0),
            duration_h=_num(r, "duration_h"),
        )
        for r in read_csv("tech_costs.csv")
    )
    growth = tuple(
        GrowthPath(path=r["path"], weight=float(r["weight"]),
                   annual_growth=float(r["annual_growth"]))
        for r in read_csv("growth.csv")
    )
    total = sum(g.weight for g in growth)
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"growth path weights sum to {total}, not 1")
    unknown = set(raw["tender"]["technologies"]) - {t.technology for t in tech_costs}
    if unknown:
        raise ValueError(f"the tender names technologies with no cost row: {', '.join(sorted(unknown))}")
    if int(raw["tender"]["contract_years"]) < 1:
        raise ValueError("the tender's contract must run for at least one year")
    ahead = list(raw["forward"]["years_ahead"])
    if not ahead or ahead != sorted(set(ahead)) or ahead[0] < 1:
        raise ValueError("the forecast's years ahead must be whole years in rising order")
    slow = [t.technology for t in tech_costs if t.lead_years > ahead[-1]]
    if slow:
        raise ValueError(f"the forecast looks {ahead[-1]} years ahead, too near to judge: {', '.join(slow)}")
    return Settings(
        market=raw["market"], time=raw["time"], demand=raw["demand"],
        storage=raw["storage"], forward=raw["forward"], investment=raw["investment"],
        tender=raw["tender"],
        fleet=fleet, dsr=dsr, tech_costs=tech_costs, growth=growth,
    )
