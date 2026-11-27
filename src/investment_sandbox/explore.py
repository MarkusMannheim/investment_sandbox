"""Sum up runs of the model and put them side by side."""

from __future__ import annotations

from typing import Any

from .config import load_settings
from .simulate import Run, draw, run
from .weather import load_weather

CASE = {"risk_neutral": "No caution", "cautious": "Caution", "tender": "Caution and tender"}
TECHNOLOGY = {"coal": "Coal", "ccgt": "Combined cycle gas", "ocgt": "Gas peakers", "wind": "Wind",
              "solar": "Solar", "battery_4h": "Four-hour batteries", "battery_8h": "Eight-hour batteries",
              "demand_response": "Demand response"}
HEADLINE = {"unserved_gwh": "Unmet demand, GWh", "price_paid": "Average price paid, $/MWh",
            "cap_hours": "Hours at the price cap", "bill_bn": "The bill, $bn",
            "underwriting_bn": "Underwriting paid in the run, $bn",
            "expected_underwriting_bn": "Underwriting expected over the contracts' terms, $bn",
            "resource_cost_bn": "Resource cost, $bn",
            "built_mw": "Committed by firms, MW", "bought_mw": "Bought by tender, MW"}


def summary(result: Run) -> dict[str, float]:
    """The numbers that sum up a run."""
    out = {
        "unserved_gwh": result.unserved_gwh,
        "price_paid": result.price_paid,
        "cap_hours": float(result.cap_hours),
        "bill_bn": result.bill / 1e9,
        "underwriting_bn": result.underwriting_cost / 1e9,
        "expected_underwriting_bn": result.expected_underwriting_cost / 1e9,
        "resource_cost_bn": result.resource_cost / 1e9,
        "built_mw": sum(result.built_mw.values()),
        "bought_mw": sum(result.bought_mw.values()),
    }
    committed = result.committed_mw
    for tech in result.settings.tech_costs:
        out[f"{tech.technology}_mw"] = committed.get(tech.technology, 0.0)
    return out


def headline(runs: dict[str, Run]) -> dict[str, dict[str, float]]:
    """The summary of each run under plain labels, one column a run."""
    out = {}
    for name, result in runs.items():
        numbers = summary(result)
        column = {label: numbers[key] for key, label in HEADLINE.items()}
        for tech in result.settings.tech_costs:
            column[f"{TECHNOLOGY[tech.technology]} committed, MW"] = numbers[f"{tech.technology}_mw"]
        out[CASE.get(name, name)] = column
    return out


def sweep(section: str, key: str, values: list[Any], seed: int = 1,
          base: dict[str, dict[str, Any]] | None = None, years: int | None = None,
          tender: bool = True, path: str | None = None) -> list[dict[str, Any]]:
    """Run the model once for each value of one setting, all on the same draw."""
    weather = load_weather()
    rows = []
    for value in values:
        overrides = {k: dict(v) for k, v in (base or {}).items()}
        overrides.setdefault(section, {})[key] = value
        settings = load_settings(overrides)
        result = run(settings, seed=seed, tender=tender, the_draw=draw(settings, weather, seed, path),
                     years=years, weather=weather)
        rows.append({key: value, **summary(result)})
    return rows
