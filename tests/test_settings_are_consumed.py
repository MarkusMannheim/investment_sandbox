"""Every declared setting is read somewhere. A setting nothing reads is a claim the
model does not honour."""

import tomllib
from importlib import resources

CONSUMED = {
    "price_cap_per_mwh": "dispatch.dispatch_year, the price when plant and demand response run out; simulate, the value of unmet demand",
    "start_year": "config.Settings.years",
    "years": "config.Settings.years and simulate.draw",
    "average_mw": "simulate.run, the starting level of demand",
    "round_trip_efficiency": "dispatch.battery_dispatch",
    "day_start_hour": "dispatch.battery_dispatch, the hour the battery's day starts",
    "years_ahead": "forward.forward_view, the years the forecast looks at; config.load_settings, the check on lead times",
    "caution": "simulate.run and investment.decide, the coefficient every firm shares",
    "block_fraction_of_peak": "investment.block_mw",
    "max_blocks_per_year": "investment.decide and tender.hold, the safety stop",
    "technologies": "tender.hold, the technologies that can bid; simulate.run, whether a tender is held",
    "reliability_standard": "tender.hold, the unmet demand the tender buys down to",
    "contract_years": "tender.hold, the length of a contract",
}


def _declared() -> set[str]:
    with (resources.files("investment_sandbox") / "data" / "settings.toml").open("rb") as fh:
        raw = tomllib.load(fh)
    return {k for section in raw.values() for k in section}


def test_every_declared_setting_is_consumed():
    declared = _declared()
    assert declared == set(CONSUMED), (
        f"unaccounted: {sorted(declared - set(CONSUMED))}; stale: {sorted(set(CONSUMED) - declared)}")


def test_every_consumer_named_here_exists():
    import importlib
    for key, where in CONSUMED.items():
        module = where.split(".")[0].split(",")[0].strip()
        importlib.import_module(f"investment_sandbox.{module}")
