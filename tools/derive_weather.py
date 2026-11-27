"""Derive the three weather years in data/weather.csv from AEMO's 2026 ISP traces.

Runs only where the traces are. It reads the New South Wales sub-region demand zips
(Step Change, POE50, operational demand for the year to June 2027), and the
half-hourly wind and solar capacity factors by sub-region, averages both to hours,
sums demand over the four sub-regions and normalises it to a mean of one, and keeps
three of the fifteen reference years: the least stressed, the median and the most
stressed. Stress is the mean of a year's 50 highest hours of residual demand after
wind and solar fleets each sized at three tenths of average demand. Nothing is tuned;
the rule is the only choice.

Usage: python tools/derive_weather.py TRACES_DIR CF_PARQUET [--out PATH]
"""

import argparse
import io
import pathlib
import zipfile

import numpy as np
import pandas as pd

SUB_REGIONS = ("CNSW", "NNSW", "SNSW", "SNW")
REFERENCE_YEARS = range(2011, 2026)
TARGET_START, TARGET_END = (2026, 7), (2027, 6)       # the year to June 2027
HALF_HOURS = [f"{i:02d}" for i in range(1, 49)]
VRE_SHARE_OF_AVERAGE_DEMAND = 0.3
STRESS_HOURS = 50


def hourly(half_hourly: np.ndarray) -> np.ndarray:
    """Average 48 half-hours a day to 24 hours and flatten to one series."""
    days = half_hourly.reshape(-1, 24, 2).mean(axis=2)
    return days.reshape(-1)


def demand_shape(traces_dir: pathlib.Path, ref_year: int) -> np.ndarray:
    total = np.zeros(365 * 24)
    for sub in SUB_REGIONS:
        archive = zipfile.ZipFile(traces_dir / f"2026 ISP Demand Traces {sub} Step Change.zip")
        member = next(n for n in archive.namelist()
                      if f"RefYear_{ref_year}_" in n and "POE50_OPSO_MODELLING.csv" in n)
        with archive.open(member) as fh:
            frame = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"))
        year, month = frame["Year"], frame["Month"]
        keep = ((year == TARGET_START[0]) & (month >= TARGET_START[1])) | \
               ((year == TARGET_END[0]) & (month <= TARGET_END[1]))
        rows = frame[keep].sort_values(["Year", "Month", "Day"])
        if len(rows) != 365:
            raise ValueError(f"{sub} {ref_year}: {len(rows)} days in the target year, not 365")
        total += hourly(rows[HALF_HOURS].to_numpy(dtype=float))
    return total / total.mean()


def capacity_factors(parquet: pathlib.Path) -> dict[tuple[int, str], np.ndarray]:
    frame = pd.read_parquet(parquet)
    frame = frame[frame["sub_region"].isin(SUB_REGIONS) & frame["tech"].isin(["wind", "solar pv"])]
    cols = [f"hh{i:02d}" for i in range(1, 49)]
    out = {}
    for (ref_year, tech), group in frame.groupby(["ref_year", "tech"]):
        days = group.groupby("day")[cols].mean().sort_index()
        if len(days) != 365:
            raise ValueError(f"{tech} {ref_year}: {len(days)} days, not 365")
        out[(int(ref_year), tech)] = hourly(days.to_numpy(dtype=float))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("traces_dir", type=pathlib.Path)
    parser.add_argument("cf_parquet", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[1]
                        / "src" / "investment_sandbox" / "data" / "weather.csv")
    args = parser.parse_args()

    cf = capacity_factors(args.cf_parquet)
    years = {}
    for ref_year in REFERENCE_YEARS:
        demand = demand_shape(args.traces_dir, ref_year)
        wind, solar = cf[(ref_year, "wind")], cf[(ref_year, "solar pv")]
        residual = demand - VRE_SHARE_OF_AVERAGE_DEMAND * (wind + solar)
        stress = float(np.sort(residual)[-STRESS_HOURS:].mean())
        years[ref_year] = (demand, wind, solar, stress)
        print(f"{ref_year}: peak/mean {demand.max():.3f}, wind {wind.mean():.3f}, "
              f"solar {solar.mean():.3f}, stress {stress:.3f}")

    ranked = sorted(years, key=lambda y: years[y][3])
    chosen = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
    print("chosen (least stressed, median, most stressed):", chosen)

    rows = []
    for ref_year in chosen:
        demand, wind, solar, _ = years[ref_year]
        for hour in range(365 * 24):
            rows.append((ref_year, hour, demand[hour], wind[hour], solar[hour]))
    table = pd.DataFrame(rows, columns=["year", "hour", "demand_shape", "wind_cf", "solar_cf"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.6f")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
