"""The three weather years: hourly demand shape and wind and solar capacity factors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import _data_dir

HOURS = 8760


@dataclass(frozen=True)
class Weather:
    years: tuple[int, ...]
    demand_shape: np.ndarray    # (years, HOURS), each with a mean of one
    wind_cf: np.ndarray         # (years, HOURS)
    solar_cf: np.ndarray        # (years, HOURS)

    def __len__(self) -> int:
        return len(self.years)


@lru_cache(maxsize=1)
def load_weather() -> Weather:
    """Read data/weather.csv, derived from AEMO's ISP traces by tools/derive_weather.py."""
    with (_data_dir() / "weather.csv").open("rb") as fh:
        table = np.loadtxt(fh, delimiter=",", skiprows=1)
    years = tuple(int(y) for y in np.unique(table[:, 0]))
    shape, wind, solar = [], [], []
    for year in years:
        rows = table[table[:, 0] == year]
        if rows.shape[0] != HOURS:
            raise ValueError(f"weather year {year} has {rows.shape[0]} hours, not {HOURS}")
        rows = rows[np.argsort(rows[:, 1])]
        shape.append(rows[:, 2])
        wind.append(rows[:, 3])
        solar.append(rows[:, 4])
    return Weather(years, np.array(shape), np.array(wind), np.array(solar))


def demand_mw(shape: np.ndarray, average_mw: float) -> np.ndarray:
    """A year's hourly demand from its shape and the year's average demand."""
    return shape * average_mw
