"""Build notebooks/investment_sandbox.ipynb from the cells below: python tools/build_notebook.py"""

import json
import pathlib

NOTEBOOK = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "investment_sandbox.ipynb"
REPOSITORY = "https://github.com/MarkusMannheim/investment_sandbox.git"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "id": f"cell-{len(cells):02d}", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "id": f"cell-{len(cells):02d}", "metadata": {},
                  "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Investment sandbox

A small agent-based model of an electricity market, in which firms decide what to build. This notebook runs it three times on the same demand and weather: with firms that ignore risk, with cautious firms, and with cautious firms and a government tender. The repository's README explains how the model works.

The model is illustrative. It follows one region with a stylised set of power stations, and nothing it produces is a forecast.
""")

code(f"""
import importlib.util, subprocess, sys

# Installs the package when the notebook runs somewhere it is not already present, such as Colab
if importlib.util.find_spec("investment_sandbox") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "git+{REPOSITORY}"])

import pandas as pd
from investment_sandbox.config import load_settings
from investment_sandbox.simulate import compare
from investment_sandbox import explore, plots

pd.options.display.float_format = "{{:,.1f}}".format
""")

md("""
## Settings

Change a setting, then run this cell and the cells below it again.
""")

code("""
CAUTION = 1.0                  # how strongly the firms dislike risk; at 0 they build whatever breaks even
DEMAND_PATH = "central"        # the demand growth that arrives: "low", "central" or "high"
SEED = 1                       # draws the weather year for each year of the run
BIDDERS = ["ocgt", "ccgt", "battery_4h", "battery_8h"]    # the technologies that can bid in the tender
CONTRACT_YEARS = 12            # how long a contract fixes a plant's income
RELIABILITY_STANDARD = 0.002   # the most unmet demand the government accepts in the forecast, % of demand

settings = load_settings({
    "investment": {"caution": CAUTION},
    "tender": {"technologies": BIDDERS, "contract_years": CONTRACT_YEARS,
               "reliability_standard": RELIABILITY_STANDARD / 100},
})
runs = compare(settings, seed=SEED, path=DEMAND_PATH)
""")

md("""
## Results

The table sums up each run over its 20 years, from 2027 to 2046. The bill is what consumers pay for energy: supply at the spot price and unmet demand at the price cap. Underwriting is what the tender's contracts cost whoever stands behind them, which is the fixed sums paid to the plants less what those plants earned in the market. The resource cost is what supplying the electricity uses up: fuel, upkeep, capital and unmet demand at the price cap.
""")

code("""
pd.DataFrame(explore.headline(runs))
""")

md("""
## Charts

The first chart shows the megawatts committed in each year, by technology. Hatched bars are plant bought by the tender.
""")

code("""
plots.committed(runs)
""")

md("""
The next chart shows the plant in service in each year.
""")

code("""
plots.capacity(runs)
""")

md("""
This chart shows the energy each technology supplied in each year.
""")

code("""
plots.generation(runs)
""")

md("""
This chart shows the average price paid for a megawatt hour in each year.
""")

code("""
plots.prices(runs)
""")

md("""
This chart shows unmet demand in each year as a share of demand. The dotted line is the reliability standard.
""")

code("""
plots.reliability(runs)
""")

md("""
This chart shows the three sums of money for each run, added up over the run.
""")

code("""
plots.costs(runs)
""")

md("""
The last chart shows the hurdle each technology faced in one year's forecast, in the run with cautious firms and no tender. Each bar is the cost of owning the plant with the firm's caution added, and the upright mark is the income the plant expects. A firm builds when the mark sits past the end of the bar. The second bar of each pair is the hurdle under a contract. Change `YEAR` and run the cell again.
""")

code("""
YEAR = 2040
plots.hurdle(runs["cautious"], YEAR, CONTRACT_YEARS)
""")


def notebook() -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    NOTEBOOK.parent.mkdir(exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook(), indent=1) + "\n")
    print(f"wrote {NOTEBOOK} with {len(cells)} cells")
