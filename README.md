# Investment sandbox

A small agent-based model of an electricity market, in which firms decide what to build. It is designed to show how uncertainty shapes those decisions and how government contracts can change them. The model follows one region, hour by hour, for 20 years. A run takes about 10 seconds.

A least-cost model asks what an electricity system *should* contain, and whatever is cheapest is built because something is assumed to build it. This model asks who *would* build it. It has no planner, and no one in it knows what the future will be. It plays out a year at a time, and each year the firms decide what to build using a forecast of what new plant might earn.

In a least-cost model, a higher cost of capital makes plant more expensive, but everything still gets built. The firms in this model dislike risk. They build only when what a plant is expected to earn clears its cost by a margin for that risk, so the plant with the least certain income is built late or not at all.

The model is illustrative. Its region is stylised, sized on public data for New South Wales. It is a much simplified version of a larger, higher-fidelity model of the whole National Electricity Market that follows the same yearly loop.

## What it shows

The table compares three runs on the same demand and weather: firms with no caution, cautious firms and cautious firms with a government reliability tender.

| | No caution | Caution | Caution and tender |
| --- | --- | --- | --- |
| Unmet demand, GWh | 0.3 | 196.5 | 0.0 |
| Years with unmet demand, of 20 | 1 | 11 | 0 |
| Average price paid, $/MWh | 97 | 151 | 98 |
| Consumers' bill, $bn | 143.1 | 226.9 | 144.0 |
| Cost of the tender's contracts during the run, $bn | 0 | 0 | 18.7 |
| Resource cost, $bn | 142.6 | 144.6 | 141.6 |

Caution leaves demand unmet and raises what consumers pay, while the system costs about the same to build and run. The tender closes most or all of the gap, at a cost to whoever stands behind its contracts.

These numbers come from one demand path and one draw of the weather. On other paths and draws the effects point the same way and differ in size.

## How it works

The model repeats five steps each year.

1. Dispatch sets a price for each hour.
2. A forecast shows what new plant would earn 4 and 8 years ahead, in nine outlooks for demand and weather.
3. Firms commit plant for as long as any technology clears its hurdle, which is the plant's cost plus the firms' caution.
4. If the forecast still shows too much unmet demand, a government tender buys enough plant to close the gap, under long contracts.
5. Demand grows and the next year begins.

[DOCUMENTATION.md](DOCUMENTATION.md) explains each step, the outputs and what the model leaves out.

## Run it

The model is run from a Python notebook. The badge below opens it in Colab, which needs only a Google account. The notebook runs the three cases, which takes about half a minute, then shows the table and a set of charts.

[![Open the notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MarkusMannheim/investment_sandbox/blob/main/notebooks/investment_sandbox.ipynb)

To run it on your own machine:

```bash
git clone https://github.com/MarkusMannheim/investment_sandbox.git
cd investment_sandbox
pip install -e ".[notebook]"
jupyter lab notebooks/investment_sandbox.ipynb
```

It needs Python 3.11 or later. There is no solver to install and no data to download.

## What to change

The notebook's settings sit in one cell. Start with these three.

- Raise caution from zero, where the firms build whatever breaks even, and unmet demand and prices rise.
- Lengthen the tender's contracts and they cost less, while reliability barely changes.
- Change the demand path to low or high, and both the amount built and the cost of the contracts change.

## Licence and data

Code is [MIT](LICENCE). The data files in [src/investment_sandbox/data/](src/investment_sandbox/data/) are derived from published sources and carry those sources' terms. [DATA_SOURCES.md](DATA_SOURCES.md) names the source, the derivation and the attribution for each. To cite the model, use [CITATION.cff](CITATION.cff).
