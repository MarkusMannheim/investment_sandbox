"""Draw the README's two figures from a run of the model: python tools/doc_figures.py"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from investment_sandbox.config import load_settings  # noqa: E402
from investment_sandbox.investment import appraise, exposure  # noqa: E402
from investment_sandbox.simulate import run  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "canonical"
SEED, YEAR, CAUTION, CONTRACT_YEARS = 1, 2040, 1.0, 12
NOTE = "A stylised one-region system, not a prediction"

LABEL = {"ocgt": "Gas peaker", "ccgt": "Combined cycle gas", "wind": "Wind", "solar": "Solar",
         "battery_4h": "Four-hour battery", "battery_8h": "Eight-hour battery"}
DEMAND_PATH = {"low": ("Low demand", "v", -0.17), "central": ("Central demand", "o", 0.0),
               "high": ("High demand", "^", 0.17)}
GROUND = {
    "light": {"surface": "#ffffff", "ink": "#333333", "muted": "#708090", "grid": "#e3e6e9",
              "low": "#375e97", "central": "#3bb16a", "high": "#fb6542",
              "cost": "#708090", "caution": "#fb6542"},
    "dark": {"surface": "#0d1117", "ink": "#f5f5f5", "muted": "#8b99a8", "grid": "#262d36",
             "low": "#6992cf", "central": "#3bb16a", "high": "#fb6542",
             "cost": "#8496a8", "caution": "#fb6542"},
}


def axes(p, height):
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(5.6, height), facecolor=p["surface"], layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.955))
    ax.set_facecolor(p["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(p["grid"])
    ax.tick_params(colors=p["muted"], length=0)
    ax.grid(axis="x", color=p["grid"], linewidth=1)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    fig.text(0.02, 0.012, NOTE, color=p["muted"], fontsize=12)
    return fig, ax


def legend(fig, p, handles, columns):
    fig.legend(handles=handles, loc="outside upper left", ncols=columns, frameon=False,
               labelcolor=p["ink"], fontsize=11.5, handletextpad=0.5, columnspacing=1.2)


def forecast(view, settings, ground="light"):
    """What a megawatt of each technology would earn in each outlook, against its cost."""
    p = GROUND[ground]
    fig, ax = axes(p, 5.2)
    techs = list(settings.tech_costs)
    for row, tech in enumerate(techs):
        incomes = view.rents[tech.technology] / 1e3
        for outlook, income in zip(view.futures, incomes):
            _, marker, offset = DEMAND_PATH[outlook.growth_path]
            ax.plot(income, row + offset, marker, color=p[outlook.growth_path], markersize=9,
                    markeredgecolor=p["surface"], markeredgewidth=1.2, zorder=3, clip_on=False)
        cost = tech.fixed_cost_per_mw_year / 1e3
        ax.plot([cost, cost], [row - 0.36, row + 0.36], color=p["ink"], linewidth=2.5, zorder=2)
    ax.set_yticks(range(len(techs)), [LABEL[t.technology] for t in techs], color=p["ink"])
    ax.invert_yaxis()
    ax.set_xlim(0, None)
    ax.set_xlabel("$ thousand per megawatt a year", color=p["muted"])
    handles = [Line2D([], [], marker=marker, color=p[path], linestyle="", markersize=9, label=name)
               for path, (name, marker, _) in DEMAND_PATH.items()]
    handles.append(Line2D([], [], marker="|", color=p["ink"], linestyle="", markersize=14,
                          markeredgewidth=2.5, label="Cost of owning the plant"))
    legend(fig, p, handles, 2)
    return fig


def hurdle(view, settings, ground="light"):
    """Each technology's hurdle, cost plus caution, with and without a contract, against expected income."""
    p = GROUND[ground]
    fig, ax = axes(p, 6.4)
    labels, row = [], 0.0
    positions = []
    for tech in settings.tech_costs:
        for contract_years in (0, CONTRACT_YEARS):
            test = appraise(tech, view, CAUTION, exposure(contract_years, tech.life_years))
            cost, caution = test.fixed_cost / 1e3, test.caution_cost / 1e3
            ax.barh(row, cost, height=0.36, color=p["cost"], zorder=2)
            ax.barh(row, caution, left=cost, height=0.36, color=p["caution"], zorder=2)
            expected = test.expected / 1e3
            ax.plot([expected, expected], [row - 0.27, row + 0.27], color=p["surface"], linewidth=5.5, zorder=3)
            ax.plot([expected, expected], [row - 0.27, row + 0.27], color=p["ink"], linewidth=2.5, zorder=4)
            labels.append(LABEL[tech.technology] if contract_years == 0
                          else f"with a {CONTRACT_YEARS}-year contract")
            positions.append(row)
            row += 0.55
        row += 0.45
    ax.set_yticks(positions, labels)
    for tick, label in zip(ax.get_yticklabels(), labels):
        tick.set_color(p["muted"] if label.startswith("with") else p["ink"])
        tick.set_fontsize(12 if label.startswith("with") else 13)
    ax.invert_yaxis()
    ax.set_xlabel("$ thousand per megawatt a year", color=p["muted"])
    handles = [Patch(color=p["cost"], label="Cost of owning the plant"),
               Patch(color=p["caution"], label="Caution"),
               Line2D([], [], marker="|", color=p["ink"], linestyle="", markersize=14, markeredgewidth=2.5,
                      label="Expected income")]
    legend(fig, p, handles, 2)
    return fig


def main():
    settings = load_settings()
    view = run(settings, seed=SEED, tender=False, years=YEAR - settings.years[0] + 1).years[-1].view
    for tech in settings.tech_costs:
        incomes = view.rents[tech.technology]
        merchant = appraise(tech, view, CAUTION, 1.0)
        contracted = appraise(tech, view, CAUTION, exposure(CONTRACT_YEARS, tech.life_years))
        print(f"{tech.technology:11s} income {incomes.min():9,.0f} to {incomes.max():9,.0f}  "
              f"expected {merchant.expected:9,.0f}  cost {merchant.fixed_cost:9,.0f}  "
              f"caution {merchant.caution_cost:8,.0f}  hurdle {merchant.hurdle:9,.0f}  "
              f"with contract: caution {contracted.caution_cost:8,.0f}  hurdle {contracted.hurdle:9,.0f}")
    OUT.mkdir(parents=True, exist_ok=True)
    for ground, suffix in (("light", ""), ("dark", "_dark")):
        for name, draw in (("forecast", forecast), ("hurdle", hurdle)):
            fig = draw(view, settings, ground)
            path = OUT / f"{name}{suffix}.png"
            fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
