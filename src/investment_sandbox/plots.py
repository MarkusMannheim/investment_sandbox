"""The notebook's charts, drawn with Plotly from runs of the model."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .explore import CASE, TECHNOLOGY
from .investment import appraise, exposure
from .simulate import Run

INK, MUTED, GRID = "#333333", "#708090", "#e5e5e5"
COLOUR = {"coal": "#333333", "ccgt": "#fb6542", "ocgt": "#fda58f", "battery_4h": "#375e97",
          "battery_8h": "#6992cf", "wind": "#3bb16a", "solar": "#ffbb00", "demand_response": "#708090"}
CASE_COLOUR = {"risk_neutral": "#708090", "cautious": "#fb6542", "tender": "#375e97"}
# The order technologies stack in, bottom to top
ORDER = ("coal", "ccgt", "ocgt", "battery_4h", "battery_8h", "wind", "solar", "demand_response")
PANEL_HEIGHT = 240


def _finish(fig: go.Figure, height: int, unit: str, rows: int = 1) -> go.Figure:
    """Apply the house layout: no title, the unit on the value axis and the legend below."""
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Open Sans, Arial, sans-serif", size=13, color=INK),
        legend=dict(orientation="h", x=0, y=-0.12 / rows, yanchor="top", traceorder="normal"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified", modebar_remove=["lasso", "select"])
    fig.update_xaxes(gridcolor=GRID, tickformat="d")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED, title_text=unit, title_font_color=MUTED)
    return fig


def _panels(runs: dict[str, Run]) -> go.Figure:
    """One panel a run, stacked down the page so that it reads on a phone."""
    fig = make_subplots(rows=len(runs), cols=1, shared_xaxes=True, vertical_spacing=0.1 / len(runs) + 0.04,
                        subplot_titles=[CASE.get(name, name) for name in runs])
    fig.update_annotations(font_size=13, x=0, xanchor="left")
    return fig


def _stacked(runs: dict[str, Run], pick, unit: str, scale: float = 1.0) -> go.Figure:
    """Stacked bars by technology for each year, one panel a run."""
    fig = _panels(runs)
    shown: set[str] = set()
    for row, result in enumerate(runs.values(), start=1):
        years = [y.year for y in result.years]
        for tech in ORDER:
            values = [pick(y).get(tech, 0.0) / scale for y in result.years]
            if not any(values):
                continue
            fig.add_trace(go.Bar(x=years, y=values, name=TECHNOLOGY[tech], marker_color=COLOUR[tech],
                                 legendgroup=tech, showlegend=tech not in shown,
                                 hovertemplate="%{y:,.0f}"), row=row, col=1)
            shown.add(tech)
    fig.update_layout(barmode="stack")
    return _finish(fig, PANEL_HEIGHT * len(runs) + 80, unit, len(runs))


def capacity(runs: dict[str, Run]) -> go.Figure:
    """Plant in service each year, by technology."""
    return _stacked(runs, lambda y: y.capacity_mw, "MW in service")


def generation(runs: dict[str, Run]) -> go.Figure:
    """Energy supplied each year, by technology."""
    return _stacked(runs, lambda y: y.generation_mwh, "GWh supplied", scale=1e3)


def committed(runs: dict[str, Run]) -> go.Figure:
    """Megawatts committed each year by technology, with plant bought by tender hatched."""
    fig = _panels(runs)
    shown: set[str] = set()
    for row, result in enumerate(runs.values(), start=1):
        years = [y.year for y in result.years]
        for tech in ORDER:
            for pick, hatch in ((lambda y: y.built_mw, ""), (lambda y: y.bought_mw, "/")):
                values = [pick(y).get(tech, 0.0) for y in result.years]
                if not any(values):
                    continue
                fig.add_trace(go.Bar(x=years, y=values, name=TECHNOLOGY[tech], legendgroup=tech,
                                     marker=dict(color=COLOUR[tech], pattern_shape=hatch, pattern_fgcolor="white"),
                                     showlegend=tech not in shown, hovertemplate="%{y:,.0f}"), row=row, col=1)
                shown.add(tech)
    if any(result.contracts for result in runs.values()):
        fig.add_trace(go.Bar(x=[None], y=[None], name="Hatched: bought by tender",
                             marker=dict(color=MUTED, pattern_shape="/", pattern_fgcolor="white")), row=1, col=1)
    fig.update_layout(barmode="stack")
    return _finish(fig, PANEL_HEIGHT * len(runs) + 80, "MW committed", len(runs))


def prices(runs: dict[str, Run]) -> go.Figure:
    """The average price paid for a megawatt hour in each year of each run."""
    fig = go.Figure()
    for name, result in runs.items():
        fig.add_trace(go.Scatter(x=[y.year for y in result.years], y=[y.price_paid for y in result.years],
                                 name=CASE.get(name, name), mode="lines+markers",
                                 line=dict(color=CASE_COLOUR.get(name, INK), width=2.5),
                                 hovertemplate="$%{y:,.0f}"))
    fig.update_yaxes(rangemode="tozero")
    return _finish(fig, 380, "$/MWh")


def reliability(runs: dict[str, Run]) -> go.Figure:
    """Unmet demand in each year as a share of demand, against the reliability standard."""
    fig = go.Figure()
    for name, result in runs.items():
        share = [100.0 * y.unserved_mwh / (y.served_mwh + y.unserved_mwh) for y in result.years]
        fig.add_trace(go.Bar(x=[y.year for y in result.years], y=share, name=CASE.get(name, name),
                             marker_color=CASE_COLOUR.get(name, INK), hovertemplate="%{y:.4f}%"))
    standard = 100.0 * float(next(iter(runs.values())).settings.tender["reliability_standard"])
    fig.add_hline(y=standard, line=dict(color=INK, width=1.5, dash="dot"),
                  annotation_text="Reliability standard", annotation_position="top left")
    fig.update_layout(barmode="group")
    return _finish(fig, 380, "% of demand unmet")


def costs(runs: dict[str, Run]) -> go.Figure:
    """The bill, underwriting and the resource cost of each run, summed over its years."""
    fig = go.Figure()
    labels = ["The bill", "Underwriting", "Resource cost"]
    for name, result in runs.items():
        values = [result.bill / 1e9, result.underwriting_cost / 1e9, result.resource_cost / 1e9]
        fig.add_trace(go.Bar(x=labels, y=values, name=CASE.get(name, name),
                             marker_color=CASE_COLOUR.get(name, INK), hovertemplate="$%{y:,.1f}bn"))
    fig.update_layout(barmode="group")
    return _finish(fig, 380, "$bn over the run")


def hurdle(result: Run, year: int, contract_years: int | None = None) -> go.Figure:
    """Each technology's hurdle in one year's forecast, with and without a contract, against expected income."""
    settings = result.settings
    years = int(settings.tender["contract_years"]) if contract_years is None else contract_years
    view = next(y.view for y in result.years if y.year == year)
    labels, cost, caution, expected = [], [], [], []
    for tech in settings.tech_costs:
        for term in (0, years):
            test = appraise(tech, view, result.caution, exposure(term, tech.life_years))
            labels.append(TECHNOLOGY[tech.technology] if term == 0 else f"with a {years}-year contract")
            cost.append(test.fixed_cost / 1e3)
            caution.append(test.caution_cost / 1e3)
            expected.append(test.expected / 1e3)
    # Rows are placed by number because the contract rows share one label
    rows = list(range(len(labels)))
    fig = go.Figure()
    fig.add_trace(go.Bar(y=rows, x=cost, orientation="h", name="Cost of owning the plant", marker_color=MUTED,
                         hovertemplate="$%{x:,.0f}k"))
    fig.add_trace(go.Bar(y=rows, x=caution, orientation="h", name="Caution", marker_color="#fb6542",
                         hovertemplate="$%{x:,.0f}k"))
    fig.add_trace(go.Scatter(y=rows, x=expected, mode="markers", name="Expected income",
                             marker=dict(symbol="line-ns", size=16, line=dict(color=INK, width=3)),
                             hovertemplate="$%{x:,.0f}k"))
    fig.update_layout(barmode="stack")
    _finish(fig, 60 + 40 * len(labels), None)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", title_text=None, tickvals=rows, ticktext=labels)
    fig.update_xaxes(title_text="$ thousand a megawatt a year", title_font_color=MUTED, tickformat=",")
    return fig
