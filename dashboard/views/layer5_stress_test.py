"""Layer 5 · Crisis & stress testing: the cascading worst-case domino scenario."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.command_center import add_derived_fields
from dashboard.config import (
    COMMAND_CENTER_STATUS_COLORS,
    COMMAND_CENTER_THRESHOLDS,
    DISTRICT_COLUMN,
    DISTRICT_KEY_COLUMN,
    INVESTMENT_MATRIX,
    MONTH_COLUMN,
    STRESS_ASSUMPTIONS,
    STRESS_CASES,
    STRESS_STAGES,
)
from dashboard.data_loader import district_options, with_district_aliases
from dashboard.district_utils import clean_district_key
from dashboard.simulation_engine import (
    PRIMARY_CARE_ONLY_MODE,
    compare_scenarios,
    default_settings,
    derive_baseline_pressure,
    package_cost,
)
from dashboard.stress_engine import (
    CITY_WIDE,
    alert_comparison,
    case_evidence,
    evaluate_triggers,
    funding_quadrants,
    run_domino,
    stage_labels,
    stage_precedents,
    trigger_history,
)
from dashboard.ui import plotly_chart
from dashboard.views.layer4_command_center import _status_grid

PRIMARY = "#0D9488"
TEXT = "#1E293B"
WATCH, CRITICAL = COMMAND_CENTER_STATUS_COLORS["Watch"], COMMAND_CENTER_STATUS_COLORS["Critical"]
CASE_COLORS = {1: "#DB2777", 2: "#F59E0B", 3: "#0D9488", 4: "#7C3AED", 5: "#2563EB", 6: "#64748B"}
FUNDING_COLORS = {
    "Private": "#0D9488", "PPP (land subsidy / tax holiday)": "#2563EB", "PPP-ready; monitor": "#93C5FD",
    "Public": "#D97706", "Public + preventive access package": "#DC2626", "No priority intervention": "#CBD5E1",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _labels(panel: pd.DataFrame) -> dict[str, str]:
    labeled = with_district_aliases(panel[[DISTRICT_COLUMN]].drop_duplicates())
    return labeled.set_index(DISTRICT_COLUMN)["district_label"].to_dict()


def _field_label(field: str) -> str:
    rule = COMMAND_CENTER_THRESHOLDS.get(field)
    return rule["label"] if rule else field


def _style(figure: go.Figure, height: int, title: str | None = None) -> go.Figure:
    figure.update_layout(
        template="plotly_white", height=height, font={"color": TEXT},
        title={"text": title, "font": {"size": 15}} if title else None,
        margin={"l": 10, "r": 10, "t": 48 if title else 16, "b": 10},
        legend={"orientation": "h", "y": -0.2, "x": 0},
    )
    return figure


def _load_lines(figure: go.Figure, vertical: bool = False) -> None:
    for value, color, name in ((0.85, WATCH, "Watch 0.85"), (1.0, CRITICAL, "Critical 1.0")):
        if vertical:
            figure.add_vline(x=value, line_dash="dash", line_color=color, line_width=1.2)
        else:
            figure.add_hline(y=value, line_dash="dash", line_color=color, line_width=1.2,
                             annotation_text=name, annotation_font_color=color, annotation_position="top left")


@st.cache_data(show_spinner=False)
def _cached_history(panel: pd.DataFrame) -> pd.DataFrame:
    return trigger_history(panel)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
def _controls(months: list[pd.Timestamp]) -> tuple[pd.Timestamp, int, dict[str, object]]:
    left, right = st.columns([2, 3])
    month = left.select_slider("Baseline month", options=months, value=months[-1],
                               format_func=lambda m: f"{m:%b %Y}", key="l5_month")
    labels = stage_labels()
    stage_label = right.select_slider("Run the domino through", options=labels, value=labels[-1], key="l5_stage")
    through = labels.index(stage_label)

    with st.expander("Stress assumptions (not measured in the panel)", expanded=False):
        spec = STRESS_ASSUMPTIONS
        cols = st.columns(4)
        horizon = cols[0].slider(spec["growth_horizon_months"]["label"], spec["growth_horizon_months"]["min"],
                                 spec["growth_horizon_months"]["max"], int(spec["growth_horizon_months"]["default"]),
                                 key="l5_horizon")
        use_derived = cols[1].checkbox("Derive respiratory multiplier from the panel", value=True, key="l5_derive")
        multiplier = None if use_derived else cols[1].slider(
            "Respiratory multiplier", float(spec["respiratory_multiplier"]["min"]),
            float(spec["respiratory_multiplier"]["max"]), 2.0, 0.05, key="l5_mult")
        cooling = cols[2].slider(spec["cooling_capacity_loss_pct"]["label"], spec["cooling_capacity_loss_pct"]["min"],
                                 spec["cooling_capacity_loss_pct"]["max"], int(spec["cooling_capacity_loss_pct"]["default"]),
                                 key="l5_cooling")
        access = cols[3].slider(spec["access_capacity_loss_pct"]["label"], spec["access_capacity_loss_pct"]["min"],
                                spec["access_capacity_loss_pct"]["max"], int(spec["access_capacity_loss_pct"]["default"]),
                                key="l5_access")
        st.caption("Set both capacity losses to 0 and the multiplier to 2.0 to reproduce Table 5 of the stress-test document.")
    assumptions = {"growth_horizon_months": horizon, "respiratory_multiplier": multiplier,
                   "cooling_capacity_loss_pct": cooling, "access_capacity_loss_pct": access}
    return month, through, assumptions


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _headline(result: dict, base_alerts: pd.DataFrame, stressed_alerts: pd.DataFrame, labels: dict[str, str]) -> None:
    final = result["final"]
    over_base = int((final["pc_only_load_base"] > 1).sum())
    over_stress = int((final["pc_only_load_stressed"] > 1).sum())
    worst = final.iloc[0]
    city_uplift = 100 * (final["pc_demand_stressed"].sum() / final["pc_demand_base"].sum() - 1)
    critical_base = int((base_alerts["status"] == "Critical").sum()) if not base_alerts.empty else 0
    critical_stress = int((stressed_alerts["status"] == "Critical").sum()) if not stressed_alerts.empty else 0
    cols = st.columns(4)
    cols[0].metric("Districts with primary care overloaded", f"{over_stress} of {len(final)}",
                   delta=f"{over_stress - over_base:+d} vs baseline", delta_color="inverse")
    cols[1].metric("Highest primary-care-only load", f"{worst['pc_only_load_stressed']:.2f}",
                   delta=f"{worst['pc_only_load_stressed'] - worst['pc_only_load_base']:+.2f} in "
                         f"{labels.get(worst[DISTRICT_COLUMN], worst[DISTRICT_COLUMN])}", delta_color="inverse")
    cols[2].metric("City primary-care demand", f"{final['pc_demand_stressed'].sum():,.0f}",
                   delta=f"{city_uplift:+.1f}% vs baseline", delta_color="inverse")
    cols[3].metric("Critical indicator alerts", f"{critical_stress}",
                   delta=f"{critical_stress - critical_base:+d} vs baseline", delta_color="inverse")


def _domino_charts(result: dict, labels: dict[str, str], districts: list[str]) -> None:
    path = result["path"].copy()
    path["District"] = path[DISTRICT_COLUMN].map(labels).fillna(path[DISTRICT_COLUMN])
    order = list(dict.fromkeys(path["stage"]))
    top = result["final"][DISTRICT_COLUMN].head(5).tolist()
    chosen = st.multiselect("Districts on the load path", districts, default=top, key="l5_path_districts",
                            format_func=lambda v: labels.get(v, v))
    left, right = st.columns([3, 2])
    with left:
        shown = path.loc[path[DISTRICT_COLUMN].isin(chosen)]
        figure = px.line(shown, x="stage", y="pc_only_load", color="District", markers=True,
                         category_orders={"stage": order},
                         labels={"stage": "", "pc_only_load": "Primary-care-only load"},
                         hover_data={"pc_demand": ":,.0f", "pc_capacity": ":,.0f", "cpr": ":.3f"})
        _load_lines(figure)
        figure.update_xaxes(tickangle=-20)
        plotly_chart(_style(figure, 420, "Load path through the domino"))
    with right:
        final = result["final"].copy()
        final["District"] = final[DISTRICT_COLUMN].map(labels).fillna(final[DISTRICT_COLUMN])
        long = final.melt(id_vars="District", value_vars=["pc_only_load_base", "pc_only_load_stressed"],
                          var_name="Series", value_name="Load")
        long["Series"] = long["Series"].map({"pc_only_load_base": "Baseline", "pc_only_load_stressed": "Stressed"})
        figure = px.bar(long, x="Load", y="District", color="Series", orientation="h", barmode="group",
                        color_discrete_map={"Baseline": "#94A3B8", "Stressed": PRIMARY})
        figure.update_yaxes(title=None, categoryorder="array",
                            categoryarray=final.sort_values("pc_only_load_stressed")["District"].tolist())
        _load_lines(figure, vertical=True)
        plotly_chart(_style(figure, 26 * len(final) + 120, "Baseline vs stressed, by district"))

    table = result["final"].copy()
    table.insert(0, "District", table[DISTRICT_COLUMN].map(labels).fillna(table[DISTRICT_COLUMN]))
    table["Status"] = np.select([table["pc_only_load_stressed"] > 1, table["pc_only_load_stressed"] > 0.85],
                                ["Critical", "Watch"], "OK")
    display = table[["District", "pc_demand_base", "pc_demand_stressed", "demand_uplift_pct", "pc_capacity_stressed",
                     "pc_only_load_base", "pc_only_load_stressed", "cpr_stressed", "Status"]].rename(columns={
        "pc_demand_base": "PC demand", "pc_demand_stressed": "PC demand stressed", "demand_uplift_pct": "Uplift %",
        "pc_capacity_stressed": "PC capacity stressed", "pc_only_load_base": "PC-only load",
        "pc_only_load_stressed": "PC-only load stressed", "cpr_stressed": "cpr stressed"})
    color = lambda v: f"color: {COMMAND_CENTER_STATUS_COLORS.get(v, TEXT)}; font-weight: 600"
    with st.expander("District stress table"):
        st.dataframe(display.style.map(color, subset=["Status"]).format({
            "PC demand": "{:,.0f}", "PC demand stressed": "{:,.0f}", "Uplift %": "{:+.1f}",
            "PC capacity stressed": "{:,.0f}", "PC-only load": "{:.3f}", "PC-only load stressed": "{:.3f}",
            "cpr stressed": "{:.3f}"}), width="stretch", hide_index=True)


def _parameters_used(result: dict, labels: dict[str, str]) -> None:
    p = result["parameters"]
    lines = [f"Primary-care share of visit demand: **{p['primary_care_share']:.3f}** (measured median)"]
    if "respiratory_multiplier" in p:
        lines.append(f"Respiratory multiplier: **{p['respiratory_multiplier']:.2f}** "
                     f"(panel-derived value {p['respiratory_multiplier_derived']:.2f})")
    if "growth_horizon_months" in p:
        lines.append(f"Growth horizon: **{p['growth_horizon_months']:.0f} months** at each district's observed rate")
    for key, setting in (("heat", "cooling_capacity_loss_pct"), ("access", "access_capacity_loss_pct")):
        districts = p.get(f"{key}_loss_districts")
        if districts is not None:
            names = ", ".join(labels.get(d, d) for d in districts) or "none flagged yet"
            lines.append(f"Assumed {p.get(setting, 0):.0f}% capacity loss in: {names}")
    st.markdown("\n".join(f"- {line}" for line in lines))


def _operational(base_alerts: pd.DataFrame, stressed_alerts: pd.DataFrame, panel: pd.DataFrame, labels: dict[str, str]) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Today**")
        _status_grid(base_alerts, panel, labels)
    with right:
        st.markdown("**Under the domino stress**")
        _status_grid(stressed_alerts, panel, labels)
    st.caption("Each stage sets its operational fields to every district's worst observed value up to the "
               "baseline month. Limits are the Layer 4 thresholds in config.py.")


def _triggers(panel: pd.DataFrame, month: pd.Timestamp, fired: pd.DataFrame, labels: dict[str, str]) -> None:
    left, right = st.columns([3, 2])
    with left:
        if fired.empty:
            st.success(f"No trigger rule fires in {month:%B %Y}.")
        else:
            table = fired.copy()
            table["District"] = table[DISTRICT_COLUMN].map(lambda d: labels.get(d, d))
            st.dataframe(table[["id", "case", "trigger", "District", "observed", "action"]].rename(
                columns={"id": "Rule", "case": "Case", "trigger": "Trigger", "observed": "Observed", "action": "Action"}),
                width="stretch", hide_index=True, height=360)
    with right:
        history = _cached_history(panel)
        if history.empty:
            st.info("No trigger has fired in the panel.")
        else:
            history = history.copy()
            history["Case"] = history["case"].map(lambda c: f"Case {c}")
            figure = px.bar(history, x=MONTH_COLUMN, y="fired", color="Case",
                            color_discrete_map={f"Case {k}": v for k, v in CASE_COLORS.items()},
                            labels={MONTH_COLUMN: "", "fired": "Triggers fired"})
            figure.add_vline(x=month.to_pydatetime(), line_dash="dot", line_color="#64748B")
            plotly_chart(_style(figure, 360, "Trigger history"))


def _case_options_engine(panel: pd.DataFrame, labels: dict[str, str], default: str) -> None:
    """Case 1: Layer 3 packages evaluated on the primary-care-only basis."""
    try:
        baseline = derive_baseline_pressure(panel, PRIMARY_CARE_ONLY_MODE)
    except (ValueError, KeyError) as error:
        st.warning(f"Layer 3 engine unavailable: {error}")
        return
    names = baseline["district_name"].tolist()
    target = st.selectbox("Evaluate Layer 3 packages for", names, index=names.index(default) if default in names else 0,
                          format_func=lambda v: labels.get(v, v), key="l5_case1_target")
    comparison = compare_scenarios(baseline, default_settings(), [target])
    st.dataframe(comparison[["Scenario", "5-yr Cost (M EGP)", "Target Avg. Pressure", "Pressure Reduction (%)",
                             "Target Overloaded Districts", "Budget Status"]].style.format({
        "5-yr Cost (M EGP)": "{:.1f}", "Target Avg. Pressure": "{:.2f}", "Pressure Reduction (%)": "{:+.1f}"}),
        width="stretch", hide_index=True)
    st.caption("Primary-care-only basis, six-month baseline, FINANCIAL_ASSUMPTIONS defaults (including "
               f"{default_settings()['population_growth_pct']:.0f}% population growth). Adjust the levers in Layer 3.")


def _case_tab(case: int, panel: pd.DataFrame, month: pd.Timestamp, fired: pd.DataFrame,
              precedents: pd.DataFrame, labels: dict[str, str], result: dict) -> None:
    spec = STRESS_CASES[case]
    stage = next(s for s in STRESS_STAGES if s["case"] == case)
    st.markdown(f"**{stage['window']}.** {spec['title']}")

    st.markdown("##### 1. Situation")
    case_precedents = precedents.loc[precedents["case"].eq(case)]
    if case_precedents.empty:
        st.caption("No precedent event for this stage in the panel up to the baseline month.")
    else:
        st.markdown("\n".join(f"- {r.event}, {getattr(r, MONTH_COLUMN):%b %Y}, {r.districts} districts"
                              for r in case_precedents.itertuples()))

    st.markdown("##### 2. Decision question")
    st.write(spec["question"])

    st.markdown("##### 3. Evidence")
    evidence = case_evidence(panel, month, spec["focus"], spec["evidence"])
    if evidence.empty:
        st.info("No evidence fields are available for the focus districts.")
    else:
        evidence["District"] = evidence[DISTRICT_COLUMN].map(lambda d: labels.get(d, d))
        evidence["Indicator"] = evidence["field"].map(_field_label)
        evidence["Worst month"] = pd.to_datetime(evidence["worst_month"]).dt.strftime("%b %Y")
        st.dataframe(evidence[["District", "Indicator", "current", "worst", "Worst month", "median"]].rename(
            columns={"current": f"{month:%b %Y}", "worst": "Worst observed", "median": "District median"}).style.format(
            {f"{month:%b %Y}": "{:,.3g}", "Worst observed": "{:,.3g}", "District median": "{:,.3g}"}),
            width="stretch", hide_index=True)

    st.markdown("##### 4. Impact chain")
    final = result["final"]
    keys = {clean_district_key(k) for k in spec["focus"]}
    focus = final.loc[final[DISTRICT_KEY_COLUMN].isin(keys)] if DISTRICT_KEY_COLUMN in final else final.head(0)
    if not focus.empty:
        st.markdown("\n".join(
            f"- {labels.get(r[DISTRICT_COLUMN], r[DISTRICT_COLUMN])}: primary-care demand "
            f"{r['pc_demand_base']:,.0f} → {r['pc_demand_stressed']:,.0f} ({r['demand_uplift_pct']:+.1f}%), "
            f"load {r['pc_only_load_base']:.2f} → {r['pc_only_load_stressed']:.2f}"
            for _, r in focus.iterrows()))
    st.caption("From the domino run through the selected stage.")

    st.markdown("##### 5. Options")
    st.markdown("\n".join(f"- **{name}:** {text}" for name, text in spec["options"]))
    if case == 1:
        _case_options_engine(panel, labels, final.iloc[0][DISTRICT_COLUMN])
    if case == 6:
        _funding(panel, month, labels)

    st.markdown("##### 6. Decision & funding output")
    st.write(f"**Decision:** {spec['decision']}")
    st.write(f"**Funding:** {spec['funding']}")

    st.markdown("##### 7. Trigger & monitoring")
    case_fired = fired.loc[fired["case"].eq(case)] if not fired.empty else fired
    if case_fired.empty:
        st.success(f"No Case {case} trigger fires in {month:%B %Y}.")
    else:
        st.markdown("\n".join(f"- **{r.id} {r.trigger}** in {labels.get(getattr(r, DISTRICT_COLUMN), getattr(r, DISTRICT_COLUMN))}: "
                              f"{r.observed}. Action: {r.action}." for r in case_fired.itertuples()))

    st.markdown("##### 8. What is measured vs assumed")
    left, right = st.columns(2)
    left.markdown("**Measured**\n" + "\n".join(f"- {item}" for item in spec["measured"]))
    right.markdown("**Assumed**\n" + "\n".join(f"- {item}" for item in spec["assumed"]))


def _funding(panel: pd.DataFrame, month: pd.Timestamp, labels: dict[str, str]) -> None:
    try:
        quadrants = funding_quadrants(panel, month)
    except (ValueError, KeyError) as error:
        st.warning(f"Funding rule unavailable: {error}")
        return
    quadrants["District"] = quadrants[DISTRICT_COLUMN].map(lambda d: labels.get(d, d))
    figure = px.scatter(quadrants, x="attractiveness", y="need", size="residents", color="funding",
                        color_discrete_map=FUNDING_COLORS, hover_name="District", size_max=40,
                        hover_data={"pc_only_load": ":.2f", "residents": ":,", "funding": True},
                        labels={"attractiveness": "Commercial attractiveness (attractiveness_index_v2)",
                                "need": "Need score", "funding": "Funding output"})
    figure.add_vline(x=quadrants.attrs["attractiveness_cut"], line_dash="dot", line_color="#64748B")
    figure.add_hline(y=quadrants.attrs["need_cut"], line_dash="dot", line_color="#64748B")
    figure.update_layout(hovermode="closest")
    plotly_chart(_style(figure, 440, "Need vs attractiveness (six-month means, medians dotted)"))
    st.dataframe(quadrants[["District", "need", "attractiveness", "residents", "pc_only_load", "quadrant", "funding"]]
                 .sort_values("need", ascending=False).style.format(
                     {"need": "{:.1f}", "attractiveness": "{:.3f}", "residents": "{:,.0f}", "pc_only_load": "{:.2f}"}),
                 width="stretch", hide_index=True)


def _investment_matrix() -> None:
    rows = []
    for item in INVESTMENT_MATRIX:
        cost = package_cost(item["package"]) if item.get("package") else np.nan
        rows.append({"Case": item["case"], "What": item["what"], "Where": item["where"], "When": item["when"],
                     "Who funds": item["who"], "5-yr cost (M EGP)": cost})
    table = pd.DataFrame(rows)
    st.dataframe(table.style.format({"5-yr cost (M EGP)": "{:.1f}"}, na_rep="Not costed"), width="stretch", hide_index=True)
    budget = default_settings()["budget_limit_m_egp"]
    costed = table["5-yr cost (M EGP)"].sum()
    message = (f"Costed items total {costed:.1f} M EGP against the default Layer 3 budget of {budget:.0f} M EGP. "
               "Water, cooling and storage items are not costed in config.py.")
    (st.warning if costed > budget else st.info)(message)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render(panel: pd.DataFrame) -> None:
    st.title("Layer 5 · Crisis & stress testing")
    st.caption("Cascading worst-case scenario: six shocks modelled on real panel episodes, stacked on one "
               "baseline month. Increments use data up to that month only; assumptions are listed separately.")
    panel = add_derived_fields(panel)
    months = [pd.Timestamp(m) for m in sorted(panel[MONTH_COLUMN].dropna().unique())]
    if not months:
        st.error("The panel has no valid months.")
        return
    labels = _labels(panel)
    month, through, assumptions = _controls(months)

    try:
        result = run_domino(panel, month, assumptions, through_stage=through)
        base_alerts, stressed_alerts = alert_comparison(result, month)
        fired = evaluate_triggers(panel, month)
        precedents = stage_precedents(panel, month)
    except (ValueError, KeyError) as error:
        st.error(f"Cannot run the stress test: {error}")
        return

    _headline(result, base_alerts, stressed_alerts, labels)
    with st.expander("Values used in this run"):
        _parameters_used(result, labels)

    st.subheader("Primary-care load through the domino")
    _domino_charts(result, labels, district_options(panel))

    st.subheader("Operational status, today vs under stress")
    _operational(base_alerts, stressed_alerts, panel, labels)

    st.subheader(f"Trigger rules, {month:%B %Y}")
    _triggers(panel, month, fired, labels)

    st.subheader("Cases")
    tabs = st.tabs([f"Case {c}: {STRESS_CASES[c]['title']}" for c in sorted(STRESS_CASES)])
    for tab, case in zip(tabs, sorted(STRESS_CASES)):
        with tab:
            _case_tab(case, panel, month, fired, precedents, labels, result)

    st.subheader("Investment matrix")
    _investment_matrix()

    st.warning("Stress-test results combine measured panel values with the assumptions shown above. They "
               "test decision logic; validate on real operational records before committing capital.")
