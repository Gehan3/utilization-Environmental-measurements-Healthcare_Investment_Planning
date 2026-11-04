"""Executive what-if simulator and budget-constrained policy comparison."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.ui import plotly_chart
from dashboard.config import FINANCIAL_ASSUMPTIONS, SCENARIO_PACKAGES
from dashboard.data_loader import with_district_aliases
from dashboard.simulation_engine import (
    CALIBRATION_MODES,
    PRIMARY_CARE_ONLY_MODE,
    RAW_MODE,
    compare_scenarios,
    derive_baseline_pressure,
    simulate_policy_package,
)



def _scenario_controls(districts: list[str]) -> tuple[dict[str, float], str, list[str]]:
    assumptions = FINANCIAL_ASSUMPTIONS
    aliases = with_district_aliases(pd.DataFrame({"district_name": districts}))
    alias_lookup = aliases.set_index("district_name")["district_alias"].to_dict()
    with st.sidebar.form("what_if_controls"):
        st.subheader("Scenario controls")
        growth = st.slider(
            "Population growth in target districts (%)",
            int(assumptions["population_growth_min_pct"]),
            int(assumptions["population_growth_max_pct"]),
            int(assumptions["default_population_growth_pct"]),
            help="Changes demand in selected districts before policy effects are applied.",
        )
        capacity = st.slider(
            "Center capacity expansion (%)",
            int(assumptions["capacity_expansion_min_pct"]),
            int(assumptions["capacity_expansion_max_pct"]),
            int(assumptions["default_capacity_expansion_pct"]),
            help="Adds effective capacity in the center and capacity-expansion packages.",
        )
        reach = st.slider(
            "Preventive program reach (% of eligible environmental demand)",
            0,
            int(assumptions["prevention_reach_max_pct"]),
            int(assumptions["default_prevention_reach_pct"]),
            help="Share of eligible environmental demand reached by prevention programs.",
        )
        access = st.slider(
            "Access-time / triage improvement (minutes)",
            int(assumptions["access_improvement_min_minutes"]),
            int(assumptions["access_improvement_max_minutes"]),
            int(assumptions["default_access_improvement_minutes"]),
            help="More negative values mean greater time improvement; modeled with diminishing returns.",
        )
        budget = st.slider(
            "5-year budget limit (M EGP)",
            float(assumptions["budget_min_m_egp"]),
            float(assumptions["budget_max_m_egp"]),
            float(assumptions["default_budget_m_egp"]),
            step=1.0,
        )
        package_label = st.selectbox("Active package", list(SCENARIO_PACKAGES))
        target_districts = st.multiselect(
            "Target districts",
            districts,
            default=districts[:1],
            format_func=lambda value: f"{alias_lookup.get(value, value)} ({value})",
        )
        st.form_submit_button("Run simulation", type="primary")

    settings = {
        "population_growth_pct": float(growth),
        "capacity_expansion_pct": float(capacity),
        "prevention_reach_pct": float(reach),
        "access_improvement_minutes": float(access),
        "budget_limit_m_egp": float(budget),
    }
    return settings, SCENARIO_PACKAGES[package_label], target_districts


def _pressure_chart(baseline: pd.DataFrame, scenario: pd.DataFrame, calibration_mode: str) -> None:
    baseline_plot = baseline[["district_name", "baseline_pressure"]].rename(
        columns={"baseline_pressure": "Pressure"}
    )
    baseline_plot["Series"] = "Baseline"
    scenario_plot = scenario[["district_name", "scenario_pressure"]].rename(
        columns={"scenario_pressure": "Pressure"}
    )
    scenario_plot["Series"] = "Active scenario"
    chart_data = pd.concat([baseline_plot, scenario_plot], ignore_index=True)
    aliases = with_district_aliases(baseline[["district_name"]].drop_duplicates())
    chart_data = chart_data.merge(aliases, on="district_name", how="left")
    figure = px.bar(
        chart_data,
        x="district_alias",
        y="Pressure",
        color="Series",
        barmode="group",
        color_discrete_map={"Baseline": "#718096", "Active scenario": "#159a83"},
        labels={"district_alias": "District", "Pressure": "Capacity pressure ratio"},
        title=f"Capacity pressure by district · {calibration_mode}",
    )
    figure.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Capacity limit = 1.0")
    figure.update_layout(height=490, legend={"orientation": "h", "y": -0.2})
    plotly_chart(figure)


def _render_calibration_note(baseline: pd.DataFrame, calibration_mode: str) -> None:
    if calibration_mode == PRIMARY_CARE_ONLY_MODE:
        st.info(
            "Primary-care-only mode uses `pc_visits_demand / pc_capacity_visits`, excluding the 25% "
            "hospital share inside `pc_effective_capacity`. Observed range is "
            f"{baseline['baseline_pressure'].min():.3f}–{baseline['baseline_pressure'].max():.3f}."
        )
    elif calibration_mode == RAW_MODE:
        st.info(
            "Raw mode uses source `pc_visits_demand / pc_effective_capacity`; "
            f"observed range is {baseline['baseline_pressure'].min():.3f}–"
            f"{baseline['baseline_pressure'].max():.3f}."
        )
    else:
        st.warning(
            "Stress-calibrated mode is a scenario visualization, not observed capacity utilization. "
            "It preserves district ordering of raw pressure and maps the six-month min/max to the "
            "configured pressure range. Effective capacity is recalculated from demand and that ratio."
        )


def render(panel: pd.DataFrame) -> None:
    st.title("Layer 3 · Executive What-If Simulation")
    st.caption(
        "Deterministic scenario analysis using configurable policy and financial assumptions; "
        "results are not a forecast or a capital-cost estimate."
    )
    calibration_mode = st.radio(
        "Capacity pressure basis",
        CALIBRATION_MODES,
        horizontal=True,
        index=1,
        help="Source capacity (with hospital share), a stress-calibrated view, or primary-care centers only.",
    )
    try:
        baseline = derive_baseline_pressure(panel, calibration_mode)
    except (ValueError, KeyError) as error:
        st.error(f"Cannot run the policy simulation: {error}")
        return

    districts = baseline["district_name"].astype(str).tolist()
    settings, active_package, targets = _scenario_controls(districts)
    if not targets:
        st.warning("Select at least one target district to run a policy scenario.")
        return

    try:
        active = simulate_policy_package(baseline, settings, active_package, targets)
        comparison = compare_scenarios(baseline, settings, targets)
    except (ValueError, KeyError) as error:
        st.error(f"Cannot calculate scenario outcomes: {error}")
        return

    _render_calibration_note(baseline, calibration_mode)
    target_baseline = baseline.loc[baseline["district_name"].isin(targets)]
    target_active = active.loc[active["targeted"]]
    baseline_avg = float(target_baseline["baseline_pressure"].mean())
    active_avg = float(target_active["scenario_pressure"].mean())
    baseline_overloaded = int((target_baseline["baseline_pressure"] > 1.0).sum())
    active_overloaded = int(target_active["overloaded"].sum())
    active_cost = float(active["cost_m_egp"].iloc[0])

    st.subheader("Active policy impact")
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Target average pressure", f"{active_avg:.3f}",
        delta=f"{active_avg - baseline_avg:+.3f} vs baseline", delta_color="inverse",
    )
    metric_cols[1].metric(
        "Overloaded target districts", f"{active_overloaded} / {len(target_active)}",
        delta=f"{active_overloaded - baseline_overloaded:+d} vs baseline", delta_color="inverse",
    )
    metric_cols[2].metric("5-year package cost", f"{active_cost:.2f} M EGP")
    metric_cols[3].metric(
        "Budget remaining", f"{settings['budget_limit_m_egp'] - active_cost:.2f} M EGP",
        delta="Within budget" if active_cost <= settings["budget_limit_m_egp"] else "Over budget",
        delta_color="normal" if active_cost <= settings["budget_limit_m_egp"] else "inverse",
    )

    _pressure_chart(baseline, active, calibration_mode)

    st.subheader("Budget-constrained package comparison")
    display_columns = [
        "Rank", "Scenario", "5-yr Cost (M EGP)", "Budget Status",
        "Target Overloaded Districts", "Stabilized Districts",
        "Target Avg. Pressure", "Pressure Reduction (%)",
        "Cost per Stabilized District (M EGP)", "Cost Efficiency Score", "Policy Score",
    ]
    st.dataframe(
        comparison[display_columns].style.format({
            "5-yr Cost (M EGP)": "{:.2f}",
            "Target Avg. Pressure": "{:.3f}",
            "Pressure Reduction (%)": "{:+.1f}%",
            "Cost per Stabilized District (M EGP)": "{:.2f}",
            "Cost Efficiency Score": "{:.1f}",
            "Policy Score": "{:.1f}",
        }, na_rep="—"),
        width="stretch",
        hide_index=True,
    )

    impact_figure = px.bar(
        comparison.sort_values("Pressure Reduction (%)", ascending=True),
        x="Pressure Reduction (%)",
        y="Scenario",
        orientation="h",
        color="Budget Status",
        color_discrete_map={"Feasible": "#159a83", "Exceeds": "#d97706"},
        title="Target-district pressure reduction by policy package",
        labels={"Pressure Reduction (%)": "Average pressure reduction (%)"},
        hover_data=["5-yr Cost (M EGP)", "Target Overloaded Districts", "Policy Score"],
    )
    impact_figure.add_vline(x=0, line_dash="dash", line_color="#718096")
    impact_figure.update_layout(height=330, showlegend=True)
    plotly_chart(impact_figure)

    if active_cost > settings["budget_limit_m_egp"]:
        st.error(
            f"Active package exceeds the configured budget by "
            f"{active_cost - settings['budget_limit_m_egp']:.2f} M EGP."
        )
    else:
        st.success(f"Active package is within budget at {active_cost:.2f} M EGP.")

    st.subheader("Model assumptions and limitations")
    st.warning(
        "District-month visit volumes, environmental encounter components and nominal capacity "
        "come from the source panel. Prevention efficacy, new-center capacity uplift, triage "
        "response curve, pressure calibration and 5-year package costs are configurable synthetic "
        "assumptions in `dashboard/config.py`, not measured causal effects or supplier quotations. "
        "The policy score is a decision-support heuristic, not a validated economic appraisal. "
        "Validate coefficients locally before using results for investment decisions."
    )