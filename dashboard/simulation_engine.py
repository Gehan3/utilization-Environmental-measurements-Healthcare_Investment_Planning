<<<<<<< HEAD
"""Deterministic, transparent district capacity and policy scenario calculations.

All intervention coefficients in FINANCIAL_ASSUMPTIONS are scenario assumptions,
not empirically validated causal effects. Keep them configurable and disclose them
in the dashboard until local evaluation data are available.
"""
=======
"""Deterministic, transparent district capacity and policy scenario calculations."""
>>>>>>> 64059cf510b061bf07d940f87eb4053e5b7efa1e

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from dashboard.config import (
<<<<<<< HEAD
    DISTRICT_KEY_COLUMN,
    DISTRICT_LABEL_COLUMN,
=======
>>>>>>> 64059cf510b061bf07d940f87eb4053e5b7efa1e
    FINANCIAL_ASSUMPTIONS,
    MONTH_COLUMN,
    SCENARIO_PACKAGES,
)
<<<<<<< HEAD
from dashboard.district_utils import clean_district_name, targeted_mask


def _finite_setting(settings: Mapping[str, float], key: str) -> float:
    if key not in settings:
        raise ValueError(f"Missing simulation setting: {key}")
    value = float(settings[key])
    if not np.isfinite(value):
        raise ValueError(f"Simulation setting {key!r} must be finite.")
    return value


def _validate_settings(settings: Mapping[str, float]) -> dict[str, float]:
    values = {
        key: _finite_setting(settings, key)
        for key in (
            "population_growth_pct",
            "capacity_expansion_pct",
            "prevention_reach_pct",
            "access_improvement_minutes",
            "budget_limit_m_egp",
        )
    }
    a = FINANCIAL_ASSUMPTIONS
    bounds = {
        "population_growth_pct": (a["population_growth_min_pct"], a["population_growth_max_pct"]),
        "capacity_expansion_pct": (a["capacity_expansion_min_pct"], a["capacity_expansion_max_pct"]),
        "prevention_reach_pct": (0, a["prevention_reach_max_pct"]),
        "access_improvement_minutes": (a["access_improvement_min_minutes"], a["access_improvement_max_minutes"]),
        "budget_limit_m_egp": (a["budget_min_m_egp"], a["budget_max_m_egp"]),
    }
    for key, (minimum, maximum) in bounds.items():
        if not minimum <= values[key] <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}; got {values[key]}.")
    return values
=======
>>>>>>> 64059cf510b061bf07d940f87eb4053e5b7efa1e


def derive_baseline_pressure(
    panel: pd.DataFrame,
    calibration_mode: str = "Peak-surge / stress-calibrated capacity",
) -> pd.DataFrame:
    """Build a district baseline from the latest available six-month monthly means."""
    required = {
<<<<<<< HEAD
        "district_name", MONTH_COLUMN, "pc_visits_demand", "pc_effective_capacity",
        "pw_waterborne_visits_total", "p3_respiratory_visits_total", "p3_heat_related_visits",
=======
        "district_key",
        MONTH_COLUMN,
        "pc_visits_demand",
        "pc_effective_capacity",
        "pw_waterborne_visits_total",
        "p3_respiratory_visits_total",
        "p3_heat_related_visits",
>>>>>>> 64059cf510b061bf07d940f87eb4053e5b7efa1e
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Simulation panel is missing required fields: {sorted(missing)}")

<<<<<<< HEAD
    months = sorted(panel[MONTH_COLUMN].dropna().drop_duplicates())
    if not months:
        raise ValueError("The simulation panel contains no valid months.")
    window = months[-int(FINANCIAL_ASSUMPTIONS["baseline_window_months"]):]
=======
    months = sorted(panel[MONTH_COLUMN].drop_duplicates())
    window = months[-int(FINANCIAL_ASSUMPTIONS["baseline_window_months"]) :]
>>>>>>> 64059cf510b061bf07d940f87eb4053e5b7efa1e
    recent = panel.loc[panel[MONTH_COLUMN].isin(window)].copy()
    if recent.empty:
        raise ValueError("No rows available for the simulation baseline window.")

<<<<<<< HEAD
    recent["district_name"] = recent["district_name"].map(clean_district_name)
    recent = recent.loc[recent["district_name"].ne("")].copy()
    numeric_fields = [
        "pc_visits_demand", "pc_effective_capacity", "pw_waterborne_visits_total",
        "p3_respiratory_visits_total", "p3_heat_related_visits",
    ]
    for column in numeric_fields:
        recent[column] = pd.to_numeric(recent[column], errors="coerce")
    if recent[numeric_fields].isna().any().any():
        bad = recent[numeric_fields].columns[recent[numeric_fields].isna().any()].tolist()
        raise ValueError(f"Simulation inputs contain missing/non-numeric values in: {bad}")
    if (recent["pc_visits_demand"] < 0).any():
        raise ValueError("Visit demand cannot be negative.")
    if (recent["pc_effective_capacity"] <= 0).any():
        raise ValueError("Effective capacity must be positive in every source row.")
    if (recent[["pw_waterborne_visits_total", "p3_respiratory_visits_total", "p3_heat_related_visits"]] < 0).any().any():
        raise ValueError("Environmental visit components cannot be negative.")

    recent["environmental_demand"] = recent[
        ["pw_waterborne_visits_total", "p3_respiratory_visits_total", "p3_heat_related_visits"]
    ].sum(axis=1)
    aggregations = {
        "demand": ("pc_visits_demand", "mean"),
        "nominal_capacity": ("pc_effective_capacity", "mean"),
        "environmental_demand": ("environmental_demand", "mean"),
    }
    if DISTRICT_KEY_COLUMN in recent.columns:
        aggregations[DISTRICT_KEY_COLUMN] = (DISTRICT_KEY_COLUMN, "first")
    baseline = recent.groupby("district_name", as_index=False).agg(**aggregations)
    if baseline.empty:
        raise ValueError("No named districts are available for the simulation baseline.")
    baseline[DISTRICT_LABEL_COLUMN] = baseline["district_name"]
    if (baseline["nominal_capacity"] <= 0).any():
        raise ValueError("Effective capacity must be positive in every district.")

    # Environmental components can overlap or exceed total visits in source data.
    # Cap the eligible pool at total demand to prevent impossible negative demand.
    baseline["environmental_share"] = (
        baseline["environmental_demand"] / baseline["demand"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)
    baseline["eligible_environmental_demand"] = baseline["demand"] * baseline["environmental_share"]
=======
    recent["environmental_demand"] = recent[
        [
            "pw_waterborne_visits_total",
            "p3_respiratory_visits_total",
            "p3_heat_related_visits",
        ]
    ].sum(axis=1)
    baseline = (
        recent.groupby("district_key", as_index=False)
        .agg(
            demand=("pc_visits_demand", "mean"),
            nominal_capacity=("pc_effective_capacity", "mean"),
            environmental_demand=("environmental_demand", "mean"),
        )
    )
    if (baseline["nominal_capacity"] <= 0).any():
        raise ValueError("Effective capacity must be positive in every district.")
    baseline["environmental_share"] = (
        baseline["environmental_demand"] / baseline["demand"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)
>>>>>>> 64059cf510b061bf07d940f87eb4053e5b7efa1e
    baseline["raw_pressure"] = baseline["demand"] / baseline["nominal_capacity"]

    if calibration_mode == "Raw nominal capacity":
        baseline["baseline_pressure"] = baseline["raw_pressure"]
        baseline["effective_capacity"] = baseline["nominal_capacity"]
        baseline["calibration_factor"] = 1.0
    elif calibration_mode == "Peak-surge / stress-calibrated capacity":
        target_min = float(FINANCIAL_ASSUMPTIONS["calibrated_pressure_min"])
        target_max = float(FINANCIAL_ASSUMPTIONS["calibrated_pressure_max"])
<<<<<<< HEAD
        if target_min <= 0 or target_max < target_min:
            raise ValueError("Calibrated pressure bounds must be positive and ordered.")
        low, high = float(baseline["raw_pressure"].min()), float(baseline["raw_pressure"].max())
        if np.isclose(low, high):
            calibrated = pd.Series((target_min + target_max) / 2, index=baseline.index, dtype=float)
        else:
            calibrated = target_min + ((baseline["raw_pressure"] - low) / (high - low)) * (target_max - target_min)
        baseline["baseline_pressure"] = calibrated
        baseline["effective_capacity"] = baseline["demand"] / baseline["baseline_pressure"].replace(0, np.nan)
        baseline["calibration_factor"] = baseline["effective_capacity"] / baseline["nominal_capacity"]
    else:
        raise ValueError(f"Unknown capacity calibration mode: {calibration_mode!r}")

    if DISTRICT_KEY_COLUMN in baseline.columns:
        baseline["_order"] = pd.to_numeric(baseline[DISTRICT_KEY_COLUMN], errors="coerce")
        baseline = baseline.sort_values(["_order", "district_name"]).drop(columns="_order")
    else:
        baseline = baseline.sort_values("district_name")
    return baseline.reset_index(drop=True)


def _scenario_cost(package: str, settings: Mapping[str, float]) -> float:
    s = _validate_settings(settings)
    assumptions = FINANCIAL_ASSUMPTIONS
    costs = assumptions["scenario_costs_m_egp"]
    default_capacity = float(assumptions["default_capacity_expansion_pct"])
    default_reach = float(assumptions["default_prevention_reach_pct"])
    default_access = abs(float(assumptions["default_access_improvement_minutes"]))
    capacity_scale = s["capacity_expansion_pct"] / default_capacity if default_capacity else 0.0
    prevention_scale = s["prevention_reach_pct"] / default_reach if default_reach else 0.0
    access_scale = abs(s["access_improvement_minutes"]) / default_access if default_access else 0.0

    if package == "center_prevention":
        cost = costs["center_fixed"] + costs["capacity_expansion_at_default"] * capacity_scale + costs["prevention_at_default"] * prevention_scale
    elif package == "capacity":
        cost = costs["capacity_expansion_at_default"] * capacity_scale
    elif package == "access":
        cost = costs["access_program_at_default"] * access_scale
    elif package == "prevention":
        cost = costs["prevention_at_default"] * prevention_scale
    else:
        raise ValueError(f"Unknown scenario package: {package!r}")
    return float(cost)


def _center_allocation_factors(baseline: pd.DataFrame, applies: pd.Series) -> pd.Series:
    """Prioritize higher-pressure target districts without changing the district schema."""
    a = FINANCIAL_ASSUMPTIONS
    minimum = float(a["center_allocation_min_factor"])
    maximum = float(a["center_allocation_max_factor"])
    if minimum <= 0 or maximum < minimum:
        raise ValueError("Center allocation factors must be positive and ordered.")
    pressure = baseline.loc[applies, "baseline_pressure"].astype(float)
    mean_pressure = float(pressure.mean())
    if len(pressure) <= 1 or np.isclose(mean_pressure, 0):
        factors = pd.Series(1.0, index=pressure.index)
    else:
        relative_need = pressure / mean_pressure
        # Moderate sensitivity prevents one outlier from absorbing the whole uplift.
        factors = (1.0 + 0.5 * (relative_need - 1.0)).clip(minimum, maximum)
        factors = factors / float(factors.mean())
        factors = factors.clip(minimum, maximum)
    return factors.reindex(baseline.index).fillna(1.0)
=======
        low = float(baseline["raw_pressure"].min())
        high = float(baseline["raw_pressure"].max())
        if np.isclose(low, high):
            calibrated = pd.Series(
                (target_min + target_max) / 2,
                index=baseline.index,
                dtype=float,
            )
        else:
            calibrated = target_min + (
                (baseline["raw_pressure"] - low) / (high - low)
            ) * (target_max - target_min)
        baseline["baseline_pressure"] = calibrated
        baseline["effective_capacity"] = (
            baseline["demand"] / baseline["baseline_pressure"].replace(0, np.nan)
        )
        baseline["calibration_factor"] = (
            baseline["effective_capacity"] / baseline["nominal_capacity"]
        )
    else:
        raise ValueError(f"Unknown capacity calibration mode: {calibration_mode!r}")

    return baseline.sort_values("district_key").reset_index(drop=True)


def _scenario_cost(package: str, settings: Mapping[str, float]) -> float:
    assumptions = FINANCIAL_ASSUMPTIONS
    costs = assumptions["scenario_costs_m_egp"]
    default_capacity = float(assumptions["default_capacity_expansion_pct"])
    selected_capacity = float(settings["capacity_expansion_pct"])
    default_reach = float(assumptions["default_prevention_reach_pct"])
    selected_reach = float(settings["prevention_reach_pct"])
    access_default_minutes = abs(float(assumptions["default_access_improvement_minutes"]))
    selected_access_minutes = abs(float(settings["access_improvement_minutes"]))

    capacity_scale = selected_capacity / default_capacity if default_capacity else 0.0
    prevention_scale = selected_reach / default_reach if default_reach else 0.0
    access_scale = selected_access_minutes / access_default_minutes if access_default_minutes else 0.0

    if package == "center_prevention":
        return float(
            costs["center_fixed"]
            + costs["capacity_expansion_at_default"] * capacity_scale
            + costs["prevention_at_default"] * prevention_scale
        )
    if package == "capacity":
        return float(costs["capacity_expansion_at_default"] * capacity_scale)
    if package == "access":
        return float(costs["access_program_at_default"] * access_scale)
    if package == "prevention":
        return float(costs["prevention_at_default"] * prevention_scale)
    raise ValueError(f"Unknown scenario package: {package!r}")
>>>>>>> 64059cf510b061bf07d940f87eb4053e5b7efa1e


def simulate_policy_package(
    baseline: pd.DataFrame,
    settings: Mapping[str, float],
    package: str,
    targeted_districts: Sequence[str],
) -> pd.DataFrame:
    """Apply configurable demand/capacity levers to a copy of baseline district metrics."""
    if package not in set(SCENARIO_PACKAGES.values()):
        raise ValueError(f"Unknown scenario package: {package!r}")
<<<<<<< HEAD
    required = {"district_name", "demand", "environmental_demand", "effective_capacity", "baseline_pressure"}
    missing = required.difference(baseline.columns)
    if missing:
        raise ValueError(f"Baseline is missing required fields: {sorted(missing)}")
    if baseline.empty:
        raise ValueError("Cannot simulate an empty district baseline.")
    s = _validate_settings(settings)
    result = baseline.copy()
    applies = targeted_mask(result, targeted_districts)
    if not applies.any():
        raise ValueError("Select at least one target district to run a policy scenario.")

    result["scenario_demand"] = pd.to_numeric(result["demand"], errors="raise").astype(float)
    result["scenario_capacity"] = pd.to_numeric(result["effective_capacity"], errors="raise").astype(float)
    if (result["scenario_demand"] < 0).any() or (result["scenario_capacity"] <= 0).any():
        raise ValueError("Baseline demand must be non-negative and effective capacity positive.")

    population_factor = 1.0 + s["population_growth_pct"] / 100.0
    result.loc[applies, "scenario_demand"] *= population_factor

    if package in {"center_prevention", "prevention"}:
        reach = s["prevention_reach_pct"] / 100.0
        efficacy = float(FINANCIAL_ASSUMPTIONS["prevention_efficacy"])
        if not 0 <= efficacy <= 1:
            raise ValueError("Prevention efficacy must be between 0 and 1.")
        eligible = result["demand"].astype(float) * result.get(
            "environmental_share",
            (result["environmental_demand"] / result["demand"].replace(0, np.nan)).fillna(0).clip(0, 1),
        )
        reduction = eligible * population_factor * reach * efficacy
        result.loc[applies, "scenario_demand"] -= reduction.loc[applies]

    if package == "center_prevention":
        center_pct = float(FINANCIAL_ASSUMPTIONS["new_center_capacity_pct"]) / 100.0
        allocation = _center_allocation_factors(result, applies)
        expansion_pct = s["capacity_expansion_pct"] / 100.0
        result.loc[applies, "scenario_capacity"] *= 1.0 + center_pct * allocation.loc[applies] + expansion_pct
    elif package == "capacity":
        result.loc[applies, "scenario_capacity"] *= 1.0 + s["capacity_expansion_pct"] / 100.0
    elif package == "access":
        minutes = abs(s["access_improvement_minutes"])
        max_gain = float(FINANCIAL_ASSUMPTIONS["triage_capacity_gain_max"])
        half_saturation = float(FINANCIAL_ASSUMPTIONS["triage_capacity_half_saturation_minutes"])
        if not 0 <= max_gain < 1 or half_saturation <= 0:
            raise ValueError("Triage gain must be in [0, 1) and half-saturation must be positive.")
        # Saturating response: each additional minute helps, with diminishing returns.
        gain = max_gain * (1.0 - np.exp(-np.log(2.0) * minutes / half_saturation))
        result.loc[applies, "scenario_capacity"] *= 1.0 + gain

    result["scenario_demand"] = result["scenario_demand"].clip(lower=0.0)
    result["scenario_pressure"] = result["scenario_demand"] / result["scenario_capacity"]
    result["cost_m_egp"] = _scenario_cost(package, s)
    result["overloaded"] = result["scenario_pressure"] > 1.0
    result["targeted"] = applies
    result["pressure_change"] = result["scenario_pressure"] - result["baseline_pressure"]
    result["pressure_reduction_pct"] = np.where(
        result["baseline_pressure"] > 0,
        (result["baseline_pressure"] - result["scenario_pressure"]) / result["baseline_pressure"] * 100.0,
        0.0,
    )
=======
    result = baseline.copy()
    targeted = set(str(value) for value in targeted_districts)
    applies = result["district_key"].astype(str).isin(targeted)

    result["scenario_demand"] = result["demand"].astype(float)
    result["scenario_capacity"] = result["effective_capacity"].astype(float)
    population_factor = 1 + float(settings["population_growth_pct"]) / 100
    result.loc[applies, "scenario_demand"] *= population_factor

    if package in {"center_prevention", "prevention"}:
        reach = float(settings["prevention_reach_pct"]) / 100
        efficacy = float(FINANCIAL_ASSUMPTIONS["prevention_efficacy"])
        reduction = (
            result["environmental_demand"] * population_factor * reach * efficacy
        )
        result.loc[applies, "scenario_demand"] -= reduction.loc[applies]

    if package == "center_prevention":
        capacity_uplift = (
            float(FINANCIAL_ASSUMPTIONS["new_center_capacity_pct"])
            + float(settings["capacity_expansion_pct"])
        ) / 100
        result.loc[applies, "scenario_capacity"] *= 1 + capacity_uplift
    elif package == "capacity":
        result.loc[applies, "scenario_capacity"] *= (
            1 + float(settings["capacity_expansion_pct"]) / 100
        )
    elif package == "access":
        minutes = abs(float(settings["access_improvement_minutes"]))
        elasticity = float(FINANCIAL_ASSUMPTIONS["triage_capacity_gain_per_minute"])
        result.loc[applies, "scenario_capacity"] *= 1 + minutes * elasticity

    result["scenario_pressure"] = (
        result["scenario_demand"] / result["scenario_capacity"]
    )
    result["cost_m_egp"] = _scenario_cost(package, settings)
    result["overloaded"] = result["scenario_pressure"] > 1.0
>>>>>>> 64059cf510b061bf07d940f87eb4053e5b7efa1e
    return result


def compare_scenarios(
    baseline: pd.DataFrame,
    settings: Mapping[str, float],
    targeted_districts: Sequence[str],
) -> pd.DataFrame:
<<<<<<< HEAD
    """Evaluate packages on targeted-district impact, budget feasibility and efficiency."""
    s = _validate_settings(settings)
    budget = s["budget_limit_m_egp"]
    target_mask = targeted_mask(baseline, targeted_districts)
    if not target_mask.any():
        raise ValueError("Select at least one target district to compare scenarios.")

    baseline_target = baseline.loc[target_mask]
    baseline_target_pressure = float(baseline_target["baseline_pressure"].mean())
    baseline_overloaded = int((baseline_target["baseline_pressure"] > 1.0).sum())
    rows: list[dict[str, object]] = []

    for label, package in SCENARIO_PACKAGES.items():
        simulated = simulate_policy_package(baseline, s, package, targeted_districts)
        targeted = simulated.loc[simulated["targeted"]]
        cost = float(simulated["cost_m_egp"].iloc[0])
        target_avg_pressure = float(targeted["scenario_pressure"].mean())
        overloaded_target = int(targeted["overloaded"].sum())
        stabilized = max(0, baseline_overloaded - overloaded_target)
        pressure_reduction_pct = (
            (baseline_target_pressure - target_avg_pressure) / baseline_target_pressure * 100.0
            if baseline_target_pressure > 0 else 0.0
        )
        overload_reduction_pct = (
            (baseline_overloaded - overloaded_target) / baseline_overloaded * 100.0
            if baseline_overloaded else 0.0
        )
        pressure_score = float(np.clip(pressure_reduction_pct / 30.0, 0.0, 1.0))
        overload_score = float(np.clip(overload_reduction_pct / 100.0, 0.0, 1.0))
        stabilization_score = stabilized / baseline_overloaded if baseline_overloaded else pressure_score
        impact_score = 100.0 * (
            0.55 * pressure_score + 0.25 * overload_score + 0.20 * stabilization_score
        )
        cost_per_stabilized = cost / stabilized if stabilized > 0 else np.nan
        rows.append({
            "Scenario": label,
            "5-yr Cost (M EGP)": cost,
            "Budget Status": "Feasible" if cost <= budget else "Exceeds",
            "Overloaded Districts": int(simulated["overloaded"].sum()),
            "Target Overloaded Districts": overloaded_target,
            "Baseline Overloaded Districts": baseline_overloaded,
            "Stabilized Districts": stabilized,
            "Avg. Pressure": float(simulated["scenario_pressure"].mean()),
            "Target Avg. Pressure": target_avg_pressure,
            "Pressure Reduction (%)": pressure_reduction_pct,
            "Overload Reduction (%)": overload_reduction_pct,
            "Cost per Stabilized District (M EGP)": cost_per_stabilized,
            "Impact Score": impact_score,
            "Package": package,
        })

    # Normalize impact delivered per EGP across the compared packages, then blend
    # it with absolute impact so a tiny/cheap intervention cannot win on efficiency alone.
    efficiencies = [
        float(row["Impact Score"]) / max(float(row["5-yr Cost (M EGP)" ]), 0.01)
        for row in rows
    ]
    max_efficiency = max(efficiencies, default=0.0)
    for row, efficiency in zip(rows, efficiencies):
        cost_efficiency_score = 100.0 * efficiency / max_efficiency if max_efficiency > 0 else 0.0
        row["Cost Efficiency Score"] = cost_efficiency_score
        row["Policy Score"] = 0.85 * float(row["Impact Score"]) + 0.15 * cost_efficiency_score

    rows.sort(key=lambda row: (
        row["Budget Status"] != "Feasible",
        -float(row["Policy Score"]),
        float(row["5-yr Cost (M EGP)"],),
        float(row["Target Avg. Pressure"]),
    ))
=======
    """Evaluate and rank all four policy packages under the selected budget limit."""
    budget = float(settings["budget_limit_m_egp"])
    rows = []
    for label, package in SCENARIO_PACKAGES.items():
        simulated = simulate_policy_package(
            baseline, settings, package, targeted_districts
        )
        cost = float(simulated["cost_m_egp"].iloc[0])
        rows.append(
            {
                "Scenario": label,
                "5-yr Cost (M EGP)": cost,
                "Budget Status": "Feasible" if cost <= budget else "Exceeds",
                "Overloaded Districts": int(simulated["overloaded"].sum()),
                "Avg. Pressure": float(simulated["scenario_pressure"].mean()),
                "Package": package,
            }
        )
    rows.sort(
        key=lambda row: (
            row["Budget Status"] != "Feasible",
            row["Overloaded Districts"],
            row["Avg. Pressure"],
            row["5-yr Cost (M EGP)"],
        )
    )
>>>>>>> 64059cf510b061bf07d940f87eb4053e5b7efa1e
    for rank, row in enumerate(rows, start=1):
        row["Rank"] = rank
    return pd.DataFrame(rows)
