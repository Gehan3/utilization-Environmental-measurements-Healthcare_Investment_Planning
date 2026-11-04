"""Configuration registry for data fields, benchmark artifacts, and policy assumptions."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "Data" / "Processed" / "master_panel.csv"

EXPERIMENTS = {
    "Experiment 1 — Macro volume": {
        "key": "exp1",
        "label": "Macro volume",
        "target": "total_visits_demand",
        "metrics": ROOT
        / "experiments/exp1_macro_volume/original_artifacts/results/benchmark_metrics.csv",
        "predictions": ROOT
        / "experiments/exp1_macro_volume/original_artifacts/results/test_predictions.csv",
        "importance": ROOT
        / "experiments/exp1_macro_volume/original_artifacts/results/top10_feature_importance.csv",
        "exclusions": ROOT
        / "experiments/exp1_macro_volume/original_artifacts/results/feature_exclusions.csv",
        "models": {},
    },
    "Experiment 2 — Controlled environmental rate": {
        "key": "exp2",
        "label": "Controlled rate",
        "target": "environmental_visits_per_1000_residents",
        "metrics": ROOT / "experiments/exp2_environmental_rate/results/benchmark_metrics.csv",
        "predictions": ROOT / "experiments/exp2_environmental_rate/results/test_predictions.csv",
        "importance": ROOT / "experiments/exp2_environmental_rate/results/top10_feature_importance.csv",
        "exclusions": ROOT / "experiments/exp2_environmental_rate/results/feature_exclusions.csv",
        "models": {
            "Dummy mean": ROOT / "experiments/exp2_environmental_rate/models/dummy_mean.joblib",
            "LightGBM (tuned)": ROOT
            / "experiments/exp2_environmental_rate/models/lightgbm_tuned.joblib",
            "Random forest": ROOT
            / "experiments/exp2_environmental_rate/models/random_forest.joblib",
            "Ridge (tuned)": ROOT
            / "experiments/exp2_environmental_rate/models/ridge_tuned.joblib",
        },
    },
    "Experiment 3 — Pure environmental ablation": {
        "key": "exp3",
        "label": "Pure environmental",
        "target": "environmental_visits_per_1000_residents",
        "metrics": ROOT
        / "experiments/exp3_pure_environmental/results/benchmark_metrics.csv",
        "predictions": ROOT
        / "experiments/exp3_pure_environmental/results/test_predictions.csv",
        "importance": ROOT
        / "experiments/exp3_pure_environmental/results/top10_feature_importance.csv",
        "exclusions": ROOT / "experiments/exp3_pure_environmental/feature_exclusions.csv",
        "models": {
            "Dummy mean": ROOT / "experiments/exp3_pure_environmental/models/dummy_mean.joblib",
            "LightGBM (tuned)": ROOT
            / "experiments/exp3_pure_environmental/models/lightgbm_tuned.joblib",
            "Random forest": ROOT
            / "experiments/exp3_pure_environmental/models/random_forest.joblib",
            "Ridge (tuned)": ROOT
            / "experiments/exp3_pure_environmental/models/ridge_tuned.joblib",
        },
    },
}

DISTRICT_ALIAS_PREFIX = "D"
DISTRICT_ALIAS_WIDTH = 2
DISTRICT_COLUMN = "district_name"
DISTRICT_KEY_COLUMN = "district_key"
DISTRICT_LABEL_COLUMN = "district_label"
# What the dashboard shows for a district: "name" -> the real district_name (default),
# "alias" -> the old D01, D02, ... labels.
DISTRICT_DISPLAY_MODE = "name"
MONTH_COLUMN = "month"

KPI_REGISTRY = {
    "Registered residents": ("residents_registered", "sum"),
    "Daytime population": ("daytime_population", "sum"),
    "Active complaints": ("p4_open_requests_end_month", "sum"),
    "Mean AQI": ("p3_avg_aqi", "mean"),
    "Water pollution index": ("pw_water_pollution_index", "mean"),
    "Total clinical visits": ("total_visits_demand", "sum"),
}

TIME_SERIES_ENVIRONMENT_FIELDS = [
    "pw_water_age_hours",
    "p3_no2_ugm3",
    "p3_pm25_ugm3",
    "p2_uncollected_tons",
    "p3_avg_aqi",
    "pw_water_pollution_index",
]
TIME_SERIES_HEALTH_FIELDS = [
    "p3_respiratory_visits_total",
    "pw_waterborne_visits_total",
    "p3_heat_related_visits",
    "total_visits_demand",
]
VULNERABILITY_GROUPS = {
    "Demographics": ["elderly_pct", "children_pct"],
    "Complaints & service": [
        "p4_total_complaints",
        "p4_complaints_per_1000",
        "p4_avg_resolution_hours",
        "p4_sla_breach_pct",
        "p4_open_requests_end_month",
        "p4_service_need_index",
    ],
    "Environmental stress": [
        "esi_heat",
        "esi_air",
        "esi_water",
        "esi_waste",
        "environmental_stress_index",
    ],
}

FINANCIAL_ASSUMPTIONS = {
    "currency": "M EGP",
    "budget_min_m_egp": 5.0,
    "budget_max_m_egp": 100.0,
    "default_budget_m_egp": 45.0,
    "capacity_expansion_min_pct": 0,
    "capacity_expansion_max_pct": 50,
    "default_capacity_expansion_pct": 15,
    "population_growth_min_pct": -10,
    "population_growth_max_pct": 50,
    "default_population_growth_pct": 20,
    "prevention_reach_max_pct": 40,
    "default_prevention_reach_pct": 20,
    "access_improvement_min_minutes": -15,
    "access_improvement_max_minutes": 0,
    "default_access_improvement_minutes": -5,
    "new_center_capacity_pct": 35.0,
    # Scenario assumptions, not empirically measured intervention effects.
    "prevention_efficacy": 0.35,
    "triage_capacity_gain_max": 0.35,
    "triage_capacity_half_saturation_minutes": 6.0,
    "center_allocation_min_factor": 0.75,
    "center_allocation_max_factor": 1.25,
    "calibrated_pressure_min": 0.75,
    "calibrated_pressure_max": 1.35,
    "baseline_window_months": 6,
    "scenario_costs_m_egp": {
        "center_fixed": 24.0,
        "capacity_expansion_at_default": 9.0,
        "access_program_at_default": 16.0,
        "prevention_at_default": 7.5,
    },
}

SCENARIO_PACKAGES = {
    "A — New center + prevention": "center_prevention",
    "B — Expand existing capacity": "capacity",
    "C — Transport + triage access": "access",
    "D — Prevention only": "prevention",
}


# ---------------------------------------------------------------------------
# Layer 4 · City command center
# ---------------------------------------------------------------------------
# Alert thresholds are planning assumptions informed by common references (WHO drinking-water
# and air-quality guidance, US EPA AQI bands, typical utility / municipal service targets).
# They are NOT legal limits for this city. Tune them here; the dashboard reads nothing else.
#
# direction "above": value > watch -> Watch, value > critical -> Critical
# direction "below": value < watch -> Watch, value < critical -> Critical
# city: how a city-wide value is built from the 15 districts
#   "ratio"         -> recomputed from its components (see command_center.CITY_RATIOS)
#   "sum" / "mean"  -> plain sum or mean across districts
#   "weighted:<col>"-> mean weighted by <col>
COMMAND_CENTER_THRESHOLDS = {
    # Water
    "pw_nrw_pct": {
        "pillar": "Water", "label": "Non-revenue water", "unit": "%",
        "direction": "above", "watch": 25.0, "critical": 35.0, "city": "ratio",
    },
    "pw_chlorine_residual_mg_l": {
        "pillar": "Water", "label": "Chlorine residual", "unit": "mg/L",
        "direction": "below", "watch": 0.3, "critical": 0.2,
        "city": "weighted:pw_water_supplied_m3",
    },
    "pw_turbidity_ntu": {
        "pillar": "Water", "label": "Turbidity", "unit": "NTU",
        "direction": "above", "watch": 1.0, "critical": 5.0,
        "city": "weighted:pw_water_supplied_m3",
    },
    "pw_ecoli_positive_pct": {
        "pillar": "Water", "label": "E. coli positive samples", "unit": "%",
        "direction": "above", "watch": 0.0, "critical": 5.0,
        "city": "weighted:pw_water_supplied_m3",
    },
    "pw_water_age_hours": {
        "pillar": "Water", "label": "Water age", "unit": "h",
        "direction": "above", "watch": 48.0, "critical": 72.0,
        "city": "weighted:pw_water_supplied_m3",
    },
    "pw_supply_interruption_hours": {
        "pillar": "Water", "label": "Supply interruption", "unit": "h/month",
        "direction": "above", "watch": 8.0, "critical": 24.0,
        "city": "weighted:residents_registered",
    },
    # Waste
    "p2_collection_efficiency_pct": {
        "pillar": "Waste", "label": "Collection efficiency", "unit": "%",
        "direction": "below", "watch": 90.0, "critical": 85.0, "city": "ratio",
    },
    "p2_bin_hours_ge85_pct": {
        "pillar": "Waste", "label": "Bin hours ≥85% full", "unit": "%",
        "direction": "above", "watch": 10.0, "critical": 20.0,
        "city": "weighted:p2_bins_installed",
    },
    # Air & heat
    "p3_avg_aqi": {
        "pillar": "Air & heat", "label": "Average AQI", "unit": "",
        "direction": "above", "watch": 100.0, "critical": 150.0,
        "city": "weighted:p3_population_exposed",
    },
    "p3_pm25_ugm3": {
        "pillar": "Air & heat", "label": "PM2.5", "unit": "µg/m³",
        "direction": "above", "watch": 35.0, "critical": 75.0,
        "city": "weighted:p3_population_exposed",
    },
    "heat_days_ge38c": {
        "pillar": "Air & heat", "label": "Days ≥38 °C", "unit": "days",
        "direction": "above", "watch": 3.0, "critical": 7.0, "city": "max",
    },
    # Citizen service
    "p4_citizen_satisfaction_pct": {
        "pillar": "Citizen service", "label": "Citizen satisfaction", "unit": "%",
        "direction": "below", "watch": 70.0, "critical": 60.0,
        "city": "weighted:p3_population_exposed",
    },
    "p4_sla_breach_pct": {
        "pillar": "Citizen service", "label": "SLA breaches", "unit": "%",
        "direction": "above", "watch": 20.0, "critical": 30.0,
        "city": "weighted:p4_total_complaints",
    },
    "p4_complaints_per_1000": {
        "pillar": "Citizen service", "label": "Complaints per 1,000", "unit": "",
        "direction": "above", "watch": 20.0, "critical": 40.0, "city": "ratio",
    },
    # Health capacity
    "cpr": {
        "pillar": "Health capacity", "label": "Primary-care pressure", "unit": "ratio",
        "direction": "above", "watch": 0.85, "critical": 1.0, "city": "ratio",
    },
}

# Pillar order, headline indicator (shown on the status card) and accent colour.
COMMAND_CENTER_PILLARS = {
    "Water": {"headline": "pw_nrw_pct", "color": "#0D9488"},
    "Waste": {"headline": "p2_collection_efficiency_pct", "color": "#65A30D"},
    "Air & heat": {"headline": "p3_avg_aqi", "color": "#7C3AED"},
    "Citizen service": {"headline": "p4_citizen_satisfaction_pct", "color": "#2563EB"},
    "Health capacity": {"headline": "cpr", "color": "#DB2777"},
}

# City-wide aggregation for fields that are charted but have no alert threshold.
# Fields not listed here and not in COMMAND_CENTER_THRESHOLDS are summed.
COMMAND_CENTER_CITY_AGGREGATION = {
    "pw_night_flow_ratio": "weighted:pw_water_supplied_m3",
    "pw_water_pollution_index": "weighted:pw_water_supplied_m3",
    "p2_avg_bin_fill_pct": "weighted:p2_bins_installed",
    "p2_recycling_diversion_pct": "weighted:p2_waste_generated_tons",
    "p2_odor_pest_risk_index": "weighted:p3_population_exposed",
    "p3_max_aqi": "max",
    "p3_pm10_ugm3": "weighted:p3_population_exposed",
    "p3_no2_ugm3": "weighted:p3_population_exposed",
    "p3_o3_ugm3": "weighted:p3_population_exposed",
    "avg_temp_c": "mean",
    "p4_avg_resolution_hours": "weighted:p4_total_complaints",
    "need_score": "weighted:residents_registered",
    "environmental_stress_index": "weighted:residents_registered",
}

# Which pillar each event flag belongs to (for timeline colours). Unlisted events -> "Other".
COMMAND_CENTER_EVENT_PILLARS = {
    "night_flow_anomaly": "Water",
    "chlorination_failure": "Water",
    "pipe_burst_campaign": "Water",
    "sewage_main_failure": "Water",
    "compound_sewage_heat_failure": "Water",
    "compound_sewage_waste_failure": "Water",
    "eid_aladha_waste_overflow": "Waste",
    "dust_storm": "Air & heat",
    "major_dust_storm": "Air & heat",
    "winter_inversion_smog": "Air & heat",
    "heatwave_jul2025": "Air & heat",
    "heatwave_aug2026": "Air & heat",
    "heat_spell": "Air & heat",
    "heat_tail": "Air & heat",
    "winter_respiratory_season": "Health capacity",
}

COMMAND_CENTER_STATUS_COLORS = {
    "OK": "#0D9488",
    "Watch": "#D97706",
    "Critical": "#DC2626",
    "No data": "#94A3B8",
}