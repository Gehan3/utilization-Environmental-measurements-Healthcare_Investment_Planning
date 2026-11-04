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
    # Derived: pc_visits_demand / pc_capacity_visits (primary-care centers only, no hospital share).
    "pc_only_load": {
        "pillar": "Health capacity", "label": "Primary-care-only load", "unit": "ratio",
        "direction": "above", "watch": 0.85, "critical": 1.0, "city": "ratio",
    },
}

# Pillar order, headline indicator (shown on the status card) and accent colour.
COMMAND_CENTER_PILLARS = {
    "Water": {"headline": "pw_nrw_pct", "color": "#0D9488"},
    "Waste": {"headline": "p2_collection_efficiency_pct", "color": "#65A30D"},
    "Air & heat": {"headline": "p3_avg_aqi", "color": "#7C3AED"},
    "Citizen service": {"headline": "p4_citizen_satisfaction_pct", "color": "#2563EB"},
    "Health capacity": {"headline": "pc_only_load", "color": "#DB2777"},
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


# ---------------------------------------------------------------------------
# Layer 5 · Crisis & stress testing
# ---------------------------------------------------------------------------
# The domino sequence. Increments are derived from the panel up to the baseline month:
#   visit_increments   -> each district's worst observed value minus its baseline-month value (>= 0)
#   respiratory_event  -> respiratory multiplier = median, over districts flagged by the event, of
#                         (event-month visits / mean of the 6 months before)
#   operational_fields -> set to each district's worst observed value (direction from thresholds)
#   capacity_loss      -> ASSUMED % loss of primary-care capacity in districts flagged by loss_events
STRESS_STAGES = [
    {
        "key": "growth", "case": 1, "label": "Demand growth", "window": "Week 0",
        "growth_field": "pc_visits_demand",
        "precedent_events": [],
    },
    {
        "key": "heat", "case": 2, "label": "Heatwave and cooling shock", "window": "Weeks 1-2",
        "visit_increments": ["p3_heat_related_visits"],
        "capacity_loss": "cooling_capacity_loss_pct",
        "loss_events": ["compound_sewage_heat_failure"],
        "operational_fields": ["heat_days_ge38c", "p2_collection_efficiency_pct", "p2_bin_hours_ge85_pct"],
        "precedent_events": ["heatwave_aug2026", "heatwave_jul2025", "compound_sewage_heat_failure"],
    },
    {
        "key": "water", "case": 3, "label": "Water outage cascade", "window": "Weeks 2-4",
        "visit_increments": ["pw_waterborne_visits_total"],
        "operational_fields": [
            "pw_nrw_pct", "pw_supply_interruption_hours", "pw_chlorine_residual_mg_l",
            "pw_ecoli_positive_pct", "pw_turbidity_ntu", "pw_water_age_hours",
        ],
        "precedent_events": ["pipe_burst_campaign", "sewage_main_failure", "chlorination_failure"],
    },
    {
        "key": "air", "case": 4, "label": "Air-quality respiratory surge", "window": "Weeks 4-6",
        "respiratory_event": "winter_inversion_smog",
        "operational_fields": ["p3_avg_aqi", "p3_pm25_ugm3"],
        "precedent_events": ["winter_inversion_smog", "major_dust_storm"],
    },
    {
        "key": "access", "case": 5, "label": "Access and road disruption", "window": "Weeks 5-7",
        "capacity_loss": "access_capacity_loss_pct",
        "loss_events": ["road_closure_ring_road"],
        "operational_fields": ["p4_sla_breach_pct", "p4_complaints_per_1000", "p4_citizen_satisfaction_pct"],
        "precedent_events": ["road_closure_ring_road"],
    },
    {
        "key": "viability", "case": 6, "label": "Viability gap", "window": "Weeks 8-12",
        "precedent_events": [],
    },
]

# Values the panel cannot measure. Shown in the UI as assumptions, with these bounds.
STRESS_ASSUMPTIONS = {
    "growth_horizon_months": {"default": 0, "min": 0, "max": 24,
                              "label": "Months of growth at each district's observed rate"},
    "respiratory_multiplier": {"default": None, "min": 1.0, "max": 3.0,
                               "label": "Respiratory multiplier (blank = derived from the panel)"},
    "cooling_capacity_loss_pct": {"default": 10, "min": 0, "max": 50,
                                  "label": "Primary-care capacity lost to cooling failure (%)"},
    "access_capacity_loss_pct": {"default": 10, "min": 0, "max": 50,
                                 "label": "Primary-care capacity lost to road closure (%)"},
}

# Trigger rules T1-T11. Condition kinds:
#   threshold        field op value (same month)
#   growth           mean month-on-month % growth over `months` months, op value
#   yoy_ratio        value / max(same month last year, 1), op value
#   own_median_ratio value / district median up to the month, op value
#   mom_change       % change vs previous month, op value
#   event            event_flags contains any of `contains`
#   median_cross     district crossed the city median of any `fields` and stayed there `months` months
# Optional "min_value": the condition only counts when the field itself is at least this large.
STRESS_TRIGGERS = [
    {"id": "T1", "case": 1, "label": "Primary-care overload", "combine": "any",
     "conditions": [{"kind": "threshold", "field": "pc_only_load", "op": ">", "value": 1.0}],
     "action": "Open Case 1 for this district"},
    {"id": "T2", "case": 1, "label": "Demand growth acceleration", "combine": "any",
     "conditions": [{"kind": "growth", "field": "pc_visits_demand", "months": 3, "op": ">", "value": 3.5,
                     "min_value": 500}],
     "action": "Re-project 12-month load"},
    {"id": "T3", "case": 2, "label": "Heat protocol", "combine": "any", "scope": "city",
     "conditions": [{"kind": "threshold", "field": "heat_days_ge38c", "op": ">=", "value": 7},
                    {"kind": "threshold", "field": "esi_heat", "op": ">=", "value": 60}],
     "action": "Activate heat protocol (Case 2)"},
    {"id": "T4", "case": 2, "label": "Heat visits doubled year on year", "combine": "any",
     "conditions": [{"kind": "yoy_ratio", "field": "p3_heat_related_visits", "op": ">", "value": 2.0,
                     "min_value": 20}],
     "action": "Review cooling contracts"},
    {"id": "T5", "case": 3, "label": "Water outage", "combine": "any",
     "conditions": [{"kind": "threshold", "field": "pw_supply_interruption_hours", "op": ">", "value": 24},
                    {"kind": "threshold", "field": "pw_nrw_pct", "op": ">", "value": 35}],
     "action": "Open Case 3"},
    {"id": "T6", "case": 3, "label": "Probable burst", "combine": "any",
     "conditions": [{"kind": "own_median_ratio", "field": "pw_night_flow_ratio", "op": ">", "value": 3.0},
                    {"kind": "own_median_ratio", "field": "pw_leak_incidents", "op": ">", "value": 2.0}],
     "action": "Flag probable burst"},
    {"id": "T7", "case": 3, "label": "Water-quality incident", "combine": "any",
     "conditions": [{"kind": "threshold", "field": "pw_chlorine_residual_mg_l", "op": "<", "value": 0.2},
                    {"kind": "threshold", "field": "pw_ecoli_positive_pct", "op": ">", "value": 0}],
     "action": "Water-quality incident response"},
    {"id": "T8", "case": 4, "label": "Severe air episode", "combine": "any",
     "conditions": [{"kind": "threshold", "field": "p3_avg_aqi", "op": ">", "value": 150},
                    {"kind": "threshold", "field": "p3_pm25_ugm3", "op": ">", "value": 75},
                    {"kind": "threshold", "field": "p3_days_aqi_gt150", "op": ">=", "value": 10}],
     "action": "Activate surge plan (Case 4)"},
    {"id": "T9", "case": 5, "label": "Access disruption", "combine": "any",
     "conditions": [{"kind": "event", "contains": ["road_closure"]},
                    {"kind": "all", "conditions": [
                        {"kind": "mom_change", "field": "p4_infrastructure_complaints", "op": ">", "value": 10,
                         "min_value": 100},
                        {"kind": "mom_change", "field": "p4_open_requests_end_month", "op": ">", "value": 25}]}],
     "action": "Activate access protocol (Case 5)"},
    {"id": "T10", "case": 6, "label": "Funding quadrant change", "combine": "any",
     "conditions": [{"kind": "median_cross", "fields": ["need_score", "attractiveness_index_v2"], "months": 6}],
     "action": "Re-classify funding quadrant (Case 6)"},
    {"id": "T11", "case": 2, "label": "Compound heat or sewage failure", "combine": "all",
     "conditions": [{"kind": "threshold", "field": "p2_collection_efficiency_pct", "op": "<", "value": 85},
                    {"kind": "event", "contains": ["heat", "sewage"]}],
     "action": "Escalate compound-failure review"},
]

# Case 6 funding rule, on six-month means up to the selected month.
FUNDING_RULES = {
    "window_months": 6,
    "attractiveness_band": 0.03,       # within +/- band of the median counts as "near median"
    "capacity_gap_load": 0.85,         # pc_only_load at or above this is a proven capacity gap
    "small_population_residents": 10000,
    "low_load_for_prevention": 0.5,
}

# Narrative per case for the Layer 5 view. Districts are referenced by district_key.
STRESS_CASES = {
    1: {
        "title": "Rapid demand growth",
        "focus": [1001, 1003, 1005],
        "evidence": ["pc_only_load", "cpr", "pc_visits_demand", "pc_capacity_visits", "residents_registered"],
        "question": "Expand the existing clinic, build a new center (with prevention), or run a triage and access programme while hospitals absorb overflow?",
        "options": [
            ("Expand existing capacity", "Layer 3 package B; at most +50% capacity."),
            ("New center + prevention", "Layer 3 package A; a new center adds 35% of existing capacity in the engine."),
            ("Triage / access programme", "Layer 3 package C; saturating capacity gain, at most 35%."),
        ],
        "decision": "Build a new primary-care center sized to bring load to 0.85, with a formal hospital-referral pathway until it opens.",
        "funding": "PPP with public land contribution; public capex if no compliant bid.",
        "measured": ["pc_visits_demand, pc_capacity_visits, pc_centers, hospital_capacity_visits, residents_registered, cpr"],
        "assumed": ["Demand and capacity on the same monthly basis", "Hospitals can absorb overflow",
                    "Package costs and the 35% new-center uplift (FINANCIAL_ASSUMPTIONS)"],
    },
    2: {
        "title": "Heatwave and cooling reliability shock",
        "focus": [1003, 1005, 1009, 1002],
        "evidence": ["p3_heat_related_visits", "pw_waterborne_visits_total", "pw_sewage_overflow_incidents",
                     "p2_collection_efficiency_pct", "pw_supply_interruption_hours", "p4_total_complaints"],
        "question": "Backup cooling, temporary service relocation, or cooling-reliability contract gating?",
        "options": [
            ("Backup cooling", "Standby cooling and power at critical clinics."),
            ("Temporary relocation", "Pre-agreed moves to cooler sites with spare capacity."),
            ("Reliability contract gating", "Cooling uptime as a condition of operating contracts."),
        ],
        "decision": "Contract gating as standing policy, plus backup cooling at critical R3 and R5 sites; relocation as fallback.",
        "funding": "PPP performance contract; public capex for the critical-site retrofit.",
        "measured": ["Heat days, esi_heat, heat and waterborne visits, sewage overflows, collection efficiency, interruption hours, complaints, event flags"],
        "assumed": ["Any cooling or power failure", "Capacity lost during heat (cooling_capacity_loss_pct)",
                    "Heat is recorded city-wide, so district exposure differs only through failures"],
    },
    3: {
        "title": "Water outage cascade",
        "focus": [1007, 1008],
        "evidence": ["pw_nrw_pct", "pw_night_flow_ratio", "pw_leak_incidents", "pw_supply_interruption_hours",
                     "pw_water_age_hours", "pw_chlorine_residual_mg_l"],
        "question": "Patch after each failure, long-term reliability replacement, or contingency storage?",
        "options": [
            ("Immediate patch", "Repair each burst as it occurs."),
            ("Reliability replacement", "Programmed network renewal with pressure and district metering."),
            ("Contingency storage", "Tanks at clinics, schools and dense blocks sized for the worst outage."),
        ],
        "decision": "Contingency storage now, sized for the worst observed outage; phased network renewal; patching only as emergency response.",
        "funding": "Public utility capex.",
        "measured": ["NRW, night-flow ratio, leaks, interruption hours, water age, chlorine, E. coli, waterborne visits, event flags"],
        "assumed": ["Pump failure as a cause (the panel records a pipe burst)", "Storage sizing and cost",
                    "That renewal lowers NRW to city levels"],
    },
    4: {
        "title": "Air-quality event and respiratory surge",
        "focus": [1006, 1007, 1008],
        "evidence": ["p3_avg_aqi", "p3_pm25_ugm3", "p3_days_aqi_gt150", "p3_respiratory_visits_total", "pc_only_load"],
        "question": "Temporary surge staffing and telehealth, or permanent capacity expansion?",
        "options": [
            ("Surge staffing and telehealth", "Seasonal staffing pool, respiratory telehealth, school air alerts."),
            ("Permanent expansion", "Add respiratory capacity in exposed districts."),
            ("Prevention", "Layer 3 package D: alerts, protective equipment, indoor-air measures."),
        ],
        "decision": "Surge staffing and telehealth plus prevention; no permanent build on air quality alone.",
        "funding": "Public operating budget; private telehealth under a framework contract.",
        "measured": ["AQI, PM2.5, PM10, days above AQI 150, respiratory visits, event flags"],
        "assumed": ["That any severe episode reaches the inversion's respiratory multiplier",
                    "Prevention efficacy (0.35) and telehealth substitution"],
    },
    5: {
        "title": "Access and road disruption",
        "focus": [1003, 1005],
        "evidence": ["p4_infrastructure_complaints", "p4_open_requests_end_month", "p4_avg_resolution_hours",
                     "p2_collection_efficiency_pct", "pc_visits_demand"],
        "question": "Temporary shuttle, a near-site center, or home care and telehealth?",
        "options": [
            ("Temporary shuttle", "Scheduled transport to open clinics during the closure."),
            ("Near-site center", "Build or expand a center on the cut-off side."),
            ("Home care and telehealth", "Remote consultations and home visits for vulnerable patients."),
        ],
        "decision": "Shuttle plus telehealth for the duration; near-site center only if closures recur or last beyond six weeks.",
        "funding": "Public opex for the shuttle; private telehealth.",
        "measured": ["Road-closure flag, complaints, open requests, resolution hours, collection efficiency, demand"],
        "assumed": ["Travel times and 15-minute coverage (not in the panel)",
                    "Lost care (unmet_visits is always 0)", "Capacity lost to closure (access_capacity_loss_pct)"],
    },
    6: {
        "title": "Need versus commercial attractiveness",
        "focus": [1001, 1002, 1003, 1005, 1007, 1008],
        "evidence": ["need_score", "attractiveness_index_v2", "residents_registered", "pc_only_load"],
        "question": "Private, PPP (land subsidy or tax holiday), public, or a preventive access package?",
        "options": [
            ("Private", "Need and attractiveness above median."),
            ("PPP", "Need above median, attractiveness near median, proven capacity gap."),
            ("Public", "Need above median, attractiveness below median."),
            ("Preventive access package", "High need, small population, ample capacity."),
        ],
        "decision": "Apply the funding rule (FUNDING_RULES) district by district.",
        "funding": "Mixed by quadrant.",
        "measured": ["need_score, attractiveness_index_v2, residents, capacity fields"],
        "assumed": ["Medians as cut-offs", "That attractiveness predicts investor appetite",
                    "Construction of need_score and attractiveness_index_v2 (precomputed, undocumented)"],
    },
}

# Investment programme. "package" rows are costed from FINANCIAL_ASSUMPTIONS via simulation_engine.package_cost.
INVESTMENT_MATRIX = [
    {"what": "Hospital-referral pathway (interim)", "where": "R1", "when": "Immediately", "who": "Public (service agreement)", "case": 1},
    {"what": "New primary-care center", "where": "R1", "when": "Procure now; open in 18-24 months", "who": "PPP, public fallback", "case": 1, "package": "center_prevention"},
    {"what": "Cooling reliability clauses", "where": "All health facilities", "when": "Next contract cycle", "who": "Operators under PPP terms", "case": 2},
    {"what": "Critical-site backup cooling", "where": "R3, R5", "when": "Before next summer", "who": "Public capex", "case": 2},
    {"what": "Contingency water storage", "where": "R7, R8", "when": "Within 6 months", "who": "Public utility", "case": 3},
    {"what": "Network renewal programme", "where": "R7, R8", "when": "Phased over 3-5 years", "who": "Public utility capex", "case": 3},
    {"what": "Seasonal surge staffing and telehealth", "where": "City-wide; R6-R8 priority", "when": "Each Nov-Mar", "who": "Public opex; private telehealth", "case": 4, "package": "prevention"},
    {"what": "Closure shuttle and telehealth protocol", "where": "R3, R5", "when": "Standing protocol", "who": "Public opex; private telehealth", "case": 5},
    {"what": "Preventive access package", "where": "R7, R8 (R2 on trigger)", "when": "Within 12 months", "who": "Public", "case": 6, "package": "prevention"},
]