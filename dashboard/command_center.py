"""City command-center calculations: status rules, alerts, city-wide aggregation and events.

No Streamlit or Plotly dependency, so it can be tested on its own (same pattern as
simulation_engine.py). Thresholds live in dashboard/config.py.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from dashboard.config import (
    COMMAND_CENTER_CITY_AGGREGATION,
    COMMAND_CENTER_EVENT_PILLARS,
    COMMAND_CENTER_PILLARS,
    COMMAND_CENTER_THRESHOLDS,
    DISTRICT_COLUMN,
    MONTH_COLUMN,
)
from dashboard.district_utils import clean_district_name

STATUS_LEVELS = {0: "OK", 1: "Watch", 2: "Critical"}
EVENT_COLUMN = "event_flags"


def _sum(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").sum())


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else np.nan


# City-wide values that must be recomputed from components, not averaged across districts.
# Each formula reproduces how the panel's own district-level field is defined.
CITY_RATIOS: dict[str, tuple[tuple[str, ...], Callable[[pd.DataFrame], float]]] = {
    "pw_nrw_pct": (
        ("pw_water_supplied_m3", "pw_metered_consumption_m3"),
        lambda f: 100
        * _safe_ratio(
            _sum(f, "pw_water_supplied_m3") - _sum(f, "pw_metered_consumption_m3"),
            _sum(f, "pw_water_supplied_m3"),
        ),
    ),
    "p2_collection_efficiency_pct": (
        ("p2_uncollected_tons", "p2_waste_generated_tons"),
        lambda f: 100
        * (1 - _safe_ratio(_sum(f, "p2_uncollected_tons"), _sum(f, "p2_waste_generated_tons"))),
    ),
    "p4_complaints_per_1000": (
        ("p4_total_complaints", "p3_population_exposed"),
        lambda f: 1000 * _safe_ratio(_sum(f, "p4_total_complaints"), _sum(f, "p3_population_exposed")),
    ),
    "cpr": (
        ("pc_visits_demand", "pc_effective_capacity"),
        lambda f: _safe_ratio(_sum(f, "pc_visits_demand"), _sum(f, "pc_effective_capacity")),
    ),
}


def validate_thresholds(thresholds: Mapping[str, Mapping[str, object]] = COMMAND_CENTER_THRESHOLDS) -> None:
    """Raise if a rule is internally inconsistent (wrong order, unknown pillar or direction)."""
    for field, rule in thresholds.items():
        direction = rule["direction"]
        watch, critical = float(rule["watch"]), float(rule["critical"])
        if direction == "above" and watch > critical:
            raise ValueError(f"{field}: for 'above' rules watch must be <= critical.")
        if direction == "below" and watch < critical:
            raise ValueError(f"{field}: for 'below' rules watch must be >= critical.")
        if direction not in {"above", "below"}:
            raise ValueError(f"{field}: direction must be 'above' or 'below'.")
        if rule["pillar"] not in COMMAND_CENTER_PILLARS:
            raise ValueError(f"{field}: unknown pillar {rule['pillar']!r}.")
        if rule.get("city") == "ratio" and field not in CITY_RATIOS:
            raise ValueError(f"{field}: city='ratio' but no formula exists in CITY_RATIOS.")


def status_level(value: object, rule: Mapping[str, object]) -> float:
    """0 = OK, 1 = Watch, 2 = Critical, NaN = no data."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return np.nan
    watch, critical = float(rule["watch"]), float(rule["critical"])
    if rule["direction"] == "above":
        return 2.0 if number > critical else 1.0 if number > watch else 0.0
    return 2.0 if number < critical else 1.0 if number < watch else 0.0


def is_better(change: float, field: str) -> bool | None:
    """Whether a positive change is good for this field (None if the field has no direction)."""
    rule = COMMAND_CENTER_THRESHOLDS.get(field)
    if rule is None or pd.isna(change) or change == 0:
        return None
    return (change < 0) if rule["direction"] == "above" else (change > 0)


def aggregation_method(field: str) -> str:
    if field in COMMAND_CENTER_THRESHOLDS:
        return str(COMMAND_CENTER_THRESHOLDS[field].get("city", "mean"))
    return str(COMMAND_CENTER_CITY_AGGREGATION.get(field, "sum"))


def city_value(frame: pd.DataFrame, field: str) -> float:
    """One city-wide value for `field` from a set of district rows (usually one month)."""
    if frame.empty or field not in frame:
        return np.nan
    method = aggregation_method(field)
    if method == "ratio" and field in CITY_RATIOS:
        components, formula = CITY_RATIOS[field]
        if all(column in frame for column in components):
            return float(formula(frame))
        method = "mean"
    values = pd.to_numeric(frame[field], errors="coerce")
    if method.startswith("weighted:"):
        weight_column = method.split(":", 1)[1]
        if weight_column in frame:
            weights = pd.to_numeric(frame[weight_column], errors="coerce")
            valid = values.notna() & weights.notna() & (weights > 0)
            if valid.any():
                return float(np.average(values[valid], weights=weights[valid]))
        method = "mean"
    if values.dropna().empty:
        return np.nan
    if method == "sum":
        return float(values.sum())
    if method == "max":
        return float(values.max())
    return float(values.mean())


def scope_frame(panel: pd.DataFrame, district: str | None) -> pd.DataFrame:
    if district is None:
        return panel
    return panel.loc[panel[DISTRICT_COLUMN].map(clean_district_name).eq(clean_district_name(district))]


def monthly_series(panel: pd.DataFrame, fields: list[str], district: str | None = None) -> pd.DataFrame:
    """Month-by-month values (city-wide or for one district), one column per field."""
    frame = scope_frame(panel, district)
    present = [field for field in fields if field in frame]
    rows = []
    for month, group in frame.groupby(MONTH_COLUMN, sort=True):
        row = {MONTH_COLUMN: month}
        row.update({field: city_value(group, field) for field in present})
        rows.append(row)
    return pd.DataFrame(rows, columns=[MONTH_COLUMN, *present])


def district_alerts(panel: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Every district x threshold-indicator for one month, with its status."""
    validate_thresholds()
    month_rows = panel.loc[panel[MONTH_COLUMN].eq(month)]
    records = []
    for _, row in month_rows.iterrows():
        for field, rule in COMMAND_CENTER_THRESHOLDS.items():
            if field not in month_rows:
                continue
            level = status_level(row[field], rule)
            records.append(
                {
                    DISTRICT_COLUMN: clean_district_name(row[DISTRICT_COLUMN]),
                    "pillar": rule["pillar"],
                    "field": field,
                    "indicator": rule["label"],
                    "unit": rule["unit"],
                    "value": pd.to_numeric(row[field], errors="coerce"),
                    "direction": rule["direction"],
                    "watch": float(rule["watch"]),
                    "critical": float(rule["critical"]),
                    "level": level,
                    "status": STATUS_LEVELS.get(level, "No data") if pd.notna(level) else "No data",
                }
            )
    return pd.DataFrame(records)


def pillar_grid(alerts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """District x pillar worst status level, plus hover text listing breached indicators."""
    pillars = list(COMMAND_CENTER_PILLARS)
    if alerts.empty:
        return pd.DataFrame(columns=pillars), pd.DataFrame(columns=pillars)
    levels = alerts.pivot_table(
        index=DISTRICT_COLUMN, columns="pillar", values="level", aggfunc="max"
    ).reindex(columns=pillars)

    def _describe(group: pd.DataFrame) -> str:
        breached = group.loc[group["level"] >= 1].sort_values("level", ascending=False)
        if breached.empty:
            return "All indicators within limits"
        return "<br>".join(
            f"{r.status}: {r.indicator} {r.value:,.2f}{(' ' + r.unit) if r.unit else ''}"
            for r in breached.itertuples()
        )

    text = (
        alerts.groupby([DISTRICT_COLUMN, "pillar"])
        .apply(_describe, include_groups=False)
        .unstack("pillar")
        .reindex(index=levels.index, columns=pillars)
        .fillna("No data")
    )
    return levels, text


def pillar_summary(
    panel: pd.DataFrame, month: pd.Timestamp, district: str | None = None
) -> pd.DataFrame:
    """Per pillar: headline value, change vs previous month, and district counts by status."""
    months = sorted(panel[MONTH_COLUMN].dropna().unique())
    position = months.index(month) if month in months else len(months) - 1
    previous = months[position - 1] if position > 0 else None
    scope = scope_frame(panel, district)
    alerts = district_alerts(panel, month)
    if district is not None and not alerts.empty:
        alerts = alerts.loc[alerts[DISTRICT_COLUMN].eq(clean_district_name(district))]
    rows = []
    for pillar, settings in COMMAND_CENTER_PILLARS.items():
        field = settings["headline"]
        current = city_value(scope.loc[scope[MONTH_COLUMN].eq(month)], field)
        before = city_value(scope.loc[scope[MONTH_COLUMN].eq(previous)], field) if previous is not None else np.nan
        pillar_alerts = alerts.loc[alerts["pillar"].eq(pillar)] if not alerts.empty else alerts
        worst_by_district = (
            pillar_alerts.groupby(DISTRICT_COLUMN)["level"].max() if not pillar_alerts.empty else pd.Series(dtype=float)
        )
        worst = worst_by_district.max() if not worst_by_district.empty else np.nan
        rows.append(
            {
                "pillar": pillar,
                "field": field,
                "label": COMMAND_CENTER_THRESHOLDS.get(field, {}).get("label", field),
                "unit": COMMAND_CENTER_THRESHOLDS.get(field, {}).get("unit", ""),
                "value": current,
                "change": current - before if pd.notna(before) else np.nan,
                "critical_districts": int((worst_by_district == 2).sum()),
                "watch_districts": int((worst_by_district == 1).sum()),
                "status": STATUS_LEVELS.get(worst, "No data") if pd.notna(worst) else "No data",
            }
        )
    return pd.DataFrame(rows)


def event_log(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per month x event with the number of districts flagged and the pillar."""
    columns = [MONTH_COLUMN, "event", "event_label", "pillar", "districts", "district_list"]
    if EVENT_COLUMN not in panel:
        return pd.DataFrame(columns=columns)
    flags = panel[[MONTH_COLUMN, DISTRICT_COLUMN, EVENT_COLUMN]].copy()
    flags[EVENT_COLUMN] = flags[EVENT_COLUMN].astype("string").str.strip()
    flags = flags.dropna(subset=[EVENT_COLUMN])
    flags = flags.loc[flags[EVENT_COLUMN].ne("")]
    if flags.empty:
        return pd.DataFrame(columns=columns)
    flags["event"] = flags[EVENT_COLUMN].str.split("|")
    flags = flags.explode("event")
    flags["event"] = flags["event"].str.strip()
    flags = flags.loc[flags["event"].ne("")]
    log = (
        flags.groupby([MONTH_COLUMN, "event"])[DISTRICT_COLUMN]
        .agg(districts="nunique", district_list=lambda s: ", ".join(sorted(set(s))))
        .reset_index()
    )
    log["event_label"] = log["event"].str.replace("_", " ").str.capitalize()
    log["pillar"] = log["event"].map(COMMAND_CENTER_EVENT_PILLARS).fillna("Other")
    return log[columns].sort_values([MONTH_COLUMN, "pillar", "event"]).reset_index(drop=True)
