"""Layer 5 crisis and stress-testing calculations.

The domino stress test stacks the six case stages on a baseline month. Every increment is derived
from the panel up to that month (no look-ahead); the only inputs not taken from the panel are the
values in STRESS_ASSUMPTIONS, which the view labels as assumptions.

No Streamlit or Plotly dependency, so this module can be unit-tested on its own.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from dashboard.command_center import (
    EVENT_COLUMN,
    add_derived_fields,
    district_alerts,
    event_log,
)
from dashboard.config import (
    COMMAND_CENTER_THRESHOLDS,
    DISTRICT_COLUMN,
    DISTRICT_KEY_COLUMN,
    FUNDING_RULES,
    MONTH_COLUMN,
    STRESS_ASSUMPTIONS,
    STRESS_STAGES,
    STRESS_TRIGGERS,
)
from dashboard.district_utils import clean_district_key

OPERATORS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}
BASELINE_STAGE = "Baseline"
CITY_WIDE = "City-wide"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _upto(panel: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    return panel.loc[panel[MONTH_COLUMN] <= month]


def _at(panel: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    return panel.loc[panel[MONTH_COLUMN].eq(month)].set_index(DISTRICT_COLUMN)


def _numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    return pd.to_numeric(frame[field], errors="coerce")


def _events_in(values: pd.Series, patterns: Sequence[str]) -> pd.Series:
    text = values.astype("string").fillna("").str.lower()
    return text.apply(lambda flags: any(pattern.lower() in flags for pattern in patterns))


def default_assumptions() -> dict[str, float | None]:
    return {key: spec["default"] for key, spec in STRESS_ASSUMPTIONS.items()}


def validate_assumptions(assumptions: Mapping[str, object]) -> dict[str, float | None]:
    values = default_assumptions()
    values.update({key: value for key, value in assumptions.items() if key in values})
    for key, spec in STRESS_ASSUMPTIONS.items():
        value = values[key]
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value) or not spec["min"] <= value <= spec["max"]:
            raise ValueError(f"{key} must be between {spec['min']} and {spec['max']}; got {value}.")
        values[key] = value
    return values


def stage_labels() -> list[str]:
    return [BASELINE_STAGE] + [f"Case {s['case']}: {s['label']}" for s in STRESS_STAGES]


# ---------------------------------------------------------------------------
# Quantities derived from the panel
# ---------------------------------------------------------------------------
def primary_care_share(panel: pd.DataFrame) -> float:
    """Share of total visit demand that reaches primary care (0.70 in every row of the current panel)."""
    total = _numeric(panel, "total_visits_demand")
    share = (_numeric(panel, "pc_visits_demand") / total.replace(0, np.nan)).dropna()
    return float(share.median()) if not share.empty else 0.7


def district_growth_rates(panel: pd.DataFrame, month: pd.Timestamp, field: str = "pc_visits_demand") -> pd.Series:
    """Compound monthly growth of `field` per district, from the first month to `month`."""
    history = _upto(panel, month).sort_values(MONTH_COLUMN)
    rates = {}
    for district, group in history.groupby(DISTRICT_COLUMN):
        values = _numeric(group, field).dropna()
        periods = len(values) - 1
        if periods < 1 or values.iloc[0] <= 0:
            rates[district] = 0.0
        else:
            rates[district] = float((values.iloc[-1] / values.iloc[0]) ** (1 / periods) - 1)
    return pd.Series(rates, dtype=float)


def event_districts(panel: pd.DataFrame, month: pd.Timestamp, events: Sequence[str]) -> set[str]:
    """Districts flagged by any of `events` at or before `month`."""
    if not events or EVENT_COLUMN not in panel:
        return set()
    history = _upto(panel, month)
    flagged = history.loc[_events_in(history[EVENT_COLUMN], events), DISTRICT_COLUMN]
    return set(flagged)


def respiratory_multiplier(panel: pd.DataFrame, month: pd.Timestamp, event: str, lookback: int = 6) -> float:
    """Median, over districts flagged by `event`, of event-month respiratory visits divided by the
    mean of the `lookback` months before. Returns 1.0 if the event has not happened by `month`."""
    if EVENT_COLUMN not in panel:
        return 1.0
    history = _upto(panel, month).sort_values(MONTH_COLUMN)
    flagged = history.loc[_events_in(history[EVENT_COLUMN], [event])]
    ratios = []
    for _, row in flagged.iterrows():
        before = history.loc[
            history[DISTRICT_COLUMN].eq(row[DISTRICT_COLUMN]) & (history[MONTH_COLUMN] < row[MONTH_COLUMN])
        ].tail(lookback)
        baseline = _numeric(before, "p3_respiratory_visits_total").mean()
        if pd.notna(baseline) and baseline > 0:
            ratios.append(float(row["p3_respiratory_visits_total"]) / baseline)
    return float(np.median(ratios)) if ratios else 1.0


def worst_observed(panel: pd.DataFrame, month: pd.Timestamp, fields: Sequence[str]) -> pd.DataFrame:
    """Each district's worst value of each field up to `month` (direction from the alert thresholds)."""
    history = _upto(panel, month)
    columns = {}
    for field in fields:
        if field not in history:
            continue
        values = pd.to_numeric(history[field], errors="coerce").groupby(history[DISTRICT_COLUMN])
        direction = COMMAND_CENTER_THRESHOLDS.get(field, {}).get("direction", "above")
        columns[field] = values.min() if direction == "below" else values.max()
    return pd.DataFrame(columns)


# ---------------------------------------------------------------------------
# The domino stress test
# ---------------------------------------------------------------------------
def run_domino(
    panel: pd.DataFrame,
    month: pd.Timestamp,
    assumptions: Mapping[str, object] | None = None,
    through_stage: int | None = None,
) -> dict[str, object]:
    """Stack the stages in STRESS_STAGES on `month`.

    Returns a dict with:
      path        district x stage: demand, capacity, pc_only_load, cpr (one row per stage step)
      final       district table at the last applied stage, with baseline columns for comparison
      stressed    the month's panel rows with operational fields set to their worst observed values
      baseline    the month's panel rows, unchanged
      parameters  values derived from the panel and assumptions actually used
    """
    panel = add_derived_fields(panel)
    a = validate_assumptions(assumptions or {})
    base = _at(panel, month)
    if base.empty:
        raise ValueError(f"No panel rows for {month:%B %Y}.")
    required = {"pc_visits_demand", "pc_capacity_visits", "pc_effective_capacity", "p3_respiratory_visits_total"}
    missing = required.difference(base.columns)
    if missing:
        raise ValueError(f"Stress test needs fields missing from the panel: {sorted(missing)}")

    stages = STRESS_STAGES if through_stage is None else STRESS_STAGES[: max(0, through_stage)]
    share = primary_care_share(panel)
    growth = district_growth_rates(panel, month)
    pc_capacity = _numeric(base, "pc_capacity_visits")
    hospital_share = (_numeric(base, "pc_effective_capacity") - pc_capacity).clip(lower=0)

    demand = _numeric(base, "pc_visits_demand").astype(float)
    capacity = pc_capacity.astype(float).copy()
    stressed = base.copy()
    parameters: dict[str, object] = {"primary_care_share": share, "baseline_month": month}

    def _snapshot(label: str, case: int | None) -> pd.DataFrame:
        effective = capacity + hospital_share
        return pd.DataFrame({
            DISTRICT_COLUMN: demand.index, "stage": label, "case": case,
            "pc_demand": demand.values, "pc_capacity": capacity.values,
            "pc_only_load": (demand / capacity.replace(0, np.nan)).values,
            "cpr": (demand / effective.replace(0, np.nan)).values,
        })

    path = [_snapshot(BASELINE_STAGE, None)]
    for stage in stages:
        label = f"Case {stage['case']}: {stage['label']}"
        # Growth at each district's own observed rate.
        if stage.get("growth_field"):
            months_ahead = float(a["growth_horizon_months"] or 0)
            factor = (1 + growth.reindex(demand.index).fillna(0)) ** months_ahead
            demand = demand * factor
            parameters["growth_horizon_months"] = months_ahead
            parameters["growth_rates"] = growth
        # Worst observed visits minus today's, passed into primary care.
        for field in stage.get("visit_increments", []):
            worst = worst_observed(panel, month, [field])
            if field in worst:
                increment = (worst[field].reindex(demand.index) - _numeric(base, field)).clip(lower=0).fillna(0)
                demand = demand + share * increment
                parameters[f"increment_{field}"] = increment
        # Respiratory multiplier from the precedent event (or the user's value).
        if stage.get("respiratory_event"):
            derived = respiratory_multiplier(panel, month, stage["respiratory_event"])
            multiplier = float(a["respiratory_multiplier"]) if a["respiratory_multiplier"] is not None else derived
            parameters["respiratory_multiplier_derived"] = derived
            parameters["respiratory_multiplier"] = multiplier
            demand = demand + share * _numeric(base, "p3_respiratory_visits_total") * (multiplier - 1)
        # Assumed capacity loss in districts the precedent event hit.
        if stage.get("capacity_loss"):
            loss = float(a[stage["capacity_loss"]] or 0) / 100
            hit = event_districts(panel, month, stage.get("loss_events", []))
            mask = demand.index.isin(list(hit))
            capacity = capacity.where(~mask, capacity * (1 - loss))
            parameters[f"{stage['key']}_loss_districts"] = sorted(hit)
            parameters[stage["capacity_loss"]] = loss * 100
        # Operational fields at their worst observed values.
        fields = stage.get("operational_fields", [])
        if fields:
            worst = worst_observed(panel, month, fields)
            for field in worst.columns:
                stressed[field] = worst[field].reindex(stressed.index)
        path.append(_snapshot(label, stage["case"]))

    stressed["pc_visits_demand"] = demand
    stressed["pc_capacity_visits"] = capacity
    stressed["pc_effective_capacity"] = capacity + hospital_share
    stressed["pc_only_load"] = demand / capacity.replace(0, np.nan)
    stressed["cpr"] = demand / stressed["pc_effective_capacity"].replace(0, np.nan)

    path_frame = pd.concat(path, ignore_index=True)
    first, last = path[0].set_index(DISTRICT_COLUMN), path[-1].set_index(DISTRICT_COLUMN)
    final = pd.DataFrame({
        "pc_demand_base": first["pc_demand"], "pc_demand_stressed": last["pc_demand"],
        "pc_capacity_base": first["pc_capacity"], "pc_capacity_stressed": last["pc_capacity"],
        "pc_only_load_base": first["pc_only_load"], "pc_only_load_stressed": last["pc_only_load"],
        "cpr_base": first["cpr"], "cpr_stressed": last["cpr"],
    })
    final["demand_uplift_pct"] = 100 * (final["pc_demand_stressed"] / final["pc_demand_base"] - 1)
    if DISTRICT_KEY_COLUMN in base:
        final[DISTRICT_KEY_COLUMN] = base[DISTRICT_KEY_COLUMN].map(clean_district_key)
    final = final.reset_index().rename(columns={"index": DISTRICT_COLUMN})
    return {
        "path": path_frame,
        "final": final.sort_values("pc_only_load_stressed", ascending=False).reset_index(drop=True),
        "stressed": stressed.reset_index(),
        "baseline": base.reset_index(),
        "parameters": parameters,
    }


def alert_comparison(result: Mapping[str, object], month: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Layer 4 alerts for the baseline month and for the stressed month (same thresholds)."""
    return district_alerts(result["baseline"], month), district_alerts(result["stressed"], month)


# ---------------------------------------------------------------------------
# Trigger rules
# ---------------------------------------------------------------------------
def _pivot(history: pd.DataFrame, field: str) -> pd.DataFrame:
    """months x districts table of a field."""
    return history.pivot_table(index=MONTH_COLUMN, columns=DISTRICT_COLUMN, values=field, aggfunc="first").sort_index()


def _condition(cond: Mapping[str, object], history: pd.DataFrame, month: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """Evaluate one condition for every district at `month`. Returns (fired, observed-value text)."""
    districts = sorted(history.loc[history[MONTH_COLUMN].eq(month), DISTRICT_COLUMN].unique())
    empty = pd.Series(False, index=districts), pd.Series("", index=districts)
    kind = cond["kind"]

    if kind == "all":
        parts = [_condition(c, history, month) for c in cond["conditions"]]
        fired = pd.concat([p[0] for p in parts], axis=1).all(axis=1)
        text = pd.concat([p[1] for p in parts], axis=1).apply(lambda r: "; ".join(v for v in r if v), axis=1)
        return fired.reindex(districts, fill_value=False), text.reindex(districts, fill_value="")

    if kind == "event":
        if EVENT_COLUMN not in history:
            return empty
        rows = history.loc[history[MONTH_COLUMN].eq(month)].set_index(DISTRICT_COLUMN)[EVENT_COLUMN]
        fired = _events_in(rows, cond["contains"]).reindex(districts, fill_value=False)
        text = rows.astype("string").fillna("").where(fired, "").reindex(districts, fill_value="")
        return fired, text.map(lambda v: f"event: {v}" if v else "")

    if kind == "median_cross":
        months = int(cond["months"])
        window = sorted(history[MONTH_COLUMN].unique())
        if len(window) < months + 1:
            return empty
        recent = window[-(months + 1):]
        fired = pd.Series(False, index=districts)
        text = pd.Series("", index=districts)
        for field in cond["fields"]:
            if field not in history:
                continue
            table = _pivot(history, field).loc[recent]
            above = table.ge(table.median(axis=1), axis=0)
            stayed = above.iloc[1:].nunique() == 1
            crossed = stayed & above.iloc[1:].iloc[0].ne(above.iloc[0])
            crossed = crossed.reindex(districts, fill_value=False)
            fired |= crossed
            side = above.iloc[-1].reindex(districts).map({True: "above", False: "below"})
            text = text.where(~crossed, text + f"{field} now {side} city median; ")
        return fired, text.str.rstrip("; ")

    field = cond["field"]
    if field not in history:
        return empty
    op = OPERATORS[cond["op"]]
    table = _pivot(history, field)
    if month not in table.index:
        return empty
    current = table.loc[month].reindex(districts)

    if kind == "threshold":
        value = current
    elif kind == "growth":
        recent = table.loc[:month].tail(int(cond["months"]) + 1)
        value = (recent.pct_change().iloc[1:].mean() * 100).reindex(districts)
    elif kind == "yoy_ratio":
        prior = month - pd.DateOffset(years=1)
        if prior not in table.index:
            return empty
        value = current / table.loc[prior].reindex(districts).clip(lower=1)
    elif kind == "own_median_ratio":
        median = table.loc[:month].median()
        value = current / median.reindex(districts).where(lambda m: m > 0)
    elif kind == "mom_change":
        earlier = table.loc[:month]
        if len(earlier) < 2:
            return empty
        value = (current / earlier.iloc[-2].reindex(districts).replace(0, np.nan) - 1) * 100
    else:
        raise ValueError(f"Unknown trigger condition kind: {kind!r}")

    if "min_value" in cond:  # ignore tiny volumes, where a few cases look like a large % change
        value = value.where(current >= float(cond["min_value"]))
    fired = value.apply(lambda v: bool(pd.notna(v) and op(float(v), float(cond["value"]))))
    suffix = {"growth": "%/month", "mom_change": "% m/m", "yoy_ratio": "x last year", "own_median_ratio": "x own median"}.get(kind, "")
    text = value.map(lambda v: f"{field} {v:,.2f}{(' ' + suffix) if suffix else ''}" if pd.notna(v) else "")
    return fired, text.where(fired, "")


def evaluate_triggers(panel: pd.DataFrame, month: pd.Timestamp, triggers: Sequence[Mapping] = STRESS_TRIGGERS) -> pd.DataFrame:
    """Trigger rules that fire at `month`, one row per trigger x district (or one city-wide row)."""
    history = _upto(add_derived_fields(panel), month)
    columns = ["id", "case", "trigger", DISTRICT_COLUMN, "observed", "action"]
    rows = []
    for rule in triggers:
        parts = [_condition(cond, history, month) for cond in rule["conditions"]]
        if not parts:
            continue
        fired_frame = pd.concat([p[0] for p in parts], axis=1)
        fired = fired_frame.all(axis=1) if rule.get("combine") == "all" else fired_frame.any(axis=1)
        text = pd.concat([p[1] for p in parts], axis=1).apply(lambda r: "; ".join(v for v in r if v), axis=1)
        hits = fired[fired].index.tolist()
        if rule.get("scope") == "city" and hits:
            rows.append([rule["id"], rule["case"], rule["label"], CITY_WIDE, text[hits[0]], rule["action"]])
            continue
        for district in hits:
            rows.append([rule["id"], rule["case"], rule["label"], district, text[district], rule["action"]])
    return pd.DataFrame(rows, columns=columns)


def trigger_history(panel: pd.DataFrame, triggers: Sequence[Mapping] = STRESS_TRIGGERS) -> pd.DataFrame:
    """Number of fired triggers per month and case, for every month in the panel."""
    panel = add_derived_fields(panel)
    frames = []
    for month in sorted(panel[MONTH_COLUMN].dropna().unique()):
        fired = evaluate_triggers(panel, pd.Timestamp(month), triggers)
        if not fired.empty:
            counts = fired.groupby("case").size().rename("fired").reset_index()
            counts[MONTH_COLUMN] = pd.Timestamp(month)
            frames.append(counts)
    if not frames:
        return pd.DataFrame(columns=[MONTH_COLUMN, "case", "fired"])
    return pd.concat(frames, ignore_index=True)[[MONTH_COLUMN, "case", "fired"]]


# ---------------------------------------------------------------------------
# Case 6 funding rule
# ---------------------------------------------------------------------------
def funding_quadrants(panel: pd.DataFrame, month: pd.Timestamp, rules: Mapping[str, float] = FUNDING_RULES) -> pd.DataFrame:
    """Classify each district by need, attractiveness, capacity gap and population (six-month means)."""
    panel = add_derived_fields(panel)
    history = _upto(panel, month)
    months = sorted(history[MONTH_COLUMN].unique())[-int(rules["window_months"]):]
    window = history.loc[history[MONTH_COLUMN].isin(months)]
    needed = {"need_score", "attractiveness_index_v2"}
    if not needed.issubset(window.columns):
        raise ValueError(f"Funding rule needs {sorted(needed)} in the panel.")
    table = window.groupby(DISTRICT_COLUMN).agg(
        need=("need_score", "mean"),
        attractiveness=("attractiveness_index_v2", "mean"),
        residents=("residents_registered", "last"),
        pc_only_load=("pc_only_load", "last"),
    )
    need_cut, attr_cut = table["need"].median(), table["attractiveness"].median()
    band = float(rules["attractiveness_band"])

    def _classify(row: pd.Series) -> tuple[str, str]:
        if row["need"] < need_cut:
            return "Lower need", "No priority intervention"
        gap = row["pc_only_load"] >= float(rules["capacity_gap_load"])
        if row["attractiveness"] > attr_cut + band:
            return "High need, high attractiveness", "Private"
        if row["attractiveness"] >= attr_cut - band:
            return ("High need, median attractiveness",
                    "PPP (land subsidy / tax holiday)" if gap else "PPP-ready; monitor")
        small = row["residents"] < float(rules["small_population_residents"])
        low_load = row["pc_only_load"] < float(rules["low_load_for_prevention"])
        if small and low_load:
            return "High need, low attractiveness", "Public + preventive access package"
        return "High need, low attractiveness", "Public"

    classified = table.apply(_classify, axis=1, result_type="expand")
    table["quadrant"], table["funding"] = classified[0], classified[1]
    table.attrs.update(need_cut=need_cut, attractiveness_cut=attr_cut, months=months)
    return table.reset_index()


# ---------------------------------------------------------------------------
# Case evidence and precedents
# ---------------------------------------------------------------------------
def case_evidence(panel: pd.DataFrame, month: pd.Timestamp, district_keys: Sequence[object], fields: Sequence[str]) -> pd.DataFrame:
    """For each focus district and field: value at `month`, worst observed (and when), district median."""
    panel = add_derived_fields(panel)
    history = _upto(panel, month)
    keys = {clean_district_key(k) for k in district_keys}
    if DISTRICT_KEY_COLUMN in history:
        history = history.loc[history[DISTRICT_KEY_COLUMN].map(clean_district_key).isin(keys)]
    rows = []
    for district, group in history.groupby(DISTRICT_COLUMN, sort=False):
        group = group.sort_values(MONTH_COLUMN)
        for field in fields:
            if field not in group:
                continue
            values = pd.to_numeric(group[field], errors="coerce")
            if values.dropna().empty:
                continue
            direction = COMMAND_CENTER_THRESHOLDS.get(field, {}).get("direction", "above")
            worst_index = values.idxmin() if direction == "below" else values.idxmax()
            current = values.loc[group[MONTH_COLUMN].eq(month)]
            rows.append({
                DISTRICT_COLUMN: district, "field": field,
                "current": float(current.iloc[0]) if not current.empty else np.nan,
                "worst": float(values.loc[worst_index]), "worst_month": group.loc[worst_index, MONTH_COLUMN],
                "median": float(values.median()),
            })
    return pd.DataFrame(rows)


def stage_precedents(panel: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """The panel events each stage is modelled on, up to `month`."""
    log = event_log(_upto(panel, month))
    rows = []
    for stage in STRESS_STAGES:
        for event in stage.get("precedent_events", []):
            hits = log.loc[log["event"].eq(event)]
            for r in hits.itertuples():
                rows.append({"case": stage["case"], "stage": stage["label"], "event": r.event_label,
                             MONTH_COLUMN: getattr(r, MONTH_COLUMN), "districts": r.districts,
                             "district_list": r.district_list})
    return pd.DataFrame(rows, columns=["case", "stage", "event", MONTH_COLUMN, "districts", "district_list"])
