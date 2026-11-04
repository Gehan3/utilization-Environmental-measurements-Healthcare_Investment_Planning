"""City command center: cross-pillar status, alerts, events and pillar drill-downs."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.ui import plotly_chart
from dashboard.command_center import (
    STATUS_LEVELS,
    district_alerts,
    event_log,
    is_better,
    monthly_series,
    pillar_grid,
    pillar_summary,
    status_level,
)
from dashboard.config import (
    COMMAND_CENTER_PILLARS,
    COMMAND_CENTER_STATUS_COLORS,
    COMMAND_CENTER_THRESHOLDS,
    DISTRICT_COLUMN,
    MONTH_COLUMN,
)
from dashboard.data_loader import district_options, with_district_aliases

# Matches .streamlit/config.toml
PRIMARY = "#0D9488"
TEXT = "#1E293B"
MUTED = "#64748B"
SURFACE = "#F0FDFA"
BORDER = "#CCFBF1"
PILLAR_COLORS = {name: cfg["color"] for name, cfg in COMMAND_CENTER_PILLARS.items()}
PILLAR_COLORS["Other"] = "#94A3B8"
STATUS_TEXT = {"OK": "Within limits", "Watch": "Watch", "Critical": "Critical", "No data": "No data"}
CITY_WIDE = "City-wide"

_CSS = f"""
<style>
.cc-card {{
  background: #FFFFFF;
  border: 1px solid {BORDER};
  border-top: 4px solid var(--cc-status);
  border-radius: 10px;
  padding: 0.85rem 1rem 0.8rem;
  height: 100%;
}}
.cc-pillar {{ color: {MUTED}; font-size: 0.85rem; font-weight: 600; display: flex; align-items: center; gap: 0.45rem; }}
.cc-dot {{ width: 0.6rem; height: 0.6rem; border-radius: 50%; background: var(--cc-accent); display: inline-block; }}
.cc-value {{ color: {TEXT}; font-size: 1.9rem; font-weight: 700; line-height: 1.15; margin-top: 0.35rem; font-variant-numeric: tabular-nums; }}
.cc-unit {{ font-size: 0.95rem; font-weight: 500; color: {MUTED}; margin-left: 0.2rem; }}
.cc-metric {{ color: {TEXT}; font-size: 0.85rem; }}
.cc-delta {{ font-size: 0.8rem; margin-top: 0.25rem; color: {MUTED}; }}
.cc-delta.good {{ color: #0F766E; }}
.cc-delta.bad {{ color: #B91C1C; }}
.cc-status {{
  display: inline-block; margin-top: 0.55rem; padding: 0.15rem 0.55rem; border-radius: 999px;
  font-size: 0.78rem; font-weight: 600; color: var(--cc-status); background: {SURFACE};
}}
</style>
"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _decimals(value: float) -> int:
    magnitude = abs(value)
    return 3 if magnitude < 1 else 2 if magnitude < 10 else 1 if magnitude < 1000 else 0


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{_decimals(float(value))}f}"


def _label(field: str) -> str:
    rule = COMMAND_CENTER_THRESHOLDS.get(field)
    if rule:
        return f"{rule['label']} ({rule['unit']})" if rule["unit"] else rule["label"]
    return field.replace("_", " ")


def _district_labels(panel: pd.DataFrame) -> dict[str, str]:
    labeled = with_district_aliases(panel[[DISTRICT_COLUMN]].drop_duplicates())
    return labeled.set_index(DISTRICT_COLUMN)["district_label"].to_dict()


def _style(figure: go.Figure, height: int = 340, title: str | None = None) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        title={"text": title, "font": {"size": 15, "color": TEXT}} if title else None,
        font={"color": TEXT},
        margin={"l": 10, "r": 10, "t": 48 if title else 16, "b": 10},
        legend={"orientation": "h", "y": -0.18, "x": 0},
        hovermode="x unified",
        colorway=[PRIMARY, "#F59E0B", "#7C3AED", "#2563EB", "#DB2777", "#65A30D"],
    )
    return figure


def _add_thresholds(figure: go.Figure, field: str, horizontal: bool = True, secondary: bool = False) -> None:
    rule = COMMAND_CENTER_THRESHOLDS.get(field)
    if not rule:
        return
    for level in ("watch", "critical"):
        color = COMMAND_CENTER_STATUS_COLORS["Watch" if level == "watch" else "Critical"]
        kwargs = {"line_dash": "dash", "line_color": color, "line_width": 1.2, "opacity": 0.8}
        if horizontal:
            if secondary:
                figure.add_shape(
                    type="line", xref="paper", x0=0, x1=1, y0=rule[level], y1=rule[level],
                    line={"dash": "dash", "color": color, "width": 1.2}, yref="y2",
                )
            else:
                figure.add_hline(y=rule[level], annotation_text=level.capitalize(),
                                 annotation_font_color=color, annotation_position="top left", **kwargs)
        else:
            figure.add_vline(x=rule[level], **kwargs)


def _mark_month(figure: go.Figure, month: pd.Timestamp) -> None:
    figure.add_vline(x=pd.Timestamp(month).to_pydatetime(), line_dash="dot", line_color=MUTED, line_width=1)


# ---------------------------------------------------------------------------
# Reusable charts
# ---------------------------------------------------------------------------
def _trend(panel: pd.DataFrame, field: str, district: str | None, month: pd.Timestamp,
           title: str | None = None, height: int = 280) -> None:
    if field not in panel:
        st.info(f"`{field}` is not in this panel version.")
        return
    series = monthly_series(panel, [field], district)
    figure = go.Figure(
        go.Scatter(x=series[MONTH_COLUMN], y=series[field], mode="lines+markers",
                   name=_label(field), line={"color": PRIMARY, "width": 2.2}, marker={"size": 5})
    )
    _add_thresholds(figure, field)
    _mark_month(figure, month)
    figure.update_yaxes(title=None)
    figure.update_xaxes(title=None)
    plotly_chart(_style(figure, height, title or _label(field)))


def _district_bar(panel: pd.DataFrame, field: str, month: pd.Timestamp, labels: dict[str, str],
                  title: str | None = None, extra_hover: list[str] | None = None) -> None:
    if field not in panel:
        st.info(f"`{field}` is not in this panel version.")
        return
    rows = panel.loc[panel[MONTH_COLUMN].eq(month)].copy()
    rows["District"] = rows[DISTRICT_COLUMN].map(labels).fillna(rows[DISTRICT_COLUMN])
    rows["Value"] = pd.to_numeric(rows[field], errors="coerce")
    rule = COMMAND_CENTER_THRESHOLDS.get(field)
    if rule:
        rows["Status"] = rows["Value"].map(lambda v: STATUS_LEVELS.get(status_level(v, rule), "No data"))
        color_args = {"color": "Status", "color_discrete_map": COMMAND_CENTER_STATUS_COLORS}
        worst_is_high = rule["direction"] == "above"
    else:
        color_args = {"color_discrete_sequence": [PRIMARY]}
        worst_is_high = True
    # Plotly draws the first category at the bottom, so sort so the worst district ends up on top.
    rows = rows.sort_values("Value", ascending=worst_is_high)
    hover = [column for column in (extra_hover or []) if column in rows]
    figure = px.bar(rows, x="Value", y="District", orientation="h", hover_data=hover,
                    labels={"Value": _label(field)}, **color_args)
    figure.update_yaxes(title=None, categoryorder="array", categoryarray=rows["District"].tolist())
    _add_thresholds(figure, field, horizontal=False)
    _style(figure, max(360, 26 * len(rows) + 90), title or f"{_label(field)} by district")
    figure.update_layout(hovermode="closest")
    plotly_chart(figure)


def _bars_with_line(series: pd.DataFrame, bar_fields: dict[str, str], line_field: str | None,
                    month: pd.Timestamp, title: str, stacked: bool = False, line_name: str | None = None,
                    bar_axis: str = "", line_axis: str = "") -> None:
    figure = go.Figure()
    palette = [PRIMARY, "#F59E0B", "#7C3AED", "#2563EB", "#DB2777", "#65A30D"]
    for index, (field, name) in enumerate(bar_fields.items()):
        if field in series:
            figure.add_trace(go.Bar(x=series[MONTH_COLUMN], y=series[field], name=name,
                                    marker_color=palette[index % len(palette)], opacity=0.85))
    if line_field and line_field in series:
        figure.add_trace(go.Scatter(x=series[MONTH_COLUMN], y=series[line_field], name=line_name or _label(line_field),
                                    yaxis="y2", mode="lines+markers", line={"color": TEXT, "width": 2}))
        figure.update_layout(yaxis2={"title": line_axis, "overlaying": "y", "side": "right", "showgrid": False})
        _add_thresholds(figure, line_field, secondary=True)
    figure.update_layout(barmode="stack" if stacked else "group", yaxis_title=bar_axis)
    _mark_month(figure, month)
    plotly_chart(_style(figure, 360, title))


# ---------------------------------------------------------------------------
# Command-center sections
# ---------------------------------------------------------------------------
def _status_cards(summary: pd.DataFrame, previous_month: pd.Timestamp | None, district: str | None) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    columns = st.columns(len(summary))
    for column, row in zip(columns, summary.itertuples()):
        status_color = COMMAND_CENTER_STATUS_COLORS.get(row.status, COMMAND_CENTER_STATUS_COLORS["No data"])
        accent = PILLAR_COLORS.get(row.pillar, PRIMARY)
        if pd.notna(row.change) and previous_month is not None:
            better = is_better(row.change, row.field)
            tone = "good" if better else "bad" if better is False else ""
            arrow = "▲" if row.change > 0 else "▼" if row.change < 0 else "■"
            delta = f'<div class="cc-delta {tone}">{arrow} {_fmt(abs(row.change))} vs {previous_month:%b %Y}</div>'
        else:
            delta = '<div class="cc-delta">No previous month</div>'
        if district is None:
            if row.critical_districts or row.watch_districts:
                parts = []
                if row.critical_districts:
                    parts.append(f"{row.critical_districts} critical")
                if row.watch_districts:
                    parts.append(f"{row.watch_districts} on watch")
                badge = ", ".join(parts)
            else:
                badge = "All districts within limits"
        else:
            badge = STATUS_TEXT.get(row.status, row.status)
        unit = f'<span class="cc-unit">{html.escape(row.unit)}</span>' if row.unit and row.unit != "ratio" else ""
        column.markdown(
            f"""
<div class="cc-card" style="--cc-status:{status_color};--cc-accent:{accent}">
  <div class="cc-pillar"><span class="cc-dot"></span>{html.escape(row.pillar)}</div>
  <div class="cc-value">{_fmt(row.value)}{unit}</div>
  <div class="cc-metric">{html.escape(row.label)}</div>
  {delta}
  <div class="cc-status">{html.escape(badge)}</div>
</div>""",
            unsafe_allow_html=True,
        )


def _status_grid(alerts: pd.DataFrame, panel: pd.DataFrame, labels: dict[str, str]) -> None:
    levels, hover = pillar_grid(alerts)
    if levels.empty:
        st.info("No threshold indicators are available for this month.")
        return
    order = [name for name in district_options(panel) if name in levels.index]
    levels, hover = levels.reindex(order), hover.reindex(order)
    text = levels.apply(lambda col: col.map(lambda v: STATUS_LEVELS.get(v, "")))
    figure = go.Figure(
        go.Heatmap(
            z=levels.values, x=levels.columns.tolist(), y=[labels.get(n, n) for n in levels.index],
            zmin=0, zmax=2, showscale=False, xgap=3, ygap=3,
            colorscale=[[0, "#CCFBF1"], [0.25, "#CCFBF1"], [0.25, "#FDE68A"], [0.75, "#FDE68A"],
                        [0.75, "#FCA5A5"], [1, "#FCA5A5"]],
            text=text.values, texttemplate="%{text}", textfont={"color": TEXT, "size": 12},
            customdata=hover.values,
            hovertemplate="<b>%{y}</b> · %{x}<br>%{customdata}<extra></extra>",
        )
    )
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(side="top")
    _style(figure, 30 * len(levels) + 90)
    figure.update_layout(hovermode="closest", margin={"l": 10, "r": 10, "t": 30, "b": 10})
    plotly_chart(figure)


def _alert_table(alerts: pd.DataFrame, labels: dict[str, str], district: str | None) -> None:
    active = alerts.loc[alerts["level"] >= 1].copy() if not alerts.empty else alerts
    if district is not None and not active.empty:
        active = active.loc[active[DISTRICT_COLUMN].eq(district)]
    if active.empty:
        st.success("No indicator is past its watch limit this month.")
        return
    active["District"] = active[DISTRICT_COLUMN].map(labels).fillna(active[DISTRICT_COLUMN])
    active["Limit"] = np.where(active["level"] == 2, active["critical"], active["watch"])
    active["Limit"] = [
        f"{'>' if d == 'above' else '<'} {_fmt(v)} {u}".strip()
        for d, v, u in zip(active["direction"], active["Limit"], active["unit"])
    ]
    active["Value"] = [f"{_fmt(v)} {u}".strip() for v, u in zip(active["value"], active["unit"])]
    table = active.sort_values(["level", "pillar", "District"], ascending=[False, True, True])[
        ["status", "District", "pillar", "indicator", "Value", "Limit"]
    ].rename(columns={"status": "Status", "pillar": "Pillar", "indicator": "Indicator"})

    def _color(value: str) -> str:
        color = COMMAND_CENTER_STATUS_COLORS.get(value)
        return f"color: {color}; font-weight: 600" if color else ""

    critical_count = int((table["Status"] == "Critical").sum())
    st.caption(f"{len(table)} active alerts, {critical_count} critical")
    st.dataframe(table.style.map(_color, subset=["Status"]), width="stretch", hide_index=True, height=420)


def _event_timeline(panel: pd.DataFrame, month: pd.Timestamp) -> None:
    log = event_log(panel)
    if log.empty:
        st.info("The panel has no event flags.")
        return
    pillar_order = [*COMMAND_CENTER_PILLARS, "Other"]
    log["pillar"] = pd.Categorical(log["pillar"], pillar_order, ordered=True)
    log = log.sort_values(["pillar", "event_label"])
    figure = px.scatter(
        log, x=MONTH_COLUMN, y="event_label", size="districts", color="pillar",
        color_discrete_map=PILLAR_COLORS, size_max=18,
        category_orders={"pillar": pillar_order, "event_label": log["event_label"].drop_duplicates().tolist()},
        hover_data={"districts": True, "pillar": True, MONTH_COLUMN: "|%b %Y", "event_label": False},
        labels={MONTH_COLUMN: "", "event_label": "", "districts": "Districts affected", "pillar": "Pillar"},
    )
    _mark_month(figure, month)
    _style(figure, 26 * log["event_label"].nunique() + 120)
    figure.update_layout(hovermode="closest")
    plotly_chart(figure)
    selected = log.loc[log[MONTH_COLUMN].eq(month)]
    if not selected.empty:
        items = ", ".join(f"{r.event_label} ({r.districts} districts)" for r in selected.itertuples())
        st.caption(f"Flagged in {month:%B %Y}: {items}")


# ---------------------------------------------------------------------------
# Pillar tabs
# ---------------------------------------------------------------------------
def _water_tab(panel: pd.DataFrame, district: str | None, month: pd.Timestamp, labels: dict[str, str]) -> None:
    series = monthly_series(panel, ["pw_water_supplied_m3", "pw_metered_consumption_m3", "pw_nrw_pct"], district)
    left, right = st.columns([3, 2])
    with left:
        _bars_with_line(series, {"pw_water_supplied_m3": "Water supplied", "pw_metered_consumption_m3": "Metered consumption"},
                        "pw_nrw_pct", month, "Supply vs metered consumption", line_name="Non-revenue water (%)",
                        bar_axis="m³", line_axis="NRW %")
    with right:
        _district_bar(panel, "pw_leak_incidents", month, labels, f"Leak incidents, {month:%b %Y}",
                      extra_hover=["pw_nrw_pct", "pw_night_flow_ratio"])
    st.markdown("**Water quality and continuity**")
    fields = ["pw_chlorine_residual_mg_l", "pw_turbidity_ntu", "pw_ecoli_positive_pct",
              "pw_water_age_hours", "pw_supply_interruption_hours", "pw_waterborne_visits_total"]
    for start in (0, 3):
        for column, field in zip(st.columns(3), fields[start:start + 3]):
            with column:
                _trend(panel, field, district, month)
    _district_bar(panel, "pw_nrw_pct", month, labels, f"Non-revenue water by district, {month:%b %Y}",
                  extra_hover=["pw_leak_incidents", "pw_night_flow_ratio"])


def _waste_tab(panel: pd.DataFrame, district: str | None, month: pd.Timestamp, labels: dict[str, str]) -> None:
    series = monthly_series(panel, ["p2_waste_generated_tons", "p2_uncollected_tons", "p2_collection_efficiency_pct"], district)
    if {"p2_waste_generated_tons", "p2_uncollected_tons"}.issubset(series.columns):
        series["collected"] = series["p2_waste_generated_tons"] - series["p2_uncollected_tons"]
    left, right = st.columns([3, 2])
    with left:
        _bars_with_line(series, {"collected": "Collected", "p2_uncollected_tons": "Uncollected"},
                        "p2_collection_efficiency_pct", month, "Waste generated and collected", stacked=True,
                        line_name="Collection efficiency (%)", bar_axis="tons", line_axis="%")
    with right:
        _district_bar(panel, "p2_collection_efficiency_pct", month, labels,
                      f"Collection efficiency, {month:%b %Y}", extra_hover=["p2_uncollected_tons"])
    for column, field in zip(st.columns(3), ["p2_bin_hours_ge85_pct", "p2_overflow_incidents", "p2_odor_pest_risk_index"]):
        with column:
            _trend(panel, field, district, month)
    needed = {"p2_waste_generated_tons", "p2_organic_tons", "p2_recyclable_tons"}
    if needed.issubset(panel.columns):
        rows = panel.loc[panel[MONTH_COLUMN].eq(month)].copy()
        rows["District"] = rows[DISTRICT_COLUMN].map(labels).fillna(rows[DISTRICT_COLUMN])
        rows["Other"] = (rows["p2_waste_generated_tons"] - rows["p2_organic_tons"] - rows["p2_recyclable_tons"]).clip(lower=0)
        long = rows.melt(id_vars="District", value_vars=["p2_organic_tons", "p2_recyclable_tons", "Other"],
                         var_name="Stream", value_name="Tons")
        long["Stream"] = long["Stream"].map({"p2_organic_tons": "Organic", "p2_recyclable_tons": "Recyclable", "Other": "Other"})
        figure = px.bar(long, x="Tons", y="District", color="Stream", orientation="h",
                        color_discrete_map={"Organic": "#65A30D", "Recyclable": PRIMARY, "Other": "#94A3B8"})
        figure.update_yaxes(title=None, categoryorder="total ascending")
        _style(figure, 26 * len(rows) + 110, f"Waste composition by district, {month:%b %Y}")
        figure.update_layout(hovermode="closest")
        plotly_chart(figure)


def _air_tab(panel: pd.DataFrame, district: str | None, month: pd.Timestamp, labels: dict[str, str]) -> None:
    left, right = st.columns(2)
    with left:
        _trend(panel, "p3_avg_aqi", district, month, "Average AQI", height=340)
    with right:
        fields = [f for f in ["p3_pm25_ugm3", "p3_pm10_ugm3", "p3_no2_ugm3", "p3_o3_ugm3"] if f in panel]
        series = monthly_series(panel, fields, district)
        long = series.melt(id_vars=MONTH_COLUMN, var_name="Pollutant", value_name="µg/m³")
        long["Pollutant"] = long["Pollutant"].map(
            {"p3_pm25_ugm3": "PM2.5", "p3_pm10_ugm3": "PM10", "p3_no2_ugm3": "NO₂", "p3_o3_ugm3": "O₃"})
        figure = px.line(long, x=MONTH_COLUMN, y="µg/m³", color="Pollutant", markers=True, labels={MONTH_COLUMN: ""})
        _mark_month(figure, month)
        plotly_chart(_style(figure, 340, "Pollutant concentrations"))
    left, right = st.columns(2)
    with left:
        _district_bar(panel, "p3_avg_aqi", month, labels, f"Average AQI by district, {month:%b %Y}",
                      extra_hover=["p3_max_aqi", "p3_days_aqi_gt150"])
    with right:
        series = monthly_series(panel, ["heat_days_ge38c", "avg_temp_c"], district)
        _bars_with_line(series, {"heat_days_ge38c": "Days ≥38 °C"}, "avg_temp_c", month, "Heat exposure",
                        line_name="Average temperature (°C)", bar_axis="days", line_axis="°C")
    series = monthly_series(panel, ["p3_respiratory_visits_total", "p3_heat_related_visits"], district)
    _bars_with_line(series, {"p3_respiratory_visits_total": "Respiratory", "p3_heat_related_visits": "Heat-related"},
                    None, month, "Air- and heat-related clinical visits", stacked=True, bar_axis="visits")


def _service_tab(panel: pd.DataFrame, district: str | None, month: pd.Timestamp, labels: dict[str, str]) -> None:
    categories = {"p4_water_complaints": "Water", "p4_waste_complaints": "Waste",
                  "p4_pollution_complaints": "Pollution", "p4_infrastructure_complaints": "Infrastructure"}
    series = monthly_series(panel, [*categories, "p4_total_complaints"], district)
    figure = go.Figure()
    palette = [PRIMARY, "#65A30D", "#7C3AED", "#F59E0B"]
    for color, (field, name) in zip(palette, categories.items()):
        if field in series:
            figure.add_trace(go.Bar(x=series[MONTH_COLUMN], y=series[field], name=name, marker_color=color))
    if "p4_total_complaints" in series:
        figure.add_trace(go.Scatter(x=series[MONTH_COLUMN], y=series["p4_total_complaints"], name="Total complaints",
                                    mode="lines+markers", line={"color": TEXT, "width": 2}))
    figure.update_layout(barmode="group")
    _mark_month(figure, month)
    plotly_chart(_style(figure, 360, "Complaints by category"))
    st.caption("Category counts are shown side by side because they do not always add up to the total in the panel.")
    for column, field in zip(st.columns(3), ["p4_citizen_satisfaction_pct", "p4_sla_breach_pct", "p4_avg_resolution_hours"]):
        with column:
            _trend(panel, field, district, month)
    left, right = st.columns(2)
    with left:
        _district_bar(panel, "p4_complaints_per_1000", month, labels, f"Complaints per 1,000, {month:%b %Y}")
    with right:
        _district_bar(panel, "p4_open_requests_end_month", month, labels, f"Open requests at month end, {month:%b %Y}")


def _health_tab(panel: pd.DataFrame, district: str | None, month: pd.Timestamp, labels: dict[str, str]) -> None:
    causes = {"nonresp_visits": "Other causes", "p3_respiratory_visits_total": "Respiratory",
              "pw_waterborne_visits_total": "Waterborne", "p3_heat_related_visits": "Heat-related"}
    series = monthly_series(panel, list(causes), district)
    left, right = st.columns([3, 2])
    with left:
        _bars_with_line(series, causes, None, month, "Clinical visits by cause", stacked=True, bar_axis="visits")
    with right:
        _district_bar(panel, "cpr", month, labels, f"Primary-care pressure, {month:%b %Y}",
                      extra_hover=["pc_visits_demand", "pc_effective_capacity"])
    left, right = st.columns(2)
    with left:
        _district_bar(panel, "need_score", month, labels, f"Need score, {month:%b %Y}")
    with right:
        components = [f for f in ["esi_heat", "esi_air", "esi_water", "esi_waste"] if f in panel]
        rows = panel.loc[panel[MONTH_COLUMN].eq(month)]
        order = [name for name in district_options(panel) if name in set(rows[DISTRICT_COLUMN])]
        matrix = rows.set_index(DISTRICT_COLUMN).reindex(order)[components]
        figure = px.imshow(
            matrix.values, x=["Heat", "Air", "Water", "Waste"][: len(components)],
            y=[labels.get(n, n) for n in matrix.index], color_continuous_scale=["#FFFFFF", "#99F6E4", PRIMARY, "#134E4A"],
            aspect="auto", text_auto=".0f", labels={"color": "Stress"},
        )
        _style(figure, 26 * len(matrix) + 110, f"Environmental stress components, {month:%b %Y}")
        figure.update_layout(hovermode="closest")
        plotly_chart(figure)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render(panel: pd.DataFrame) -> None:
    st.title("Layer 4 · City command center")
    st.caption(
        "Cross-pillar operating picture from the monthly district panel. Alert limits are "
        "configurable planning assumptions in `dashboard/config.py`, not legal standards."
    )
    months = sorted(panel[MONTH_COLUMN].dropna().unique())
    if not months:
        st.error("The panel has no valid months.")
        return
    months = [pd.Timestamp(m) for m in months]
    labels = _district_labels(panel)
    districts = district_options(panel)

    left, right = st.columns([3, 2])
    month = left.select_slider("Month", options=months, value=months[-1],
                               format_func=lambda m: f"{m:%b %Y}", key="cc_month")
    scope = right.selectbox("Scope", [CITY_WIDE, *districts], key="cc_scope",
                            format_func=lambda v: v if v == CITY_WIDE else labels.get(v, v))
    district = None if scope == CITY_WIDE else scope
    position = months.index(month)
    previous = months[position - 1] if position > 0 else None

    try:
        summary = pillar_summary(panel, month, district)
        alerts = district_alerts(panel, month)
    except (ValueError, KeyError) as error:
        st.error(f"Cannot build the command center: {error}")
        return

    _status_cards(summary, previous, district)
    st.write("")

    grid_col, alert_col = st.columns([1, 1])
    with grid_col:
        st.subheader("District status by pillar")
        st.caption("Each cell shows the worst indicator in that pillar. Hover for details.")
        _status_grid(alerts, panel, labels)
    with alert_col:
        st.subheader("Active alerts")
        _alert_table(alerts, labels, district)

    st.subheader("Event timeline")
    _event_timeline(panel, month)

    st.subheader("Pillar detail" + ("" if district is None else f" for {labels.get(district, district)}"))
    tabs = st.tabs(list(COMMAND_CENTER_PILLARS))
    renderers = [_water_tab, _waste_tab, _air_tab, _service_tab, _health_tab]
    for tab, renderer in zip(tabs, renderers):
        with tab:
            renderer(panel, district, month, labels)

    with st.expander("Alert limits in use"):
        limits = pd.DataFrame(
            [{"Pillar": r["pillar"], "Indicator": r["label"], "Field": f, "Unit": r["unit"],
              "Alert when": "above" if r["direction"] == "above" else "below",
              "Watch": r["watch"], "Critical": r["critical"], "City-wide value": r.get("city", "mean")}
             for f, r in COMMAND_CENTER_THRESHOLDS.items()]
        )
        st.dataframe(limits, width="stretch", hide_index=True)
        st.caption(
            "Edit COMMAND_CENTER_THRESHOLDS in dashboard/config.py to change limits. City-wide values "
            "for ratios (NRW, collection efficiency, complaints per 1,000, primary-care pressure) are "
            "recomputed from district totals rather than averaged."
        )