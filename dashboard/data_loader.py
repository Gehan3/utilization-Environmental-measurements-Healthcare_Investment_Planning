"""Cached, schema-aware loading and validation for dashboard inputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from dashboard.config import (
    DISTRICT_COLUMN,
    DISTRICT_KEY_COLUMN,
    DISTRICT_LABEL_COLUMN,
    EXPERIMENTS,
    MONTH_COLUMN,
    PANEL_PATH,
)
from dashboard.district_utils import (
    build_district_lookup,
    clean_district_key,
    clean_district_name,
    district_label,
    ordered_district_names,
    resolve_district_names,
    short_district_code,
)

# Columns that may hold a district identifier in a prediction file, in order of preference.
DISTRICT_SOURCE_COLUMNS = ("district_name", "district_key", "district_id", "district")


@st.cache_data(show_spinner=False)
def load_panel(path: str | Path = PANEL_PATH) -> pd.DataFrame:
    """Load the panel, normalize its key fields, and validate its structural schema."""
    panel = pd.read_csv(path, encoding="utf-8-sig")
    panel.columns = [str(column).strip() for column in panel.columns]
    missing = {DISTRICT_COLUMN, MONTH_COLUMN}.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing required key columns: {sorted(missing)}")
    panel[MONTH_COLUMN] = pd.to_datetime(panel[MONTH_COLUMN], errors="raise")
    panel[DISTRICT_COLUMN] = panel[DISTRICT_COLUMN].map(clean_district_name)
    if (panel[DISTRICT_COLUMN] == "").any():
        raise ValueError("Panel contains rows with a blank district_name.")
    if panel.duplicated([DISTRICT_COLUMN, MONTH_COLUMN]).any():
        raise ValueError("Panel contains duplicate district-month keys.")
    return panel.sort_values([MONTH_COLUMN, DISTRICT_COLUMN]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_csv_artifact(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_resource(show_spinner=False)
def load_model_artifacts() -> dict[str, dict[str, object]]:
    """Load only model files present in the configured experiment artifact registry."""
    loaded: dict[str, dict[str, object]] = {}
    for experiment_name, config in EXPERIMENTS.items():
        loaded[experiment_name] = {}
        for model_name, path in config["models"].items():
            model_path = Path(path)
            if model_path.is_file():
                loaded[experiment_name][model_name] = joblib.load(model_path)
    return loaded


def validate_panel_schema(panel: pd.DataFrame) -> list[str]:
    """Return non-fatal schema observations; unknown numeric columns remain supported."""
    observations = []
    if len(panel) != 315:
        observations.append(f"Expected 315 rows in the benchmark panel; found {len(panel)}.")
    district_count = panel[DISTRICT_COLUMN].nunique()
    if district_count != 15:
        observations.append(f"Expected 15 districts; found {district_count}.")
    if panel[MONTH_COLUMN].nunique() != 21:
        observations.append(
            f"Expected 21 monthly timestamps; found {panel[MONTH_COLUMN].nunique()}."
        )
    if panel[DISTRICT_COLUMN].str.fullmatch(r"\d+").any():
        observations.append("Some district_name values look like numeric IDs, not names.")
    if DISTRICT_KEY_COLUMN in panel.columns:
        pairs = pd.DataFrame(
            {
                "key": panel[DISTRICT_KEY_COLUMN].map(clean_district_key),
                "name": panel[DISTRICT_COLUMN].map(clean_district_name),
            }
        ).drop_duplicates()
        if pairs["key"].duplicated().any() or pairs["name"].duplicated().any():
            observations.append("district_key and district_name are not one-to-one.")
    return observations


def district_alias(district_name: object, ordered_districts: list[str]) -> str:
    """Kept for backward compatibility. Returns the display label for a district.

    With DISTRICT_DISPLAY_MODE == "name" (the default) this is the real district_name.
    """
    return district_label(district_name, ordered_districts)


def with_district_aliases(panel: pd.DataFrame) -> pd.DataFrame:
    """Add display columns. `district_label` and the legacy `district_alias` both hold the name
    (unless DISTRICT_DISPLAY_MODE == "alias"); `district_code` is the short code, e.g. R1."""
    result = panel.copy()
    names = result[DISTRICT_COLUMN].map(clean_district_name)
    legacy_order = sorted(set(names) - {""})
    result[DISTRICT_LABEL_COLUMN] = names.map(lambda value: district_label(value, legacy_order))
    result["district_alias"] = result[DISTRICT_LABEL_COLUMN]
    result["district_code"] = names.map(short_district_code)
    return result


with_district_labels = with_district_aliases


def district_options(panel: pd.DataFrame) -> list[str]:
    """District names in natural order (by district_key), for selectboxes and chart axes."""
    return ordered_district_names(panel)


def normalize_prediction_columns(
    frame: pd.DataFrame, lookup: Mapping[str, str] | None = None
) -> pd.DataFrame:
    """Standardise a prediction file. The district column always ends up holding the district name,
    even when the file only has district_key or a legacy D01 alias."""
    if frame.empty:
        return frame.copy()

    actual_candidates = [col for col in frame if col.startswith("actual_")]
    predicted_candidates = [col for col in frame if col.startswith("predicted_")]

    normalized = frame.copy()
    if actual_candidates:
        normalized["actual_value"] = pd.to_numeric(normalized[actual_candidates[0]], errors="coerce")
    if predicted_candidates:
        normalized["predicted_value"] = pd.to_numeric(
            normalized[predicted_candidates[0]], errors="coerce"
        )

    source = next((col for col in DISTRICT_SOURCE_COLUMNS if col in normalized.columns), None)
    if source is None:
        normalized[DISTRICT_COLUMN] = "Unknown"
    else:
        normalized[DISTRICT_COLUMN] = resolve_district_names(normalized[source], lookup or {})
    normalized[DISTRICT_LABEL_COLUMN] = normalized[DISTRICT_COLUMN]
    return normalized


def load_experiment_artifacts(
    panel: pd.DataFrame | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load every experiment's artifacts. Pass the panel to resolve district keys to names;
    if omitted it is loaded from PANEL_PATH."""
    if panel is None:
        try:
            panel = load_panel()
        except Exception:  # panel unavailable: names are kept as they appear in the files
            panel = None
    lookup = build_district_lookup(panel) if panel is not None else {}
    return {
        name: {
            "metrics": load_csv_artifact(config["metrics"]),
            "predictions": normalize_prediction_columns(
                load_csv_artifact(config["predictions"]), lookup
            ),
            "importance": load_csv_artifact(config["importance"]),
            "exclusions": load_csv_artifact(config["exclusions"]),
        }
        for name, config in EXPERIMENTS.items()
    }


def model_artifact_summary(
    models: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    rows = []
    for experiment_name, config in EXPERIMENTS.items():
        loaded = models.get(experiment_name, {})
        configured = config["models"]
        if not configured:
            rows.append(
                {
                    "Experiment": experiment_name,
                    "Model": "—",
                    "Artifact": "—",
                    "Input feature count": None,
                    "Status": "No serialized model in source artifacts",
                }
            )
            continue
        for model_name, path in configured.items():
            model = loaded.get(model_name)
            input_features = (
                getattr(model, "feature_names_in_", None) if model is not None else None
            )
            feature_count = (
                len(input_features)
                if input_features is not None
                else getattr(model, "n_features_in_", None)
                if model is not None
                else None
            )
            rows.append(
                {
                    "Experiment": experiment_name,
                    "Model": model_name,
                    "Artifact": Path(path).name,
                    "Input feature count": feature_count,
                    "Status": "Loaded" if model is not None else "Missing",
                }
            )
    return pd.DataFrame(rows)
