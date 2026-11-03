"""Run from the project root:  python -m dashboard.check_district_names

Shows how every district identifier is resolved, and flags any file that still contains
IDs or names that do not exist in the master panel."""

from __future__ import annotations

import sys

import pandas as pd

from dashboard.config import DISTRICT_COLUMN, DISTRICT_KEY_COLUMN, EXPERIMENTS
from dashboard.data_loader import (
    DISTRICT_SOURCE_COLUMNS,
    district_options,
    load_csv_artifact,
    load_panel,
    normalize_prediction_columns,
    validate_panel_schema,
)
from dashboard.district_utils import build_district_lookup, clean_district_name, short_district_code


def main() -> int:
    problems = 0
    panel = load_panel()
    names = district_options(panel)
    print(f"Panel: {len(panel)} rows, {len(names)} districts\n")
    key_by_name = (
        panel.drop_duplicates(DISTRICT_COLUMN).set_index(DISTRICT_COLUMN)[DISTRICT_KEY_COLUMN].to_dict()
        if DISTRICT_KEY_COLUMN in panel.columns
        else {}
    )
    for name in names:
        print(f"  {str(key_by_name.get(name, '')):>6}  {short_district_code(name):<6} {name}")
    for note in validate_panel_schema(panel):
        print(f"\n  NOTE: {note}")

    lookup = build_district_lookup(panel)
    known = set(names)
    print("\nPrediction files")
    for experiment, config in EXPERIMENTS.items():
        raw = load_csv_artifact(config["predictions"])
        if raw.empty:
            print(f"  [missing] {experiment}: {config['predictions']}")
            problems += 1
            continue
        found = [c for c in DISTRICT_SOURCE_COLUMNS if c in raw.columns]
        fixed = normalize_prediction_columns(raw, lookup)
        unresolved = sorted(set(fixed[DISTRICT_COLUMN]) - known)
        status = "OK" if not unresolved else "UNRESOLVED"
        print(f"  [{status}] {experiment}: id columns in file = {found or 'NONE'}; "
              f"{fixed[DISTRICT_COLUMN].nunique()} districts")
        if unresolved:
            problems += 1
            print(f"      not in panel: {unresolved[:5]}")
        print(f"      sample: {fixed[DISTRICT_COLUMN].head(2).tolist()}")
    print("\nResult:", "all districts resolve to names." if not problems else f"{problems} problem(s) found.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
