"""District identifier helpers: one place that turns keys, aliases and raw names into display names.

This module has no Streamlit dependency, so the simulation engine and the data loader can share it.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

import pandas as pd

from dashboard.config import (
    DISTRICT_ALIAS_PREFIX,
    DISTRICT_ALIAS_WIDTH,
    DISTRICT_COLUMN,
    DISTRICT_DISPLAY_MODE,
    DISTRICT_KEY_COLUMN,
)

# Invisible bidirectional control characters that can sneak into Arabic text.
_BIDI_MARKS = dict.fromkeys(
    map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"), None
)
_WHITESPACE = re.compile(r"\s+")
_CODE_AT_END = re.compile(r"([A-Za-z]+\d+)\s*$")
_EMPTY_TEXT = {"", "nan", "none", "<na>", "nat"}


def clean_district_name(value: object) -> str:
    """Normalise a district name so the panel and the prediction files always match.

    Unicode NFC, invisible bidi marks removed, runs of whitespace collapsed, ends stripped.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value)).translate(_BIDI_MARKS)
    text = _WHITESPACE.sub(" ", text).strip()
    return "" if text.lower() in _EMPTY_TEXT else text


def clean_district_key(value: object) -> str:
    """Normalise a district key so 1001, '1001' and 1001.0 all become '1001'."""
    text = clean_district_name(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def short_district_code(name: object) -> str:
    """Return the trailing code of a name such as 'R1', or the full name if there is none."""
    cleaned = clean_district_name(name)
    match = _CODE_AT_END.search(cleaned)
    return match.group(1) if match else cleaned


def ordered_district_names(
    panel: pd.DataFrame,
    name_column: str = DISTRICT_COLUMN,
    key_column: str = DISTRICT_KEY_COLUMN,
) -> list[str]:
    """District names in natural order (by district_key when available), for widgets and axes.

    Sorting Arabic ordinals alphabetically gives a meaningless order, so the key is preferred.
    """
    names = panel[name_column].map(clean_district_name)
    if key_column in panel.columns:
        pairs = pd.DataFrame(
            {"name": names, "key": pd.to_numeric(panel[key_column].map(clean_district_key), errors="coerce")}
        ).drop_duplicates("name")
        return pairs.sort_values(["key", "name"], na_position="last")["name"].tolist()
    return sorted(set(names) - {""})


def build_district_lookup(
    frame: pd.DataFrame,
    name_column: str = DISTRICT_COLUMN,
    key_column: str = DISTRICT_KEY_COLUMN,
) -> dict[str, str]:
    """Map every identifier a file might contain (name, key, legacy D01 alias) to the district name."""
    names = frame[name_column].map(clean_district_name)
    unique_names = sorted(set(names) - {""})
    lookup: dict[str, str] = {}
    for ordinal, name in enumerate(unique_names, start=1):  # legacy alias order: sorted names
        lookup[f"{DISTRICT_ALIAS_PREFIX}{ordinal:0{DISTRICT_ALIAS_WIDTH}d}"] = name
    if key_column in frame.columns:
        for key, name in zip(frame[key_column].map(clean_district_key), names):
            if key and name:
                lookup[key] = name
    for name in unique_names:
        lookup[name] = name
    return lookup


def resolve_district_names(values: pd.Series, lookup: Mapping[str, str]) -> pd.Series:
    """Turn a column of names, keys or aliases into district names. Unknown values are kept (cleaned)."""

    def _resolve(value: object) -> str:
        name = clean_district_name(value)
        for candidate in (name, clean_district_key(value), name.upper()):
            if candidate in lookup:
                return lookup[candidate]
        return name

    return values.map(_resolve)


def district_label(
    name: object,
    ordered_districts: Sequence[str] | None = None,
    mode: str = DISTRICT_DISPLAY_MODE,
) -> str:
    """The text shown to the user: the district name, or a D01-style alias if mode == 'alias'."""
    cleaned = clean_district_name(name)
    if mode == "alias" and ordered_districts is not None:
        ordered = [clean_district_name(item) for item in ordered_districts]
        if cleaned in ordered:
            return f"{DISTRICT_ALIAS_PREFIX}{ordered.index(cleaned) + 1:0{DISTRICT_ALIAS_WIDTH}d}"
    return cleaned


def targeted_mask(
    frame: pd.DataFrame,
    targets: Sequence[object] | None,
    name_column: str = DISTRICT_COLUMN,
    key_column: str = DISTRICT_KEY_COLUMN,
) -> pd.Series:
    """Boolean mask of rows whose district is in `targets`.

    `targets` may contain names, keys or legacy aliases. If targets are given but none match,
    raise instead of silently simulating nothing.
    """
    requested = list(targets) if targets is not None else []
    names = frame[name_column].map(clean_district_name)
    if not requested:
        return pd.Series(False, index=frame.index)
    lookup = build_district_lookup(frame, name_column, key_column)
    resolved = set(resolve_district_names(pd.Series(requested, dtype=object), lookup))
    mask = names.isin(resolved)
    if not mask.any():
        raise ValueError(
            f"None of the selected districts {requested!r} match the baseline districts "
            f"{sorted(set(names))!r}."
        )
    return mask
