"""Small Streamlit helpers shared by all dashboard views."""

from __future__ import annotations

import inspect

import streamlit as st

# Plotly toolbar options. Passed through `config`, which is what Streamlit asks for.
PLOTLY_CONFIG = {"displaylogo": False}

# Recent Streamlit versions stretch charts to full width by default and warn when `width` is passed.
# Older versions have no `width` parameter and need use_container_width=True instead.
_PLOTLY_PARAMS = inspect.signature(st.plotly_chart).parameters


def plotly_chart(figure, **kwargs) -> None:
    """Show a Plotly figure at full container width without Streamlit deprecation warnings."""
    kwargs.setdefault("config", PLOTLY_CONFIG)
    if "width" not in _PLOTLY_PARAMS and "use_container_width" in _PLOTLY_PARAMS:
        kwargs.setdefault("use_container_width", True)
    st.plotly_chart(figure, **kwargs)
