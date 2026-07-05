"""Streamlit entry point for SynapSight.

This module only coordinates the GUI flow: read UI configuration, run the
pipeline, and display results. Sidebar widgets, result tabs, style, summaries,
and exports live in nnsimviz.ui helper modules.

Run with:
    streamlit run src/nnsimviz/app.py
"""

from __future__ import annotations

import streamlit as st

from nnsimviz.models import MODEL_REGISTRY
from nnsimviz.pipeline import run_pipeline
from nnsimviz.ui.exports import show_exports, show_summary
from nnsimviz.ui.results import show_continuous_result, show_event_result
from nnsimviz.ui.sidebar import read_config_from_sidebar
from nnsimviz.ui.style import configure_page, show_header


def main() -> None:
    """Run the SynapSight Streamlit application."""
    configure_page()
    show_header()

    config, imported_weight_matrix = read_config_from_sidebar()

    try:
        config.validate()
    except ValueError as exc:
        st.error(f"Invalid parameters: {exc}")
        st.stop()

    if st.sidebar.button("\u25b6 Run simulation", type="primary", use_container_width=True):
        with st.spinner("Running simulation..."):
            result = run_pipeline(config, imported_weight_matrix)
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if result is None:
        st.info("Set your parameters in the sidebar and press **Run simulation**.")
        return

    model_name = result.metadata.get("network_source_name")
    if model_name is None:
        model_name = MODEL_REGISTRY[result.config.network.model_type].name

    if result.config.simulation.simulation_type == "event_based":
        show_event_result(result, model_name)
    else:
        show_continuous_result(result, model_name)

    show_summary(result)
    show_exports(result)


if __name__ == "__main__":
    main()