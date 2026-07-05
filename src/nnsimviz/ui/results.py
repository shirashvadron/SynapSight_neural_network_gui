"""Result tabs for continuous and event-based simulations."""

from __future__ import annotations

import streamlit as st

from nnsimviz.ui.constants import PLOT_CONFIG
from nnsimviz.visualization import (
    build_activity_figure,
    build_animated_figure,
    build_event_raster_figure,
    build_figure,
    build_pca_animation_figure,
)


def show_continuous_result(result, model_name: str) -> None:
    """Render the original continuous-simulation tabs."""
    tab_graph, tab_activity, tab_anim, tab_pca = st.tabs(
        ["Network graph", "Activity over time", "Animation", "Animated PCA"]
    )

    with tab_graph:
        st.subheader(f"Network graph — {model_name}")
        st.caption(
            "Tip: use the toolbar (top-right of the plot) to zoom, pan, or "
            "download a PNG with the camera icon. Double-click the plot to reset."
        )
        fig = build_figure(result)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    with tab_activity:
        st.subheader("Activity over time")
        converged_at = result.metadata.get("converged_at")
        if converged_at is not None:
            conv_time = round(converged_at * result.config.simulation.dt, 4)
            st.info(
                f"✅ Converged early at step {converged_at} "
                f"(t ≈ {conv_time}). "
                "Remaining time steps are filled with the settled state."
            )
        act_fig = build_activity_figure(result)
        st.plotly_chart(act_fig, use_container_width=True, config=PLOT_CONFIG)

    with tab_anim:
        st.subheader("Activity animation")
        st.caption("Press ▶ Play to watch node activity evolve over time.")
        anim_fig = build_animated_figure(result)
        st.plotly_chart(anim_fig, use_container_width=True, config=PLOT_CONFIG)

    with tab_pca:
        st.subheader("Animated PCA — state-space trajectory")
        st.caption(
            "The network's N-dimensional activity vector is projected onto its "
            "first three principal components. The moving dot traces the collective "
            "state through this 3-D subspace over time -- click and drag to rotate. "
            "🟢 Green stars = stable fixed points. "
            "🟠 Orange open diamonds = unstable fixed points."
        )
        pca_fig = build_pca_animation_figure(result)
        st.plotly_chart(pca_fig, use_container_width=True, config=PLOT_CONFIG)


def show_event_result(result, model_name: str) -> None:
    """Render event-based/spike-like tabs."""
    tab_graph, tab_raster, tab_activity, tab_anim, tab_pca = st.tabs(
        ["Network graph", "Spike raster", "Activation over time", "Animation", "Animated PCA"]
    )

    with tab_graph:
        st.subheader(f"Network graph — {model_name}")
        fig = build_figure(result)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    with tab_raster:
        st.subheader("Spike raster")
        st.caption("Each marker is one threshold-crossing spike event.")
        raster_fig = build_event_raster_figure(result)
        st.plotly_chart(raster_fig, use_container_width=True, config=PLOT_CONFIG)

    with tab_activity:
        st.subheader("Activation over event steps")
        act_fig = build_activity_figure(result)
        st.plotly_chart(act_fig, use_container_width=True, config=PLOT_CONFIG)
        with st.expander("Event log"):
            st.dataframe(result.metadata.get("event_log", []), use_container_width=True)

    with tab_anim:
        st.subheader("Event-state animation")
        anim_fig = build_animated_figure(result)
        st.plotly_chart(anim_fig, use_container_width=True, config=PLOT_CONFIG)

    with tab_pca:
        st.subheader("Animated PCA — event-state trajectory")
        st.caption(
            "The event simulation activity vector is projected onto the first "
            "three principal components over event steps -- click and drag to "
            "rotate the 3-D view."
        )
        pca_fig = build_pca_animation_figure(result)
        st.plotly_chart(pca_fig, use_container_width=True, config=PLOT_CONFIG)

