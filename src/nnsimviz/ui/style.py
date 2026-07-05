"""Page setup and header rendering for the Streamlit UI."""

from __future__ import annotations

import streamlit as st

from nnsimviz.ui.constants import FAVICON_PATH, LOGO_PATH


def configure_page() -> None:
    """Configure the Streamlit page and inject small sidebar CSS tweaks."""
    page_icon = str(FAVICON_PATH) if FAVICON_PATH.exists() else "\U0001f9e0"
    st.set_page_config(
        page_title="SynapSight",
        page_icon=page_icon,
        layout="wide",
    )

    st.markdown(
        """
        <style>
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px;
            width: auto !important;
            max-width: none !important;
        }
        [data-testid="stExpandSidebarButton"]::after,
        [data-testid="stSidebarCollapsedControl"]::after,
        [data-testid="collapsedControl"]::after {
            content: "Parameters";
            font-weight: 600;
            font-size: 0.9rem;
            white-space: nowrap;
            color: rgb(49, 51, 63);
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
        section[data-testid="stSidebar"] details summary p {
            font-weight: 700;
            font-size: 1.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_header() -> None:
    """Render the application logo, title, and short explanation."""
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=500)
    else:
        st.title("\U0001f9e0 SynapSight")

    st.subheader("Neural Network Simulation & Visualization GUI")
    st.markdown(
        "**SynapSight** is an interactive tool for defining, simulating, "
        "and visualizing simple recurrent neural networks. Build a network, "
        "run continuous or event-based dynamics, add neurobiological motifs, "
        "and explore the result as an interactive graph, an activity timeline, "
        "and an animation."
    )
    st.caption(
        "Define a network, run a simple recurrent simulation, and explore it "
        "as a graph. Blue = positive weight, red = negative, thickness = strength."
    )
