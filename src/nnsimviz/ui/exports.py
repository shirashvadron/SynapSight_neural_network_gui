"""Summary metrics and export controls for simulation results."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from nnsimviz import io_utils
from nnsimviz.ui.constants import METHOD_LABELS
from nnsimviz.visualization import NetworkVisualizer, build_figure


def show_summary(result) -> None:
    """Render compact summary metrics and mode/integration badges."""
    _, edges = NetworkVisualizer(
        result.config.visualization
    ).build_graph(result.weight_matrix)
    n_pos = sum(1 for e in edges if e.weight > 0)
    n_neg = sum(1 for e in edges if e.weight < 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Neurons", result.metadata["n_neurons"])
    c2.metric("Visible edges", len(edges))
    c3.metric("Positive edges", n_pos)
    c4.metric("Negative edges", n_neg)
    c5.metric("Time steps", result.metadata["n_steps"])

    method_used = result.metadata.get("integration_method", "euler")
    converged_at = result.metadata.get("converged_at")

    badge_col1, badge_col2, _spacer = st.columns([1, 1, 6])

    if result.config.simulation.simulation_type == "event_based":
        badge_col1.caption("**Mode:** event-based")
        badge_col2.caption(f"**Spike threshold:** {result.metadata.get('threshold')}")
    else:
        badge_col1.caption(
            f"**Integration:** {METHOD_LABELS.get(method_used, method_used)}"
        )

        if converged_at is not None:
            badge_col2.caption(f"**Converged at step:** {converged_at}")
        else:
            badge_col2.caption("**Convergence:** not triggered")


def show_exports(result) -> None:
    """Render JSON/CSV/PNG export controls."""
    st.subheader("Export")

    graph_fig = build_figure(result)
    hidden_div = graph_fig.to_html(include_plotlyjs="cdn", full_html=False,
                                   div_id="png_export_fig")
    png_component = f"""
    <div style="position:absolute; width:1200px; height:700px;
                left:-10000px; top:0; visibility:hidden;">
      {hidden_div}
    </div>
    <button id="png_btn" style="
        width:100%; box-sizing:border-box;
        padding:0.25rem 0.75rem; height:38.4px; margin:0; margin-top:-5px;
        font-size:0.875rem; font-weight:400; line-height:1.6; cursor:pointer;
        border:1px solid rgba(49,51,63,0.2); border-radius:0.5rem;
        background:#fff; color:rgb(49,51,63);
        font-family:'Source Sans Pro','Source Sans',-apple-system,
                    BlinkMacSystemFont,sans-serif;
        transition:border-color 0.2s, color 0.2s;"
        onmouseover="this.style.borderColor='rgb(255,75,75)';this.style.color='rgb(255,75,75)';"
        onmouseout="this.style.borderColor='rgba(49,51,63,0.2)';this.style.color='rgb(49,51,63)';">
      Image-PNG
    </button>
    <script>
      document.getElementById("png_btn").addEventListener("click", function() {{
        var gd = document.getElementById("png_export_fig");
        Plotly.downloadImage(gd, {{
          format: "png", scale: 2, filename: "neural_network_graph"
        }});
      }});
    </script>
    """

    e1, e2, e3, e4, _spacer = st.columns([1, 1, 1, 1, 4], vertical_alignment="bottom")
    e1.download_button("Config (JSON)", io_utils.config_to_json(result.config),
                       "config.json", "application/json")
    e2.download_button("Weights (CSV)", io_utils.weight_matrix_to_csv(result.weight_matrix),
                       "weights.csv", "text/csv")
    e3.download_button("Activity (CSV)", io_utils.activity_to_csv(result),
                       "activity.csv", "text/csv")
    with e4:
        components.html(png_component, height=39)