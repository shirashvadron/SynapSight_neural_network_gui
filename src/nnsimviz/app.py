"""Streamlit GUI: a thin orchestrator over the pipeline modules.

This layer only reads widget values, builds a ProjectConfig, and calls the
model -> simulation -> visualization modules in sequence, then displays the
figures and a short summary. It contains NO graph-building or simulation
logic of its own.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from nnsimviz.configs import (
    NetworkConfig,
    ProjectConfig,
    SimulationConfig,
    VisualizationConfig,
)
from nnsimviz.models import MODEL_REGISTRY, build_weight_matrix
from nnsimviz.simulation import Simulator
from nnsimviz.visualization import (
    build_figure,
    build_activity_figure,
    build_animated_figure,
    NetworkVisualizer,
    AVAILABLE_LAYOUTS,
)
from nnsimviz.help_texts import get_help_texts
from nnsimviz import io_utils


st.set_page_config(page_title="Neural Network Simulator", layout="wide")

# Feature #2: show the word "Parameters" next to the sidebar-expand arrow.
# When the sidebar is collapsed, Streamlit shows only a small ">>" button; on
# its own, nothing tells the user a parameters panel is hidden there. This CSS
# appends a "Parameters" label to that button. The real element id in current
# Streamlit is `stExpandSidebarButton`; older names are kept as fallbacks so
# the label still appears on other versions your teammates might run.
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
    </style>
    """,
    unsafe_allow_html=True,
)

# Human-readable labels for the layout dropdown (feature #5).
_LAYOUT_LABELS = {
    "spring": "Spring (force-directed)",
    "circular": "Circular",
    "shell": "Shell (concentric)",
    "spiral": "Spiral",
    "random": "Random",
}

# Plotly modebar config: enables zoom/pan/box-zoom/reset, and sets the
# built-in camera button to download a PNG (feature #1, dependency-free).
_PLOT_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,  # supports scroll/double-click zoom (feature #8)
    "toImageButtonOptions": {"format": "png", "filename": "neural_network_graph",
                             "scale": 2},
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


# --------------------------------------------------------------------------- #
# Sidebar controls -> ProjectConfig
# --------------------------------------------------------------------------- #
def read_config_from_sidebar() -> ProjectConfig:
    """Read every sidebar widget and assemble a ProjectConfig.

    Each widget carries a `help=` string, which Streamlit renders as a small
    "i"/"?" icon that reveals a short explanation on hover (feature #3). The
    strings come from the help_texts module: once the user picks a model, the
    matching HelpTexts object is fetched and used for every control, so the
    tooltips adapt automatically to the selected network.
    """
    st.sidebar.title("Parameters")

    # ---- Network ----
    st.sidebar.header("Network")
    model_keys = list(MODEL_REGISTRY)
    # The model tooltip itself uses the shared default (no model chosen yet).
    model_type = st.sidebar.selectbox(
        "Model", model_keys,
        format_func=lambda k: MODEL_REGISTRY[k].name,
        help=get_help_texts("").model,
    )

    # Fetch the help texts for the chosen model; every control below uses them.
    txt = get_help_texts(model_type)

    n_neurons = st.sidebar.slider(
        "Number of neurons", 2, 100, 20, help=txt.n_neurons,
    )
    connection_probability = st.sidebar.slider(
        "Connection probability", 0.0, 1.0, 0.3, 0.05,
        help=txt.connection_probability,
    )
    weight_scale = st.sidebar.slider(
        "Weight scale", 0.1, 3.0, 1.0, 0.1, help=txt.weight_scale,
    )
    positive_connection_ratio = st.sidebar.slider(
        "Positive / excitatory ratio", 0.0, 1.0, 0.7, 0.05,
        help=txt.positive_connection_ratio,
    )
    random_seed = st.sidebar.number_input(
        "Random seed", 0, 10_000, 42, 1, help=txt.random_seed,
    )

    n_modules = 4
    inter_module_probability = 0.05

    if model_type == "modular":
        st.sidebar.subheader("Modular network")
        n_modules = st.sidebar.slider(
            "Number of modules", 1, max(1, int(n_neurons)), min(4, int(n_neurons)), 1,
            help=txt.n_modules,
        )
        inter_module_probability = st.sidebar.slider(
            "Inter-module connection probability", 0.0, 1.0, 0.05, 0.01,
            help=txt.inter_module_probability,
        )

    network = NetworkConfig(
        n_neurons=int(n_neurons),
        connection_probability=connection_probability,
        weight_scale=weight_scale,
        positive_connection_ratio=positive_connection_ratio,
        model_type=model_type,
        random_seed=int(random_seed),
        n_modules=n_modules,
        inter_module_probability=inter_module_probability,
    )

    # ---- Simulation ----
    st.sidebar.header("Simulation")
    duration = st.sidebar.slider(
        "Duration", 1.0, 50.0, 10.0, 1.0, help=txt.duration,
    )
    dt = st.sidebar.slider(
        "Time step (dt)", 0.01, 1.0, 0.1, 0.01, help=txt.dt,
    )
    input_type = st.sidebar.selectbox(
        "Input type", SimulationConfig.VALID_INPUT_TYPES, index=1,
        help=txt.input_type,
    )
    input_amplitude = st.sidebar.slider(
        "Input amplitude", 0.0, 5.0, 1.0, 0.1, help=txt.input_amplitude,
    )
    noise_level = st.sidebar.slider(
        "Noise level", 0.0, 1.0, 0.05, 0.01, help=txt.noise_level,
    )

    simulation = SimulationConfig(
        duration=duration,
        dt=dt,
        input_type=input_type,
        input_amplitude=input_amplitude,
        noise_level=noise_level,
    )

    # ---- Visualization ----
    st.sidebar.header("Visualization")
    layout_type = st.sidebar.selectbox(
        "Layout", AVAILABLE_LAYOUTS,
        format_func=lambda k: _LAYOUT_LABELS.get(k, k),
        help=txt.layout,
    )
    show_labels = st.sidebar.checkbox(
        "Show neuron labels", True, help=txt.show_labels,
    )
    edge_width_scale = st.sidebar.slider(
        "Edge width scale", 1.0, 10.0, 3.0, 0.5, help=txt.edge_width_scale,
    )
    min_edge_abs_weight = st.sidebar.slider(
        "Min edge |weight| to show", 0.0, 3.0, 0.0, 0.1,
        help=txt.min_edge_abs_weight,
    )
    node_size_scale = st.sidebar.slider(
        "Node size scale", 0.5, 3.0, 1.0, 0.1, help=txt.node_size_scale,
    )
    show_activity_on_nodes = st.sidebar.checkbox(
        "Color nodes by activity", True, help=txt.show_activity_on_nodes,
    )

    visualization = VisualizationConfig(
        layout_type=layout_type,
        show_labels=show_labels,
        edge_width_scale=edge_width_scale,
        min_edge_abs_weight=min_edge_abs_weight,
        node_size_scale=node_size_scale,
        show_activity_on_nodes=show_activity_on_nodes,
    )

    return ProjectConfig(network=network, simulation=simulation, visualization=visualization)


# --------------------------------------------------------------------------- #
# Pipeline (orchestration only)
# --------------------------------------------------------------------------- #
def run_pipeline(config: ProjectConfig):
    """model -> simulation -> result. Returns the SimulationResult."""
    weight_matrix = build_weight_matrix(config.network)
    return Simulator().run(config, weight_matrix)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("\U0001f9e0 Neural Network Simulation & Visualization")
    st.caption(
        "Define a network, run a simple recurrent simulation, and explore it "
        "as a graph. Blue = positive weight, red = negative, thickness = strength."
    )

    config = read_config_from_sidebar()

    # Validate up front so the user gets a clear message, not a traceback.
    try:
        config.validate()
    except ValueError as exc:
        st.error(f"Invalid parameters: {exc}")
        st.stop()

    if st.sidebar.button("\u25b6 Run simulation", type="primary"):
        with st.spinner("Running simulation..."):
            result = run_pipeline(config)
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if result is None:
        st.info("Set your parameters in the sidebar and press **Run simulation**.")
        return

    model_name = MODEL_REGISTRY[result.config.network.model_type].name

    # ---- three tabs: static graph, activity over time, animation ---- #
    tab_graph, tab_activity, tab_anim = st.tabs(
        ["Network graph", "Activity over time", "Animation"]
    )

    with tab_graph:
        # Feature #7: the model name is shown above the graph (also in title).
        st.subheader(f"Network graph \u2014 {model_name}")
        st.caption(
            "Tip: use the toolbar (top-right of the plot) to zoom, pan, or "
            "download a PNG with the camera icon. Double-click the plot to reset."
        )
        fig = build_figure(result)
        st.plotly_chart(fig, use_container_width=True, config=_PLOT_CONFIG)

    with tab_activity:
        st.subheader("Activity over time")
        act_fig = build_activity_figure(result)
        st.plotly_chart(act_fig, use_container_width=True, config=_PLOT_CONFIG)

    with tab_anim:
        st.subheader("Activity animation")
        st.caption("Press \u25b6 Play to watch node activity evolve over time.")
        anim_fig = build_animated_figure(result)
        st.plotly_chart(anim_fig, use_container_width=True, config=_PLOT_CONFIG)

    # ---- summary ----
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

    # ---- exports ----
    st.subheader("Export")

    # Feature #1: an "Image - PNG" button placed in the export row, alongside
    # the JSON/CSV buttons. It triggers the browser's own Plotly image
    # download (the same mechanism as the camera icon on the toolbar), so it
    # needs no server-side image library (no kaleido / Chrome).
    #
    # The button needs a Plotly graph in the page to capture, but we do NOT
    # want a second visible graph. So we render the figure inside a zero-size,
    # hidden container and only surface the button itself.
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
        padding:0.25rem 0.75rem; height:38.4px; margin:0;
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

    # Narrow columns sized close to each button's own width, with a spacer
    # column at the end absorbing the leftover space -- so the four buttons
    # sit next to each other instead of being stretched across the full row.
    e1, e2, e3, e4, _spacer = st.columns([1, 1, 1, 1, 4], vertical_alignment="bottom")
    with e1:
        components.html(png_component, height=39)
    e2.download_button("Config (JSON)", io_utils.config_to_json(result.config),
                       "config.json", "application/json")
    e3.download_button("Weights (CSV)", io_utils.weight_matrix_to_csv(result.weight_matrix),
                       "weights.csv", "text/csv")
    e4.download_button("Activity (CSV)", io_utils.activity_to_csv(result),
                       "activity.csv", "text/csv")


if __name__ == "__main__":
    main()
