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
    EventSimulationConfig,
    VALID_SIMULATION_TYPES,
    VALID_INTEGRATION_METHODS,
)

from nnsimviz.models import MODEL_REGISTRY
from nnsimviz.pipeline import run_pipeline

from nnsimviz.visualization import (
    build_figure,
    build_activity_figure,
    build_animated_figure,
    build_pca_animation_figure,
    NetworkVisualizer,
    build_event_raster_figure,
    AVAILABLE_LAYOUTS,
)
from nnsimviz.help_texts import get_help_texts
from nnsimviz import io_utils


st.set_page_config(page_title="Neural Network Simulator", layout="wide")

# Feature #2: show the word "Parameters" next to the sidebar-expand arrow.
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

# Human-readable labels for the layout dropdown.
_LAYOUT_LABELS = {
    "spring": "Spring (force-directed)",
    "circular": "Circular",
    "shell": "Shell (concentric)",
    "spiral": "Spiral",
    "random": "Random",
}

# Human-readable labels for integration method dropdown.
_METHOD_LABELS = {
    "euler": "Euler-Maruyama (1st order, fast)",
    "heun": "Heun / Improved Euler (2nd order)",
    "rk4": "Runge-Kutta 4 (4th order, accurate)",
}

# Plotly modebar config.
_PLOT_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "toImageButtonOptions": {"format": "png", "filename": "neural_network_graph",
                             "scale": 2},
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

_SIMULATION_TYPE_LABELS = {
    "continuous": "Continuous recurrent dynamics",
    "event_based": "Event-based / spike-like",
}

# --------------------------------------------------------------------------- #
# Sidebar controls -> ProjectConfig
# --------------------------------------------------------------------------- #
def read_config_from_sidebar() -> tuple[ProjectConfig, object | None]:
    """Read every sidebar widget and assemble a ProjectConfig.

    Each widget carries a `help=` string rendered as a tooltip (feature #3).
    Strings come from help_texts and adapt to the selected model automatically.
    """
    st.sidebar.title("Parameters")
    simulation_type = st.sidebar.selectbox(
        "Simulation type",
        VALID_SIMULATION_TYPES,
        index=0,
        format_func=lambda k: _SIMULATION_TYPE_LABELS.get(k, k),
        help=(
            "Choose the simulation engine: original continuous dynamics or "
            "event-based spike propagation."
        ),
    )

    # ---- Network ----
    st.sidebar.header("Network")
    model_keys = list(MODEL_REGISTRY)
    model_type = st.sidebar.selectbox(
        "Model", model_keys,
        format_func=lambda k: MODEL_REGISTRY[k].name,
        help=get_help_texts("").model,
    )

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

    imported_weight_matrix = read_imported_weight_matrix_from_sidebar()

    if imported_weight_matrix is not None:
        network.n_neurons = int(imported_weight_matrix.shape[0])

    # ---- Simulation ----
    
    st.sidebar.header("Simulation")
    event = EventSimulationConfig()

    if simulation_type == "continuous":
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

        integration_method = st.sidebar.selectbox(
            "Integration method",
            list(VALID_INTEGRATION_METHODS),
            index=0,
            format_func=lambda k: _METHOD_LABELS.get(k, k),
            help=txt.integration_method,
        )

        use_convergence = st.sidebar.checkbox(
            "Enable early convergence stop",
            value=False,
            help=txt.convergence_eps,
        )
        convergence_eps: float | None = None
        if use_convergence:
            convergence_eps = st.sidebar.number_input(
                "Convergence epsilon (ε)",
                min_value=1e-8,
                max_value=1.0,
                value=1e-4,
                format="%e",
                step=1e-5,
                help=(
                    "The simulation stops early once max|x[t] − x[t−1]| < ε. "
                    "The remaining time steps are filled with the settled state."
                ),
            )
    else:
        st.sidebar.caption(
            "Event mode uses the same network/imported W, but updates mainly "
            "when spike events occur."
        )
        event_max_steps = st.sidebar.slider(
            "Event steps", 1, 200, 20, 1,
            help="Number of discrete event steps to simulate.",
        )
        event_threshold = st.sidebar.number_input(
            "Spike threshold", min_value=0.001, value=1.0, step=0.1,
            help="A neuron emits a spike when activation is at or above this value.",
        )
        event_reset_value = st.sidebar.number_input(
            "Reset value", value=0.0, step=0.1,
            help="Neuron activation after it spikes.",
        )
        event_decay = st.sidebar.slider(
            "Activation decay per step", 0.0, 1.0, 0.0, 0.05,
            help="Fraction of activation removed at the end of each event step.",
        )
        default_input_neuron = st.sidebar.number_input(
            "Default input neuron",
            min_value=0,
            max_value=max(0, int(network.n_neurons) - 1),
            value=0,
            step=1,
            help="Target neuron for the default external event at step 0.",
        )
        default_input_value = st.sidebar.number_input(
            "Default input value",
            value=1.0,
            step=0.1,
            help="Injected at step 0 if no custom events are provided.",
        )

        event = EventSimulationConfig(
            max_steps=int(event_max_steps),
            threshold=float(event_threshold),
            reset_value=float(event_reset_value),
            decay=float(event_decay),
            default_input_neuron=int(default_input_neuron),
            default_input_value=float(default_input_value),
        )

        duration = float(event.max_steps + 1)
        dt = 1.0
        input_type = "none"
        input_amplitude = 0.0
        noise_level = 0.0
        integration_method = "euler"
        convergence_eps = None

    simulation = SimulationConfig(
        duration=duration,
        dt=dt,
        input_type=input_type,
        input_amplitude=input_amplitude,
        noise_level=noise_level,
        integration_method=integration_method,
        convergence_eps=convergence_eps,
        simulation_type=simulation_type,
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

    config = ProjectConfig(
        network=network,
        simulation=simulation,
        visualization=visualization,
        event=event,
    )

    return config, imported_weight_matrix


# --------------------------------------------------------------------------- #
# Streamlit upload control
# --------------------------------------------------------------------------- #
def read_imported_weight_matrix_from_sidebar():
    """Read optional imported weight matrix from the sidebar."""
    st.sidebar.subheader("Import network")

    use_imported = st.sidebar.checkbox(
        "Use imported weight matrix",
        value=False,
        help=(
            "Upload a square recurrent weight matrix W. "
            "If enabled, this overrides the generated network model."
        ),
    )

    if not use_imported:
        return None

    uploaded_file = st.sidebar.file_uploader(
        "Upload W",
        type=["csv", "npy", "npz"],
        help=(
            "CSV, NPY, or NPZ file containing a square matrix. "
            "Convention: W[i, j] is the connection from neuron j to neuron i."
        ),
    )

    if uploaded_file is None:
        st.sidebar.info("Upload a weight matrix to use import mode.")
        return None

    try:
        W = io_utils.load_weight_matrix_from_upload(
            uploaded_file.name,
            uploaded_file.getvalue(),
        )
    except ValueError as exc:
        st.sidebar.error(f"Could not import network: {exc}")
        st.stop()

    st.sidebar.success(f"Imported W with shape {W.shape[0]} x {W.shape[1]}")
    return W


def _show_continuous_result(result, model_name: str) -> None:
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
        st.plotly_chart(fig, use_container_width=True, config=_PLOT_CONFIG)

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
        st.plotly_chart(act_fig, use_container_width=True, config=_PLOT_CONFIG)

    with tab_anim:
        st.subheader("Activity animation")
        st.caption("Press ▶ Play to watch node activity evolve over time.")
        anim_fig = build_animated_figure(result)
        st.plotly_chart(anim_fig, use_container_width=True, config=_PLOT_CONFIG)

    with tab_pca:
        st.subheader("Animated PCA — state-space trajectory")
        st.caption(
            "The network’s N-dimensional activity vector is projected onto its "
            "first two principal components. The moving dot traces the collective "
            "state through PC-space over time. "
            "🟢 Green stars = stable fixed points · "
            "🟠 Orange diamonds = unstable fixed points."
        )
        pca_fig = build_pca_animation_figure(result)
        st.plotly_chart(pca_fig, use_container_width=True, config=_PLOT_CONFIG)


def _show_event_result(result, model_name: str) -> None:
    """Render event-based/spike-like tabs."""
    tab_graph, tab_raster, tab_activity, tab_anim, tab_pca = st.tabs(
        ["Network graph", "Spike raster", "Activation over time", "Animation", "Animated PCA"]
    )

    with tab_graph:
        st.subheader(f"Network graph — {model_name}")
        fig = build_figure(result)
        st.plotly_chart(fig, use_container_width=True, config=_PLOT_CONFIG)

    with tab_raster:
        st.subheader("Spike raster")
        st.caption("Each marker is one threshold-crossing spike event.")
        raster_fig = build_event_raster_figure(result)
        st.plotly_chart(raster_fig, use_container_width=True, config=_PLOT_CONFIG)

    with tab_activity:
        st.subheader("Activation over event steps")
        act_fig = build_activity_figure(result)
        st.plotly_chart(act_fig, use_container_width=True, config=_PLOT_CONFIG)
        with st.expander("Event log"):
            st.dataframe(result.metadata.get("event_log", []), use_container_width=True)

    with tab_anim:
        st.subheader("Event-state animation")
        anim_fig = build_animated_figure(result)
        st.plotly_chart(anim_fig, use_container_width=True, config=_PLOT_CONFIG)

    with tab_pca:
        st.subheader("Animated PCA — event-state trajectory")
        st.caption(
            "The event simulation activity vector is projected onto the first "
            "two principal components over event steps."
        )
        pca_fig = build_pca_animation_figure(result)
        st.plotly_chart(pca_fig, use_container_width=True, config=_PLOT_CONFIG)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("\U0001f9e0 Neural Network Simulation & Visualization")
    st.caption(
        "Define a network, run a simple recurrent simulation, and explore it "
        "as a graph. Blue = positive weight, red = negative, thickness = strength."
    )

    config, imported_weight_matrix = read_config_from_sidebar()

    try:
        config.validate()
    except ValueError as exc:
        st.error(f"Invalid parameters: {exc}")
        st.stop()

    if st.sidebar.button("\u25b6 Run simulation", type="primary"):
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
        _show_event_result(result, model_name)
    else:
        _show_continuous_result(result, model_name)

    converged_at = result.metadata.get("converged_at")
   
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

    # ---- convergence / method badges ----
    method_used = result.metadata.get("integration_method", "euler")
    badge_col1, badge_col2, _spacer2 = st.columns([1, 1, 6])
    if result.config.simulation.simulation_type == "event_based":
        badge_col1.caption("**Mode:** event-based")
        badge_col2.caption(f"**Spike threshold:** {result.metadata.get('threshold')}")
    else:
        method_used = result.metadata.get("integration_method", "euler")
        badge_col1.caption(f"**Integration:** {_METHOD_LABELS.get(method_used, method_used)}")

        if converged_at is not None:
            badge_col2.caption(f"**Converged at step:** {converged_at}")
        else:
            badge_col2.caption("**Convergence:** not triggered")
    # ---- exports ----
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
