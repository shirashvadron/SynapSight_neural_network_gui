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
from nnsimviz.motifs import MotifConfig, MOTIF_LABELS, MotifType
from nnsimviz.motif_icons import motif_icon_svg

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


from pathlib import Path

# Repo-root paths for brand assets (logo + favicon live at the repo root,
# while this app runs from src/nnsimviz/, so we resolve up two parents).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOGO_PATH = _REPO_ROOT / "SynapSight.png"
_FAVICON_PATH = _REPO_ROOT / "SynapSightfavicon.png"

_page_icon = str(_FAVICON_PATH) if _FAVICON_PATH.exists() else "\U0001f9e0"
st.set_page_config(
    page_title="SynapSight",
    page_icon=_page_icon,
    layout="wide",
)

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
    /* Bold + larger sidebar section headers (the collapsible expanders:
       Network, Motifs, Simulation, Visualization). */
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
    section[data-testid="stSidebar"] details summary p {
        font-weight: 700;
        font-size: 1.15rem;
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
        "**Simulation type**",
        VALID_SIMULATION_TYPES,
        index=0,
        format_func=lambda k: _SIMULATION_TYPE_LABELS.get(k, k),
        help=(
            "Choose the simulation engine: original continuous dynamics or "
            "event-based spike propagation."
        ),
    )

    # ---- Network (collapsible, starts closed) ----
    net = st.sidebar.expander("Network", expanded=False)
    model_keys = list(MODEL_REGISTRY)
    model_type = net.selectbox(
        "Model", model_keys,
        format_func=lambda k: MODEL_REGISTRY[k].name,
        help=get_help_texts("").model,
    )

    txt = get_help_texts(model_type)

    n_neurons = net.slider(
        "Number of neurons", 2, 100, 20, help=txt.n_neurons,
    )
    connection_probability = net.slider(
        "Connection probability", 0.0, 1.0, 0.3, 0.05,
        help=txt.connection_probability,
    )
    weight_scale = net.slider(
        "Weight scale", 0.1, 3.0, 1.0, 0.1, help=txt.weight_scale,
    )
    positive_connection_ratio = net.slider(
        "Positive / excitatory ratio", 0.0, 1.0, 0.7, 0.05,
        help=txt.positive_connection_ratio,
    )
    random_seed = net.number_input(
        "Random seed", 0, 10_000, 42, 1, help=txt.random_seed,
    )

    n_modules = 4
    inter_module_probability = 0.05

    if model_type == "modular":
        net.subheader("Modular network")
        n_modules = net.slider(
            "Number of modules", 1, max(1, int(n_neurons)), min(4, int(n_neurons)), 1,
            help=txt.n_modules,
        )
        inter_module_probability = net.slider(
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

    # ---- Motifs ----
    motifs = read_motif_config_from_sidebar()

    # ---- Simulation (collapsible, starts closed) ----
    sim = st.sidebar.expander("Simulation", expanded=False)
    event = EventSimulationConfig()

    if simulation_type == "continuous":
        duration = sim.slider(
            "Duration", 1.0, 50.0, 10.0, 1.0, help=txt.duration,
        )
        dt = sim.slider(
            "Time step (dt)", 0.01, 1.0, 0.1, 0.01, help=txt.dt,
        )
        input_type = sim.selectbox(
            "Input type", SimulationConfig.VALID_INPUT_TYPES, index=1,
            help=txt.input_type,
        )
        input_amplitude = sim.slider(
            "Input amplitude", 0.0, 5.0, 1.0, 0.1, help=txt.input_amplitude,
        )
        noise_level = sim.slider(
            "Noise level", 0.0, 1.0, 0.05, 0.01, help=txt.noise_level,
        )

        integration_method = sim.selectbox(
            "Integration method",
            list(VALID_INTEGRATION_METHODS),
            index=0,
            format_func=lambda k: _METHOD_LABELS.get(k, k),
            help=txt.integration_method,
        )

        use_convergence = sim.checkbox(
            "Enable early convergence stop",
            value=False,
            help=txt.convergence_eps,
        )
        convergence_eps: float | None = None
        if use_convergence:
            convergence_eps = sim.number_input(
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
        sim.caption(
            "Event mode uses the same network/imported W, but updates mainly "
            "when spike events occur."
        )
        event_max_steps = sim.slider(
            "Event steps", 1, 200, 20, 1,
            help="Number of discrete event steps to simulate.",
        )
        event_threshold = sim.number_input(
            "Spike threshold", min_value=0.001, value=1.0, step=0.1,
            help="A neuron emits a spike when activation is at or above this value.",
        )
        event_reset_value = sim.number_input(
            "Reset value", value=0.0, step=0.1,
            help="Neuron activation after it spikes.",
        )
        event_decay = sim.slider(
            "Activation decay per step", 0.0, 1.0, 0.0, 0.05,
            help="Fraction of activation removed at the end of each event step.",
        )
        default_input_neuron = sim.number_input(
            "Default input neuron",
            min_value=0,
            max_value=max(0, int(network.n_neurons) - 1),
            value=0,
            step=1,
            help="Target neuron for the default external event at step 0.",
        )
        default_input_value = sim.number_input(
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
    viz = st.sidebar.expander("Visualization", expanded=False)
    layout_type = viz.selectbox(
        "Layout", AVAILABLE_LAYOUTS,
        format_func=lambda k: _LAYOUT_LABELS.get(k, k),
        help=txt.layout,
    )
    show_labels = viz.checkbox(
        "Show neuron labels", True, help=txt.show_labels,
    )
    edge_width_scale = viz.slider(
        "Edge width scale", 1.0, 10.0, 3.0, 0.5, help=txt.edge_width_scale,
    )
    min_edge_abs_weight = viz.slider(
        "Min edge |weight| to show", 0.0, 3.0, 0.0, 0.1,
        help=txt.min_edge_abs_weight,
    )
    node_size_scale = viz.slider(
        "Node size scale", 0.5, 3.0, 1.0, 0.1, help=txt.node_size_scale,
    )
    show_activity_on_nodes = viz.checkbox(
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
        motifs=motifs,
    )

    return config, imported_weight_matrix


def _motif_number_input(container, motif_type: MotifType, default: int,
                        help_text: str) -> int:
    """Render a motif's schematic icon + a count number input, side by side.

    The small SVG icon (green = excitatory, red = inhibitory) shows the motif's
    structure at a glance; the number input sets how many to add. Rendered
    inside the given container (the collapsible Motifs expander).
    """
    label = MOTIF_LABELS[motif_type]
    svg = motif_icon_svg(motif_type.value)
    icon_col, input_col = container.columns([1, 2], vertical_alignment="center")
    with icon_col:
        st.markdown(
            f'<div title="{label}" style="display:flex;align-items:center;'
            f'height:100%;">{svg}</div>',
            unsafe_allow_html=True,
        )
    with input_col:
        return int(st.number_input(
            f"{label} count",
            min_value=0, max_value=50, value=default, step=1,
            help=help_text,
        ))


def read_motif_config_from_sidebar() -> MotifConfig:
    """Read the Motifs section of the sidebar into a MotifConfig.

    Kept as a small, self-contained reader: it only gathers widget values and
    returns a config. All motif-building logic lives in the motifs module.
    The whole section is a collapsible expander that starts closed.
    """
    mot = st.sidebar.expander("Motifs", expanded=False)
    enabled = mot.checkbox(
        "Add motifs", value=False,
        help="Append small repeated connectivity patterns (motifs) as extra "
             "neurons on top of the base network.",
    )

    if not enabled:
        return MotifConfig(enabled=False)

    n_coincidence = _motif_number_input(
        mot, MotifType.COINCIDENCE_DETECTOR, default=1,
        help_text="Several excitatory neurons converge (positive edges) onto "
                  "one target neuron.")
    n_lateral = _motif_number_input(
        mot, MotifType.LATERAL_INHIBITION, default=0,
        help_text="Excitatory neurons that mutually inhibit each other through "
                  "negative edges (competition).")
    n_feedback = _motif_number_input(
        mot, MotifType.NEGATIVE_FEEDBACK_LOOP, default=0,
        help_text="An excitatory neuron drives another, which sends negative "
                  "feedback back, forming a regulatory loop.")
    n_ffl = _motif_number_input(
        mot, MotifType.FEEDFORWARD_LOOP, default=0,
        help_text="A drives C both directly and via B. Acts as a filter that "
                  "passes persistent signals and ignores brief ones.")
    n_ffi = _motif_number_input(
        mot, MotifType.FEEDFORWARD_INHIBITION, default=0,
        help_text="A excites a target and, via an inhibitory interneuron, also "
                  "inhibits it -- creating a narrow timing window.")
    n_mutex = _motif_number_input(
        mot, MotifType.MUTUAL_EXCITATION, default=0,
        help_text="Two neurons that excite each other, latching into a "
                  "sustained 'on' state (bistability / simple memory).")
    strength = mot.slider(
        "Motif connection strength", 0.1, 3.0, 1.0, 0.1,
        help="Scale of the motif edge weights (their signs come from the "
             "motif structure).",
    )
    n_external = mot.slider(
        "External connections per motif", 0, 10, 2, 1,
        help="How many edges link each motif to the base network.",
    )

    return MotifConfig(
        enabled=True,
        n_coincidence_detector=int(n_coincidence),
        n_lateral_inhibition=int(n_lateral),
        n_negative_feedback_loop=int(n_feedback),
        n_feedforward_loop=int(n_ffl),
        n_feedforward_inhibition=int(n_ffi),
        n_mutual_excitation=int(n_mutex),
        connection_strength=strength,
        n_external_connections=int(n_external),
        random_seed=42,
    )


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
    if _LOGO_PATH.exists():
        st.image(str(_LOGO_PATH), width=500)
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
    config, imported_weight_matrix = read_config_from_sidebar()

    try:
        config.validate()
    except ValueError as exc:
        st.error(f"Invalid parameters: {exc}")
        st.stop()

    if st.sidebar.button("\u25b6 Run simulation", type="primary",
                         use_container_width=True):
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


if __name__ == "__main__":
    main()