"""Sidebar controls that build ProjectConfig objects."""

from __future__ import annotations

import streamlit as st

from nnsimviz import io_utils
from nnsimviz.configs import (
    EventSimulationConfig,
    NetworkConfig,
    ProjectConfig,
    SimulationConfig,
    VALID_INTEGRATION_METHODS,
    VALID_SIMULATION_TYPES,
    VisualizationConfig,
)
from nnsimviz.help_texts import get_help_texts
from nnsimviz.models import MODEL_REGISTRY
from nnsimviz.motif_icons import motif_icon_svg
from nnsimviz.motifs import MOTIF_LABELS, MotifConfig, MotifType
from nnsimviz.ui.constants import (
    LAYOUT_LABELS,
    METHOD_LABELS,
    SIMULATION_TYPE_LABELS,
)
from nnsimviz.visualization import AVAILABLE_LAYOUTS


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
        format_func=lambda k: SIMULATION_TYPE_LABELS.get(k, k),
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

    imported_weight_matrix = read_imported_weight_matrix_from_sidebar(net)

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
            format_func=lambda k: METHOD_LABELS.get(k, k),
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
        format_func=lambda k: LAYOUT_LABELS.get(k, k),
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
def read_imported_weight_matrix_from_sidebar(container):
    """Read optional imported weight matrix from the Network sidebar expander."""
    container.divider()
    container.subheader("Import network")

    use_imported = container.checkbox(
        "Use imported weight matrix",
        value=False,
        help=(
            "Upload a square recurrent weight matrix W. "
            "If enabled, this overrides the generated network model."
        ),
    )

    if not use_imported:
        return None

    uploaded_file = container.file_uploader(
        "Upload W",
        type=["csv", "npy", "npz"],
        help=(
            "CSV, NPY, or NPZ file containing a square matrix. "
            "Convention: W[i, j] is the connection from neuron j to neuron i."
        ),
    )

    if uploaded_file is None:
        container.info("Upload a weight matrix to use import mode.")
        return None

    try:
        W = io_utils.load_weight_matrix_from_upload(
            uploaded_file.name,
            uploaded_file.getvalue(),
        )
    except ValueError as exc:
        container.error(f"Could not import network: {exc}")
        st.stop()

    container.success(f"Imported W with shape {W.shape[0]} x {W.shape[1]}")
    return W