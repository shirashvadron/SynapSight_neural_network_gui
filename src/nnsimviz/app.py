"""Streamlit GUI: a thin orchestrator over the pipeline modules.

This layer only reads widget values, builds a ProjectConfig, and calls the
model -> simulation -> visualization modules in sequence, then displays the
figure and a short summary. It contains NO graph-building or simulation
logic of its own.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from nnsimviz.configs import (
    NetworkConfig,
    ProjectConfig,
    SimulationConfig,
    VisualizationConfig,
)
from nnsimviz.models import MODEL_REGISTRY, build_weight_matrix
from nnsimviz.simulation import Simulator
from nnsimviz.visualization import build_figure, NetworkVisualizer
from nnsimviz import io_utils


st.set_page_config(page_title="Neural Network Simulator", layout="wide")


# --------------------------------------------------------------------------- #
# Sidebar controls -> ProjectConfig
# --------------------------------------------------------------------------- #
def read_config_from_sidebar() -> ProjectConfig:
    """Read every sidebar widget and assemble a ProjectConfig."""
    st.sidebar.title("Parameters")

    # ---- Network ----
    st.sidebar.header("Network")
    model_keys = list(MODEL_REGISTRY)
    model_type = st.sidebar.selectbox(
        "Model", model_keys,
        format_func=lambda k: MODEL_REGISTRY[k].name,
    )
    n_neurons = st.sidebar.slider("Number of neurons", 2, 100, 20)
    connection_probability = st.sidebar.slider("Connection probability", 0.0, 1.0, 0.3, 0.05)
    weight_scale = st.sidebar.slider("Weight scale", 0.1, 3.0, 1.0, 0.1)
    positive_connection_ratio = st.sidebar.slider(
    "Positive / excitatory ratio",
    0.0,        
    1.0,
    0.7,
    0.05,
    help=(
        "For Random Weighted Network: fraction of positive edges. "
        "For Excitatory/Inhibitory Network: fraction of excitatory neurons."
        ),
    )
    random_seed = st.sidebar.number_input("Random seed", 0, 10_000, 42, 1)

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

    n_modules = st.sidebar.slider(
        "Number of modules",
        min_value=1,
        max_value=max(1, n_neurons),
        value=min(4, n_neurons),
        step=1,
        help="Used by the Modular Network model.",
    )

    inter_module_probability = st.sidebar.slider(
        "Inter-module connection probability",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        help="Used by the Modular Network model. Lower values create more separated modules.",
    )

    # ---- Simulation ----
    st.sidebar.header("Simulation")
    duration = st.sidebar.slider("Duration", 1.0, 50.0, 10.0, 1.0)
    dt = st.sidebar.slider("Time step (dt)", 0.01, 1.0, 0.1, 0.01)
    input_type = st.sidebar.selectbox("Input type", SimulationConfig.VALID_INPUT_TYPES, index=1)
    input_amplitude = st.sidebar.slider("Input amplitude", 0.0, 5.0, 1.0, 0.1)
    noise_level = st.sidebar.slider("Noise level", 0.0, 1.0, 0.05, 0.01)

    simulation = SimulationConfig(
        duration=duration,
        dt=dt,
        input_type=input_type,
        input_amplitude=input_amplitude,
        noise_level=noise_level,
    )

    # ---- Visualization ----
    st.sidebar.header("Visualization")
    layout_type = st.sidebar.selectbox("Layout", VisualizationConfig.VALID_LAYOUTS)
    show_labels = st.sidebar.checkbox("Show neuron labels", True)
    edge_width_scale = st.sidebar.slider("Edge width scale", 1.0, 10.0, 3.0, 0.5)
    min_edge_abs_weight = st.sidebar.slider("Min edge |weight| to show", 0.0, 3.0, 0.0, 0.1)
    node_size_scale = st.sidebar.slider("Node size scale", 0.5, 3.0, 1.0, 0.1)
    show_activity_on_nodes = st.sidebar.checkbox("Color nodes by activity", True)

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
    st.title("🧠 Neural Network Simulation & Visualization")
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

    if st.sidebar.button("▶ Run simulation", type="primary"):
        with st.spinner("Running simulation..."):
            result = run_pipeline(config)
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if result is None:
        st.info("Set your parameters in the sidebar and press **Run simulation**.")
        return

    fig = build_figure(result)
    st.plotly_chart(fig, use_container_width=True)

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
    e1, e2, e3 = st.columns(3)
    e1.download_button("Config (JSON)", io_utils.config_to_json(result.config),
                       "config.json", "application/json")
    e2.download_button("Weights (CSV)", io_utils.weight_matrix_to_csv(result.weight_matrix),
                       "weights.csv", "text/csv")
    e3.download_button("Activity (CSV)", io_utils.activity_to_csv(result),
                       "activity.csv", "text/csv")


if __name__ == "__main__":
    main()
