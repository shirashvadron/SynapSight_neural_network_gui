"""Integration tests for simulation-type dispatch and project contracts."""

import numpy as np
import plotly.graph_objects as go

from nnsimviz.pipeline import run_pipeline
from nnsimviz.configs import (
    EventSimulationConfig,
    NetworkConfig,
    ProjectConfig,
    SimulationConfig,
    VisualizationConfig,
)
from nnsimviz.visualization import build_figure, build_event_raster_figure


def test_run_pipeline_keeps_continuous_mode_working():
    cfg = ProjectConfig(
        network=NetworkConfig(n_neurons=8, random_seed=0),
        simulation=SimulationConfig(
            simulation_type="continuous",
            duration=1.0,
            dt=0.1,
            noise_level=0.0,
        ),
        visualization=VisualizationConfig(layout_type="circular"),
    )
    result = run_pipeline(cfg)
    assert result.metadata["model_type"] == "random_weighted"
    assert result.weight_matrix.shape == (8, 8)
    assert isinstance(build_figure(result), go.Figure)


def test_run_pipeline_dispatches_event_based_mode():
    cfg = ProjectConfig(
        network=NetworkConfig(n_neurons=3, random_seed=0),
        simulation=SimulationConfig(simulation_type="event_based"),
        visualization=VisualizationConfig(layout_type="circular"),
        event=EventSimulationConfig(max_steps=4, threshold=1.0),
    )
    W = np.zeros((3, 3))
    W[1, 0] = 1.2
    result = run_pipeline(cfg, imported_weight_matrix=W)
    assert result.metadata["simulation_type"] == "event_based"
    assert result.metadata["network_source"] == "imported"
    assert result.weight_matrix.shape == (3, 3)
    assert (0, 0) in result.metadata["spike_times"]
    assert (1, 1) in result.metadata["spike_times"]
    assert isinstance(build_event_raster_figure(result), go.Figure)
