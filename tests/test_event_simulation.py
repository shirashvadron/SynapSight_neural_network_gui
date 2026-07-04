
"""Tests for event-based/spike-like simulation."""

import numpy as np
import plotly.graph_objects as go

from nnsimviz.configs import (
    EventSimulationConfig,
    NetworkConfig,
    ProjectConfig,
    SimulationConfig,
)
from nnsimviz.event_simulation import EventBasedSimulator
from nnsimviz.visualization import build_event_raster_figure


def _event_config(**event_kwargs):
    return ProjectConfig(
        network=NetworkConfig(n_neurons=3, random_seed=7),
        simulation=SimulationConfig(simulation_type="event_based"),
        event=EventSimulationConfig(max_steps=5, threshold=1.0, **event_kwargs),
    )


def test_threshold_crossing_creates_spike():
    W = np.zeros((3, 3))
    cfg = _event_config(default_input_neuron=0, default_input_value=1.0)
    result = EventBasedSimulator().run(cfg, W)
    assert result.metadata["spike_times"] == [(0, 0)]
    assert result.metadata["spike_counts"][0] == 1
    assert result.metadata["spike_train"][0, 0] == 1


def test_spike_recording_preserves_pre_reset_activation():
    W = np.zeros((3, 3))
    cfg = _event_config(default_input_neuron=0, default_input_value=1.0)
    result = EventBasedSimulator().run(cfg, W)

    pre_reset = result.metadata["activation_before_reset"]
    spike_train = result.metadata["spike_train"]

    assert pre_reset.shape == result.activity.shape
    assert spike_train.shape == result.activity.shape
    assert pre_reset[0, 0] == 1.0
    assert spike_train[0, 0] == 1
    assert result.activity[0, 0] == 0.0


def test_downstream_spike_is_visible_before_reset():
    W = np.zeros((3, 3))
    W[1, 0] = 1.2
    cfg = _event_config(default_input_neuron=0, default_input_value=1.0)
    result = EventBasedSimulator().run(cfg, W)

    assert result.metadata["activation_before_reset"][1, 1] == 1.2
    assert result.metadata["spike_train"][1, 1] == 1
    assert result.activity[1, 1] == 0.0


def test_excitatory_connection_increases_target_activation():
    W = np.zeros((3, 3))
    W[1, 0] = 0.6  # source 0 -> target 1
    cfg = _event_config(default_input_neuron=0, default_input_value=1.0)
    result = EventBasedSimulator().run(cfg, W)
    assert result.activity[1, 1] == 0.6


def test_inhibitory_connection_decreases_target_activation():
    W = np.zeros((3, 3))
    W[2, 0] = -0.4  # source 0 -> target 2
    cfg = _event_config(default_input_neuron=0, default_input_value=1.0)
    result = EventBasedSimulator().run(cfg, W)
    assert result.activity[2, 1] == -0.4


def test_spike_propagation_can_trigger_downstream_spike():
    W = np.zeros((3, 3))
    W[1, 0] = 1.2  # source 0 -> target 1, enough to spike next step
    cfg = _event_config(default_input_neuron=0, default_input_value=1.0)
    result = EventBasedSimulator().run(cfg, W)
    assert (0, 0) in result.metadata["spike_times"]
    assert (1, 1) in result.metadata["spike_times"]


def test_external_events_override_default_input():
    W = np.zeros((3, 3))
    cfg = _event_config(external_events=[(2, 2, 1.5)])
    result = EventBasedSimulator().run(cfg, W)
    assert result.metadata["spike_times"] == [(2, 2)]


def test_event_simulation_is_deterministic():
    W = np.zeros((3, 3))
    W[1, 0] = 0.5
    cfg = _event_config(default_input_neuron=0, default_input_value=1.0)
    sim = EventBasedSimulator()
    r1 = sim.run(cfg, W)
    r2 = sim.run(cfg, W)
    np.testing.assert_array_equal(r1.activity, r2.activity)
    assert r1.metadata["spike_times"] == r2.metadata["spike_times"]


def test_event_raster_visualization_returns_figure():
    W = np.zeros((3, 3))
    cfg = _event_config(default_input_neuron=0, default_input_value=1.0)
    result = EventBasedSimulator().run(cfg, W)
    assert isinstance(build_event_raster_figure(result), go.Figure)