"""Tests for the simulation engine and SimulationResult."""

import numpy as np
import pytest

from nnsimviz.configs import (
    NetworkConfig,
    SimulationConfig,
    ProjectConfig,
)
from nnsimviz.models import build_weight_matrix
from nnsimviz.simulation import Simulator, SimulationResult


def _make_config(n=10, **sim_kwargs):
    return ProjectConfig(
        network=NetworkConfig(n_neurons=n, random_seed=42),
        simulation=SimulationConfig(**sim_kwargs),
    )


def test_result_shapes():
    cfg = _make_config(n=10, duration=5.0, dt=0.1)
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert isinstance(result, SimulationResult)
    assert result.activity.shape == (10, cfg.simulation.n_steps)
    assert result.final_state.shape == (10,)
    assert result.time.shape == (cfg.simulation.n_steps,)


def test_final_state_matches_last_column():
    cfg = _make_config(n=8)
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert np.array_equal(result.final_state, result.activity[:, -1])


def test_simulation_is_finite():
    cfg = _make_config(n=20, duration=20.0, dt=0.1, noise_level=0.1)
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert np.isfinite(result.activity).all()


def test_clipping_prevents_explosion():
    # Large weights would blow up without clipping.
    cfg = _make_config(n=30, duration=30.0, dt=0.5)
    cfg.network.weight_scale = 3.0
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert np.abs(result.activity).max() <= Simulator.CLIP


@pytest.mark.parametrize("input_type", SimulationConfig.VALID_INPUT_TYPES)
def test_all_input_types_run(input_type):
    cfg = _make_config(n=10, input_type=input_type)
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert np.isfinite(result.activity).all()


def test_metadata_populated():
    cfg = _make_config(n=14)
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert result.metadata["n_neurons"] == 14
    assert result.metadata["n_steps"] == cfg.simulation.n_steps


def test_shape_mismatch_raises():
    cfg = _make_config(n=10)
    wrong_W = np.zeros((5, 5))
    with pytest.raises(ValueError):
        Simulator().run(cfg, wrong_W)


def test_reproducible_runs():
    cfg = _make_config(n=12)
    W = build_weight_matrix(cfg.network)
    r1 = Simulator().run(cfg, W)
    r2 = Simulator().run(cfg, W)
    assert np.array_equal(r1.activity, r2.activity)
