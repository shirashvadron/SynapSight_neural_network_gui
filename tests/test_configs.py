"""Tests for the configuration dataclasses and their validation."""

import json

import pytest

from nnsimviz.configs import (
    NetworkConfig,
    SimulationConfig,
    VisualizationConfig,
    ProjectConfig,
)


class TestNetworkConfig:
    def test_defaults_are_valid(self):
        NetworkConfig().validate()  # should not raise

    @pytest.mark.parametrize("n", [0, -1, -10])
    def test_non_positive_neurons_rejected(self, n):
        with pytest.raises(ValueError):
            NetworkConfig(n_neurons=n).validate()

    @pytest.mark.parametrize("p", [-0.1, 1.5])
    def test_connection_probability_out_of_range(self, p):
        with pytest.raises(ValueError):
            NetworkConfig(connection_probability=p).validate()

    def test_non_positive_weight_scale_rejected(self):
        with pytest.raises(ValueError):
            NetworkConfig(weight_scale=0).validate()

    @pytest.mark.parametrize("r", [-0.2, 1.1])
    def test_positive_ratio_out_of_range(self, r):
        with pytest.raises(ValueError):
            NetworkConfig(positive_connection_ratio=r).validate()


class TestSimulationConfig:
    def test_defaults_are_valid(self):
        SimulationConfig().validate()

    def test_n_steps(self):
        assert SimulationConfig(duration=10.0, dt=0.1).n_steps == 100

    def test_dt_must_be_smaller_than_duration(self):
        with pytest.raises(ValueError):
            SimulationConfig(duration=1.0, dt=2.0).validate()

    def test_invalid_input_type_rejected(self):
        with pytest.raises(ValueError):
            SimulationConfig(input_type="banana").validate()

    def test_negative_noise_rejected(self):
        with pytest.raises(ValueError):
            SimulationConfig(noise_level=-0.1).validate()


class TestVisualizationConfig:
    def test_defaults_are_valid(self):
        VisualizationConfig().validate()

    def test_invalid_layout_rejected(self):
        with pytest.raises(ValueError):
            VisualizationConfig(layout_type="hexagonal").validate()

    def test_non_positive_edge_width_rejected(self):
        with pytest.raises(ValueError):
            VisualizationConfig(edge_width_scale=0).validate()


class TestProjectConfig:
    def test_validate_cascades(self):
        cfg = ProjectConfig(network=NetworkConfig(n_neurons=-5))
        with pytest.raises(ValueError):
            cfg.validate()

    def test_roundtrip_dict(self):
        cfg = ProjectConfig(network=NetworkConfig(n_neurons=7, random_seed=3))
        restored = ProjectConfig.from_dict(cfg.to_dict())
        assert restored.network.n_neurons == 7
        assert restored.network.random_seed == 3

    def test_roundtrip_json(self):
        cfg = ProjectConfig()
        text = json.dumps(cfg.to_dict())
        restored = ProjectConfig.from_dict(json.loads(text))
        assert restored.simulation.duration == cfg.simulation.duration
