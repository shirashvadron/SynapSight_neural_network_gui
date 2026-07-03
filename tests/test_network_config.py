
"""Unit tests for NetworkConfig validation.

These tests define the public contract of NetworkConfig:
valid network parameters should validate cleanly, and invalid parameter
ranges should fail with clear ValueError messages.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from nnsimviz.configs import NetworkConfig


def test_network_config_defaults_are_valid() -> None:
    """The default config should be usable without manual changes."""
    config = NetworkConfig()

    config.validate()

    assert config.n_neurons == 20
    assert config.connection_probability == pytest.approx(0.3)
    assert config.weight_scale == pytest.approx(1.0)
    assert config.positive_connection_ratio == pytest.approx(0.7)
    assert config.model_type == "random_weighted"
    assert config.random_seed == 42


@pytest.mark.parametrize(
    "config",
    [
        NetworkConfig(n_neurons=1),
        NetworkConfig(connection_probability=0.0),
        NetworkConfig(connection_probability=1.0),
        NetworkConfig(positive_connection_ratio=0.0),
        NetworkConfig(positive_connection_ratio=1.0),
        NetworkConfig(weight_scale=1e-12),
    ],
    ids=[
        "single-neuron-network",
        "zero-connection-probability",
        "full-connection-probability",
        "all-negative-connections",
        "all-positive-connections",
        "very-small-positive-weight-scale",
    ],
)
def test_network_config_accepts_valid_boundary_values(config: NetworkConfig) -> None:
    """Boundary values are part of the valid API and should remain valid."""
    config.validate()


@pytest.mark.parametrize("n_neurons", [0, -1, -100])
def test_network_config_rejects_non_positive_neuron_count(n_neurons: int) -> None:
    """A network must contain at least one neuron."""
    with pytest.raises(ValueError, match="n_neurons"):
        NetworkConfig(n_neurons=n_neurons).validate()


@pytest.mark.parametrize("connection_probability", [-0.01, 1.01])
def test_network_config_rejects_connection_probability_outside_unit_interval(
    connection_probability: float,
) -> None:
    """Connection probability is a probability, so it must be in [0, 1]."""
    with pytest.raises(ValueError, match="connection_probability"):
        NetworkConfig(connection_probability=connection_probability).validate()


@pytest.mark.parametrize("weight_scale", [0.0, -0.01, -10.0])
def test_network_config_rejects_non_positive_weight_scale(weight_scale: float) -> None:
    """Weight scale controls magnitudes and must be strictly positive."""
    with pytest.raises(ValueError, match="weight_scale"):
        NetworkConfig(weight_scale=weight_scale).validate()


@pytest.mark.parametrize("positive_connection_ratio", [-0.01, 1.01])
def test_network_config_rejects_positive_connection_ratio_outside_unit_interval(
    positive_connection_ratio: float,
) -> None:
    """The positive connection ratio is a fraction, so it must be in [0, 1]."""
    with pytest.raises(ValueError, match="positive_connection_ratio"):
        NetworkConfig(positive_connection_ratio=positive_connection_ratio).validate()


def test_network_config_is_plain_data_and_serializable() -> None:
    """NetworkConfig should remain easy to convert to a JSON-compatible dict."""
    config = NetworkConfig(
        n_neurons=7,
        connection_probability=0.25,
        weight_scale=2.5,
        positive_connection_ratio=0.4,
        model_type="random_weighted",
        random_seed=123,
    )

    config.validate()

    assert asdict(config) == {
        "n_neurons": 7,
        "connection_probability": 0.25,
        "weight_scale": 2.5,
        "positive_connection_ratio": 0.4,
        "model_type": "random_weighted",
        "random_seed": 123,
    }