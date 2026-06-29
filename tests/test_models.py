"""Tests for the model library (weight-matrix generation)."""

import numpy as np
import pytest

from nnsimviz.configs import NetworkConfig
from nnsimviz.models import (
    RandomWeightedNetwork,
    MODEL_REGISTRY,
    get_model,
    build_weight_matrix,
)


def test_weight_matrix_shape():
    cfg = NetworkConfig(n_neurons=12)
    W = build_weight_matrix(cfg)
    assert W.shape == (12, 12)


def test_no_self_connections():
    W = build_weight_matrix(NetworkConfig(n_neurons=15))
    assert np.allclose(np.diag(W), 0.0)


def test_reproducible_with_seed():
    cfg = NetworkConfig(n_neurons=10, random_seed=123)
    W1 = build_weight_matrix(cfg)
    W2 = build_weight_matrix(cfg)
    assert np.array_equal(W1, W2)


def test_different_seeds_differ():
    W1 = build_weight_matrix(NetworkConfig(n_neurons=10, random_seed=1))
    W2 = build_weight_matrix(NetworkConfig(n_neurons=10, random_seed=2))
    assert not np.array_equal(W1, W2)


def test_zero_probability_gives_empty_network():
    W = build_weight_matrix(NetworkConfig(n_neurons=10, connection_probability=0.0))
    assert np.count_nonzero(W) == 0


def test_full_probability_connects_all_off_diagonal():
    n = 8
    W = build_weight_matrix(NetworkConfig(n_neurons=n, connection_probability=1.0))
    off_diagonal = n * n - n
    assert np.count_nonzero(W) == off_diagonal


def test_all_positive_ratio_gives_no_negative_weights():
    W = build_weight_matrix(NetworkConfig(
        n_neurons=20, connection_probability=1.0, positive_connection_ratio=1.0))
    assert (W >= 0).all()


def test_all_negative_ratio_gives_no_positive_weights():
    W = build_weight_matrix(NetworkConfig(
        n_neurons=20, connection_probability=1.0, positive_connection_ratio=0.0))
    assert (W <= 0).all()


def test_registry_contains_default_model():
    assert "random_weighted" in MODEL_REGISTRY
    assert isinstance(get_model("random_weighted"), RandomWeightedNetwork)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model("does_not_exist")


def test_invalid_config_propagates():
    with pytest.raises(ValueError):
        build_weight_matrix(NetworkConfig(n_neurons=0))
