"""Tests for the model library (weight-matrix generation)."""

import numpy as np
import pytest

from nnsimviz.configs import NetworkConfig
from nnsimviz.models import (
    RandomWeightedNetwork,
    ExcitatoryInhibitoryNetwork,
    SymmetricWeightedNetwork,
    ModularNetwork,
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


def test_registry_contains_excitatory_inhibitory_model():
    assert "excitatory_inhibitory" in MODEL_REGISTRY
    assert isinstance(
        get_model("excitatory_inhibitory"),
        ExcitatoryInhibitoryNetwork,
    )


def test_excitatory_inhibitory_model_has_source_based_signs():
    n = 12
    positive_ratio = 0.25

    cfg = NetworkConfig(
        n_neurons=n,
        connection_probability=1.0,
        positive_connection_ratio=positive_ratio,
        model_type="excitatory_inhibitory",
        random_seed=123,
    )

    W = build_weight_matrix(cfg)

    source_signs = []

    for source in range(n):
        outgoing_weights = W[:, source]
        outgoing_weights = np.delete(outgoing_weights, source)

        assert np.all(outgoing_weights > 0) or np.all(outgoing_weights < 0)

        if np.all(outgoing_weights > 0):
            source_signs.append(1)
        else:
            source_signs.append(-1)

    expected_excitatory = int(round(positive_ratio * n))

    assert source_signs.count(1) == expected_excitatory
    assert source_signs.count(-1) == n - expected_excitatory


def test_excitatory_inhibitory_model_has_no_self_connections():
    cfg = NetworkConfig(
        n_neurons=20,
        model_type="excitatory_inhibitory",
        connection_probability=1.0,
    )

    W = build_weight_matrix(cfg)

    assert np.allclose(np.diag(W), 0.0)


def test_excitatory_inhibitory_model_is_reproducible_with_seed():
    cfg = NetworkConfig(
        n_neurons=20,
        model_type="excitatory_inhibitory",
        random_seed=999,
    )

    W1 = build_weight_matrix(cfg)
    W2 = build_weight_matrix(cfg)

    assert np.array_equal(W1, W2)


def test_registry_contains_symmetric_weighted_model():
    assert "symmetric_weighted" in MODEL_REGISTRY
    assert isinstance(get_model("symmetric_weighted"), SymmetricWeightedNetwork)


def test_symmetric_weighted_model_returns_symmetric_matrix():
    cfg = NetworkConfig(
        n_neurons=20,
        connection_probability=0.5,
        model_type="symmetric_weighted",
        random_seed=123,
    )

    W = build_weight_matrix(cfg)

    assert W.shape == (20, 20)
    assert np.allclose(W, W.T)


def test_symmetric_weighted_model_has_no_self_connections():
    cfg = NetworkConfig(
        n_neurons=20,
        connection_probability=1.0,
        model_type="symmetric_weighted",
        random_seed=123,
    )

    W = build_weight_matrix(cfg)

    assert np.allclose(np.diag(W), 0.0)


def test_symmetric_weighted_model_is_reproducible_with_seed():
    cfg = NetworkConfig(
        n_neurons=20,
        connection_probability=0.4,
        model_type="symmetric_weighted",
        random_seed=999,
    )

    W1 = build_weight_matrix(cfg)
    W2 = build_weight_matrix(cfg)

    assert np.array_equal(W1, W2)


def test_symmetric_weighted_model_zero_connection_probability_gives_empty_matrix():
    cfg = NetworkConfig(
        n_neurons=20,
        connection_probability=0.0,
        model_type="symmetric_weighted",
        random_seed=123,
    )

    W = build_weight_matrix(cfg)

    assert np.allclose(W, 0.0)


def test_symmetric_weighted_model_all_positive_when_ratio_is_one():
    cfg = NetworkConfig(
        n_neurons=10,
        connection_probability=1.0,
        positive_connection_ratio=1.0,
        model_type="symmetric_weighted",
        random_seed=123,
    )

    W = build_weight_matrix(cfg)
    off_diagonal = W[~np.eye(W.shape[0], dtype=bool)]

    assert np.all(off_diagonal > 0)


def test_symmetric_weighted_model_all_negative_when_ratio_is_zero():
    cfg = NetworkConfig(
        n_neurons=10,
        connection_probability=1.0,
        positive_connection_ratio=0.0,
        model_type="symmetric_weighted",
        random_seed=123,
    )

    W = build_weight_matrix(cfg)
    off_diagonal = W[~np.eye(W.shape[0], dtype=bool)]

    assert np.all(off_diagonal < 0)


def test_registry_contains_modular_model():
    assert "modular" in MODEL_REGISTRY
    assert isinstance(get_model("modular"), ModularNetwork)


def test_modular_network_has_no_self_connections():
    cfg = NetworkConfig(
        n_neurons=20,
        n_modules=4,
        connection_probability=1.0,
        inter_module_probability=1.0,
        model_type="modular",
        random_seed=123,
    )

    W = build_weight_matrix(cfg)

    assert np.allclose(np.diag(W), 0.0)


def test_modular_network_is_reproducible_with_seed():
    cfg = NetworkConfig(
        n_neurons=24,
        n_modules=4,
        connection_probability=0.8,
        inter_module_probability=0.1,
        model_type="modular",
        random_seed=999,
    )

    W1 = build_weight_matrix(cfg)
    W2 = build_weight_matrix(cfg)

    assert np.array_equal(W1, W2)


def test_modular_network_respects_zero_inter_module_probability():
    cfg = NetworkConfig(
        n_neurons=12,
        n_modules=3,
        connection_probability=1.0,
        inter_module_probability=0.0,
        model_type="modular",
        random_seed=123,
    )

    W = build_weight_matrix(cfg)

    module_ids = np.arange(cfg.n_neurons) % cfg.n_modules
    different_module = module_ids[:, np.newaxis] != module_ids[np.newaxis, :]

    assert np.allclose(W[different_module], 0.0)


def test_modular_network_connects_within_modules_when_probability_is_one():
    cfg = NetworkConfig(
        n_neurons=12,
        n_modules=3,
        connection_probability=1.0,
        inter_module_probability=0.0,
        model_type="modular",
        random_seed=123,
    )

    W = build_weight_matrix(cfg)

    module_ids = np.arange(cfg.n_neurons) % cfg.n_modules
    same_module = module_ids[:, np.newaxis] == module_ids[np.newaxis, :]
    not_diagonal = ~np.eye(cfg.n_neurons, dtype=bool)

    within_module_off_diagonal = same_module & not_diagonal

    assert np.all(W[within_module_off_diagonal] != 0.0)


if __name__ == "__main__":
    test_names = [name for name in globals() if name.startswith("test_")]
    # the -v flag is for verbose output, and the -k flag allows us to specify which tests to run
    pytest.main(["-v", "-k", " or ".join(test_names)])