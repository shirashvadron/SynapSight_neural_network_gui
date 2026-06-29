"""Model library: generates network structure (weight matrices).

Each model takes a NetworkConfig and produces a NumPy weight matrix W of
shape (n_neurons, n_neurons), where W[i, j] is the weight of the directed
connection from neuron j to neuron i (so the simulation can compute W @ x).

The library is kept deliberately small for the hackathon MVP: one concrete
model plus a tiny registry so the GUI can list and select models, and so
new models can be added later without touching GUI or simulation code.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .configs import NetworkConfig


class NetworkModel(Protocol):
    """Protocol every model must follow: turn a config into a weight matrix."""

    name: str
    description: str
    equation: str

    def generate(self, config: NetworkConfig) -> np.ndarray:
        """Return a (n, n) weight matrix from the given network config."""
        ...


class RandomWeightedNetwork:
    """A random recurrent network with signed, weighted connections.

    For each ordered pair (i, j), i != j, a directed connection j -> i exists
    with probability `connection_probability`. Existing connections are
    excitatory (positive) with probability `positive_connection_ratio`,
    otherwise inhibitory (negative). Magnitudes are drawn from a half-normal
    scaled by `weight_scale`. The diagonal (self-connections) is zero.
    """

    name = "Random Weighted Network"
    description = (
        "A randomly connected recurrent network with a configurable "
        "density and balance of excitatory (positive) and inhibitory "
        "(negative) weights."
    )
    equation = "W[i, j] ~ sign * |N(0, weight_scale)|,  P(edge) = p"

    def generate(self, config: NetworkConfig) -> np.ndarray:
        """Build and return the weight matrix W (shape n_neurons x n_neurons)."""
        config.validate()
        n = config.n_neurons
        rng = np.random.default_rng(config.random_seed)

        # 1. Connection mask: which directed edges exist.
        mask = rng.random((n, n)) < config.connection_probability
        np.fill_diagonal(mask, False)  # no self-connections

        # 2. Magnitudes (always positive), scaled.
        magnitudes = np.abs(rng.normal(0.0, config.weight_scale, size=(n, n)))

        # 3. Signs: positive with prob positive_connection_ratio, else negative.
        positive = rng.random((n, n)) < config.positive_connection_ratio
        signs = np.where(positive, 1.0, -1.0)

        # 4. Combine: weight only where an edge exists.
        weights = magnitudes * signs * mask
        return weights


# --------------------------------------------------------------------------- #
# Registry — lets the GUI list available models by key.
# --------------------------------------------------------------------------- #
MODEL_REGISTRY: dict[str, NetworkModel] = {
    "random_weighted": RandomWeightedNetwork(),
}


def get_model(model_type: str) -> NetworkModel:
    """Look up a model instance by its registry key.

    Raises:
        KeyError: if the model_type is not registered.
    """
    if model_type not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type]


def build_weight_matrix(config: NetworkConfig) -> np.ndarray:
    """Convenience: pick the model named in the config and generate W."""
    model = get_model(config.model_type)
    return model.generate(config)
