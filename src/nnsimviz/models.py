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


class ExcitatoryInhibitoryNetwork:
    """A recurrent network with source-neuron excitatory/inhibitory identity.

    Unlike RandomWeightedNetwork, where each edge independently chooses its
    sign, this model assigns each source neuron a fixed type.

    If neuron j is excitatory, all outgoing weights W[i, j] are positive.
    If neuron j is inhibitory, all outgoing weights W[i, j] are negative.

    Matrix convention:
        W[i, j] is the connection from source neuron j to target neuron i.
    """

    name = "Excitatory/Inhibitory Network"
    description = (
        "A randomly connected recurrent network where each neuron has a fixed "
        "excitatory or inhibitory identity. Excitatory neurons have positive "
        "outgoing weights; inhibitory neurons have negative outgoing weights."
    )
    equation = "W[i, j] = neuron_sign[j] * |N(0, weight_scale)|"

    def generate(self, config: NetworkConfig) -> np.ndarray:
        """Build and return the weight matrix W."""
        config.validate()
        n = config.n_neurons
        rng = np.random.default_rng(config.random_seed)

        # 1. Connection mask: which directed edges exist.
        mask = rng.random((n, n)) < config.connection_probability
        np.fill_diagonal(mask, False)

        # 2. Magnitudes are always positive.
        magnitudes = np.abs(rng.normal(0.0, config.weight_scale, size=(n, n)))

        # 3. Assign neuron identities.
        # positive_connection_ratio is interpreted here as the fraction
        # of excitatory source neurons.
        n_excitatory = int(round(config.positive_connection_ratio * n))

        source_signs = np.full(n, -1.0)
        excitatory_sources = rng.choice(n, size=n_excitatory, replace=False)
        source_signs[excitatory_sources] = 1.0

        # 4. Apply source-neuron sign by column.
        weights = magnitudes * source_signs[np.newaxis, :] * mask
        return weights


class SymmetricWeightedNetwork:
    """A random weighted network with symmetric reciprocal connections.

    This model creates an undirected-style recurrent network where every
    connection is reciprocal and has the same weight in both directions.

    Matrix convention:
        W[i, j] is the connection from source neuron j to target neuron i.

    Symmetry means:
        W[i, j] == W[j, i]
    """

    name = "Symmetric Weighted Network"
    description = (
        "A random weighted network with reciprocal connections. "
        "Every connection has the same weight in both directions."
    )
    equation = "W[i, j] = W[j, i]"

    def generate(self, config: NetworkConfig) -> np.ndarray:
        """Build and return a symmetric weight matrix."""
        config.validate()

        n = config.n_neurons
        rng = np.random.default_rng(config.random_seed)

        # Sample only the upper triangle, then mirror it.
        connection_mask = rng.random((n, n)) < config.connection_probability
        connection_mask = np.triu(connection_mask, k=1)

        magnitudes = np.abs(
            rng.normal(
                loc=0.0,
                scale=config.weight_scale,
                size=(n, n),
            )
        )

        signs = np.where(
            rng.random((n, n)) < config.positive_connection_ratio,
            1.0,
            -1.0,
        )

        upper_weights = connection_mask * magnitudes * signs

        weights = upper_weights + upper_weights.T
        return weights


class ModularNetwork:
    """A random weighted network with community structure.

    Neurons are divided into modules. Connections within the same module use
    `connection_probability`, while connections between different modules use
    `inter_module_probability`.

    Matrix convention:
        W[i, j] is the connection from source neuron j to target neuron i.
    """

    name = "Modular Network"
    description = (
        "A community-structured recurrent network. Neurons are divided into "
        "modules with dense within-module connectivity and sparse between-module "
        "connectivity."
    )
    equation = "P(edge) = p_in if same module else p_out"

    def generate(self, config: NetworkConfig) -> np.ndarray:
        """Build and return a modular weight matrix."""
        config.validate()

        n = config.n_neurons
        rng = np.random.default_rng(config.random_seed)

        module_ids = np.arange(n) % config.n_modules

        same_module = module_ids[:, np.newaxis] == module_ids[np.newaxis, :]

        connection_probabilities = np.where(
            same_module,
            config.connection_probability,
            config.inter_module_probability,
        )

        connection_mask = rng.random((n, n)) < connection_probabilities
        np.fill_diagonal(connection_mask, False)

        magnitudes = np.abs(
            rng.normal(
                loc=0.0,
                scale=config.weight_scale,
                size=(n, n),
            )
        )

        signs = np.where(
            rng.random((n, n)) < config.positive_connection_ratio,
            1.0,
            -1.0,
        )

        return connection_mask * magnitudes * signs


# --------------------------------------------------------------------------- #
# Registry — lets the GUI list available models by key.
# --------------------------------------------------------------------------- #
MODEL_REGISTRY: dict[str, NetworkModel] = {
    "random_weighted": RandomWeightedNetwork(),
    "excitatory_inhibitory": ExcitatoryInhibitoryNetwork(),
    "symmetric_weighted": SymmetricWeightedNetwork(),
    "modular": ModularNetwork(),
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
