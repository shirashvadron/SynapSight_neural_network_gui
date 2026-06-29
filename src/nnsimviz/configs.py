"""Configuration objects for the neural-network simulation GUI.

This module is the single source of truth for all parameters that flow
between the GUI, model, simulation, and visualization layers. Every other
module agrees on the field names defined here; if an output does not match
these shapes, that is a bug to fix rather than a reason to add a converter.

All configs are plain dataclasses with light validation, so they can be
created from GUI widgets, serialized to JSON, and passed around freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
@dataclass
class NetworkConfig:
    """Parameters that define the network structure (the weight matrix).

    Attributes:
        n_neurons: Number of neurons (nodes) in the network. Must be > 0.
        connection_probability: Probability that any directed pair (i, j)
            has a connection. In [0, 1].
        weight_scale: Scale of the random connection weights. Must be > 0.
        positive_connection_ratio: Fraction of existing connections that are
            excitatory (positive). In [0, 1]; the rest are inhibitory.
        model_type: Identifier of the model in the model library.
        random_seed: Seed for reproducible weight generation.
    """

    n_neurons: int = 20
    connection_probability: float = 0.3
    weight_scale: float = 1.0
    positive_connection_ratio: float = 0.7
    model_type: str = "random_weighted"
    random_seed: int = 42

    def validate(self) -> None:
        """Raise ValueError if any field is out of its valid range."""
        if self.n_neurons <= 0:
            raise ValueError("n_neurons must be a positive integer.")
        if not 0.0 <= self.connection_probability <= 1.0:
            raise ValueError("connection_probability must be in [0, 1].")
        if self.weight_scale <= 0:
            raise ValueError("weight_scale must be positive.")
        if not 0.0 <= self.positive_connection_ratio <= 1.0:
            raise ValueError("positive_connection_ratio must be in [0, 1].")


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass
class SimulationConfig:
    """Parameters that control the forward-in-time network dynamics.

    Attributes:
        duration: Total simulated time (arbitrary units). Must be > 0.
        dt: Integration time step. Must be > 0 and < duration.
        input_type: One of {"none", "constant", "noise", "sine"}.
        input_amplitude: Amplitude of the external input signal.
        noise_level: Standard deviation of additive Gaussian process noise.
    """

    duration: float = 10.0
    dt: float = 0.1
    input_type: str = "constant"
    input_amplitude: float = 1.0
    noise_level: float = 0.05

    VALID_INPUT_TYPES = ("none", "constant", "noise", "sine")

    def validate(self) -> None:
        """Raise ValueError if any field is out of its valid range."""
        if self.duration <= 0:
            raise ValueError("duration must be positive.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.dt >= self.duration:
            raise ValueError("dt must be smaller than duration.")
        if self.input_type not in self.VALID_INPUT_TYPES:
            raise ValueError(
                f"input_type must be one of {self.VALID_INPUT_TYPES}."
            )
        if self.noise_level < 0:
            raise ValueError("noise_level must be non-negative.")

    @property
    def n_steps(self) -> int:
        """Number of time steps in the simulation."""
        return int(round(self.duration / self.dt))


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
@dataclass
class VisualizationConfig:
    """Parameters that control how the network graph is drawn.

    Attributes:
        layout_type: One of {"spring", "circular"}.
        show_labels: Whether to draw neuron index labels on nodes.
        edge_width_scale: Multiplier mapping abs(weight) -> line width.
        min_edge_abs_weight: Edges with abs(weight) below this are hidden.
        node_size_scale: Multiplier mapping activity -> node marker size.
        show_activity_on_nodes: Color/size nodes by final activity if True.
    """

    layout_type: str = "spring"
    show_labels: bool = True
    edge_width_scale: float = 3.0
    min_edge_abs_weight: float = 0.0
    node_size_scale: float = 1.0
    show_activity_on_nodes: bool = True

    VALID_LAYOUTS = ("spring", "circular")

    def validate(self) -> None:
        """Raise ValueError if any field is out of its valid range."""
        if self.layout_type not in self.VALID_LAYOUTS:
            raise ValueError(f"layout_type must be one of {self.VALID_LAYOUTS}.")
        if self.edge_width_scale <= 0:
            raise ValueError("edge_width_scale must be positive.")
        if self.min_edge_abs_weight < 0:
            raise ValueError("min_edge_abs_weight must be non-negative.")
        if self.node_size_scale <= 0:
            raise ValueError("node_size_scale must be positive.")


# --------------------------------------------------------------------------- #
# Project (top-level container)
# --------------------------------------------------------------------------- #
@dataclass
class ProjectConfig:
    """Top-level config bundling the three sub-configs.

    This is the object the GUI creates and passes down the pipeline:
    GUI -> ProjectConfig -> model -> simulation -> visualization.
    """

    network: NetworkConfig = field(default_factory=NetworkConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def validate(self) -> None:
        """Validate all sub-configs."""
        self.network.validate()
        self.simulation.validate()
        self.visualization.validate()

    def to_dict(self) -> dict[str, Any]:
        """Return a nested plain-dict representation (JSON-serializable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        """Rebuild a ProjectConfig from a nested dict (e.g. loaded JSON)."""
        return cls(
            network=NetworkConfig(**data.get("network", {})),
            simulation=SimulationConfig(**data.get("simulation", {})),
            visualization=VisualizationConfig(**data.get("visualization", {})),
        )
