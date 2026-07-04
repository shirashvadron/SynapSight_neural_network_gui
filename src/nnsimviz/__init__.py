"""nnsimviz: neural-network simulation & visualization.

A small toolkit for defining simple recurrent neural networks, simulating
their dynamics, and visualizing them as interactive signed, weighted graphs
(blue = positive, red = negative, thickness = strength).

Public API::

    from nnsimviz import (
        NetworkConfig, SimulationConfig, VisualizationConfig, ProjectConfig,
        build_weight_matrix, Simulator, SimulationResult,
        NetworkVisualizer, build_figure,
    )
"""

from __future__ import annotations

from .configs import (
    NetworkConfig,
    SimulationConfig,
    VisualizationConfig,
    ProjectConfig,
    EventSimulationConfig,
)
from .models import (
    NetworkModel,
    RandomWeightedNetwork,
    ExcitatoryInhibitoryNetwork,
    SymmetricWeightedNetwork,
    ModularNetwork,
    MODEL_REGISTRY,
    get_model,
    build_weight_matrix,
)
from .simulation import Simulator, SimulationResult
from .visualization import NetworkVisualizer, build_figure
from .event_simulation import EventBasedSimulator
from .pipeline import run_pipeline
from .motifs import (
    MotifType,
    MotifConfig,
    MotifTemplate,
    MotifInstance,
    add_motifs,
    build_network_with_motifs,
    get_template,
    MOTIF_LABELS,
    MOTIF_SIZES,
)

__version__ = "0.1.0"

__all__ = [
    "NetworkConfig",
    "SimulationConfig",
    "VisualizationConfig",
    "ProjectConfig",
    "NetworkModel",
    "RandomWeightedNetwork",
    "ExcitatoryInhibitoryNetwork",
    "SymmetricWeightedNetwork",
    "ModularNetwork",
    "MODEL_REGISTRY",
    "get_model",
    "build_weight_matrix",
    "Simulator",
    "SimulationResult",
    "NetworkVisualizer",
    "build_figure",
    "EventSimulationConfig",
    "EventBasedSimulator",
    "run_pipeline",
    "MotifType",
    "MotifConfig",
    "MotifTemplate",
    "MotifInstance",
    "add_motifs",
    "build_network_with_motifs",
    "get_template",
    "MOTIF_LABELS",
    "MOTIF_SIZES",
    "__version__",
]
