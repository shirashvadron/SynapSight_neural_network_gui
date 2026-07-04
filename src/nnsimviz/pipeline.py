"""Pipeline orchestration independent of Streamlit.

This module dispatches from ProjectConfig to the selected simulation engine. It
keeps the GUI thin and also gives tests/CLI code a Streamlit-free integration
entry point.
"""

from __future__ import annotations

import numpy as np

from . import io_utils
from .configs import ProjectConfig
from .event_simulation import EventBasedSimulator
from .models import build_weight_matrix
from .motifs import build_network_with_motifs
from .simulation import Simulator, SimulationResult


def run_pipeline(
    config: ProjectConfig,
    imported_weight_matrix: np.ndarray | None = None,
) -> SimulationResult:
    """Build/import a network, run the selected simulation, and return a result."""
    if imported_weight_matrix is None:
        weight_matrix = build_weight_matrix(config.network)
        network_source = "generated"
        network_source_name = None
    else:
        weight_matrix = io_utils.validate_weight_matrix(imported_weight_matrix)
        config.network.n_neurons = weight_matrix.shape[0]
        network_source = "imported"
        network_source_name = "Imported weight matrix"

    # Apply motifs (no-op when disabled). They expand the weight matrix with
    # extra neurons before simulation, so they work for both the continuous
    # and event-based engines. n_neurons is grown to match.
    weight_matrix, motif_meta = build_network_with_motifs(
        weight_matrix, config.motifs
    )
    config.network.n_neurons = weight_matrix.shape[0]

    if config.simulation.simulation_type == "event_based":
        result = EventBasedSimulator().run(config, weight_matrix)
    else:
        result = Simulator().run(config, weight_matrix)

    result.metadata["network_source"] = network_source
    if network_source_name is not None:
        result.metadata["network_source_name"] = network_source_name
    result.metadata["motifs"] = motif_meta["motifs"]
    return result
