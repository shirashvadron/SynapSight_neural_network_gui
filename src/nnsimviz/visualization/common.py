"""Small shared helpers used across the visualization figures."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import MODEL_REGISTRY
from ..simulation import SimulationResult


@dataclass
class EdgeData:
    """A single visible directed edge, ready for plotting."""

    source: int
    target: int
    weight: float

    @property
    def is_positive(self) -> bool:
        return self.weight > 0


def _model_display_name(result: SimulationResult) -> str:
    """Human-readable model/source name for figure titles."""
    if result.metadata.get("network_source") == "imported":
        return result.metadata.get("network_source_name", "Imported network")

    key = result.config.network.model_type
    model = MODEL_REGISTRY.get(key)
    return model.name if model is not None else key


def result_time_label(result: SimulationResult, step: int) -> str:
    """Short label for a time step, showing the actual simulated time."""
    return f"{result.time[step]:.2f}"
