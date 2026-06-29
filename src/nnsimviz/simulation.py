"""Simulation engine: runs network dynamics forward in time.

Given a ProjectConfig and a weight matrix, this module integrates a simple
continuous-time recurrent rule and returns a SimulationResult. It depends
only on configs.py and NumPy -- never on the GUI or visualization layers.

Update rule (Euler integration):

    x[t+1] = x[t] + dt * (-x[t] + W @ tanh(x[t]) + input[t]) + noise

State values are clipped each step to keep the demo numerically stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .configs import ProjectConfig, SimulationConfig


# --------------------------------------------------------------------------- #
# Result object (shared data contract with the visualization module)
# --------------------------------------------------------------------------- #
@dataclass
class SimulationResult:
    """Everything the visualization layer needs to draw the network.

    Attributes:
        time: 1-D array of time points, shape (n_steps,).
        activity: Neuron activity over time, shape (n_neurons, n_steps).
        final_state: Activity at the last time step, shape (n_neurons,).
        weight_matrix: The (n_neurons, n_neurons) matrix that was simulated.
        config: The ProjectConfig used to produce this result.
        metadata: Free-form dict (seed, model type, counts, etc.).
    """

    time: np.ndarray
    activity: np.ndarray
    final_state: np.ndarray
    weight_matrix: np.ndarray
    config: ProjectConfig
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Input signal generation
# --------------------------------------------------------------------------- #
def _build_input(sim: SimulationConfig, n_neurons: int, n_steps: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Return an (n_neurons, n_steps) external-input array for the run."""
    if sim.input_type == "none":
        return np.zeros((n_neurons, n_steps))
    if sim.input_type == "constant":
        return np.full((n_neurons, n_steps), sim.input_amplitude)
    if sim.input_type == "noise":
        return rng.normal(0.0, sim.input_amplitude, size=(n_neurons, n_steps))
    if sim.input_type == "sine":
        t = np.arange(n_steps) * sim.dt
        wave = sim.input_amplitude * np.sin(2.0 * np.pi * t / max(sim.duration / 4, sim.dt))
        return np.tile(wave, (n_neurons, 1))
    raise ValueError(f"Unsupported input_type: {sim.input_type}")


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class Simulator:
    """Runs a recurrent network forward in time and packages the result."""

    CLIP = 1e3  # hard cap on state magnitude to prevent blow-ups

    def run(self, config: ProjectConfig, weight_matrix: np.ndarray) -> SimulationResult:
        """Integrate the dynamics and return a SimulationResult.

        Args:
            config: Full project config (uses .simulation and .network).
            weight_matrix: (n, n) matrix from the model library.
        """
        config.validate()
        sim = config.simulation
        n = config.network.n_neurons

        if weight_matrix.shape != (n, n):
            raise ValueError(
                f"weight_matrix shape {weight_matrix.shape} does not match "
                f"n_neurons={n}."
            )

        rng = np.random.default_rng(config.network.random_seed)
        n_steps = sim.n_steps

        time = np.arange(n_steps) * sim.dt
        activity = np.zeros((n, n_steps))
        x = rng.normal(0.0, 0.1, size=n)  # small random initial state
        inputs = _build_input(sim, n, n_steps, rng)

        for t in range(n_steps):
            activity[:, t] = x
            drive = -x + weight_matrix @ np.tanh(x) + inputs[:, t]
            noise = rng.normal(0.0, sim.noise_level, size=n) if sim.noise_level > 0 else 0.0
            x = x + sim.dt * drive + noise
            x = np.clip(x, -self.CLIP, self.CLIP)

        metadata = {
            "model_type": config.network.model_type,
            "n_neurons": n,
            "n_steps": n_steps,
            "duration": sim.duration,
            "dt": sim.dt,
            "random_seed": config.network.random_seed,
            "input_type": sim.input_type,
        }

        return SimulationResult(
            time=time,
            activity=activity,
            final_state=activity[:, -1].copy(),
            weight_matrix=weight_matrix,
            config=config,
            metadata=metadata,
        )
