"""Simulation engine: runs network dynamics forward in time.

Given a ProjectConfig and a weight matrix, this module integrates a simple
continuous-time recurrent rule and returns a SimulationResult. It depends
only on configs.py and NumPy -- never on the GUI or visualization layers.

Update rule (Euler-Maruyama integration):
Supported integration methods (configured via SimulationConfig.integration_method):
    - "euler": Euler-Maruyama (first-order SDE integrator):
        x[t+1] = x[t] + dt * (-x[t] + W @ tanh(x[t]) + input[t])
             + noise_level * sqrt(dt) * N(0, 1)
    - "heun":  Heun / Improved Euler (second-order, noise added once at end)
    - "rk4":   Classic 4th-order Runge-Kutta (noise added once at end)

Noise is scaled by sqrt(dt) (Euler-Maruyama convention) so the effective
noise intensity remains constant as dt shrinks.
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
# ODE right-hand side
# --------------------------------------------------------------------------- #
def _rhs(x: np.ndarray, u: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Continuous-time RNN derivative: dx/dt = -x + W @ tanh(x) + u."""
    return -x + W @ np.tanh(x) + u


# --------------------------------------------------------------------------- #
# Step functions  (deterministic part + pre-scaled noise injected at end)
# --------------------------------------------------------------------------- #
def _euler_step(x: np.ndarray, u: np.ndarray, W: np.ndarray,
                dt: float, noise: np.ndarray) -> np.ndarray:
    """Euler-Maruyama step."""
    return x + dt * _rhs(x, u, W) + noise


def _heun_step(x: np.ndarray, u: np.ndarray, W: np.ndarray,
               dt: float, noise: np.ndarray) -> np.ndarray:
    """Heun (Improved Euler / trapezoidal) step with EM noise at end."""
    k1 = _rhs(x, u, W)
    x_pred = x + dt * k1
    k2 = _rhs(x_pred, u, W)
    return x + dt * 0.5 * (k1 + k2) + noise


def _rk4_step(x: np.ndarray, u: np.ndarray, W: np.ndarray,
              dt: float, noise: np.ndarray) -> np.ndarray:
    """Classic 4th-order Runge-Kutta step with EM noise at end."""
    k1 = _rhs(x,               u, W)
    k2 = _rhs(x + 0.5*dt*k1,  u, W)
    k3 = _rhs(x + 0.5*dt*k2,  u, W)
    k4 = _rhs(x +      dt*k3, u, W)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4) + noise


# Dispatch table: method name -> step function
_STEP_FN = {
    "euler": _euler_step,
    "heun":  _heun_step,
    "rk4":   _rk4_step,
}


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

        method = getattr(sim, "integration_method", "euler")
        step_fn = _STEP_FN.get(method)
        if step_fn is None:
            raise ValueError(
                f"Unknown integration_method '{method}'. "
                f"Valid options: {list(_STEP_FN)}"
            )

        rng = np.random.default_rng(config.network.random_seed)
        n_steps = sim.n_steps
        sqrt_dt = np.sqrt(sim.dt)

        time = np.arange(n_steps) * sim.dt
        activity = np.zeros((n, n_steps))
        x = rng.normal(0.0, 0.1, size=n)  # small random initial state
        inputs = _build_input(sim, n, n_steps, rng)

        for t in range(n_steps):
            activity[:, t] = x
            noise = (
                rng.normal(0.0, sim.noise_level * sqrt_dt, size=n)
                if sim.noise_level > 0
                else np.zeros(n)
            )
            x = step_fn(x, inputs[:, t], weight_matrix, sim.dt, noise)
            x = np.clip(x, -self.CLIP, self.CLIP)

        metadata = {
            "model_type": config.network.model_type,
            "n_neurons": n,
            "n_steps": n_steps,
            "duration": sim.duration,
            "dt": sim.dt,
            "random_seed": config.network.random_seed,
            "input_type": sim.input_type,
            "integration_method": method,
        }

        return SimulationResult(
            time=time,
            activity=activity,
            final_state=activity[:, -1].copy(),
            weight_matrix=weight_matrix,
            config=config,
            metadata=metadata,
        )
