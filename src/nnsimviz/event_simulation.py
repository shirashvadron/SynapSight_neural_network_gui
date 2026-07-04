"""Discrete event/spike-like simulation engine.

This is intentionally lightweight rather than a detailed biophysical spiking
model. Neurons hold scalar activation. External or propagated events add values
to neurons. When activation crosses threshold, the neuron emits a spike, resets,
and sends its outgoing weights to target neurons one discrete step later.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict

import numpy as np

from .configs import ProjectConfig
from .simulation import SimulationResult


@dataclass(frozen=True)
class SpikeEvent:
    """A single spike emitted by a neuron at a discrete step."""

    step: int
    neuron: int


@dataclass(frozen=True)
class PropagationEvent:
    """A value delivered to one target neuron at a discrete step."""

    step: int
    source: int | None
    target: int
    value: float


class EventBasedSimulator:
    """Runs spike-like event propagation on a weighted directed network."""

    def run(self, config: ProjectConfig, weight_matrix: np.ndarray) -> SimulationResult:
        """Run event-based propagation and return a SimulationResult."""
        config.validate()
        if config.simulation.simulation_type != "event_based":
            raise ValueError("EventBasedSimulator requires simulation_type='event_based'.")

        event_cfg = config.event
        n_neurons = config.network.n_neurons
        W = np.asarray(weight_matrix, dtype=float)
        if W.shape != (n_neurons, n_neurons):
            raise ValueError(
                f"weight_matrix shape {W.shape} does not match n_neurons={n_neurons}."
            )

        scheduled: DefaultDict[int, list[PropagationEvent]] = defaultdict(list)
        external_events = list(event_cfg.external_events)
        if not external_events:
            external_events = [
                (0, event_cfg.default_input_neuron, event_cfg.default_input_value)
            ]

        for step, target, value in external_events:
            scheduled[int(step)].append(
                PropagationEvent(
                    step=int(step),
                    source=None,
                    target=int(target),
                    value=float(value),
                )
            )

        state = np.zeros(n_neurons, dtype=float)

        # activity stores the post-reset/post-decay state used for normal
        # activity plots and final_state. For event-based simulations this can
        # hide a spike, because a neuron that crosses threshold is immediately
        # reset in the same step. The two arrays below preserve the event view:
        # activation_before_reset shows the state after inputs arrive but before
        # threshold reset, and spike_train explicitly records emitted spikes.
        activity = np.zeros((n_neurons, event_cfg.max_steps), dtype=float)
        activation_before_reset = np.zeros(
            (n_neurons, event_cfg.max_steps), dtype=float
        )
        spike_train = np.zeros((n_neurons, event_cfg.max_steps), dtype=int)

        spike_times: list[tuple[int, int]] = []
        event_log: list[dict[str, int | float | None | str]] = []
        
        for step in range(event_cfg.max_steps):
            # Apply all events scheduled for this step.
            for event in scheduled.get(step, []):
                if not 0 <= event.target < n_neurons:
                    raise ValueError("Event target index is outside the network.")
                state[event.target] += event.value
                event_log.append({
                    "kind": "external" if event.source is None else "propagation",
                    "step": step,
                    "source": event.source,
                    "target": event.target,
                    "value": event.value,
                })

            # Record the pre-reset state so threshold-crossing spikes remain
            # visible even after the spiking neurons are reset below.
            activation_before_reset[:, step] = state

            # Threshold check after all input for this step has arrived.
            spiking_neurons = np.flatnonzero(state >= event_cfg.threshold)
            for source in spiking_neurons:
                spike_train[source, step] = 1
                spike_times.append((step, int(source)))
                event_log.append({
                    "kind": "spike",
                    "step": step,
                    "source": int(source),
                    "target": int(source),
                    "value": float(state[source]),
                })
                state[source] = event_cfg.reset_value

                # W[target, source] is the connection source -> target.
                outgoing_targets = np.flatnonzero(W[:, source] != 0.0)
                for target in outgoing_targets:
                    scheduled[step + 1].append(
                        PropagationEvent(
                            step=step + 1,
                            source=int(source),
                            target=int(target),
                            value=float(W[target, source]),
                        )
                    )

            if event_cfg.decay > 0:
                state *= (1.0 - event_cfg.decay)

            activity[:, step] = state

        spike_counts = np.zeros(n_neurons, dtype=int)
        for _, neuron in spike_times:
            spike_counts[neuron] += 1

        metadata = {
            "simulation_type": "event_based",
            "model_type": config.network.model_type,
            "n_neurons": n_neurons,
            "n_steps": event_cfg.max_steps,
            "duration": event_cfg.max_steps,
            "dt": 1.0,
            "random_seed": config.network.random_seed,
            "threshold": event_cfg.threshold,
            "reset_value": event_cfg.reset_value,
            "decay": event_cfg.decay,
            "spike_times": spike_times,
            "spike_counts": spike_counts.tolist(),
            "spike_train": spike_train,
            "activation_before_reset": activation_before_reset,
            "event_log": event_log,
            "final_state": state.copy(),
            "converged_at": None,
        }

        return SimulationResult(
            time=np.arange(event_cfg.max_steps, dtype=float),
            activity=activity,
            final_state=activity[:, -1].copy(),
            weight_matrix=W,
            config=config,
            metadata=metadata,
        )
