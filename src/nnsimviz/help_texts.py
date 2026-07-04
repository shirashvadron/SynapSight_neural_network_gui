"""Help texts (tooltips) for the GUI, organized per network model.

This module lets teammates add or edit the little "i" tooltip explanations
shown next to each sidebar control WITHOUT touching app.py. Each network model
can supply its own set of texts; anything it does not override falls back to a
shared default set.

How to add texts for a new model
--------------------------------
1. Create a HelpTexts(...) instance, overriding only the fields whose meaning
   differs from the defaults (leave the rest as they are).
2. Register it in HELP_TEXTS_BY_MODEL under the model's registry key
   (the same key used in models.MODEL_REGISTRY, e.g. "modular").

That's it -- the GUI calls get_help_texts(model_type) and uses whatever it
gets back, so the new texts appear automatically when that model is selected.

Design note: we use a plain dataclass (not pydantic) to stay dependency-free
and consistent with the rest of the project. All fields are plain strings.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class HelpTexts:
    """Tooltip strings for every parameter control in the sidebar.

    One field per control. A model only needs to override the fields whose
    meaning changes for it; unspecified fields keep the shared defaults via
    :func:`merge`.
    """

    # ---- Network ----
    model: str = (
        "The rule used to generate the connection weight matrix. Each model "
        "wires the neurons together differently."
    )
    n_neurons: str = (
        "How many nodes (neurons) the network has. Each becomes one circle "
        "in the graph."
    )
    connection_probability: str = (
        "Chance that any given pair of neurons is connected. Higher values "
        "create a denser graph with more edges."
    )
    weight_scale: str = (
        "Overall strength of the connections. Larger values make edges "
        "thicker and push the dynamics harder."
    )
    positive_connection_ratio: str = (
        "Fraction of connections that are positive (excitatory, blue). "
        "The rest are negative (inhibitory, red)."
    )
    random_seed: str = (
        "Fixes the randomness so the same settings always produce the same "
        "network. Change it to get a different random network."
    )

    # ---- Modular-network extras (only shown for the modular model) ----
    n_modules: str = (
        "Number of communities the neurons are split into. Connections are "
        "dense inside a module and sparse between modules."
    )
    inter_module_probability: str = (
        "Connection probability between neurons in different modules. Lower "
        "values create more clearly separated modules."
    )

    # ---- Simulation ----
    duration: str = (
        "Total simulated time. Longer runs let the network settle or reveal "
        "slower dynamics."
    )
    dt: str = (
        "Integration step size. Smaller values are more accurate but take "
        "more steps to cover the same duration."
    )
    input_type: str = (
        "External signal fed into every neuron: none, a constant drive, "
        "random noise, or a sine wave."
    )
    input_amplitude: str = "Strength of the external input signal."
    noise_level: str = "Amount of random fluctuation added at each time step."

    # ---- Integration method (step function) ----
    integration_method: str = (
        "Numerical integrator used for the simulation time-loop. "
        "Euler-Maruyama is fast; Heun adds a corrector step (2nd order); "
        "RK4 uses four evaluations per step for the highest accuracy. "
        "Higher-order methods are more stable with larger dt values."
    )

    # ---- Convergence epsilon ----
    convergence_eps: str = (
        "Enable early-exit: if the maximum state change between consecutive "
        "steps falls below \u03b5, the simulation is considered converged and "
        "stops. The remaining time steps are filled with the settled state. "
        "Leave disabled to always run the full duration."
    )

    # ---- Visualization ----
    layout: str = (
        "How node positions are arranged. Different layouts reveal different "
        "structure; none changes the network itself."
    )
    show_labels: str = "Print each neuron's index number inside its circle."
    edge_width_scale: str = (
        "Multiplier for edge thickness. Thicker edges = stronger weights."
    )
    min_edge_abs_weight: str = (
        "Hide edges weaker than this value to declutter dense graphs."
    )
    node_size_scale: str = (
        "Multiplier for node size. Nodes also grow with their activity."
    )
    show_activity_on_nodes: str = (
        "Color and size each node by its final activity (red/blue scale)."
    )

    def merge(self, **overrides: str) -> "HelpTexts":
        """Return a copy with the given fields replaced (ignores unknown keys)."""
        valid = {f.name for f in fields(self)}
        clean = {k: v for k, v in overrides.items() if k in valid}
        return replace(self, **clean)


# Shared defaults used by any model that does not override them.
DEFAULT_HELP_TEXTS = HelpTexts()


# --------------------------------------------------------------------------- #
# Per-model help texts.
#
# Each entry overrides ONLY the fields whose meaning differs for that model.
# Everything else falls back to DEFAULT_HELP_TEXTS automatically.
# --------------------------------------------------------------------------- #
HELP_TEXTS_BY_MODEL: dict[str, HelpTexts] = {
    "random_weighted": DEFAULT_HELP_TEXTS.merge(
        positive_connection_ratio=(
            "Fraction of edges that are positive (excitatory, blue). Each edge "
            "picks its sign independently; the rest are negative (red)."
        ),
    ),
    "excitatory_inhibitory": DEFAULT_HELP_TEXTS.merge(
        positive_connection_ratio=(
            "Fraction of neurons that are excitatory. An excitatory neuron's "
            "outgoing edges are all positive; an inhibitory neuron's are all "
            "negative (Dale's principle)."
        ),
    ),
    "symmetric_weighted": DEFAULT_HELP_TEXTS.merge(
        positive_connection_ratio=(
            "Fraction of connections that are positive. Connections are "
            "reciprocal, so W[i, j] equals W[j, i]."
        ),
    ),
    "modular": DEFAULT_HELP_TEXTS.merge(
        connection_probability=(
            "Connection probability WITHIN a module (same community). "
            "Between-module density is set separately below."
        ),
    ),
}


def get_help_texts(model_type: str) -> HelpTexts:
    """Return the HelpTexts for a model, falling back to defaults.

    The GUI calls this after the user picks a model, so the right tooltips are
    used automatically. Unknown model keys simply get the shared defaults,
    which keeps the GUI working even before a teammate has written custom text.
    """
    return HELP_TEXTS_BY_MODEL.get(model_type, DEFAULT_HELP_TEXTS)
