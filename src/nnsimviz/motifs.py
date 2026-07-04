"""Network motif system: repeated small connectivity patterns.

A *motif* is a small, reusable pattern of connected neurons that is appended
to a base network. This module is completely independent of Streamlit and of
the GUI; it only needs NumPy and the config objects, so it can be tested on
its own.

Design
------
The public entry point is :func:`add_motifs`, which takes a base weight matrix
and a :class:`MotifConfig`, appends the requested motif instances as *new*
neurons, and returns the expanded weight matrix together with a metadata
dictionary describing every motif (its type, node indices, edges, and the
excitatory/inhibitory sign of each node).

Matrix convention (identical to models.py):
    ``W[i, j]`` is the weight of the directed connection FROM neuron ``j`` TO
    neuron ``i``. Motif edges are expressed as ``(source, target)`` pairs and
    written as ``W[target, source]``.

Supported motif types (see the reference figure):
    * ``coincidence_detector`` -- several excitatory neurons converge, with
      positive edges, onto one target neuron.
    * ``lateral_inhibition``   -- excitatory neurons mutually inhibit each
      other through negative edges (competition).
    * ``negative_feedback_loop`` -- an excitatory neuron drives a second
      neuron, which sends a negative edge back, forming a feedback loop with
      at least one inhibitory connection.

Motifs only reshape connectivity (the weight matrix). They introduce no
event-based or spiking behaviour, so the existing simulator runs unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Motif types
# --------------------------------------------------------------------------- #
class MotifType(str, Enum):
    """The kinds of motif this module can build."""

    COINCIDENCE_DETECTOR = "coincidence_detector"
    LATERAL_INHIBITION = "lateral_inhibition"
    NEGATIVE_FEEDBACK_LOOP = "negative_feedback_loop"
    FEEDFORWARD_LOOP = "feedforward_loop"
    FEEDFORWARD_INHIBITION = "feedforward_inhibition"
    MUTUAL_EXCITATION = "mutual_excitation"


# Number of neurons each motif type contributes.
MOTIF_SIZES: dict[MotifType, int] = {
    MotifType.COINCIDENCE_DETECTOR: 4,   # 3 sources + 1 target
    MotifType.LATERAL_INHIBITION: 3,     # 3 mutually-inhibiting neurons
    MotifType.NEGATIVE_FEEDBACK_LOOP: 3,  # A -> B -> C -| A
    MotifType.FEEDFORWARD_LOOP: 3,       # A -> C and A -> B -> C
    MotifType.FEEDFORWARD_INHIBITION: 3,  # A -> target, A -> inh -| target
    MotifType.MUTUAL_EXCITATION: 2,        # A <-> B mutual excitation
}

# Human-readable labels, handy for the GUI and legends.
MOTIF_LABELS: dict[MotifType, str] = {
    MotifType.COINCIDENCE_DETECTOR: "Coincidence detector",
    MotifType.LATERAL_INHIBITION: "Lateral inhibition",
    MotifType.NEGATIVE_FEEDBACK_LOOP: "Negative feedback loop",
    MotifType.FEEDFORWARD_LOOP: "Feedforward loop",
    MotifType.FEEDFORWARD_INHIBITION: "Feedforward inhibition",
    MotifType.MUTUAL_EXCITATION: "Mutual excitation",
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class MotifConfig:
    """Parameters controlling how motifs are added to the base network.

    Attributes:
        enabled: Master switch. When False, add_motifs is a no-op and the
            network is left exactly as it was.
        n_coincidence_detector: Number of coincidence-detector motifs to add.
        n_lateral_inhibition: Number of lateral-inhibition motifs to add.
        n_negative_feedback_loop: Number of negative-feedback-loop motifs.
        connection_strength: Scale of the (positive) magnitude used for motif
            edges. Signs are decided by the motif structure. Must be > 0.
        n_external_connections: How many edges connect each motif to the base
            network (motif -> base and base -> motif combined, split evenly).
        random_seed: Seed for reproducible motif wiring.
    """

    enabled: bool = False
    n_coincidence_detector: int = 0
    n_lateral_inhibition: int = 0
    n_negative_feedback_loop: int = 0
    n_feedforward_loop: int = 0
    n_feedforward_inhibition: int = 0
    n_mutual_excitation: int = 0
    connection_strength: float = 1.0
    n_external_connections: int = 2
    random_seed: int = 42

    def validate(self) -> None:
        """Raise ValueError if any field is out of its valid range."""
        counts = {
            "n_coincidence_detector": self.n_coincidence_detector,
            "n_lateral_inhibition": self.n_lateral_inhibition,
            "n_negative_feedback_loop": self.n_negative_feedback_loop,
            "n_feedforward_loop": self.n_feedforward_loop,
            "n_feedforward_inhibition": self.n_feedforward_inhibition,
            "n_mutual_excitation": self.n_mutual_excitation,
        }
        for name, value in counts.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")
        if self.connection_strength <= 0:
            raise ValueError("connection_strength must be positive.")
        if self.n_external_connections < 0:
            raise ValueError("n_external_connections cannot be negative.")

    @property
    def total_motifs(self) -> int:
        """Total number of motif instances requested."""
        return (self.n_coincidence_detector
                + self.n_lateral_inhibition
                + self.n_negative_feedback_loop
                + self.n_feedforward_loop
                + self.n_feedforward_inhibition
                + self.n_mutual_excitation)

    def requested_types(self) -> list[MotifType]:
        """Expand the per-type counts into a flat list of motif types."""
        ordered = [
            (MotifType.COINCIDENCE_DETECTOR, self.n_coincidence_detector),
            (MotifType.LATERAL_INHIBITION, self.n_lateral_inhibition),
            (MotifType.NEGATIVE_FEEDBACK_LOOP, self.n_negative_feedback_loop),
            (MotifType.FEEDFORWARD_LOOP, self.n_feedforward_loop),
            (MotifType.FEEDFORWARD_INHIBITION, self.n_feedforward_inhibition),
            (MotifType.MUTUAL_EXCITATION, self.n_mutual_excitation),
        ]
        result: list[MotifType] = []
        for motif_type, count in ordered:
            result.extend([motif_type] * count)
        return result


# --------------------------------------------------------------------------- #
# Template + instance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MotifTemplate:
    """A reusable description of one motif's internal wiring.

    Node indices are LOCAL (0-based within the motif). The builder offsets
    them into global indices when the motif is placed into the network.

    Attributes:
        motif_type: Which motif this template describes.
        n_nodes: Number of neurons in the motif.
        edges: Local (source, target) pairs with their sign (+1 or -1).
        node_signs: Local node index -> "positive" or "negative".
    """

    motif_type: MotifType
    n_nodes: int
    edges: tuple[tuple[int, int, int], ...]   # (source, target, sign)
    node_signs: dict[int, str]


@dataclass
class MotifInstance:
    """A motif placed into the network at concrete global node indices.

    Attributes:
        motif_id: Sequential id of this instance.
        motif_type: Which motif this is.
        nodes: Global node indices belonging to this motif.
        edges: Global (source, target) pairs internal to this motif.
        node_signs: Global node index -> "positive" or "negative".
    """

    motif_id: int
    motif_type: MotifType
    nodes: list[int]
    edges: list[tuple[int, int]]
    node_signs: dict[int, str]

    def to_metadata(self) -> dict[str, Any]:
        """Return a plain-dict view for visualization / serialization."""
        return {
            "motif_id": self.motif_id,
            "motif_type": self.motif_type.value,
            "nodes": list(self.nodes),
            "edges": [list(e) for e in self.edges],
            "node_signs": dict(self.node_signs),
        }


# --------------------------------------------------------------------------- #
# Templates for each motif type
# --------------------------------------------------------------------------- #
def _coincidence_detector_template() -> MotifTemplate:
    """3 excitatory sources -> 1 target, all positive edges."""
    # nodes 0,1,2 = sources; node 3 = target
    edges = tuple((src, 3, +1) for src in (0, 1, 2))
    signs = {0: "positive", 1: "positive", 2: "positive", 3: "positive"}
    return MotifTemplate(MotifType.COINCIDENCE_DETECTOR, 4, edges, signs)


def _lateral_inhibition_template() -> MotifTemplate:
    """3 excitatory neurons that mutually inhibit each other (negative edges)."""
    nodes = (0, 1, 2)
    edges = tuple(
        (a, b, -1) for a in nodes for b in nodes if a != b
    )
    signs = {0: "positive", 1: "positive", 2: "positive"}
    return MotifTemplate(MotifType.LATERAL_INHIBITION, 3, edges, signs)


def _negative_feedback_loop_template() -> MotifTemplate:
    """A -> B (positive), B -> C (positive), C -| A (negative feedback)."""
    edges = (
        (0, 1, +1),   # A activates B
        (1, 2, +1),   # B activates C
        (2, 0, -1),   # C inhibits A  (the negative feedback)
    )
    signs = {0: "positive", 1: "positive", 2: "negative"}
    return MotifTemplate(MotifType.NEGATIVE_FEEDBACK_LOOP, 3, edges, signs)


def _feedforward_loop_template() -> MotifTemplate:
    """Coherent feedforward loop: A -> C directly, and A -> B -> C indirectly.

    All three edges are positive. Because C is reached by both a fast direct
    path and a slower two-step path, this motif acts as a sign-sensitive
    filter that responds mainly to persistent input (the classic FFL).
    """
    edges = (
        (0, 1, +1),   # A -> B
        (1, 2, +1),   # B -> C
        (0, 2, +1),   # A -> C (direct)
    )
    signs = {0: "positive", 1: "positive", 2: "positive"}
    return MotifTemplate(MotifType.FEEDFORWARD_LOOP, 3, edges, signs)


def _feedforward_inhibition_template() -> MotifTemplate:
    """A drives a target directly (+) and via an inhibitory interneuron (-).

    Node 0 = source (A), node 1 = inhibitory interneuron, node 2 = target.
    A excites both the target and the interneuron; the interneuron inhibits
    the target. This creates a narrow time window for the signal to pass.
    """
    edges = (
        (0, 2, +1),   # A -> target (fast excitation)
        (0, 1, +1),   # A -> inhibitory interneuron
        (1, 2, -1),   # interneuron -| target (delayed inhibition)
    )
    signs = {0: "positive", 1: "negative", 2: "positive"}
    return MotifTemplate(MotifType.FEEDFORWARD_INHIBITION, 3, edges, signs)


def _mutual_excitation_template() -> MotifTemplate:
    """Two neurons that mutually excite each other (positive both ways).

    Reciprocal excitation makes the pair bistable: once driven, the pair
    latches into a sustained high-activity state (a simple memory element).
    """
    edges = (
        (0, 1, +1),   # A -> B
        (1, 0, +1),   # B -> A
    )
    signs = {0: "positive", 1: "positive"}
    return MotifTemplate(MotifType.MUTUAL_EXCITATION, 2, edges, signs)


_TEMPLATE_BUILDERS = {
    MotifType.COINCIDENCE_DETECTOR: _coincidence_detector_template,
    MotifType.LATERAL_INHIBITION: _lateral_inhibition_template,
    MotifType.NEGATIVE_FEEDBACK_LOOP: _negative_feedback_loop_template,
    MotifType.FEEDFORWARD_LOOP: _feedforward_loop_template,
    MotifType.FEEDFORWARD_INHIBITION: _feedforward_inhibition_template,
    MotifType.MUTUAL_EXCITATION: _mutual_excitation_template,
}


def get_template(motif_type: MotifType) -> MotifTemplate:
    """Return the wiring template for a motif type."""
    return _TEMPLATE_BUILDERS[motif_type]()


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def add_motifs(
    weight_matrix: np.ndarray,
    config: MotifConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Append motif instances to a base network's weight matrix.

    Args:
        weight_matrix: The base (n, n) weight matrix.
        config: Motif configuration.

    Returns:
        A tuple ``(new_weight_matrix, metadata)`` where ``metadata`` has the
        shape ``{"motifs": [ {motif_id, motif_type, nodes, edges,
        node_signs}, ... ]}``. When motifs are disabled or none are
        requested, the original matrix is returned unchanged with an empty
        motif list.
    """
    config.validate()

    base_n = weight_matrix.shape[0]
    if not config.enabled or config.total_motifs == 0:
        return weight_matrix.copy(), {"motifs": []}

    requested = config.requested_types()
    extra = sum(MOTIF_SIZES[t] for t in requested)
    new_n = base_n + extra

    # Grow the matrix, copying the base network into the top-left block.
    new_w = np.zeros((new_n, new_n), dtype=float)
    new_w[:base_n, :base_n] = weight_matrix

    rng = np.random.default_rng(config.random_seed)
    strength = config.connection_strength

    instances: list[MotifInstance] = []
    cursor = base_n

    for motif_id, motif_type in enumerate(requested):
        template = get_template(motif_type)
        offset = cursor
        global_nodes = [offset + i for i in range(template.n_nodes)]

        # Write internal motif edges into the matrix.
        global_edges: list[tuple[int, int]] = []
        for (src, tgt, sign) in template.edges:
            g_src = offset + src
            g_tgt = offset + tgt
            magnitude = abs(rng.normal(0.0, strength))
            new_w[g_tgt, g_src] = sign * magnitude   # W[target, source]
            global_edges.append((g_src, g_tgt))

        global_signs = {
            offset + local: sign for local, sign in template.node_signs.items()
        }

        # Connect this motif to the base network (external connections).
        _wire_external(new_w, rng, base_n, global_nodes, global_signs,
                       config.n_external_connections, strength)

        instances.append(MotifInstance(
            motif_id=motif_id,
            motif_type=motif_type,
            nodes=global_nodes,
            edges=global_edges,
            node_signs=global_signs,
        ))
        cursor += template.n_nodes

    metadata = {"motifs": [inst.to_metadata() for inst in instances]}
    return new_w, metadata


def build_network_with_motifs(
    base_weight_matrix: np.ndarray,
    config: MotifConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Public pipeline helper: base matrix + motif config -> (matrix, metadata).

    Thin wrapper around :func:`add_motifs` that gives the GUI and the
    simulation pipeline a single, clearly named entry point. Keeping this here
    (rather than in app.py) means the Streamlit layer never contains motif or
    matrix-building logic of its own.
    """
    return add_motifs(base_weight_matrix, config)


def _wire_external(
    new_w: np.ndarray,
    rng: np.random.Generator,
    base_n: int,
    motif_nodes: list[int],
    motif_signs: dict[int, str],
    n_external: int,
    strength: float,
) -> None:
    """Add edges linking a motif to the base network, in place.

    Half the connections go motif -> base and half base -> motif (rounded).
    The sign of a motif -> base edge follows the motif source neuron's
    identity; base -> motif edges are positive (excitatory drive in).
    """
    if n_external <= 0 or base_n == 0:
        return

    n_out = n_external // 2
    n_in = n_external - n_out

    # motif -> base
    for _ in range(n_out):
        src = int(rng.choice(motif_nodes))
        tgt = int(rng.integers(0, base_n))
        sign = 1.0 if motif_signs.get(src, "positive") == "positive" else -1.0
        new_w[tgt, src] = sign * abs(rng.normal(0.0, strength))

    # base -> motif (positive drive into the motif)
    for _ in range(n_in):
        src = int(rng.integers(0, base_n))
        tgt = int(rng.choice(motif_nodes))
        new_w[tgt, src] = abs(rng.normal(0.0, strength))
