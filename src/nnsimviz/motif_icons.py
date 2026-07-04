"""Small schematic SVG icons for each motif type.

These are tiny inline diagrams (circles + arrows) that show the structure of
each motif at a glance, using the same colour convention as the network graph:
green = excitatory (positive), red = inhibitory (negative).

The icons are pure SVG strings with no dependencies, so they can be rendered
in the Streamlit sidebar (via st.markdown/st.image) or anywhere else. Keeping
them here means the GUI holds no drawing code of its own.

Public API::

    from nnsimviz.motif_icons import motif_icon_svg
    svg = motif_icon_svg("feedforward_loop")   # -> "<svg ...>...</svg>"
"""

from __future__ import annotations

# Shared colours (match visualization.py).
_EXC = "#2ca02c"   # green, excitatory (+)
_INH = "#d62728"   # red, inhibitory (-)
_NODE = "#4a4a4a"  # neuron outline
_NODE_FILL = "#ffffff"

_W = 96   # icon width
_H = 40   # icon height


def _defs() -> str:
    """Arrowhead markers: one green (excitatory), one red (inhibitory bar)."""
    return (
        '<defs>'
        f'<marker id="exc" markerWidth="6" markerHeight="6" refX="5" refY="3" '
        f'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="{_EXC}"/></marker>'
        f'<marker id="inh" markerWidth="6" markerHeight="6" refX="5" refY="3" '
        f'orient="auto"><rect x="4" y="0" width="2" height="6" fill="{_INH}"/>'
        '</marker>'
        '</defs>'
    )


def _node(cx: float, cy: float, r: float = 6.0, fill: str = _NODE_FILL) -> str:
    """A single neuron circle."""
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
            f'stroke="{_NODE}" stroke-width="1.5"/>')


def _edge(x1: float, y1: float, x2: float, y2: float,
          excitatory: bool = True, curve: float | None = None) -> str:
    """An arrow between two points; green (exc) or red-bar (inh).

    If ``curve`` is given, draw a quadratic curve bowing by that amount (used
    for feedback / reciprocal edges so they don't overlap).
    """
    color = _EXC if excitatory else _INH
    marker = "exc" if excitatory else "inh"
    if curve is None:
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{color}" stroke-width="2" '
                f'marker-end="url(#{marker})"/>')
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2 + curve
    return (f'<path d="M{x1},{y1} Q{mx},{my} {x2},{y2}" fill="none" '
            f'stroke="{color}" stroke-width="2" '
            f'marker-end="url(#{marker})"/>')


def _wrap(body: str) -> str:
    """Wrap icon body in a sized SVG root."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" '
            f'height="{_H}" viewBox="0 0 {_W} {_H}">{_defs()}{body}</svg>')


# --------------------------------------------------------------------------- #
# One builder per motif.
# --------------------------------------------------------------------------- #
def _coincidence_detector() -> str:
    # three sources on the left converge to one target on the right
    body = (
        _edge(12, 8, 60, 20, True)
        + _edge(12, 20, 60, 20, True)
        + _edge(12, 32, 60, 20, True)
        + _node(10, 8) + _node(10, 20) + _node(10, 32)
        + _node(66, 20, r=7)
    )
    return _wrap(body)


def _lateral_inhibition() -> str:
    # three neurons in a row that inhibit each other (red bars)
    body = (
        _edge(24, 14, 44, 14, False, curve=-6)
        + _edge(44, 26, 24, 26, False, curve=6)
        + _edge(52, 14, 72, 14, False, curve=-6)
        + _edge(72, 26, 52, 26, False, curve=6)
        + _node(20, 20) + _node(48, 20) + _node(76, 20)
    )
    return _wrap(body)


def _negative_feedback_loop() -> str:
    # A -> B -> C (green), C -| A (red, curved back)
    body = (
        _edge(18, 20, 42, 20, True)
        + _edge(54, 20, 78, 20, True)
        + _edge(78, 28, 18, 28, False, curve=10)
        + _node(12, 20) + _node(48, 20) + _node(84, 20)
    )
    return _wrap(body)


def _feedforward_loop() -> str:
    # A -> B -> C (top path, green) and A -> C direct (bottom, green)
    body = (
        _edge(18, 14, 42, 14, True)
        + _edge(54, 14, 78, 18, True)
        + _edge(18, 24, 78, 22, True, curve=8)
        + _node(12, 16) + _node(48, 14) + _node(84, 20)
    )
    return _wrap(body)


def _feedforward_inhibition() -> str:
    # A -> target (green direct), A -> inh (green), inh -| target (red)
    body = (
        _edge(18, 20, 78, 20, True)              # A -> target direct
        + _edge(16, 14, 44, 8, True)             # A -> interneuron
        + _edge(50, 10, 78, 16, False)           # interneuron -| target
        + _node(12, 20) + _node(47, 8, fill="#fde0e0") + _node(84, 20)
    )
    return _wrap(body)


def _mutual_excitation() -> str:
    # two neurons exciting each other (two green curved arrows)
    body = (
        _edge(30, 14, 62, 14, True, curve=-7)
        + _edge(62, 26, 30, 26, True, curve=7)
        + _node(26, 20) + _node(66, 20)
    )
    return _wrap(body)


_ICON_BUILDERS = {
    "coincidence_detector": _coincidence_detector,
    "lateral_inhibition": _lateral_inhibition,
    "negative_feedback_loop": _negative_feedback_loop,
    "feedforward_loop": _feedforward_loop,
    "feedforward_inhibition": _feedforward_inhibition,
    "mutual_excitation": _mutual_excitation,
}


def motif_icon_svg(motif_type: str) -> str:
    """Return the schematic SVG string for a motif type.

    Args:
        motif_type: The motif type string (e.g. "feedforward_loop"). Accepts
            the same values used as metadata keys and MotifType values.

    Returns:
        An SVG string, or an empty string if the type is unknown.
    """
    builder = _ICON_BUILDERS.get(motif_type)
    return builder() if builder is not None else ""


def all_motif_icons() -> dict[str, str]:
    """Return {motif_type: svg} for every known motif type."""
    return {key: builder() for key, builder in _ICON_BUILDERS.items()}
