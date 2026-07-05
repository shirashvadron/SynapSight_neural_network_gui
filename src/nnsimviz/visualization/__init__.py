"""Visualization module: turns network structure + activity into Plotly figures.

This module is fully independent of the GUI. It receives a SimulationResult
(and the VisualizationConfig inside it) and nothing else, and returns Plotly
Figures. It never imports Streamlit.

The network graph follows the standard two-trace Plotly + NetworkX pattern:
  * one Scatter trace for edges, drawn as disconnected line segments with
    `None` separators between coordinate pairs;
  * one Scatter trace for nodes, drawn as markers at NetworkX layout
    positions.

On top of that base pattern it adds the project's core visual encoding:
  * edge color by connection sign (blue = positive, red = negative);
  * edge width scaled by absolute weight;
  * optional node size/color by simulation activity.

This module also provides:
  * multiple layout options (spring, circular, kamada_kawai, shell, random);
  * the model name shown as the figure title;
  * an activity-over-time line figure;
  * an animated version of the network graph (nodes pulse with activity);
  * an animated, rotatable 3-D PCA projection of network state-space
    dynamics, with fixed-point markers overlaid (build_pca_animation_figure);
  * a static-image (PNG) export helper.

The implementation is split across a few focused submodules, each covering
one of the concerns above:
  * constants.py     -- shared colors and layout names
  * common.py         -- EdgeData and small figure-title/label helpers
  * pca.py            -- PCA projection and fixed-point computation
  * network_graph.py  -- NetworkVisualizer and its Plotly entry points
  * png_export.py      -- static PNG export helper
  * event_raster.py    -- event-based/spike raster figure
This file re-exports the combined public API so callers can keep doing
`from nnsimviz.visualization import ...` exactly as before.
"""

from __future__ import annotations

from .common import EdgeData, result_time_label
from .constants import (
    AVAILABLE_LAYOUTS,
    MOTIF_NEGATIVE_COLOR,
    MOTIF_POSITIVE_COLOR,
    MOTIF_RING_COLOR,
    MOTIF_TYPE_COLORS,
    MOTIF_TYPE_LABELS,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
)
from .event_raster import build_event_raster_figure
from .network_graph import (
    NetworkVisualizer,
    build_activity_figure,
    build_animated_figure,
    build_figure,
    build_pca_animation_figure,
)
from .pca import _compute_pca_projection, _find_fixed_points_pca
from .png_export import figure_to_png_bytes

__all__ = [
    "EdgeData",
    "result_time_label",
    "AVAILABLE_LAYOUTS",
    "POSITIVE_COLOR",
    "NEGATIVE_COLOR",
    "MOTIF_POSITIVE_COLOR",
    "MOTIF_NEGATIVE_COLOR",
    "MOTIF_RING_COLOR",
    "MOTIF_TYPE_COLORS",
    "MOTIF_TYPE_LABELS",
    "NetworkVisualizer",
    "build_figure",
    "build_activity_figure",
    "build_animated_figure",
    "build_pca_animation_figure",
    "build_event_raster_figure",
    "figure_to_png_bytes",
    "_compute_pca_projection",
    "_find_fixed_points_pca",
]
