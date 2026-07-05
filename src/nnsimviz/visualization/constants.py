"""Shared color and layout constants for the visualization figures."""

from __future__ import annotations

POSITIVE_COLOR = "#1f77b4"  # blue
NEGATIVE_COLOR = "#d62728"  # red
MOTIF_POSITIVE_COLOR = "#2ca02c"  # green (motif excitatory)
MOTIF_NEGATIVE_COLOR = "#d62728"  # red   (motif inhibitory)
MOTIF_RING_COLOR = "#2ca02c"      # default green ring (fallback)

# Distinct ring color per motif type, so different motifs are told apart at a
# glance. Keys are the motif_type string values used in the metadata.
MOTIF_TYPE_COLORS: dict[str, str] = {
    "coincidence_detector": "#2ca02c",     # green
    "lateral_inhibition": "#9467bd",       # purple
    "negative_feedback_loop": "#ff7f0e",   # orange
    "feedforward_loop": "#17becf",         # cyan
    "feedforward_inhibition": "#e377c2",   # pink
    "mutual_excitation": "#bcbd22",        # olive
}

# Readable labels for the motif legend.
MOTIF_TYPE_LABELS: dict[str, str] = {
    "coincidence_detector": "Coincidence detector",
    "lateral_inhibition": "Lateral inhibition",
    "negative_feedback_loop": "Negative feedback loop",
    "feedforward_loop": "Feedforward loop",
    "feedforward_inhibition": "Feedforward inhibition",
    "mutual_excitation": "Mutual excitation",
}
_N_WIDTH_BUCKETS = 5         # discrete line-width levels per sign
_N_PCA_COMPONENTS = 3        # number of leading principal components used for PCA views

# Layouts offered by the visualizer. The GUI reads this list so the two
# never drift apart.
AVAILABLE_LAYOUTS = ("spring", "circular", "shell", "spiral", "random")
