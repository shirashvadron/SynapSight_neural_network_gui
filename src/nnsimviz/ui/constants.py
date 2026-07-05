"""Shared UI constants for the SynapSight Streamlit app."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
LOGO_PATH = _REPO_ROOT / "SynapSight.png"
FAVICON_PATH = _REPO_ROOT / "SynapSightfavicon.png"

LAYOUT_LABELS = {
    "spring": "Spring (force-directed)",
    "circular": "Circular",
    "shell": "Shell (concentric)",
    "spiral": "Spiral",
    "random": "Random",
}

METHOD_LABELS = {
    "euler": "Euler-Maruyama (1st order, fast)",
    "heun": "Heun / Improved Euler (2nd order)",
    "rk4": "Runge-Kutta 4 (4th order, accurate)",
}

PLOT_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "neural_network_graph",
        "scale": 2,
    },
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

SIMULATION_TYPE_LABELS = {
    "continuous": "Continuous recurrent dynamics",
    "event_based": "Event-based / spike-like",
}
