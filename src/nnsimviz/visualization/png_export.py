"""Static-image (PNG) export helper for Plotly figures."""

from __future__ import annotations

import plotly.graph_objects as go


def figure_to_png_bytes(fig: go.Figure, scale: float = 2.0) -> bytes:
    """Render a Plotly figure to PNG bytes (for download).

    Requires the `kaleido` package. Raises a clear RuntimeError if it is
    missing so the GUI can show a helpful message instead of crashing.
    """
    try:
        return fig.to_image(format="png", scale=scale)
    except Exception as exc:  # kaleido missing or render failure
        raise RuntimeError(
            "PNG export requires the 'kaleido' package. "
            "Install it with: pip install kaleido"
        ) from exc
