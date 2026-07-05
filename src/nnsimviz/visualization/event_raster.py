"""Event-based/spike-specific figures."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..simulation import SimulationResult


def build_event_raster_figure(result: SimulationResult) -> go.Figure:
    """Return a spike raster figure from an event-based simulation result."""
    spike_train = result.metadata.get("spike_train")
    spike_times = result.metadata.get("spike_times", [])

    if spike_train is not None:
        neurons_arr, steps_arr = np.nonzero(spike_train)
        steps = steps_arr.tolist()
        neurons = neurons_arr.tolist()
    else:
        # Backward compatibility for older event results.
        steps = [step for step, _ in spike_times]
        neurons = [neuron for _, neuron in spike_times]

    fig = go.Figure()

    if steps:
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=neurons,
                mode="markers",
                name="spike",
                marker=dict(size=9, symbol="line-ns-open"),
                hovertemplate="step=%{x}<br>neuron=%{y}<extra></extra>",
            )
        )
    else:
        fig.add_annotation(
            text="No spikes emitted",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    fig.update_layout(
        title=dict(text="Event-based spike raster", x=0.5),
        xaxis_title="event step",
        yaxis_title="neuron",
        margin=dict(b=45, l=55, r=20, t=60),
        plot_bgcolor="white",
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig
