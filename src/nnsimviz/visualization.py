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
  * a static-image (PNG) export helper.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from .configs import VisualizationConfig
from .models import MODEL_REGISTRY
from .simulation import SimulationResult

POSITIVE_COLOR = "#1f77b4"  # blue
NEGATIVE_COLOR = "#d62728"  # red
_N_WIDTH_BUCKETS = 5         # discrete line-width levels per sign

# Layouts offered by the visualizer. The GUI reads this list so the two
# never drift apart.
AVAILABLE_LAYOUTS = ("spring", "circular", "shell", "spiral", "random")


@dataclass
class EdgeData:
    """A single visible directed edge, ready for plotting."""

    source: int
    target: int
    weight: float

    @property
    def is_positive(self) -> bool:
        return self.weight > 0


def _model_display_name(result: SimulationResult) -> str:
    """Human-readable model name from the registry, for figure titles."""
    key = result.config.network.model_type
    model = MODEL_REGISTRY.get(key)
    return model.name if model is not None else key


def result_time_label(result: SimulationResult, step: int) -> str:
    """Short label for a time step, showing the actual simulated time."""
    return f"{result.time[step]:.2f}"


class NetworkVisualizer:
    """Builds interactive Plotly figures from a SimulationResult."""

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config
        self.config.validate()

    # ---- step 1: structure ------------------------------------------------ #
    def build_graph(self, weight_matrix: np.ndarray) -> tuple[nx.DiGraph, list[EdgeData]]:
        """Build a NetworkX DiGraph and a list of visible edges.

        Edges with abs(weight) below `min_edge_abs_weight` are skipped.
        Every neuron becomes a node even if it has no visible edges.
        """
        n = weight_matrix.shape[0]
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))

        edges: list[EdgeData] = []
        threshold = self.config.min_edge_abs_weight
        for i in range(n):           # target neuron (row)
            for j in range(n):       # source neuron (col): W[i, j] is j -> i
                w = float(weight_matrix[i, j])
                if i != j and abs(w) > threshold and w != 0.0:
                    graph.add_edge(j, i, weight=w)
                    edges.append(EdgeData(source=j, target=i, weight=w))
        return graph, edges

    # ---- step 2: layout --------------------------------------------------- #
    def compute_layout(self, graph: nx.DiGraph,
                       seed: int = 42) -> dict[int, tuple[float, float]]:
        """Return {node: (x, y)} positions for the chosen layout type."""
        layout = self.config.layout_type
        if layout == "circular":
            return nx.circular_layout(graph)
        if layout == "shell":
            return nx.shell_layout(graph)
        if layout == "spiral":
            return nx.spiral_layout(graph)
        if layout == "random":
            return nx.random_layout(graph, seed=seed)
        # default: spring
        return nx.spring_layout(graph, seed=seed, k=None)

    # ---- step 3: figure --------------------------------------------------- #
    def create_figure(self, result: SimulationResult) -> go.Figure:
        """Assemble the full network Plotly figure from a SimulationResult."""
        weight_matrix = result.weight_matrix
        graph, edges = self.build_graph(weight_matrix)
        seed = result.config.network.random_seed
        pos = self.compute_layout(graph, seed=seed)

        edge_traces = self._edge_traces(edges, pos)
        node_trace = self._node_trace(graph, pos, result)
        legend_traces = self._legend_traces()

        fig = go.Figure(data=[*edge_traces, node_trace, *legend_traces])
        title = f"Neural Network Graph \u2014 {_model_display_name(result)}"
        fig.update_layout(
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    # ---- activity over time (feature #4) ---------------------------------- #
    def create_activity_figure(self, result: SimulationResult,
                               max_lines: int = 30) -> go.Figure:
        """Line figure of each neuron's activity over time.

        To keep the plot readable for large networks, at most `max_lines`
        neurons are drawn (evenly sampled); the rest are omitted.
        """
        n = result.activity.shape[0]
        if n <= max_lines:
            indices = list(range(n))
        else:
            indices = list(np.linspace(0, n - 1, max_lines, dtype=int))

        fig = go.Figure()
        for i in indices:
            fig.add_trace(go.Scatter(
                x=result.time,
                y=result.activity[i],
                mode="lines",
                name=f"neuron {i}",
                line=dict(width=1),
            ))
        note = "" if n <= max_lines else f" (showing {max_lines} of {n})"
        fig.update_layout(
            title=dict(text=f"Activity over time{note}", x=0.5, xanchor="center"),
            xaxis_title="time",
            yaxis_title="activity",
            margin=dict(b=40, l=50, r=20, t=50),
            plot_bgcolor="white",
            hovermode="closest",
            showlegend=n <= 12,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eee")
        fig.update_yaxes(showgrid=True, gridcolor="#eee")
        return fig

    # ---- animation (feature #6) ------------------------------------------- #
    def create_animated_figure(self, result: SimulationResult,
                               max_frames: int = 40) -> go.Figure:
        """Animated network graph: node color/size follow activity over time.

        The layout is fixed (computed once); only the node marker updates each
        frame. Time steps are evenly sampled down to at most `max_frames` so
        the animation stays light.

        The figure has a small media-player interface: Play, Pause, three
        speed presets (0.5x, 1x, 2x), and a slider that scrubs through frames
        manually (drag it, or click anywhere on its track).
        """
        weight_matrix = result.weight_matrix
        graph, edges = self.build_graph(weight_matrix)
        seed = result.config.network.random_seed
        pos = self.compute_layout(graph, seed=seed)
        nodes = list(graph.nodes())
        xs = [pos[k][0] for k in nodes]
        ys = [pos[k][1] for k in nodes]

        n_steps = result.activity.shape[1]
        if n_steps <= max_frames:
            frame_idx = list(range(n_steps))
        else:
            frame_idx = list(np.linspace(0, n_steps - 1, max_frames, dtype=int))

        amax = float(np.abs(result.activity).max()) or 1.0
        edge_traces = self._edge_traces(edges, pos)

        def node_marker(step: int) -> go.Scatter:
            act = result.activity[nodes, step]
            sizes = 10 + np.abs(act) * 8 * self.config.node_size_scale
            return go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=sizes.tolist(), color=act.tolist(),
                            colorscale="RdBu", cmin=-amax, cmax=amax, cmid=0,
                            showscale=True, colorbar=dict(title="activity", thickness=12),
                            line=dict(width=1, color="#333")),
                hoverinfo="skip", showlegend=False,
            )

        frame_names = [str(int(s)) for s in frame_idx]
        first = node_marker(frame_idx[0])
        frames = [
            go.Frame(name=name, data=[node_marker(int(s))],
                     traces=[len(edge_traces)])
            for name, s in zip(frame_names, frame_idx)
        ]

        # Base frame duration in ms at 1x speed; halved/doubled for 2x/0.5x.
        base_ms = 120
        speeds = [("0.5x", base_ms * 2), ("1x", base_ms), ("2x", base_ms // 2)]

        play_buttons = [
            dict(label=f"\u25b6 {label}", method="animate",
                 args=[None, dict(frame=dict(duration=ms, redraw=True),
                                  transition=dict(duration=0),
                                  fromcurrent=True, mode="immediate")])
            for label, ms in speeds
        ]
        pause_button = dict(
            label="\u23f8 Pause", method="animate",
            args=[[None], dict(frame=dict(duration=0, redraw=False),
                               mode="immediate")],
        )

        # Scrub slider: one step per frame, jumps straight to that frame
        # without re-triggering continuous playback.
        slider_steps = [
            dict(label=result_time_label(result, int(s)), method="animate",
                 args=[[name], dict(frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=0))])
            for name, s in zip(frame_names, frame_idx)
        ]

        fig = go.Figure(data=[*edge_traces, first], frames=frames)
        fig.update_layout(
            title=dict(text=f"Activity animation \u2014 {_model_display_name(result)}",
                       x=0.5, xanchor="center"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white", margin=dict(b=20, l=20, r=20, t=90),
            updatemenus=[dict(
                type="buttons", showactive=False, direction="left",
                x=0.0, xanchor="left", y=1.12, yanchor="top", pad=dict(r=6, t=0),
                buttons=[*play_buttons, pause_button],
            )],
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="t = ", visible=True, xanchor="right"),
                pad=dict(t=40, b=0),
                x=0.0, len=1.0,
                steps=slider_steps,
            )],
        )
        return fig

    # ---- helpers ---------------------------------------------------------- #
    def _edge_traces(self, edges: list[EdgeData],
                     pos: dict[int, tuple[float, float]]) -> list[go.Scatter]:
        """Build width-bucketed line traces, separated by sign (color)."""
        if not edges:
            return []

        max_abs = max(abs(e.weight) for e in edges) or 1.0
        traces: list[go.Scatter] = []

        for is_positive, color in ((True, POSITIVE_COLOR), (False, NEGATIVE_COLOR)):
            group = [e for e in edges if e.is_positive == is_positive]
            if not group:
                continue
            buckets: dict[int, list[EdgeData]] = {}
            for e in group:
                level = min(_N_WIDTH_BUCKETS - 1,
                            int(abs(e.weight) / max_abs * _N_WIDTH_BUCKETS))
                buckets.setdefault(level, []).append(e)

            for level, bucket_edges in buckets.items():
                xs: list[float | None] = []
                ys: list[float | None] = []
                for e in bucket_edges:
                    x0, y0 = pos[e.source]
                    x1, y1 = pos[e.target]
                    xs += [x0, x1, None]
                    ys += [y0, y1, None]
                width = (level + 1) / _N_WIDTH_BUCKETS * self.config.edge_width_scale
                traces.append(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=color, width=width),
                    hoverinfo="none", showlegend=False,
                ))
        return traces

    def _node_trace(self, graph: nx.DiGraph,
                    pos: dict[int, tuple[float, float]],
                    result: SimulationResult) -> go.Scatter:
        """Build the node marker trace, optionally encoding activity."""
        nodes = list(graph.nodes())
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]

        activity = result.final_state
        cfg = self.config

        if cfg.show_activity_on_nodes:
            sizes = 10 + np.abs(activity[nodes]) * 8 * cfg.node_size_scale
            colors = activity[nodes]
            marker = dict(
                size=sizes.tolist(),
                color=colors.tolist(),
                colorscale="RdBu",
                cmid=0,
                showscale=True,
                colorbar=dict(title="activity", thickness=12),
                line=dict(width=1, color="#333"),
            )
        else:
            marker = dict(
                size=14 * cfg.node_size_scale,
                color="#888",
                line=dict(width=1, color="#333"),
            )

        text = [str(n) for n in nodes] if cfg.show_labels else None
        hover = [
            f"neuron {n}<br>activity: {activity[n]:.3f}<br>"
            f"out-degree: {graph.out_degree(n)}<br>in-degree: {graph.in_degree(n)}"
            for n in nodes
        ]
        return go.Scatter(
            x=xs, y=ys, mode="markers+text" if cfg.show_labels else "markers",
            marker=marker, text=text, textposition="middle center",
            textfont=dict(size=9, color="white"),
            hovertext=hover, hoverinfo="text", showlegend=False,
        )

    def _legend_traces(self) -> list[go.Scatter]:
        """Invisible traces that produce a clean legend for edge colors."""
        return [
            go.Scatter(x=[None], y=[None], mode="lines",
                       line=dict(color=POSITIVE_COLOR, width=3),
                       name="positive (excitatory)"),
            go.Scatter(x=[None], y=[None], mode="lines",
                       line=dict(color=NEGATIVE_COLOR, width=3),
                       name="negative (inhibitory)"),
        ]


def build_figure(result: SimulationResult) -> go.Figure:
    """Convenience entry point used by the GUI: result -> network figure."""
    visualizer = NetworkVisualizer(result.config.visualization)
    return visualizer.create_figure(result)


def build_activity_figure(result: SimulationResult) -> go.Figure:
    """Convenience entry point: result -> activity-over-time figure."""
    visualizer = NetworkVisualizer(result.config.visualization)
    return visualizer.create_activity_figure(result)


def build_animated_figure(result: SimulationResult) -> go.Figure:
    """Convenience entry point: result -> animated network figure."""
    visualizer = NetworkVisualizer(result.config.visualization)
    return visualizer.create_animated_figure(result)


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
