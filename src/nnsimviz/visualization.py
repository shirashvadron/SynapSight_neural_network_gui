"""Visualization module: turns network structure + activity into a Plotly figure.

This module is fully independent of the GUI. It receives a SimulationResult
(and the VisualizationConfig inside it) and nothing else, and returns a
Plotly Figure. It never imports Streamlit.

It follows the standard two-trace Plotly + NetworkX pattern:
  * one Scatter trace for edges, drawn as disconnected line segments with
    `None` separators between coordinate pairs;
  * one Scatter trace for nodes, drawn as markers at NetworkX layout
    positions.

On top of that base pattern it adds the project's core visual encoding:
  * edge color by connection sign (blue = positive, red = negative);
  * edge width scaled by absolute weight;
  * optional node size/color by simulation activity.

Because Plotly colors a whole Scatter trace uniformly, positive and negative
edges are drawn as two separate edge traces (one blue, one red), each itself
split into width-bucketed sub-traces so thickness can reflect weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from .configs import VisualizationConfig
from .simulation import SimulationResult

POSITIVE_COLOR = "#1f77b4"  # blue
NEGATIVE_COLOR = "#d62728"  # red
_N_WIDTH_BUCKETS = 5         # discrete line-width levels per sign


@dataclass
class EdgeData:
    """A single visible directed edge, ready for plotting."""

    source: int
    target: int
    weight: float

    @property
    def is_positive(self) -> bool:
        return self.weight > 0


class NetworkVisualizer:
    """Builds an interactive Plotly graph from a SimulationResult."""

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
    def compute_layout(self, graph: nx.DiGraph, seed: int = 42) -> dict[int, tuple[float, float]]:
        """Return {node: (x, y)} positions for the chosen layout type."""
        if self.config.layout_type == "circular":
            return nx.circular_layout(graph)
        # default: spring
        return nx.spring_layout(graph, seed=seed, k=None)

    # ---- step 3: figure --------------------------------------------------- #
    def create_figure(self, result: SimulationResult) -> go.Figure:
        """Assemble the full Plotly figure from a SimulationResult."""
        weight_matrix = result.weight_matrix
        graph, edges = self.build_graph(weight_matrix)
        seed = result.config.network.random_seed
        pos = self.compute_layout(graph, seed=seed)

        edge_traces = self._edge_traces(edges, pos)
        node_trace = self._node_trace(graph, pos, result)
        legend_traces = self._legend_traces()

        fig = go.Figure(data=[*edge_traces, node_trace, *legend_traces])
        fig.update_layout(
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=40),
            title="Neural Network Graph",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
            # Bucket edges by normalized magnitude so each trace has one width.
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
    """Convenience entry point used by the GUI: result -> Plotly figure."""
    visualizer = NetworkVisualizer(result.config.visualization)
    return visualizer.create_figure(result)
