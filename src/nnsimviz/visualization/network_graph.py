"""NetworkVisualizer: the core network-graph figure and its Plotly entry points.

Builds interactive Plotly figures from a SimulationResult, following the
standard two-trace Plotly + NetworkX pattern:
  * one Scatter trace for edges, drawn as disconnected line segments with
    `None` separators between coordinate pairs;
  * one Scatter trace for nodes, drawn as markers at NetworkX layout
    positions.

On top of that base pattern it adds the project's core visual encoding:
  * edge color by connection sign (blue = positive, red = negative);
  * edge width scaled by absolute weight;
  * optional node size/color by simulation activity.

It also provides multiple layout options, an activity-over-time line figure,
an animated version of the network graph (nodes pulse with activity), and an
animated, rotatable 3-D PCA projection of network state-space dynamics with
fixed-point markers overlaid.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from ..configs import VisualizationConfig
from ..simulation import SimulationResult
from .common import EdgeData, _model_display_name, result_time_label
from .constants import (
    MOTIF_NEGATIVE_COLOR,
    MOTIF_POSITIVE_COLOR,
    MOTIF_RING_COLOR,
    MOTIF_TYPE_COLORS,
    MOTIF_TYPE_LABELS,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    _N_WIDTH_BUCKETS,
)
from .pca import _compute_pca_projection, _find_fixed_points_pca


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

        motif_meta = result.metadata.get("motifs", [])
        motif_traces = self._motif_traces(motif_meta, pos)

        fig = go.Figure(data=[*edge_traces, *motif_traces, node_trace, *legend_traces])
        title = f"Neural Network Graph \u2014 {_model_display_name(result)}"
        fig.update_layout(
            showlegend=True,
            hovermode="closest",
            margin=dict(b=70, l=20, r=20, t=50),
            title=dict(text=title, x=0.5, xanchor="center", y=0.97, yanchor="top"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="top", y=-0.02,
                        xanchor="center", x=0.5),
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

        # Motif marking overlay (static): type-colored rings + green/red motif
        # edges. Appended AFTER the node marker so the marker stays at index
        # len(edge_traces) -- the trace the animation frames update.
        motif_meta = result.metadata.get("motifs", [])
        motif_traces = self._motif_traces(motif_meta, pos)

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

        fig = go.Figure(data=[*edge_traces, first, *motif_traces], frames=frames)
        fig.update_layout(
            title=dict(text=f"Activity animation \u2014 {_model_display_name(result)}",
                       x=0.5, xanchor="center"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white", margin=dict(b=70, l=20, r=20, t=90),
            legend=dict(orientation="h", yanchor="top", y=-0.02,
                        xanchor="center", x=0.5),
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

    # ---- animated PCA projection ------------------------------------------ #
    def create_pca_animated_figure(
        self,
        result: SimulationResult,
        max_frames: int = 60,
        n_fp_candidates: int = 10,
    ) -> go.Figure:
        """Animated, rotatable 3-D PCA projection of the network's
        state-space trajectory.

        What this shows
        ---------------
        The network's full N-dimensional activity vector is projected onto
        its first three principal components (PCs), computed from the
        entire simulation run. Each animation frame moves a marker along
        this 3-D trajectory, revealing how the network's collective state
        travels through its dominant subspace over time. The scene can be
        freely rotated, panned, and zoomed by dragging on the plot.

        Fixed points are overlaid as larger background markers:
          * Green filled diamond = stable fixed point (all Jacobian eigenvalues Re < 0)
          * Orange open diamond  = unstable fixed point (at least one Re \u2265 0)

        Fixed points are found by relaxing a set of candidate states sampled
        from the trajectory, then projecting the converged points into the
        same PC space.

        Controls
        --------
        Same Play/Pause/speed (0.5x, 1x, 2x) buttons and scrub-slider as
        the existing Animation tab.
        """
        activity = result.activity                            # (n, T)
        n_steps = activity.shape[1]

        # Evenly sample frames.
        if n_steps <= max_frames:
            frame_idx = list(range(n_steps))
        else:
            frame_idx = list(np.linspace(0, n_steps - 1, max_frames, dtype=int))

        # ---------- PCA ------------------------------------------------------ #
        coords, components, explained = _compute_pca_projection(activity)
        # coords: (T, 3), full trajectory

        pc1_label = f"PC1 ({explained[0] * 100:.1f}% var)"
        pc2_label = f"PC2 ({explained[1] * 100:.1f}% var)"
        pc3_label = f"PC3 ({explained[2] * 100:.1f}% var)"

        # ---------- fixed points --------------------------------------------- #
        fp_pca, fp_stable = _find_fixed_points_pca(
            result.weight_matrix, activity, components,
            n_candidates=n_fp_candidates,
        )

        # ---------- full ghost trajectory trace (static background) ---------- #
        ghost_trace = go.Scatter3d(
            x=coords[:, 0].tolist(),
            y=coords[:, 1].tolist(),
            z=coords[:, 2].tolist(),
            mode="lines",
            line=dict(color="#cccccc", width=1.5),
            name="Full trajectory",
            hoverinfo="skip",
            showlegend=True,
        )

        # ---------- fixed point traces (static background) ------------------- #
        fp_traces: list[go.Scatter3d] = []
        if fp_pca.shape[0] > 0:
            stable_mask = fp_stable
            unstable_mask = ~fp_stable

            if stable_mask.any():
                fp_traces.append(go.Scatter3d(
                    x=fp_pca[stable_mask, 0].tolist(),
                    y=fp_pca[stable_mask, 1].tolist(),
                    z=fp_pca[stable_mask, 2].tolist(),
                    mode="markers",
                    marker=dict(
                        symbol="diamond", size=10,
                        color="#2ca02c",          # green
                        line=dict(width=1.5, color="#1a7a1a"),
                    ),
                    name="Stable fixed point",
                    hovertemplate="Stable FP<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>",
                ))

            if unstable_mask.any():
                fp_traces.append(go.Scatter3d(
                    x=fp_pca[unstable_mask, 0].tolist(),
                    y=fp_pca[unstable_mask, 1].tolist(),
                    z=fp_pca[unstable_mask, 2].tolist(),
                    mode="markers",
                    marker=dict(
                        symbol="diamond-open", size=9,
                        color="#ff7f0e",          # orange
                        line=dict(width=1.5, color="#b35900"),
                    ),
                    name="Unstable fixed point",
                    hovertemplate="Unstable FP<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>",
                ))

        # ---------- color-encode time along the trajectory ------------------- #
        # Use a sequential colorscale so the moving dot inherits the frame's
        # normalized time position.
        n_frames = len(frame_idx)
        color_vals = list(range(n_frames))    # 0 .. n_frames-1 for colorscale

        def _moving_dot(fi: int) -> go.Scatter3d:
            """Scatter3d trace for the animated state-dot at frame index fi."""
            step = frame_idx[fi]
            x_val = float(coords[step, 0])
            y_val = float(coords[step, 1])
            z_val = float(coords[step, 2])
            t_val = float(result.time[step])
            return go.Scatter3d(
                x=[x_val],
                y=[y_val],
                z=[z_val],
                mode="markers",
                marker=dict(
                    size=8,
                    color=[color_vals[fi]],
                    colorscale="Plasma",
                    cmin=0, cmax=n_frames - 1,
                    showscale=True,
                    colorbar=dict(
                        title="frame",
                        thickness=12,
                        len=0.6,
                        tickvals=[0, (n_frames - 1) // 2, n_frames - 1],
                        ticktext=["start", "mid", "end"],
                    ),
                    line=dict(width=2, color="white"),
                    symbol="circle",
                ),
                name=f"State (t={t_val:.2f})",
                hovertemplate=(
                    f"t = {t_val:.3f}<br>"
                    "PC1 = %{x:.3f}<br>PC2 = %{y:.3f}<br>PC3 = %{z:.3f}<extra></extra>"
                ),
                showlegend=False,
            )

        # Static traces: ghost + fixed points; animated trace is appended last.
        static_traces = [ghost_trace, *fp_traces]
        anim_trace_idx = len(static_traces)   # index of the animated dot trace

        first_dot = _moving_dot(0)
        frames = [
            go.Frame(
                name=str(int(frame_idx[fi])),
                data=[_moving_dot(fi)],
                traces=[anim_trace_idx],
            )
            for fi in range(n_frames)
        ]

        # ---------- media controls ------------------------------------------- #
        base_ms = 100
        speeds = [("0.5x", base_ms * 2), ("1x", base_ms), ("2x", base_ms // 2)]
        play_buttons = [
            dict(
                label=f"\u25b6 {label}", method="animate",
                args=[None, dict(
                    frame=dict(duration=ms, redraw=True),
                    transition=dict(duration=0),
                    fromcurrent=True, mode="immediate",
                )],
            )
            for label, ms in speeds
        ]
        pause_button = dict(
            label="\u23f8 Pause", method="animate",
            args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
        )

        frame_names = [str(int(s)) for s in frame_idx]
        slider_steps = [
            dict(
                label=result_time_label(result, int(s)),
                method="animate",
                args=[[name], dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0),
                )],
            )
            for name, s in zip(frame_names, frame_idx)
        ]

        # ---------- assemble figure ------------------------------------------ #
        fig = go.Figure(
            data=[*static_traces, first_dot],
            frames=frames,
        )
        fig.update_layout(
            title=dict(
                text=(
                    f"PCA state-space trajectory \u2014 {_model_display_name(result)}"
                ),
                x=0.5, xanchor="center",
            ),
            scene=dict(
                xaxis=dict(
                    title=pc1_label,
                    showgrid=True, gridcolor="#eee", zeroline=True, zerolinecolor="#ccc",
                ),
                yaxis=dict(
                    title=pc2_label,
                    showgrid=True, gridcolor="#eee", zeroline=True, zerolinecolor="#ccc",
                ),
                zaxis=dict(
                    title=pc3_label,
                    showgrid=True, gridcolor="#eee", zeroline=True, zerolinecolor="#ccc",
                ),
                bgcolor="white",
            ),
            margin=dict(b=20, l=60, r=20, t=90),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="closest",
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

    def _motif_traces(self, motif_meta: list[dict],
                      pos: dict[int, tuple[float, float]]) -> list[go.Scatter]:
        """Build highlight traces for motif nodes and edges from metadata.

        Uses the motif metadata directly (node/edge indices and signs) rather
        than trying to rediscover motifs from the weight matrix. Motif edges
        are emphasized in green (positive) or red (negative). Motif nodes get
        a ring whose color identifies the motif TYPE, with one legend entry
        per type actually present. Returns an empty list when there are no
        motifs.
        """
        if not motif_meta:
            return []

        traces: list[go.Scatter] = []

        # 1. Emphasized motif edges, split by sign for color.
        for want_positive, color in ((True, MOTIF_POSITIVE_COLOR),
                                     (False, MOTIF_NEGATIVE_COLOR)):
            xs: list[float | None] = []
            ys: list[float | None] = []
            for motif in motif_meta:
                signs = motif.get("node_signs", {})
                for src, tgt in motif.get("edges", []):
                    # An edge is "positive" if its source node is excitatory.
                    src_sign = signs.get(src) or signs.get(str(src)) or "positive"
                    is_pos = src_sign == "positive"
                    if is_pos != want_positive:
                        continue
                    if src not in pos or tgt not in pos:
                        continue
                    x0, y0 = pos[src]
                    x1, y1 = pos[tgt]
                    xs += [x0, x1, None]
                    ys += [y0, y1, None]
            if xs:
                traces.append(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=color, width=3.5),
                    hoverinfo="none", showlegend=False, opacity=0.9,
                ))

        # 2. Rings around motif nodes, colored by motif type. One marker
        #    trace per type keeps colors grouped and hover text meaningful.
        by_type: dict[str, list[int]] = {}
        for motif in motif_meta:
            mtype = motif.get("motif_type", "motif")
            by_type.setdefault(mtype, []).extend(motif.get("nodes", []))

        for mtype, nodes in by_type.items():
            ring_color = MOTIF_TYPE_COLORS.get(mtype, MOTIF_RING_COLOR)
            label = MOTIF_TYPE_LABELS.get(mtype, mtype)
            rx = [pos[n][0] for n in nodes if n in pos]
            ry = [pos[n][1] for n in nodes if n in pos]
            hover = [f"{label}<br>neuron {n}" for n in nodes if n in pos]
            if not rx:
                continue
            traces.append(go.Scatter(
                x=rx, y=ry, mode="markers",
                marker=dict(size=22, color="rgba(0,0,0,0)",
                            line=dict(color=ring_color, width=3)),
                hovertext=hover, hoverinfo="text",
                name=label, showlegend=True,
            ))

        # 3. Legend entries for the edge sign encoding (shared by all motifs).
        traces.append(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=MOTIF_POSITIVE_COLOR, width=3.5),
            name="motif excitatory (+)",
        ))
        traces.append(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=MOTIF_NEGATIVE_COLOR, width=3.5),
            name="motif inhibitory (\u2212)",
        ))
        return traces

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


def build_pca_animation_figure(result: SimulationResult) -> go.Figure:
    """Convenience entry point: result -> animated PCA state-space figure.

    Projects the full activity trajectory onto its first three principal
    components and animates the network state moving through that 3-D
    subspace, which can be freely rotated. Fixed points (stable/unstable)
    are overlaid as background markers so you can see where the trajectory
    is attracted to or repelled from.
    """
    visualizer = NetworkVisualizer(result.config.visualization)
    return visualizer.create_pca_animated_figure(result)
