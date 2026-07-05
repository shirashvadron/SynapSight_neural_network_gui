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
  * an animated PCA projection of network state-space dynamics, with
    fixed-point markers overlaid (new: build_pca_animation_figure);
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
    """Human-readable model/source name for figure titles."""
    if result.metadata.get("network_source") == "imported":
        return result.metadata.get("network_source_name", "Imported network")

    key = result.config.network.model_type
    model = MODEL_REGISTRY.get(key)
    return model.name if model is not None else key


def result_time_label(result: SimulationResult, step: int) -> str:
    """Short label for a time step, showing the actual simulated time."""
    return f"{result.time[step]:.2f}"


# --------------------------------------------------------------------------- #
# PCA helpers
# --------------------------------------------------------------------------- #

def _compute_pca_projection(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (n_neurons, n_steps) activity matrix onto its first 2 PCs.

    Returns
    -------
    coords   : (n_steps, 2)  PC coordinates for every time step
    components : (2, n_neurons)  the two principal directions
    explained  : (2,)  fraction of variance explained by each PC
    """
    X = activity - activity.mean(axis=1, keepdims=True)   # (n, T)
    n, T = activity.shape

    U, s, Vt = np.linalg.svd(X.T, full_matrices=False)    # U:(T,k), s:(k,), Vt:(k,n)
    k = s.shape[0]
    total_var = (s ** 2).sum() or 1.0

    coords = U * s  # (T, k)

    if k < 2:
        coords = np.hstack([coords, np.zeros((T, 2 - k))])
        components = np.vstack([Vt, np.zeros((2 - k, n))])
        explained = np.concatenate([(s ** 2) / total_var, np.zeros(2 - k)])
    else:
        components = Vt[:2]
        explained = (s[:2] ** 2) / total_var

    return coords[:, :2], components[:2], explained[:2]


def _find_fixed_points_pca(
    weight_matrix: np.ndarray,
    activity: np.ndarray,
    components: np.ndarray,
    n_candidates: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate fixed points and project them into PCA space.

    Strategy
    --------
    We take several candidate states from the activity trajectory (evenly
    spaced), then refine each one via a short fixed-point iteration:
        x <- tanh-inverse of the effective drive  (Newton-like relaxation)
    In practice we use a simple gradient-descent relaxation:
        x <- x - alpha * (-x + W @ tanh(x))
    until convergence, which is fast for well-conditioned networks.

    Each converged point is then classified as stable or unstable by the
    spectral radius of its Jacobian  J = -I + W * diag(sech²(x*)).
    If all eigenvalues of J have Re < 0 the point is stable; otherwise unstable.

    Returns
    -------
    fp_pca    : (m, 2)  unique fixed points in PC space
    stability : (m,)  bool array, True = stable
    """
    n, T = activity.shape
    W = weight_matrix
    alpha = 0.05
    max_iter = 400
    tol = 1e-6

    # Candidate starting points: evenly sampled from trajectory.
    cand_indices = np.linspace(0, T - 1, n_candidates, dtype=int)
    candidates = [activity[:, i].copy() for i in cand_indices]

    fps_raw: list[np.ndarray] = []
    stabilities: list[bool] = []

    for x0 in candidates:
        x = x0.copy()
        for _ in range(max_iter):
            dx = -x + W @ np.tanh(x)
            x = x + alpha * dx
            if np.linalg.norm(dx) < tol:
                break
        # Deduplicate: skip if too close to an existing fixed point.
        is_dup = any(np.linalg.norm(x - fp) < 0.1 for fp in fps_raw)
        if not is_dup:
            fps_raw.append(x.copy())
            # Stability: Jacobian eigenvalues.
            sech2 = 1.0 / (np.cosh(x) ** 2)          # (n,)
            J = -np.eye(n) + W * sech2[np.newaxis, :]  # (n, n)
            eigvals = np.linalg.eigvals(J)
            stable = bool(np.all(eigvals.real < 0))
            stabilities.append(stable)

    if not fps_raw:
        return np.empty((0, 2)), np.empty(0, dtype=bool)

    # Project fixed points into PCA space using the same components.
    X_mean = activity.mean(axis=1)                         # (n,)
    fp_matrix = np.array(fps_raw)                          # (m, n)
    fp_centred = fp_matrix - X_mean[np.newaxis, :]         # (m, n)
    fp_pca = fp_centred @ components.T                     # (m, 2)

    return fp_pca, np.array(stabilities)


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
        """Animated PCA projection of the network's state-space trajectory.

        What this shows
        ---------------
        The network's full N-dimensional activity vector is projected onto
        its first two principal components (PCs), computed from the entire
        simulation run. Each animation frame moves a marker along this
        2-D trajectory, revealing how the network's collective state
        travels through its dominant subspace over time.

        Fixed points are overlaid as larger background markers:
          * Green star  = stable fixed point  (all Jacobian eigenvalues Re < 0)
          * Orange diamond = unstable fixed point (at least one Re \u2265 0)

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
        # coords: (T, 2), full trajectory

        pc1_label = f"PC1 ({explained[0] * 100:.1f}% var)"
        pc2_label = f"PC2 ({explained[1] * 100:.1f}% var)"

        # ---------- fixed points --------------------------------------------- #
        fp_pca, fp_stable = _find_fixed_points_pca(
            result.weight_matrix, activity, components,
            n_candidates=n_fp_candidates,
        )

        # ---------- full ghost trajectory trace (static background) ---------- #
        ghost_trace = go.Scatter(
            x=coords[:, 0].tolist(),
            y=coords[:, 1].tolist(),
            mode="lines",
            line=dict(color="#cccccc", width=1.5),
            name="Full trajectory",
            hoverinfo="skip",
            showlegend=True,
        )

        # ---------- fixed point traces (static background) ------------------- #
        fp_traces: list[go.Scatter] = []
        if fp_pca.shape[0] > 0:
            stable_mask = fp_stable
            unstable_mask = ~fp_stable

            if stable_mask.any():
                fp_traces.append(go.Scatter(
                    x=fp_pca[stable_mask, 0].tolist(),
                    y=fp_pca[stable_mask, 1].tolist(),
                    mode="markers",
                    marker=dict(
                        symbol="star", size=18,
                        color="#2ca02c",          # green
                        line=dict(width=1.5, color="#1a7a1a"),
                    ),
                    name="Stable fixed point",
                    hovertemplate="Stable FP<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
                ))

            if unstable_mask.any():
                fp_traces.append(go.Scatter(
                    x=fp_pca[unstable_mask, 0].tolist(),
                    y=fp_pca[unstable_mask, 1].tolist(),
                    mode="markers",
                    marker=dict(
                        symbol="diamond", size=14,
                        color="#ff7f0e",          # orange
                        line=dict(width=1.5, color="#b35900"),
                    ),
                    name="Unstable fixed point",
                    hovertemplate="Unstable FP<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
                ))

        # ---------- color-encode time along the trajectory ------------------- #
        # Use a sequential colorscale so the moving dot inherits the frame's
        # normalized time position.
        n_frames = len(frame_idx)
        color_vals = list(range(n_frames))    # 0 .. n_frames-1 for colorscale

        def _moving_dot(fi: int) -> go.Scatter:
            """Scatter trace for the animated state-dot at frame index fi."""
            step = frame_idx[fi]
            x_val = float(coords[step, 0])
            y_val = float(coords[step, 1])
            t_val = float(result.time[step])
            return go.Scatter(
                x=[x_val],
                y=[y_val],
                mode="markers",
                marker=dict(
                    size=16,
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
                    "PC1 = %{x:.3f}<br>PC2 = %{y:.3f}<extra></extra>"
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
            xaxis=dict(
                title=pc1_label,
                showgrid=True, gridcolor="#eee", zeroline=True, zerolinecolor="#ccc",
            ),
            yaxis=dict(
                title=pc2_label,
                showgrid=True, gridcolor="#eee", zeroline=True, zerolinecolor="#ccc",
            ),
            plot_bgcolor="white",
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

    Projects the full activity trajectory onto its first two principal
    components and animates the network state moving through that 2-D
    subspace. Fixed points (stable/unstable) are overlaid as background
    markers so you can see where the trajectory is attracted to or repelled
    from.
    """
    visualizer = NetworkVisualizer(result.config.visualization)
    return visualizer.create_pca_animated_figure(result)


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


# --------------------------------------------------------------------------- #
# Event-based/spike-specific figures
# --------------------------------------------------------------------------- #
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
