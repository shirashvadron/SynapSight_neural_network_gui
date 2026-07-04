"""Tests for visualization, IO utilities, and full pipeline integration."""

import numpy as np
import plotly.graph_objects as go
import io
import pytest

from nnsimviz.configs import (
    NetworkConfig,
    SimulationConfig,
    VisualizationConfig,
    ProjectConfig,
)
from nnsimviz.models import build_weight_matrix
from nnsimviz.simulation import Simulator
from nnsimviz.visualization import (
    NetworkVisualizer,
    build_figure,
    build_pca_animation_figure,
    _compute_pca_projection,
    _find_fixed_points_pca,
)
from nnsimviz import io_utils


@pytest.fixture
def result():
    cfg = ProjectConfig(
        network=NetworkConfig(n_neurons=12, connection_probability=0.4, random_seed=7),
        simulation=SimulationConfig(duration=5.0, dt=0.1),
        visualization=VisualizationConfig(layout_type="spring"),
    )
    W = build_weight_matrix(cfg.network)
    return Simulator().run(cfg, W)


@pytest.fixture
def pca_result():
    """A clean, deterministic SimulationResult used across all PCA tests."""
    cfg = ProjectConfig(
        network=NetworkConfig(n_neurons=10, connection_probability=0.5, random_seed=42),
        simulation=SimulationConfig(duration=10.0, dt=0.1, noise_level=0.0),
        visualization=VisualizationConfig(layout_type="spring"),
    )
    W = build_weight_matrix(cfg.network)
    return Simulator().run(cfg, W)


class TestVisualization:
    def test_build_graph_node_count(self, result):
        viz = NetworkVisualizer(result.config.visualization)
        graph, edges = viz.build_graph(result.weight_matrix)
        assert graph.number_of_nodes() == 12

    def test_edge_filtering(self, result):
        cfg = VisualizationConfig(min_edge_abs_weight=100.0)  # filter everything
        viz = NetworkVisualizer(cfg)
        _, edges = viz.build_graph(result.weight_matrix)
        assert edges == []

    @pytest.mark.parametrize("layout", VisualizationConfig.VALID_LAYOUTS)
    def test_layouts_produce_positions(self, result, layout):
        cfg = VisualizationConfig(layout_type=layout)
        viz = NetworkVisualizer(cfg)
        graph, _ = viz.build_graph(result.weight_matrix)
        pos = viz.compute_layout(graph)
        assert set(pos.keys()) == set(graph.nodes())

    def test_create_figure_returns_figure(self, result):
        fig = build_figure(result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_figure_has_legend_traces(self, result):
        fig = build_figure(result)
        names = [t.name for t in fig.data if t.name]
        assert any("positive" in n for n in names)
        assert any("negative" in n for n in names)


# =========================================================================== #
# PCA Logic Tests
# =========================================================================== #

class TestComputePcaProjection:
    """Unit tests for the _compute_pca_projection helper."""

    def test_output_shapes(self, pca_result):
        """coords must be (T, 2), components (2, n), explained (2,)."""
        activity = pca_result.activity  # (n, T)
        n, T = activity.shape
        coords, components, explained = _compute_pca_projection(activity)
        assert coords.shape == (T, 2)
        assert components.shape == (2, n)
        assert explained.shape == (2,)

    def test_explained_variance_sums_to_at_most_one(self, pca_result):
        """The two explained-variance fractions must be in [0, 1] and sum ≤ 1."""
        _, _, explained = _compute_pca_projection(pca_result.activity)
        assert np.all(explained >= 0.0)
        assert np.all(explained <= 1.0)
        assert explained.sum() <= 1.0 + 1e-9  # allow tiny float error

    def test_explained_variance_descending(self, pca_result):
        """PC1 must explain at least as much variance as PC2."""
        _, _, explained = _compute_pca_projection(pca_result.activity)
        assert explained[0] >= explained[1] - 1e-9

    def test_coords_are_finite(self, pca_result):
        """All projected coordinates must be finite numbers."""
        coords, _, _ = _compute_pca_projection(pca_result.activity)
        assert np.isfinite(coords).all()

    def test_components_are_orthonormal(self, pca_result):
        """The two PC directions must be orthogonal and unit-length."""
        _, components, _ = _compute_pca_projection(pca_result.activity)
        # Gram-matrix of the two rows should be the 2x2 identity.
        gram = components @ components.T
        assert np.allclose(gram, np.eye(2), atol=1e-6)

    def test_known_rank1_signal_concentrates_variance_in_pc1(self):
        """Rank-1 activity → PC1 should capture almost all variance."""
        rng = np.random.default_rng(0)
        # Activity = outer product of a neuron direction and a time signal
        n, T = 8, 200
        neuron_dir = rng.normal(size=n)
        time_sig = rng.normal(size=T)
        activity = np.outer(neuron_dir, time_sig)  # (n, T)
        _, _, explained = _compute_pca_projection(activity)
        # First PC should capture >99% of variance for a rank-1 signal.
        assert explained[0] > 0.99

    def test_isotropic_noise_spreads_variance(self):
        """White-noise activity → explained[0] should be well below 1."""
        rng = np.random.default_rng(1)
        activity = rng.normal(size=(20, 500))
        _, _, explained = _compute_pca_projection(activity)
        # For i.i.d. data the first PC holds only a small fraction of variance.
        assert explained[0] < 0.5

    def test_mean_centering_is_applied(self):
        """Adding a constant offset to every neuron should not change the
        coordinates (because the function centres each neuron across time)."""
        rng = np.random.default_rng(2)
        activity = rng.normal(size=(6, 100))
        coords_base, _, _ = _compute_pca_projection(activity)
        offset = rng.normal(size=(6, 1)) * 10  # large per-neuron constant
        coords_shifted, _, _ = _compute_pca_projection(activity + offset)
        # The sign of each PC can flip; compare absolute coordinates.
        assert np.allclose(np.abs(coords_base), np.abs(coords_shifted), atol=1e-6)

    def test_single_timestep_does_not_crash(self):
        """Edge case: only one time step should return zeros without error."""
        activity = np.ones((5, 1))  # (n=5, T=1)
        coords, components, explained = _compute_pca_projection(activity)
        assert coords.shape == (1, 2)
        assert np.isfinite(coords).all()


class TestFindFixedPointsPca:
    """Unit tests for the _find_fixed_points_pca helper."""

    def test_output_shapes_consistent(self, pca_result):
        """fp_pca and stability arrays must have matching first dimension."""
        activity = pca_result.activity
        _, components, _ = _compute_pca_projection(activity)
        fp_pca, stability = _find_fixed_points_pca(
            pca_result.weight_matrix, activity, components
        )
        assert fp_pca.ndim == 2
        assert fp_pca.shape[1] == 2
        assert fp_pca.shape[0] == stability.shape[0]

    def test_stability_is_bool_array(self, pca_result):
        """The stability array must be a boolean dtype array."""
        activity = pca_result.activity
        _, components, _ = _compute_pca_projection(activity)
        _, stability = _find_fixed_points_pca(
            pca_result.weight_matrix, activity, components
        )
        assert stability.dtype == bool

    def test_fixed_points_satisfy_equilibrium(self):
        """In a 2-neuron system, back-projection from 2 PCs is exact, so each
        recovered fixed point should satisfy ||f(x*)|| ≈ 0."""
        W = np.array([
            [0.0, 0.25],
            [-0.15, 0.0],
        ])

        dt = 0.1
        T = 120
        x = np.array([0.3, -0.2], dtype=float)
        activity = np.zeros((2, T))

        for t in range(T):
            activity[:, t] = x
            x = x + dt * (-x + W @ np.tanh(x))

        _, components, _ = _compute_pca_projection(activity)
        fp_pca, _ = _find_fixed_points_pca(W, activity, components, n_candidates=6)

        assert fp_pca.shape[0] > 0

        X_mean = activity.mean(axis=1)
        fp_neuron = fp_pca @ components + X_mean[np.newaxis, :]

        for x_star in fp_neuron:
            residual = np.linalg.norm(-x_star + W @ np.tanh(x_star))
            assert residual < 1e-3, f"Fixed point residual too large: {residual:.6f}"

    def test_empty_activity_returns_empty(self):
        """When activity has zero time steps the function should not crash."""
        n = 4
        W = np.zeros((n, n))
        activity = np.zeros((n, 1))  # single frozen state
        _, components, _ = _compute_pca_projection(activity)
        fp_pca, stability = _find_fixed_points_pca(W, activity, components, n_candidates=2)
        assert fp_pca.shape[1] == 2
        assert fp_pca.shape[0] == stability.shape[0]

    def test_zero_weight_matrix_has_stable_fixed_point_at_origin(self):
        """With W=0 the only fixed point is x=0, which must be stable."""
        n = 5
        rng = np.random.default_rng(99)
        W = np.zeros((n, n))
        # Activity that starts away from zero and decays to zero
        T = 100
        activity = np.outer(rng.normal(size=n), np.exp(-np.linspace(0, 4, T)))
        _, components, _ = _compute_pca_projection(activity)
        fp_pca, stability = _find_fixed_points_pca(W, activity, components, n_candidates=8)
        assert fp_pca.shape[0] > 0, "Expected at least one fixed point for W=0"
        # The zero fixed point must be stable (eigenvalues all == -1)
        assert np.any(stability), "Expected at least one stable fixed point"

    def test_fp_pca_coords_are_finite(self, pca_result):
        """All returned PCA-space coordinates must be finite."""
        activity = pca_result.activity
        _, components, _ = _compute_pca_projection(activity)
        fp_pca, _ = _find_fixed_points_pca(
            pca_result.weight_matrix, activity, components
        )
        assert np.isfinite(fp_pca).all()


# =========================================================================== #
# PCA Visualization Tests
# =========================================================================== #

class TestPcaAnimatedFigure:
    """Tests for the animated PCA figure produced by NetworkVisualizer."""

    def test_returns_plotly_figure(self, pca_result):
        """build_pca_animation_figure must return a go.Figure."""
        fig = build_pca_animation_figure(pca_result)
        assert isinstance(fig, go.Figure)

    def test_figure_has_frames(self, pca_result):
        """The animated figure must contain at least one frame."""
        fig = build_pca_animation_figure(pca_result)
        assert len(fig.frames) > 0

    def test_frame_count_bounded_by_max_frames(self, pca_result):
        """Frame count must not exceed max_frames."""
        max_frames = 15
        viz = NetworkVisualizer(pca_result.config.visualization)
        fig = viz.create_pca_animated_figure(pca_result, max_frames=max_frames)
        assert len(fig.frames) <= max_frames

    def test_figure_has_ghost_trajectory_trace(self, pca_result):
        """The static ghost trajectory trace must be present in fig.data."""
        fig = build_pca_animation_figure(pca_result)
        trace_names = [t.name for t in fig.data if hasattr(t, "name") and t.name]
        assert any("trajectory" in n.lower() for n in trace_names)

    def test_xaxis_and_yaxis_titles_contain_pc_labels(self, pca_result):
        """X/Y axis titles must mention PC1 and PC2."""
        fig = build_pca_animation_figure(pca_result)
        xaxis_title = fig.layout.xaxis.title.text or ""
        yaxis_title = fig.layout.yaxis.title.text or ""
        assert "PC1" in xaxis_title
        assert "PC2" in yaxis_title

    def test_figure_title_contains_pca(self, pca_result):
        """The figure title must reference PCA or state-space."""
        fig = build_pca_animation_figure(pca_result)
        title_text = (fig.layout.title.text or "").lower()
        assert "pca" in title_text or "state" in title_text

    def test_figure_has_play_button(self, pca_result):
        """The figure layout must contain at least one updatemenus entry
        with a Play button."""
        fig = build_pca_animation_figure(pca_result)
        assert fig.layout.updatemenus, "No updatemenus found"
        all_labels = [
            btn.label
            for menu in fig.layout.updatemenus
            for btn in (menu.buttons or [])
        ]
        assert any("\u25b6" in lbl or "Play" in lbl for lbl in all_labels)

    def test_figure_has_slider(self, pca_result):
        """A time scrub slider must be present."""
        fig = build_pca_animation_figure(pca_result)
        assert fig.layout.sliders, "No sliders found in figure layout"

    def test_first_frame_dot_within_trajectory_bounds(self, pca_result):
        """The first frame's animated dot must lie within the full trajectory."""
        activity = pca_result.activity
        coords, _, _ = _compute_pca_projection(activity)
        fig = build_pca_animation_figure(pca_result)

        # The last trace in fig.data is the initial moving dot.
        dot_trace = fig.data[-1]
        x_dot = dot_trace.x[0]
        y_dot = dot_trace.y[0]

        assert coords[:, 0].min() <= x_dot <= coords[:, 0].max()
        assert coords[:, 1].min() <= y_dot <= coords[:, 1].max()

    def test_stable_fixed_point_trace_color_is_green(self, pca_result):
        """If a stable fixed-point trace is present it must use green."""
        fig = build_pca_animation_figure(pca_result)
        for trace in fig.data:
            if hasattr(trace, "name") and trace.name == "Stable fixed point":
                assert trace.marker.color == "#2ca02c"
                break  # found it

    def test_unstable_fixed_point_trace_color_is_orange(self, pca_result):
        """If an unstable fixed-point trace is present it must use orange."""
        fig = build_pca_animation_figure(pca_result)
        for trace in fig.data:
            if hasattr(trace, "name") and trace.name == "Unstable fixed point":
                assert trace.marker.color == "#ff7f0e"
                break

    def test_explained_variance_in_axis_labels(self, pca_result):
        """Axis labels must include the '% var' string from the explained
        variance formatting."""
        fig = build_pca_animation_figure(pca_result)
        xaxis_title = fig.layout.xaxis.title.text or ""
        yaxis_title = fig.layout.yaxis.title.text or ""
        assert "% var" in xaxis_title
        assert "% var" in yaxis_title

    def test_small_network_does_not_crash(self):
        """A 2-neuron network (minimum valid) must produce a figure without error."""
        cfg = ProjectConfig(
            network=NetworkConfig(n_neurons=2, random_seed=1),
            simulation=SimulationConfig(duration=3.0, dt=0.1, noise_level=0.0),
            visualization=VisualizationConfig(),
        )
        W = build_weight_matrix(cfg.network)
        result = Simulator().run(cfg, W)
        fig = build_pca_animation_figure(result)
        assert isinstance(fig, go.Figure)

    def test_max_frames_one_produces_single_frame(self, pca_result):
        """max_frames=1 must result in exactly one animation frame."""
        viz = NetworkVisualizer(pca_result.config.visualization)
        fig = viz.create_pca_animated_figure(pca_result, max_frames=1)
        assert len(fig.frames) == 1


class TestIO:
    def test_config_json_roundtrip(self):
        cfg = ProjectConfig(network=NetworkConfig(n_neurons=9, random_seed=11))
        text = io_utils.config_to_json(cfg)
        restored = io_utils.config_from_json(text)
        assert restored.network.n_neurons == 9
        assert restored.network.random_seed == 11

    def test_save_and_load_config(self, tmp_path):
        cfg = ProjectConfig(network=NetworkConfig(n_neurons=6))
        path = tmp_path / "cfg.json"
        io_utils.save_config(cfg, path)
        restored = io_utils.load_config(path)
        assert restored.network.n_neurons == 6

    def test_weight_matrix_csv(self, result):
        csv = io_utils.weight_matrix_to_csv(result.weight_matrix)
        n_rows = result.weight_matrix.shape[0]
        assert csv.strip().count("\n") == n_rows - 1

    def test_activity_csv_has_header_and_rows(self, result):
        csv = io_utils.activity_to_csv(result)
        lines = csv.strip().split("\n")
        # 1 header + one row per neuron
        assert len(lines) == result.activity.shape[0] + 1

    def test_import_weight_matrix_from_csv(self):
        data = b"0.0,1.0\n-2.0,0.0\n"
        W = io_utils.load_weight_matrix_from_upload("weights.csv", data)
        expected = np.array([
            [0.0, 1.0],
            [-2.0, 0.0],
        ])
        assert np.array_equal(W, expected)

    def test_import_weight_matrix_from_npy(self):
        original = np.array([
            [0.0, 0.5],
            [-0.2, 0.0],
        ])
        buf = io.BytesIO()
        np.save(buf, original)
        W = io_utils.load_weight_matrix_from_upload("weights.npy", buf.getvalue())
        assert np.array_equal(W, original)

    def test_import_weight_matrix_from_npz_with_W_key(self):
        original = np.array([
            [0.0, 0.5],
            [-0.2, 0.0],
        ])
        buf = io.BytesIO()
        np.savez(buf, W=original)
        W = io_utils.load_weight_matrix_from_upload("weights.npz", buf.getvalue())
        assert np.array_equal(W, original)

    def test_import_weight_matrix_rejects_non_square_matrix(self):
        data = b"1.0,2.0,3.0\n4.0,5.0,6.0\n"
        with pytest.raises(ValueError, match="square"):
            io_utils.load_weight_matrix_from_upload("bad.csv", data)

    def test_import_weight_matrix_rejects_nan_values(self):
        data = b"0.0,nan\n1.0,0.0\n"
        with pytest.raises(ValueError, match="finite"):
            io_utils.load_weight_matrix_from_upload("bad.csv", data)

    def test_import_weight_matrix_rejects_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported"):
            io_utils.load_weight_matrix_from_upload("weights.txt", b"0,1\n1,0")


class TestIntegration:
    def test_full_pipeline_runs_end_to_end(self):
        cfg = ProjectConfig(
            network=NetworkConfig(n_neurons=15, random_seed=42),
            simulation=SimulationConfig(duration=10.0, dt=0.1, input_type="constant"),
            visualization=VisualizationConfig(layout_type="circular"),
        )
        cfg.validate()
        W = build_weight_matrix(cfg.network)
        result = Simulator().run(cfg, W)
        fig = build_figure(result)
        assert isinstance(fig, go.Figure)
        assert result.metadata["n_neurons"] == 15
        assert np.isfinite(result.activity).all()

    def test_pipeline_respects_data_contract_fields(self):
        cfg = ProjectConfig(network=NetworkConfig(n_neurons=8))
        W = build_weight_matrix(cfg.network)
        result = Simulator().run(cfg, W)
        # SimulationResult must carry exactly the contract fields.
        for field in ("time", "activity", "final_state",
                      "weight_matrix", "config", "metadata"):
            assert hasattr(result, field)

    def test_imported_weight_matrix_runs_end_to_end(self):
        W = np.array([
            [0.0, 0.7, 0.0],
            [-0.3, 0.0, 0.2],
            [0.1, 0.0, 0.0],
        ])
        cfg = ProjectConfig(
            network=NetworkConfig(n_neurons=W.shape[0]),
            simulation=SimulationConfig(
                duration=2.0,
                dt=0.1,
                input_type="constant",
                noise_level=0.0,
            ),
            visualization=VisualizationConfig(layout_type="circular"),
        )
        result = Simulator().run(cfg, io_utils.validate_weight_matrix(W))
        result.metadata["network_source"] = "imported"
        result.metadata["network_source_name"] = "Imported weight matrix"
        fig = build_figure(result)
        assert isinstance(fig, go.Figure)
        assert result.weight_matrix.shape == (3, 3)
        assert np.array_equal(result.weight_matrix, W)
        assert np.isfinite(result.activity).all()


if __name__ == "__main__":
    test_names = [name for name in globals() if name.startswith("test_")]
    # the -v flag is for verbose output, and the -k flag allows us to specify which tests to run
    pytest.main(["-v", "-k", " or ".join(test_names)])
