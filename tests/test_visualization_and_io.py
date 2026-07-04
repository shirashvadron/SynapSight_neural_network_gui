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
from nnsimviz.visualization import NetworkVisualizer, build_figure
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