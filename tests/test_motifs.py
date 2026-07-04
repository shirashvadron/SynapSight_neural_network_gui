"""Tests for the network motif feature."""

import numpy as np
import pytest

from nnsimviz.configs import ProjectConfig, NetworkConfig, MotifConfig
from nnsimviz.models import build_weight_matrix
from nnsimviz.simulation import Simulator
from nnsimviz.motifs import (
    MotifType,
    MotifInstance,
    add_motifs,
    build_network_with_motifs,
    get_template,
    MOTIF_SIZES,
)
from nnsimviz.visualization import build_figure


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _base_matrix(n: int = 10, seed: int = 0) -> np.ndarray:
    """A small deterministic base network for motif tests."""
    return build_weight_matrix(NetworkConfig(n_neurons=n, random_seed=seed))


# --------------------------------------------------------------------------- #
# 1. Motif construction
# --------------------------------------------------------------------------- #
class TestMotifConstruction:
    def test_coincidence_detector_counts(self):
        cfg = MotifConfig(enabled=True, n_coincidence_detector=1,
                          n_external_connections=0)
        _, meta = add_motifs(_base_matrix(), cfg)
        m = meta["motifs"][0]
        assert m["motif_type"] == "coincidence_detector"
        assert len(m["nodes"]) == 4        # 3 sources + 1 target
        assert len(m["edges"]) == 3        # each source -> target

    def test_coincidence_edges_are_positive(self):
        cfg = MotifConfig(enabled=True, n_coincidence_detector=1,
                          n_external_connections=0, random_seed=1)
        W, meta = add_motifs(_base_matrix(), cfg)
        for src, tgt in meta["motifs"][0]["edges"]:
            assert W[tgt, src] > 0

    def test_lateral_inhibition_edges_are_negative(self):
        cfg = MotifConfig(enabled=True, n_lateral_inhibition=1,
                          n_external_connections=0, random_seed=2)
        W, meta = add_motifs(_base_matrix(), cfg)
        edges = meta["motifs"][0]["edges"]
        assert len(edges) > 0
        for src, tgt in edges:
            assert W[tgt, src] < 0

    def test_negative_feedback_loop_has_feedback_and_negative(self):
        cfg = MotifConfig(enabled=True, n_negative_feedback_loop=1,
                          n_external_connections=0, random_seed=3)
        W, meta = add_motifs(_base_matrix(), cfg)
        m = meta["motifs"][0]
        signs = [np.sign(W[t, s]) for s, t in m["edges"]]
        assert -1 in signs, "feedback loop must contain a negative edge"
        assert +1 in signs, "feedback loop must contain a positive edge"
        # the edges must form a closed path A->B->C->A over 3 nodes
        assert len(m["nodes"]) == 3
        assert len(m["edges"]) == 3

    def test_template_sizes_match_registry(self):
        for motif_type, size in MOTIF_SIZES.items():
            template = get_template(motif_type)
            assert template.n_nodes == size


# --------------------------------------------------------------------------- #
# 2. Weight-matrix integration
# --------------------------------------------------------------------------- #
class TestWeightMatrixIntegration:
    def test_shape_grows_by_motif_sizes(self):
        base = _base_matrix(10)
        cfg = MotifConfig(enabled=True, n_coincidence_detector=1,
                          n_lateral_inhibition=1)
        W, _ = add_motifs(base, cfg)
        expected = 10 + MOTIF_SIZES[MotifType.COINCIDENCE_DETECTOR] \
            + MOTIF_SIZES[MotifType.LATERAL_INHIBITION]
        assert W.shape == (expected, expected)

    def test_base_network_preserved(self):
        base = _base_matrix(10, seed=4)
        cfg = MotifConfig(enabled=True, n_coincidence_detector=1)
        W, _ = add_motifs(base, cfg)
        assert np.array_equal(W[:10, :10], base)

    def test_positive_motif_connections_positive(self):
        cfg = MotifConfig(enabled=True, n_coincidence_detector=1,
                          n_external_connections=0, random_seed=5)
        W, meta = add_motifs(_base_matrix(), cfg)
        signs = meta["motifs"][0]["node_signs"]
        for src, tgt in meta["motifs"][0]["edges"]:
            if signs[src] == "positive":
                assert W[tgt, src] > 0

    def test_negative_motif_connections_negative(self):
        cfg = MotifConfig(enabled=True, n_negative_feedback_loop=1,
                          n_external_connections=0, random_seed=6)
        W, meta = add_motifs(_base_matrix(), cfg)
        signs = meta["motifs"][0]["node_signs"]
        for src, tgt in meta["motifs"][0]["edges"]:
            if signs[src] == "negative":
                assert W[tgt, src] < 0

    def test_deterministic_with_seed(self):
        base = _base_matrix()
        cfg = MotifConfig(enabled=True, n_coincidence_detector=2,
                          n_lateral_inhibition=1, random_seed=7)
        W1, _ = add_motifs(base, cfg)
        W2, _ = add_motifs(base, cfg)
        assert np.array_equal(W1, W2)

    def test_build_network_with_motifs_wrapper(self):
        base = _base_matrix()
        cfg = MotifConfig(enabled=True, n_coincidence_detector=1)
        W, meta = build_network_with_motifs(base, cfg)
        assert W.shape[0] == base.shape[0] + 4
        assert len(meta["motifs"]) == 1


# --------------------------------------------------------------------------- #
# 3. Disabled motifs
# --------------------------------------------------------------------------- #
class TestDisabledMotifs:
    def test_disabled_leaves_matrix_unchanged(self):
        base = _base_matrix()
        W, meta = add_motifs(base, MotifConfig(enabled=False))
        assert np.array_equal(W, base)
        assert meta["motifs"] == []

    def test_zero_counts_leaves_matrix_unchanged(self):
        base = _base_matrix()
        cfg = MotifConfig(enabled=True)  # all counts default to 0
        W, meta = add_motifs(base, cfg)
        assert np.array_equal(W, base)
        assert meta["motifs"] == []

    def test_disabled_pipeline_matches_no_motif_pipeline(self):
        # A full run with motifs disabled must equal the old behavior.
        cfg_off = ProjectConfig(network=NetworkConfig(n_neurons=12, random_seed=9))
        W = build_weight_matrix(cfg_off.network)
        r_plain = Simulator().run(cfg_off, W)

        cfg_motif = ProjectConfig(network=NetworkConfig(n_neurons=12, random_seed=9),
                                  motifs=MotifConfig(enabled=False))
        W2, meta = add_motifs(build_weight_matrix(cfg_motif.network),
                              cfg_motif.motifs)
        r_motif = Simulator().run(cfg_motif, W2)
        assert np.array_equal(r_plain.activity, r_motif.activity)


# --------------------------------------------------------------------------- #
# 4. Metadata
# --------------------------------------------------------------------------- #
class TestMetadata:
    def test_metadata_structure(self):
        cfg = MotifConfig(enabled=True, n_coincidence_detector=1,
                          n_lateral_inhibition=1, n_negative_feedback_loop=1)
        _, meta = add_motifs(_base_matrix(), cfg)
        assert "motifs" in meta
        assert len(meta["motifs"]) == 3
        for m in meta["motifs"]:
            assert set(m.keys()) >= {"motif_id", "motif_type", "nodes",
                                     "edges", "node_signs"}
            assert isinstance(m["nodes"], list)
            assert isinstance(m["edges"], list)
            assert isinstance(m["node_signs"], dict)

    def test_node_signs_cover_all_nodes(self):
        cfg = MotifConfig(enabled=True, n_negative_feedback_loop=1,
                          n_external_connections=0)
        _, meta = add_motifs(_base_matrix(), cfg)
        m = meta["motifs"][0]
        for node in m["nodes"]:
            assert m["node_signs"][node] in ("positive", "negative")

    def test_visualization_accepts_metadata(self):
        cfg = ProjectConfig(network=NetworkConfig(n_neurons=12),
                            motifs=MotifConfig(enabled=True,
                                               n_coincidence_detector=1))
        W = build_weight_matrix(cfg.network)
        W2, meta = add_motifs(W, cfg.motifs)
        cfg.network.n_neurons = W2.shape[0]
        result = Simulator().run(cfg, W2)
        result.metadata["motifs"] = meta["motifs"]
        fig = build_figure(result)          # must not raise
        names = [t.name for t in fig.data if t.name]
        assert any("motif" in n for n in names)

    def test_instance_to_metadata(self):
        inst = MotifInstance(
            motif_id=0, motif_type=MotifType.COINCIDENCE_DETECTOR,
            nodes=[10, 11, 12, 13], edges=[(10, 13), (11, 13), (12, 13)],
            node_signs={10: "positive", 11: "positive",
                        12: "positive", 13: "positive"},
        )
        d = inst.to_metadata()
        assert d["motif_type"] == "coincidence_detector"
        assert d["nodes"] == [10, 11, 12, 13]


# --------------------------------------------------------------------------- #
# 5. App / config integration + validation
# --------------------------------------------------------------------------- #
class TestConfigIntegration:
    def test_negative_counts_rejected(self):
        with pytest.raises(ValueError):
            MotifConfig(enabled=True, n_coincidence_detector=-1).validate()

    def test_invalid_strength_rejected(self):
        with pytest.raises(ValueError):
            MotifConfig(connection_strength=0).validate()

    def test_negative_external_connections_rejected(self):
        with pytest.raises(ValueError):
            MotifConfig(n_external_connections=-2).validate()

    def test_project_config_without_motifs_still_works(self):
        cfg = ProjectConfig(network=NetworkConfig(n_neurons=8))
        cfg.validate()                       # must not raise
        assert cfg.motifs.enabled is False

    def test_project_config_validate_cascades_to_motifs(self):
        cfg = ProjectConfig(
            network=NetworkConfig(n_neurons=8),
            motifs=MotifConfig(enabled=True, n_lateral_inhibition=-3),
        )
        with pytest.raises(ValueError):
            cfg.validate()

    def test_project_config_roundtrip_with_motifs(self):
        cfg = ProjectConfig(
            network=NetworkConfig(n_neurons=8),
            motifs=MotifConfig(enabled=True, n_coincidence_detector=2,
                               connection_strength=1.5),
        )
        restored = ProjectConfig.from_dict(cfg.to_dict())
        assert restored.motifs.enabled is True
        assert restored.motifs.n_coincidence_detector == 2
        assert restored.motifs.connection_strength == 1.5

    def test_total_motifs_and_requested_types(self):
        cfg = MotifConfig(enabled=True, n_coincidence_detector=2,
                          n_lateral_inhibition=1)
        assert cfg.total_motifs == 3
        types = cfg.requested_types()
        assert types.count(MotifType.COINCIDENCE_DETECTOR) == 2
        assert types.count(MotifType.LATERAL_INHIBITION) == 1


# --------------------------------------------------------------------------- #
# 6. New motif types (feedforward loop, feedforward inhibition, WTA)
# --------------------------------------------------------------------------- #
class TestNewMotifTypes:
    def test_feedforward_loop_all_positive_with_two_paths(self):
        cfg = MotifConfig(enabled=True, n_feedforward_loop=1,
                          n_external_connections=0, random_seed=1)
        W, meta = add_motifs(_base_matrix(), cfg)
        m = meta["motifs"][0]
        assert m["motif_type"] == "feedforward_loop"
        assert len(m["nodes"]) == 3 and len(m["edges"]) == 3
        for src, tgt in m["edges"]:
            assert W[tgt, src] > 0
        # must contain both the direct edge (A->C) and the two-step path
        a, b, c = m["nodes"]
        edge_set = {tuple(e) for e in m["edges"]}
        assert (a, c) in edge_set          # direct
        assert (a, b) in edge_set and (b, c) in edge_set  # indirect

    def test_feedforward_inhibition_has_inhibitory_interneuron(self):
        cfg = MotifConfig(enabled=True, n_feedforward_inhibition=1,
                          n_external_connections=0, random_seed=2)
        W, meta = add_motifs(_base_matrix(), cfg)
        m = meta["motifs"][0]
        assert m["motif_type"] == "feedforward_inhibition"
        signs = [np.sign(W[t, s]) for s, t in m["edges"]]
        assert +1 in signs and -1 in signs
        # middle node is the inhibitory interneuron
        assert m["node_signs"][m["nodes"][1]] == "negative"

    def test_mutual_excitation_reciprocal(self):
        cfg = MotifConfig(enabled=True, n_mutual_excitation=1,
                          n_external_connections=0, random_seed=3)
        W, meta = add_motifs(_base_matrix(), cfg)
        m = meta["motifs"][0]
        assert m["motif_type"] == "mutual_excitation"
        assert len(m["nodes"]) == 2 and len(m["edges"]) == 2
        for src, tgt in m["edges"]:
            assert W[tgt, src] > 0
        # reciprocal: both directions present
        a, b = m["nodes"]
        edge_set = {tuple(e) for e in m["edges"]}
        assert (a, b) in edge_set and (b, a) in edge_set

    def test_all_six_types_together(self):
        cfg = MotifConfig(
            enabled=True, n_coincidence_detector=1, n_lateral_inhibition=1,
            n_negative_feedback_loop=1, n_feedforward_loop=1,
            n_feedforward_inhibition=1, n_mutual_excitation=1, random_seed=7,
        )
        base = _base_matrix(10)
        W, meta = add_motifs(base, cfg)
        assert len(meta["motifs"]) == 6
        expected = 10 + sum(MOTIF_SIZES[t] for t in cfg.requested_types())
        assert W.shape == (expected, expected)
        # every motif type appears exactly once
        types = {m["motif_type"] for m in meta["motifs"]}
        assert types == {
            "coincidence_detector", "lateral_inhibition",
            "negative_feedback_loop", "feedforward_loop",
            "feedforward_inhibition", "mutual_excitation",
        }

    def test_new_types_deterministic(self):
        base = _base_matrix()
        cfg = MotifConfig(enabled=True, n_feedforward_loop=2,
                          n_mutual_excitation=1, random_seed=11)
        W1, _ = add_motifs(base, cfg)
        W2, _ = add_motifs(base, cfg)
        assert np.array_equal(W1, W2)

    def test_new_count_validation(self):
        with pytest.raises(ValueError):
            MotifConfig(enabled=True, n_feedforward_loop=-1).validate()
        with pytest.raises(ValueError):
            MotifConfig(enabled=True, n_mutual_excitation=-5).validate()

    def test_visualization_colors_by_type(self):
        from nnsimviz.visualization import MOTIF_TYPE_COLORS
        cfg = ProjectConfig(
            network=NetworkConfig(n_neurons=12),
            motifs=MotifConfig(enabled=True, n_feedforward_loop=1,
                               n_mutual_excitation=1),
        )
        W = build_weight_matrix(cfg.network)
        W2, meta = add_motifs(W, cfg.motifs)
        cfg.network.n_neurons = W2.shape[0]
        result = Simulator().run(cfg, W2)
        result.metadata["motifs"] = meta["motifs"]
        fig = build_figure(result)
        names = [t.name for t in fig.data if t.name]
        # per-type legend entries present
        assert "Feedforward loop" in names
        assert "Mutual excitation" in names
        # the two types use different ring colors
        assert MOTIF_TYPE_COLORS["feedforward_loop"] != \
            MOTIF_TYPE_COLORS["mutual_excitation"]


# --------------------------------------------------------------------------- #
# 7. Motif icons
# --------------------------------------------------------------------------- #
class TestMotifIcons:
    def test_every_motif_type_has_an_icon(self):
        from nnsimviz.motif_icons import motif_icon_svg
        for motif_type in MotifType:
            svg = motif_icon_svg(motif_type.value)
            assert svg.startswith("<svg") and svg.endswith("</svg>")

    def test_unknown_type_returns_empty(self):
        from nnsimviz.motif_icons import motif_icon_svg
        assert motif_icon_svg("does_not_exist") == ""

    def test_all_motif_icons_covers_registry(self):
        from nnsimviz.motif_icons import all_motif_icons
        icons = all_motif_icons()
        for motif_type in MotifType:
            assert motif_type.value in icons

    def test_icons_are_well_formed_xml(self):
        import xml.dom.minidom as minidom
        from nnsimviz.motif_icons import all_motif_icons
        for svg in all_motif_icons().values():
            minidom.parseString(svg)   # raises if malformed


# --------------------------------------------------------------------------- #
# 8. Motif marking in the animation
# --------------------------------------------------------------------------- #
class TestMotifAnimation:
    def _result_with_motifs(self):
        cfg = ProjectConfig(
            network=NetworkConfig(n_neurons=12),
            motifs=MotifConfig(enabled=True, n_coincidence_detector=1,
                               n_mutual_excitation=1),
        )
        W = build_weight_matrix(cfg.network)
        W2, meta = add_motifs(W, cfg.motifs)
        cfg.network.n_neurons = W2.shape[0]
        result = Simulator().run(cfg, W2)
        result.metadata["motifs"] = meta["motifs"]
        return result

    def test_animation_includes_motif_marking(self):
        from nnsimviz.visualization import build_animated_figure
        fig = build_animated_figure(self._result_with_motifs())
        names = [t.name for t in fig.data if t.name]
        assert any("Coincidence" in n or "Mutual" in n for n in names)

    def test_animation_frames_still_target_node_marker(self):
        from nnsimviz.visualization import (
            build_animated_figure, NetworkVisualizer,
        )
        result = self._result_with_motifs()
        fig = build_animated_figure(result)
        viz = NetworkVisualizer(result.config.visualization)
        graph, edges = viz.build_graph(result.weight_matrix)
        n_edge_traces = len(viz._edge_traces(edges, viz.compute_layout(graph)))
        # frames must update the node-marker trace, which sits right after
        # the edge traces -- adding motif traces after it must not shift this.
        assert list(fig.frames[0].traces) == [n_edge_traces]
        assert fig.data[n_edge_traces].mode == "markers"

    def test_animation_without_motifs_still_builds(self):
        from nnsimviz.visualization import build_animated_figure
        cfg = ProjectConfig(network=NetworkConfig(n_neurons=10))
        W = build_weight_matrix(cfg.network)
        result = Simulator().run(cfg, W)
        fig = build_animated_figure(result)   # must not raise
        assert len(fig.frames) > 0
