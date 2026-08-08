import unittest

import torch

from transattack.attack import adaptive_addition_attack, select_attack_targets
from transattack.data import adjacency_from_edge_index, count_edges, make_sbm, undirected_pairs
from transattack.localize import (
    GaussianProfile,
    edge_features,
    feature_views,
    fit_profiles,
    ranking_metrics,
)
from transattack.model import evaluate_trace, train_model
from transattack.pe import build_pe, rms_scale
from transattack.phase2 import _round_robin, candidate_rankings


class Phase0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.graph = make_sbm(32, seed=19)
        cls.clean_pe = build_pe("rw", cls.graph.edge_index, cls.graph.num_nodes, 4)
        cls.scale = rms_scale(cls.clean_pe.values)
        cls.trained = train_model(
            cls.graph,
            cls.clean_pe.values / cls.scale,
            cls.device,
            seed=19,
            hidden_dim=16,
            heads=4,
            layers=2,
            dropout=0.0,
            epochs=4,
            patience=4,
            learning_rate=0.004,
            weight_decay=0.0,
        )
        cls.trace = evaluate_trace(cls.trained.model, cls.graph, cls.clean_pe.values / cls.scale, cls.device)

    def test_trace_and_edge_feature_shapes(self):
        pairs = undirected_pairs(self.graph.edge_index)
        features = edge_features(self.trace, pairs)
        self.assertEqual(self.trace.attentions.shape[:2], (2, 4))
        self.assertEqual(self.trace.hidden.size(0), 3)
        self.assertEqual(features.per_layer.shape, (pairs.size(1), 2, 7))
        self.assertTrue(torch.isfinite(features.groups["full_dynamics"]).all())

    def test_profile_contains_parameters_not_edge_identities(self):
        features = edge_features(self.trace, undirected_pairs(self.graph.edge_index))
        profile = fit_profiles(features)["full_dynamics"]
        self.assertEqual(set(vars(profile)), {"mean", "precision"})
        self.assertEqual(profile.mean.numel(), 14)

    def test_phase1_views_separate_layers_and_channels(self):
        features = edge_features(self.trace, undirected_pairs(self.graph.edge_index))
        views = feature_views(features)
        edges = features.pairs.size(1)
        self.assertEqual(views["first_layer_full"].shape, (edges, 7))
        self.assertEqual(views["last_layer_full"].shape, (edges, 7))
        self.assertEqual(views["all_layer_full"].shape, (edges, 14))
        self.assertEqual(views["temporal_residual"].shape, (edges, 7))
        self.assertEqual(views["attention_trajectory"].shape, (edges, 6))
        self.assertEqual(views["value_trajectory"].shape, (edges, 2))
        self.assertEqual(views["hidden_logit_trajectory"].shape, (edges, 6))

    def test_addition_attack_never_deletes(self):
        targets = select_attack_targets(self.trace, self.graph, 1)
        if not targets:
            self.skipTest("tiny training run produced no correctly classified test target")
        snapshots = adaptive_addition_attack(
            self.trained.model,
            self.graph,
            self.trace,
            "rw",
            4,
            self.scale,
            targets[0],
            [1],
            3,
            self.device,
        )
        self.assertIn(1, snapshots)
        self.assertEqual(count_edges(snapshots[1].edge_index), count_edges(self.graph.edge_index) + 1)
        adj = adjacency_from_edge_index(self.graph.edge_index, self.graph.num_nodes)
        u, v = snapshots[1].added_edges[0]
        self.assertEqual(float(adj[u, v]), 0.0)

    def test_ranking_metrics_known_order(self):
        pairs = torch.tensor([[0, 0, 1], [1, 2, 2]])
        metrics = ranking_metrics(pairs, torch.tensor([0.1, 0.9, 0.2]), ((0, 2),), 1)
        self.assertEqual(metrics["edge_auprc"], 1.0)
        self.assertEqual(metrics["recall_at_b"], 1.0)
        self.assertEqual(metrics["first_positive_rank"], 1.0)

    def test_mahalanobis_uses_off_diagonal_precision(self):
        profile = GaussianProfile(
            mean=torch.zeros(2, dtype=torch.float64),
            precision=torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
        )
        score = profile.score(torch.tensor([[1.0, 1.0]]))
        self.assertAlmostEqual(float(score.item()), 3.0, places=6)

    def test_phase2_round_robin_is_unique_and_deterministic(self):
        rankings = [[0, 1, 2, 3], [1, 3, 0, 2], [2, 0, 3, 1]]
        self.assertEqual(_round_robin(rankings, 4), [0, 1, 2, 3])

    def test_phase2_target_incident_ranking_uses_known_target(self):
        pairs = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
        all_scores = torch.tensor([0.1, 0.2, 9.0, 8.0])
        temporal_scores = torch.tensor([0.2, 0.1, 8.0, 9.0])
        rankings = candidate_rankings(pairs, all_scores, temporal_scores, target=0)
        self.assertEqual(set(rankings["target_incident"][:2]), {0, 1})
        self.assertEqual(len(rankings["multiview_union"]), 4)
        self.assertEqual(len(set(rankings["multiview_union"])), 4)


if __name__ == "__main__":
    unittest.main()
