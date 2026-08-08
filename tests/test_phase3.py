import unittest

import torch

from transattack.data import adjacency_from_edge_index, edge_index_from_adjacency, make_sbm, undirected_pairs
from transattack.dshield_aug import augmented_graph_views, stable_view_seed, view_statistics
from transattack.gps_attack import adaptive_gps_attack, candidate_additions
from transattack.heuristic_search import heuristic_candidates, heuristic_remote_search
from transattack.gps_model import (
    GraphGPSNodeClassifier,
    candidate_trace_views_for_adjacencies,
    evaluate_gps_trace,
    logits_for_adjacencies,
    rwse_from_adjacency,
)
from transattack.localize import edge_features, feature_views, fit_view_profiles


class Phase3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = make_sbm(32, seed=73)
        cls.model = GraphGPSNodeClassifier(
            cls.graph.num_features,
            cls.graph.num_classes,
            channels=24,
            pe_channels=8,
            walk_length=4,
            layers=2,
            heads=4,
            dropout=0.0,
        ).eval()
        cls.adjacency = adjacency_from_edge_index(cls.graph.edge_index, cls.graph.num_nodes).float()

    def test_dshield_views_are_deterministic_symmetric_subgraphs(self):
        seed = stable_view_seed("phase6-test")
        left_adj, left_x = augmented_graph_views(self.graph, self.adjacency, 4, 0.2, 0.2, seed)
        right_adj, right_x = augmented_graph_views(self.graph, self.adjacency, 4, 0.2, 0.2, seed)
        self.assertTrue(torch.equal(left_adj, right_adj))
        self.assertTrue(torch.equal(left_x, right_x))
        self.assertTrue(torch.equal(left_adj, left_adj.transpose(1, 2)))
        self.assertTrue(torch.all(left_adj <= self.adjacency.unsqueeze(0)))

    def test_dshield_disagreement_is_zero_for_identical_views(self):
        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]).repeat(5, 1, 1)
        stats = view_statistics(logits, target=0, true_label=0)
        self.assertEqual(stats.prediction, 0)
        self.assertAlmostEqual(stats.disagreement, 0.0, places=12)

    def test_rwse_shape_and_structural_sensitivity(self):
        pe = rwse_from_adjacency(self.adjacency, 4)
        self.assertEqual(pe.shape, (self.graph.num_nodes, 4))
        self.assertTrue(torch.isfinite(pe).all())
        changed = self.adjacency.clone()
        nonedge = torch.nonzero(torch.triu(changed == 0, diagonal=1), as_tuple=False)[0]
        u, v = (int(value) for value in nonedge.tolist())
        changed[u, v] = changed[v, u] = 1.0
        changed_pe = rwse_from_adjacency(changed, 4)
        self.assertFalse(torch.allclose(pe, changed_pe))

    def test_trace_path_preserves_logits(self):
        x = self.graph.x
        pe = rwse_from_adjacency(self.adjacency, 4)
        batch = torch.zeros(self.graph.num_nodes, dtype=torch.long)
        with torch.no_grad():
            ordinary = self.model(x, pe, self.graph.edge_index, batch)
            trace = self.model(x, pe, self.graph.edge_index, batch, return_trace=True)
        self.assertTrue(torch.allclose(ordinary, trace.logits, atol=1e-6, rtol=1e-5))
        self.assertEqual(trace.attentions.shape, (2, 4, 32, 32))
        self.assertEqual(trace.values.shape, (2, 32, 4, 6))
        self.assertEqual(trace.hidden.shape, (3, 32, 24))

    def test_batched_logits_match_single_graph_evaluation(self):
        changed = self.adjacency.clone()
        nonedge = torch.nonzero(torch.triu(changed == 0, diagonal=1), as_tuple=False)[0]
        u, v = (int(value) for value in nonedge.tolist())
        changed[u, v] = changed[v, u] = 1.0
        batched = logits_for_adjacencies(
            self.model,
            self.graph,
            torch.stack((self.adjacency, changed)),
            torch.device("cpu"),
            batch_size=2,
        )
        first = evaluate_gps_trace(self.model, self.graph, self.graph.edge_index, torch.device("cpu")).logits
        second_edges = edge_index_from_adjacency(changed)
        second = evaluate_gps_trace(self.model, self.graph, second_edges, torch.device("cpu")).logits
        self.assertTrue(torch.allclose(batched[0], first, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(batched[1], second, atol=1e-6, rtol=1e-5))

    def test_batched_candidate_trace_views_match_single_graph_views(self):
        candidates = candidate_additions(
            self.graph,
            self.adjacency,
            evaluate_gps_trace(self.model, self.graph, self.graph.edge_index, torch.device("cpu")).logits,
            int(self.graph.test_idx[0]),
            pool_size=2,
            attack_type="remote",
        )
        variants = self.adjacency.unsqueeze(0).repeat(2, 1, 1)
        for index, (u, v) in enumerate(candidates):
            variants[index, u, v] = variants[index, v, u] = 1.0
        batched = candidate_trace_views_for_adjacencies(
            self.model,
            self.graph,
            variants,
            candidates,
            torch.device("cpu"),
            batch_size=2,
        )
        for index, pair in enumerate(candidates):
            trace = evaluate_gps_trace(
                self.model,
                self.graph,
                edge_index_from_adjacency(variants[index]),
                torch.device("cpu"),
            )
            views = feature_views(edge_features(trace, torch.tensor(pair).reshape(2, 1)))
            self.assertTrue(torch.allclose(batched.logits[index], trace.logits, atol=1e-6, rtol=1e-5))
            self.assertTrue(torch.allclose(batched.all_layer_full[index], views["all_layer_full"][0], atol=1e-6, rtol=1e-5))
            self.assertTrue(torch.allclose(batched.temporal_residual[index], views["temporal_residual"][0], atol=1e-6, rtol=1e-5))

    def test_remote_candidates_are_nonincident_and_two_hop(self):
        trace = evaluate_gps_trace(self.model, self.graph, self.graph.edge_index, torch.device("cpu"))
        target = int(self.graph.test_idx[0])
        candidates = candidate_additions(
            self.graph,
            self.adjacency,
            trace.logits,
            target,
            pool_size=12,
            attack_type="remote",
        )
        self.assertTrue(candidates)
        for u, v in candidates:
            self.assertNotIn(target, (u, v))
            self.assertEqual(float(self.adjacency[u, v]), 0.0)
            self.assertTrue(float(self.adjacency[target, u]) > 0 or float(self.adjacency[target, v]) > 0)

    def test_multi_rival_candidates_preserve_remote_threat_model(self):
        trace = evaluate_gps_trace(self.model, self.graph, self.graph.edge_index, torch.device("cpu"))
        target = int(self.graph.test_idx[0])
        candidates = heuristic_candidates(
            self.graph,
            self.adjacency,
            trace.logits,
            target,
            pool_size=20,
            strategy="multi_rival",
        )
        self.assertTrue(candidates)
        self.assertEqual(len(candidates), len(set(candidates)))
        for u, v in candidates:
            self.assertNotIn(target, (u, v))
            self.assertEqual(float(self.adjacency[u, v]), 0.0)
            self.assertTrue(float(self.adjacency[target, u]) > 0 or float(self.adjacency[target, v]) > 0)

    def test_greedy_cross_entropy_search_matches_legacy_first_step(self):
        clean_trace = evaluate_gps_trace(self.model, self.graph, self.graph.edge_index, torch.device("cpu"))
        target = int(self.graph.test_idx[0])
        legacy = adaptive_gps_attack(
            self.model,
            self.graph,
            clean_trace,
            target,
            [1],
            12,
            "remote",
            torch.device("cpu"),
            12,
        )
        searched = heuristic_remote_search(
            self.model,
            self.graph,
            clean_trace,
            target,
            [1],
            "cross_entropy",
            1,
            "single_rival",
            "fixed",
            12,
            12,
            torch.device("cpu"),
            12,
        )
        self.assertEqual(legacy[1].added_edges, searched.exact[1].added_edges)
        self.assertAlmostEqual(legacy[1].margin, searched.exact[1].margin, places=5)

    def test_within_budget_search_is_monotone_in_canonical_attack_score(self):
        clean_trace = evaluate_gps_trace(self.model, self.graph, self.graph.edge_index, torch.device("cpu"))
        searched = heuristic_remote_search(
            self.model,
            self.graph,
            clean_trace,
            int(self.graph.test_idx[0]),
            [1, 2, 3],
            "normalized_margin",
            2,
            "multi_rival",
            "adaptive",
            8,
            16,
            torch.device("cpu"),
            16,
        )
        scores = [searched.within_budget[budget].attack_score for budget in (1, 2, 3)]
        self.assertEqual(scores, sorted(scores))
        for budget, snapshot in searched.within_budget.items():
            self.assertLessEqual(snapshot.used_edges, budget)
            self.assertTrue(torch.isfinite(torch.tensor(snapshot.objective_score)))

    def test_zero_strength_adaptive_attack_matches_classification_attack(self):
        clean_trace = evaluate_gps_trace(self.model, self.graph, self.graph.edge_index, torch.device("cpu"))
        profiles = fit_view_profiles(edge_features(clean_trace, undirected_pairs(self.graph.edge_index)))
        target = int(self.graph.test_idx[0])
        baseline = adaptive_gps_attack(
            self.model,
            self.graph,
            clean_trace,
            target,
            [1, 2],
            12,
            "remote",
            torch.device("cpu"),
            12,
        )
        adaptive = adaptive_gps_attack(
            self.model,
            self.graph,
            clean_trace,
            target,
            [1, 2],
            12,
            "remote",
            torch.device("cpu"),
            12,
            attack_objective="adaptive_stealth",
            profiles=profiles,
            adaptive_stealth_strength=0.0,
        )
        self.assertEqual(baseline[2].added_edges, adaptive[2].added_edges)

    def test_unit_retention_constrained_attack_matches_classification_attack(self):
        clean_trace = evaluate_gps_trace(self.model, self.graph, self.graph.edge_index, torch.device("cpu"))
        profiles = fit_view_profiles(edge_features(clean_trace, undirected_pairs(self.graph.edge_index)))
        target = int(self.graph.test_idx[0])
        baseline = adaptive_gps_attack(
            self.model,
            self.graph,
            clean_trace,
            target,
            [1, 2],
            12,
            "remote",
            torch.device("cpu"),
            12,
        )
        constrained = adaptive_gps_attack(
            self.model,
            self.graph,
            clean_trace,
            target,
            [1, 2],
            12,
            "remote",
            torch.device("cpu"),
            12,
            attack_objective="classification_constrained_stealth",
            profiles=profiles,
            classification_retention_ratio=1.0,
        )
        self.assertEqual(baseline[2].added_edges, constrained[2].added_edges)
        self.assertEqual(constrained[2].minimum_selected_gain_ratio, 1.0)
        self.assertEqual(constrained[2].eligible_candidates, 1)

    def test_constrained_attack_rejects_invalid_retention_ratio(self):
        clean_trace = evaluate_gps_trace(self.model, self.graph, self.graph.edge_index, torch.device("cpu"))
        with self.assertRaises(ValueError):
            adaptive_gps_attack(
                self.model,
                self.graph,
                clean_trace,
                int(self.graph.test_idx[0]),
                [1],
                12,
                "remote",
                torch.device("cpu"),
                12,
                classification_retention_ratio=0.0,
            )


if __name__ == "__main__":
    unittest.main()
