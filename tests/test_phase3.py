import unittest

import torch

from transattack.data import adjacency_from_edge_index, edge_index_from_adjacency, make_sbm
from transattack.gps_attack import candidate_additions
from transattack.gps_model import GraphGPSNodeClassifier, evaluate_gps_trace, logits_for_adjacencies, rwse_from_adjacency


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


if __name__ == "__main__":
    unittest.main()

