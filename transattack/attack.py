from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .data import GraphData, add_edge, adjacency_from_edge_index
from .model import DenseGraphTransformer, ModelTrace, true_margin
from .pe import PEBundle, build_pe


@dataclass(frozen=True)
class AttackSnapshot:
    target: int
    budget: int
    edge_index: Tensor
    pe: PEBundle
    added_edges: tuple[tuple[int, int], ...]
    target_loss: float
    prediction: int
    margin: float
    success: bool


def select_attack_targets(trace: ModelTrace, graph: GraphData, count: int) -> list[int]:
    """Pick correctly classified test nodes nearest to the decision boundary."""
    candidates: list[tuple[float, int]] = []
    for target in graph.test_idx.tolist():
        label = int(graph.y[target])
        if int(trace.logits[target].argmax()) != label:
            continue
        candidates.append((true_margin(trace.logits[target], label), int(target)))
    candidates.sort()
    return [target for _, target in candidates[:count]]


def _candidate_nodes(
    graph: GraphData,
    edge_index: Tensor,
    logits: Tensor,
    target: int,
    pool_size: int,
) -> list[int]:
    adj = adjacency_from_edge_index(edge_index, graph.num_nodes)
    available = [
        node
        for node in range(graph.num_nodes)
        if node != target and adj[target, node] == 0
    ]
    if len(available) <= pool_size:
        return available

    target_label = int(graph.y[target])
    target_row = logits[target].clone()
    target_row[target_label] = -float("inf")
    rival = int(target_row.argmax())
    probabilities = torch.softmax(logits, dim=1)
    features = torch.nn.functional.normalize(graph.x.float(), dim=1)
    similarity = features @ features[target]

    # Model-derived heuristic only: prefer nodes representing the current rival
    # class and nodes far from the target in feature space. The true labels of
    # candidate nodes are deliberately not used.
    ordered = sorted(
        available,
        key=lambda node: (
            int(logits[node].argmax() == rival),
            float(probabilities[node, rival]),
            -float(similarity[node]),
        ),
        reverse=True,
    )
    half = max(1, pool_size // 2)
    chosen = ordered[:half]
    by_dissimilarity = sorted(available, key=lambda node: float(similarity[node]))
    for node in by_dissimilarity:
        if node not in chosen:
            chosen.append(node)
        if len(chosen) >= pool_size:
            break
    return chosen


@torch.no_grad()
def _evaluate_candidate(
    model: DenseGraphTransformer,
    graph: GraphData,
    pe: Tensor,
    pe_scale: float,
    target: int,
    device: torch.device,
) -> tuple[float, int, float, Tensor]:
    model.eval()
    logits = model(graph.x.to(device), (pe / pe_scale).to(device))
    label = graph.y[target].to(device).reshape(1)
    row = logits[target].reshape(1, -1)
    loss = float(torch.nn.functional.cross_entropy(row, label).item())
    cpu_row = row.detach().cpu().flatten()
    return loss, int(cpu_row.argmax()), true_margin(cpu_row, int(label.item())), logits.detach().cpu()


def adaptive_addition_attack(
    model: DenseGraphTransformer,
    graph: GraphData,
    clean_trace: ModelTrace,
    pe_name: str,
    pe_dim: int,
    pe_scale: float,
    target: int,
    budgets: list[int],
    candidate_pool: int,
    device: torch.device,
) -> dict[int, AttackSnapshot]:
    """Greedily maximize target CE, recomputing PE after every edge addition."""
    maximum = max(budgets)
    current_edges = graph.edge_index
    current_logits = clean_trace.logits
    added: list[tuple[int, int]] = []
    snapshots: dict[int, AttackSnapshot] = {}
    target_label = int(graph.y[target])

    for step in range(1, maximum + 1):
        candidates = _candidate_nodes(
            graph, current_edges, current_logits, target, candidate_pool
        )
        if not candidates:
            break
        best: tuple[float, int, Tensor, PEBundle, int, float, Tensor] | None = None
        for other in candidates:
            candidate_edges = add_edge(current_edges, graph.num_nodes, target, other)
            candidate_pe = build_pe(pe_name, candidate_edges, graph.num_nodes, pe_dim)
            loss, prediction, margin, logits = _evaluate_candidate(
                model,
                graph,
                candidate_pe.values,
                pe_scale,
                target,
                device,
            )
            if best is None or loss > best[0]:
                best = (
                    loss,
                    int(other),
                    candidate_edges,
                    candidate_pe,
                    prediction,
                    margin,
                    logits,
                )
        assert best is not None
        loss, other, current_edges, current_pe, prediction, margin, current_logits = best
        added.append(tuple(sorted((int(target), int(other)))))
        if step in budgets:
            snapshots[step] = AttackSnapshot(
                target=int(target),
                budget=step,
                edge_index=current_edges,
                pe=current_pe,
                added_edges=tuple(added),
                target_loss=loss,
                prediction=prediction,
                margin=margin,
                success=prediction != target_label,
            )
    return snapshots

