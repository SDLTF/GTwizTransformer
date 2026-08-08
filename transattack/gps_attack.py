from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .data import GraphData, adjacency_from_edge_index, edge_index_from_adjacency
from .gps_model import BatchTelemetry, GraphGPSNodeClassifier, logits_for_adjacencies
from .model import ModelTrace, true_margin


ATTACK_TYPES = ("incident", "remote")


@dataclass(frozen=True)
class GPSAttackSnapshot:
    attack_type: str
    target: int
    budget: int
    edge_index: Tensor
    adjacency: Tensor
    added_edges: tuple[tuple[int, int], ...]
    target_loss: float
    prediction: int
    margin: float
    success: bool


def _rival_and_scores(graph: GraphData, logits: Tensor, target: int) -> tuple[int, Tensor, Tensor]:
    target_label = int(graph.y[target])
    row = logits[target].clone().float()
    row[target_label] = -float("inf")
    rival = int(row.argmax())
    probabilities = torch.softmax(logits.float(), dim=-1)
    features = F.normalize(graph.x.float(), dim=-1)
    similarity = features @ features[target]
    return rival, probabilities, similarity


def candidate_additions(
    graph: GraphData,
    adjacency: Tensor,
    logits: Tensor,
    target: int,
    pool_size: int,
    attack_type: str,
) -> list[tuple[int, int]]:
    """Model-derived candidates without using labels of candidate nodes."""
    if attack_type not in ATTACK_TYPES:
        raise ValueError(f"unsupported attack type: {attack_type}")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    adjacency = adjacency.detach().cpu()
    rival, probabilities, similarity = _rival_and_scores(graph, logits.detach().cpu(), target)
    candidates: list[tuple[tuple[float, ...], tuple[int, int]]] = []
    if attack_type == "incident":
        for other in range(graph.num_nodes):
            if other == target or float(adjacency[target, other]) > 0:
                continue
            pair = tuple(sorted((int(target), int(other))))
            score = (
                float(probabilities[other, rival]),
                -float(similarity[other]),
                -float(other),
            )
            candidates.append((score, pair))
    else:
        anchors = torch.nonzero(adjacency[target] > 0, as_tuple=False).flatten().tolist()
        seen: set[tuple[int, int]] = set()
        for anchor in anchors:
            for other in range(graph.num_nodes):
                if other == target or other == anchor or float(adjacency[anchor, other]) > 0:
                    continue
                pair = tuple(sorted((int(anchor), int(other))))
                if target in pair or pair in seen:
                    continue
                seen.add(pair)
                score = (
                    float(probabilities[other, rival]),
                    -float(similarity[other]),
                    float(probabilities[anchor, rival]),
                    -float(anchor),
                    -float(other),
                )
                candidates.append((score, pair))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [pair for _, pair in candidates[:pool_size]]


def _candidate_adjacencies(adjacency: Tensor, candidates: list[tuple[int, int]]) -> Tensor:
    variants = adjacency.unsqueeze(0).repeat(len(candidates), 1, 1)
    for index, (u, v) in enumerate(candidates):
        variants[index, u, v] = 1.0
        variants[index, v, u] = 1.0
    return variants


@torch.no_grad()
def adaptive_gps_attack(
    model: GraphGPSNodeClassifier,
    graph: GraphData,
    clean_trace: ModelTrace,
    target: int,
    budgets: list[int],
    candidate_pool: int,
    attack_type: str,
    device: torch.device,
    graph_batch_size: int,
    telemetry: BatchTelemetry | None = None,
) -> dict[int, GPSAttackSnapshot]:
    if attack_type not in ATTACK_TYPES:
        raise ValueError(f"unsupported attack type: {attack_type}")
    maximum = max(budgets)
    current_adjacency = adjacency_from_edge_index(graph.edge_index, graph.num_nodes).float()
    current_logits = clean_trace.logits
    target_label = int(graph.y[target])
    added: list[tuple[int, int]] = []
    snapshots: dict[int, GPSAttackSnapshot] = {}
    for step in range(1, maximum + 1):
        candidates = candidate_additions(
            graph,
            current_adjacency,
            current_logits,
            target,
            candidate_pool,
            attack_type,
        )
        if not candidates:
            break
        variants = _candidate_adjacencies(current_adjacency, candidates)
        logits = logits_for_adjacencies(
            model,
            graph,
            variants,
            device,
            graph_batch_size,
            telemetry,
        )
        target_logits = logits[:, target, :]
        labels = torch.full((len(candidates),), target_label, dtype=torch.long)
        losses = F.cross_entropy(target_logits, labels, reduction="none")
        best_index = int(losses.argmax())
        best_pair = candidates[best_index]
        current_adjacency = variants[best_index]
        current_logits = logits[best_index]
        added.append(best_pair)
        if step in budgets:
            row = current_logits[target]
            prediction = int(row.argmax())
            snapshots[step] = GPSAttackSnapshot(
                attack_type=attack_type,
                target=int(target),
                budget=step,
                edge_index=edge_index_from_adjacency(current_adjacency),
                adjacency=current_adjacency.clone(),
                added_edges=tuple(added),
                target_loss=float(losses[best_index]),
                prediction=prediction,
                margin=true_margin(row, target_label),
                success=prediction != target_label,
            )
    return snapshots

