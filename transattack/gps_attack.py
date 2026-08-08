from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .data import GraphData, adjacency_from_edge_index, edge_index_from_adjacency
from .gps_model import (
    BatchTelemetry,
    GraphGPSNodeClassifier,
    candidate_trace_views_for_adjacencies,
    logits_for_adjacencies,
)
from .localize import GaussianProfile
from .model import ModelTrace, true_margin


ATTACK_TYPES = ("incident", "remote")
ATTACK_OBJECTIVES = (
    "classification_only",
    "adaptive_stealth",
    "classification_constrained_stealth",
)


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
    selected_gain_ratio: float = float("nan")
    minimum_selected_gain_ratio: float = float("nan")
    eligible_candidates: int = 0
    mean_eligible_candidates: float = float("nan")


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


def _robust_z(values: Tensor) -> Tensor:
    values = values.detach().cpu().float()
    median = values.median()
    scale = (values - median).abs().median().clamp_min(1e-6)
    return (values - median) / (1.4826 * scale)


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
    attack_objective: str = "classification_only",
    profiles: dict[str, GaussianProfile] | None = None,
    adaptive_stealth_strength: float = 1.0,
    classification_retention_ratio: float = 0.85,
) -> dict[int, GPSAttackSnapshot]:
    if attack_type not in ATTACK_TYPES:
        raise ValueError(f"unsupported attack type: {attack_type}")
    if attack_objective not in ATTACK_OBJECTIVES:
        raise ValueError(f"unsupported attack objective: {attack_objective}")
    if adaptive_stealth_strength < 0:
        raise ValueError("adaptive stealth strength must be nonnegative")
    if not 0 < classification_retention_ratio <= 1:
        raise ValueError("classification retention ratio must lie in (0, 1]")
    if attack_objective != "classification_only" and profiles is None:
        raise ValueError("stealth-aware attacks require clean trace profiles")
    maximum = max(budgets)
    current_adjacency = adjacency_from_edge_index(graph.edge_index, graph.num_nodes).float()
    current_logits = clean_trace.logits
    target_label = int(graph.y[target])
    added: list[tuple[int, int]] = []
    selected_gain_ratios: list[float] = []
    eligible_counts: list[int] = []
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
        if attack_objective == "classification_only":
            logits = logits_for_adjacencies(
                model,
                graph,
                variants,
                device,
                graph_batch_size,
                telemetry,
            )
            all_scores = temporal_scores = cf_signal = None
        else:
            traced = candidate_trace_views_for_adjacencies(
                model,
                graph,
                variants,
                candidates,
                device,
                graph_batch_size,
                telemetry,
            )
            logits = traced.logits
            assert profiles is not None
            all_scores = profiles["all_layer_full"].score(traced.all_layer_full)
            temporal_scores = profiles["temporal_residual"].score(traced.temporal_residual)
        target_logits = logits[:, target, :]
        labels = torch.full((len(candidates),), target_label, dtype=torch.long)
        losses = F.cross_entropy(target_logits, labels, reduction="none")
        if attack_objective == "classification_only":
            selection_scores = losses
        else:
            predictions = target_logits.argmax(dim=-1)
            candidate_probabilities = torch.softmax(target_logits, dim=-1)
            attacked_probability = candidate_probabilities.gather(1, predictions.unsqueeze(1)).squeeze(1)
            previous_probability = torch.softmax(current_logits[target].float(), dim=-1)[predictions]
            cf_signal = (attacked_probability - previous_probability).clamp_min(0.0)
            assert all_scores is not None and temporal_scores is not None
            stealth_penalty = (
                _robust_z(all_scores)
                + _robust_z(temporal_scores)
                + _robust_z(cf_signal)
            ) / 3.0
            if attack_objective == "adaptive_stealth":
                selection_scores = _robust_z(losses) - adaptive_stealth_strength * stealth_penalty
            else:
                current_row = current_logits[target].float().reshape(1, -1)
                current_loss = float(F.cross_entropy(current_row, torch.tensor([target_label])))
                gains = losses - current_loss
                best_loss_index = int(losses.argmax())
                best_gain = float(gains[best_loss_index])
                if classification_retention_ratio == 1.0:
                    eligible = torch.zeros_like(losses, dtype=torch.bool)
                    eligible[best_loss_index] = True
                elif best_gain > 1e-12:
                    eligible = gains >= classification_retention_ratio * best_gain
                else:
                    eligible = torch.zeros_like(losses, dtype=torch.bool)
                    eligible[best_loss_index] = True
                eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
                eligible_penalties = stealth_penalty[eligible_indices]
                minimum_penalty = eligible_penalties.min()
                tied = eligible_indices[eligible_penalties <= minimum_penalty + 1e-12]
                best_index = int(tied[losses[tied].argmax()])
                selected_gain_ratios.append(
                    float(gains[best_index] / best_gain) if best_gain > 1e-12 else 1.0
                )
                eligible_counts.append(int(eligible.sum()))
                selection_scores = None
        if attack_objective != "classification_constrained_stealth":
            best_index = int(selection_scores.argmax())
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
                selected_gain_ratio=(selected_gain_ratios[-1] if selected_gain_ratios else float("nan")),
                minimum_selected_gain_ratio=(min(selected_gain_ratios) if selected_gain_ratios else float("nan")),
                eligible_candidates=(eligible_counts[-1] if eligible_counts else 0),
                mean_eligible_candidates=(
                    sum(eligible_counts) / len(eligible_counts) if eligible_counts else float("nan")
                ),
            )
    return snapshots
