from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .data import GraphData, adjacency_from_edge_index
from .gps_attack import _candidate_adjacencies, candidate_additions
from .gps_model import BatchTelemetry, GraphGPSNodeClassifier, logits_for_adjacencies
from .model import ModelTrace


OBJECTIVES = ("cross_entropy", "normalized_margin")
CANDIDATE_STRATEGIES = ("single_rival", "multi_rival")
POOL_MODES = ("fixed", "adaptive")


@dataclass(frozen=True)
class HeuristicState:
    adjacency: Tensor
    logits: Tensor
    added_edges: tuple[tuple[int, int], ...]
    objective_score: float
    attack_score: float
    target_loss: float


@dataclass(frozen=True)
class HeuristicSnapshot:
    budget: int
    added_edges: tuple[tuple[int, int], ...]
    margin: float
    attack_score: float
    objective_score: float
    target_loss: float
    prediction: int
    success: bool
    used_edges: int


@dataclass(frozen=True)
class HeuristicSearchResult:
    exact: dict[int, HeuristicSnapshot]
    within_budget: dict[int, HeuristicSnapshot]
    expanded_states: int
    evaluated_candidates: int
    adaptive_expansions: int


def _scores(rows: Tensor, target_label: int, objective: str) -> tuple[Tensor, Tensor, Tensor]:
    if objective not in OBJECTIVES:
        raise ValueError(f"unsupported objective: {objective}")
    rows = rows.float()
    labels = torch.full((rows.size(0),), int(target_label), dtype=torch.long, device=rows.device)
    losses = F.cross_entropy(rows, labels, reduction="none")
    rivals = rows.clone()
    rivals[:, target_label] = -float("inf")
    raw_attack = rivals.max(dim=-1).values - rows[:, target_label]
    if objective == "cross_entropy":
        objective_scores = losses
    else:
        if rows.size(1) < 3:
            raise ValueError("normalized margin requires at least three classes")
        ordered = rows.sort(dim=-1, descending=True).values
        scale = (ordered[:, 0] - ordered[:, 2]).clamp_min(1e-4)
        objective_scores = raw_attack / scale
    return objective_scores, raw_attack, losses


def _multi_rival_candidates(
    graph: GraphData,
    adjacency: Tensor,
    logits: Tensor,
    target: int,
    pool_size: int,
) -> list[tuple[int, int]]:
    adjacency = adjacency.detach().cpu()
    probabilities = torch.softmax(logits.detach().cpu().float(), dim=-1)
    features = F.normalize(graph.x.float(), dim=-1)
    similarity = features @ features[target]
    target_label = int(graph.y[target])
    anchors = torch.nonzero(adjacency[target] > 0, as_tuple=False).flatten().tolist()
    by_class: dict[int, list[tuple[tuple[float, ...], tuple[int, int]]]] = {
        cls: [] for cls in range(graph.num_classes) if cls != target_label
    }
    global_rows: list[tuple[tuple[float, ...], tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for anchor in anchors:
        for other in range(graph.num_nodes):
            if other == target or other == anchor or float(adjacency[anchor, other]) > 0:
                continue
            pair = tuple(sorted((int(anchor), int(other))))
            if target in pair or pair in seen:
                continue
            seen.add(pair)
            class_scores = []
            for cls in by_class:
                score = (
                    float(probabilities[other, cls]),
                    -float(similarity[other]),
                    float(probabilities[anchor, cls]),
                    -float(anchor),
                    -float(other),
                )
                by_class[cls].append((score, pair))
                class_scores.append(score)
            global_rows.append((max(class_scores), pair))
    rivals = max(1, len(by_class))
    quota = max(1, int(math.ceil(pool_size / rivals)))
    selected: list[tuple[int, int]] = []
    selected_set: set[tuple[int, int]] = set()
    for cls in sorted(by_class):
        by_class[cls].sort(key=lambda item: item[0], reverse=True)
        for _, pair in by_class[cls][:quota]:
            if pair not in selected_set:
                selected.append(pair)
                selected_set.add(pair)
    global_rows.sort(key=lambda item: item[0], reverse=True)
    for _, pair in global_rows:
        if len(selected) >= pool_size:
            break
        if pair not in selected_set:
            selected.append(pair)
            selected_set.add(pair)
    return selected[:pool_size]


def heuristic_candidates(
    graph: GraphData,
    adjacency: Tensor,
    logits: Tensor,
    target: int,
    pool_size: int,
    strategy: str,
) -> list[tuple[int, int]]:
    if strategy == "single_rival":
        return candidate_additions(graph, adjacency, logits, target, pool_size, "remote")
    if strategy == "multi_rival":
        return _multi_rival_candidates(graph, adjacency, logits, target, pool_size)
    raise ValueError(f"unsupported candidate strategy: {strategy}")


def _state_from_logits(
    adjacency: Tensor,
    logits: Tensor,
    added_edges: tuple[tuple[int, int], ...],
    target: int,
    target_label: int,
    objective: str,
) -> HeuristicState:
    objective_scores, attack_scores, losses = _scores(logits[target].reshape(1, -1), target_label, objective)
    return HeuristicState(
        adjacency.detach().cpu(),
        logits.detach().cpu(),
        added_edges,
        float(objective_scores[0]),
        float(attack_scores[0]),
        float(losses[0]),
    )


def _snapshot(state: HeuristicState, graph: GraphData, target: int, budget: int) -> HeuristicSnapshot:
    label = int(graph.y[target])
    prediction = int(state.logits[target].argmax())
    return HeuristicSnapshot(
        budget=int(budget),
        added_edges=state.added_edges,
        margin=-state.attack_score,
        attack_score=state.attack_score,
        objective_score=state.objective_score,
        target_loss=state.target_loss,
        prediction=prediction,
        success=prediction != label,
        used_edges=len(state.added_edges),
    )


@torch.no_grad()
def heuristic_remote_search(
    model: GraphGPSNodeClassifier,
    graph: GraphData,
    clean_trace: ModelTrace,
    target: int,
    budgets: list[int],
    objective: str,
    beam_width: int,
    candidate_strategy: str,
    pool_mode: str,
    candidate_pool: int,
    maximum_candidate_pool: int,
    device: torch.device,
    graph_batch_size: int,
    telemetry: BatchTelemetry | None = None,
) -> HeuristicSearchResult:
    if objective not in OBJECTIVES:
        raise ValueError(f"unsupported objective: {objective}")
    if candidate_strategy not in CANDIDATE_STRATEGIES:
        raise ValueError(f"unsupported candidate strategy: {candidate_strategy}")
    if pool_mode not in POOL_MODES:
        raise ValueError(f"unsupported pool mode: {pool_mode}")
    if beam_width < 1 or candidate_pool < 1 or maximum_candidate_pool < candidate_pool:
        raise ValueError("invalid beam or candidate-pool configuration")
    target_label = int(graph.y[target])
    clean_adjacency = adjacency_from_edge_index(graph.edge_index, graph.num_nodes).float()
    clean = _state_from_logits(clean_adjacency, clean_trace.logits, tuple(), target, target_label, objective)
    beam = [clean]
    best_within = clean
    exact: dict[int, HeuristicSnapshot] = {}
    within: dict[int, HeuristicSnapshot] = {}
    expanded_states = evaluated_candidates = adaptive_expansions = 0

    def evaluate(parent_rows: list[tuple[int, tuple[int, int], Tensor]]) -> list[HeuristicState]:
        nonlocal evaluated_candidates
        if not parent_rows:
            return []
        variants = torch.stack([item[2] for item in parent_rows], dim=0)
        evaluated = logits_for_adjacencies(model, graph, variants, device, graph_batch_size, telemetry)
        evaluated_candidates += len(parent_rows)
        output: list[HeuristicState] = []
        for row_index, (parent_index, pair, adjacency) in enumerate(parent_rows):
            parent = beam[parent_index]
            output.append(_state_from_logits(
                adjacency,
                evaluated[row_index],
                parent.added_edges + (pair,),
                target,
                target_label,
                objective,
            ))
        return output

    for step in range(1, max(budgets) + 1):
        base_rows: list[tuple[int, tuple[int, int], Tensor]] = []
        base_pairs: dict[int, set[tuple[int, int]]] = {}
        for parent_index, state in enumerate(beam):
            candidates = heuristic_candidates(
                graph, state.adjacency, state.logits, target, candidate_pool, candidate_strategy
            )
            base_pairs[parent_index] = set(candidates)
            variants = _candidate_adjacencies(state.adjacency, candidates)
            base_rows.extend((parent_index, pair, variants[index]) for index, pair in enumerate(candidates))
        expanded_states += len(beam)
        children = evaluate(base_rows)

        if pool_mode == "adaptive" and maximum_candidate_pool > candidate_pool:
            best_gain = {index: -float("inf") for index in range(len(beam))}
            cursor = 0
            for parent_index, pairs in base_pairs.items():
                count = len(pairs)
                parent_children = children[cursor : cursor + count]
                if parent_children:
                    best_gain[parent_index] = max(item.objective_score - beam[parent_index].objective_score for item in parent_children)
                cursor += count
            extra_rows: list[tuple[int, tuple[int, int], Tensor]] = []
            for parent_index, state in enumerate(beam):
                if best_gain[parent_index] > 1e-12:
                    continue
                expanded = heuristic_candidates(
                    graph, state.adjacency, state.logits, target, maximum_candidate_pool, candidate_strategy
                )
                extras = [pair for pair in expanded if pair not in base_pairs[parent_index]]
                if not extras:
                    continue
                adaptive_expansions += 1
                variants = _candidate_adjacencies(state.adjacency, extras)
                extra_rows.extend((parent_index, pair, variants[index]) for index, pair in enumerate(extras))
            children.extend(evaluate(extra_rows))

        unique: dict[tuple[tuple[int, int], ...], HeuristicState] = {}
        for child in children:
            key = tuple(sorted(child.added_edges))
            previous = unique.get(key)
            if previous is None or child.objective_score > previous.objective_score:
                unique[key] = child
        ranked = sorted(
            unique.values(),
            key=lambda state: (state.objective_score, state.attack_score, tuple(state.added_edges)),
            reverse=True,
        )
        if not ranked:
            break
        beam = ranked[:beam_width]
        canonical_best = max(beam, key=lambda state: (state.attack_score, state.objective_score))
        if canonical_best.attack_score > best_within.attack_score:
            best_within = canonical_best
        if step in budgets:
            exact[step] = _snapshot(canonical_best, graph, target, step)
            within[step] = _snapshot(best_within, graph, target, step)
    return HeuristicSearchResult(exact, within, expanded_states, evaluated_candidates, adaptive_expansions)
