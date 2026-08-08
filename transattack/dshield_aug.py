from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor

from .data import GraphData
from .gps_model import BatchTelemetry, GraphGPSNodeClassifier, batched_edge_index, rwse_from_adjacency


@dataclass(frozen=True)
class ViewStatistics:
    prediction: int
    target_probability: float
    disagreement: float


def stable_view_seed(key: str, base_seed: int = 61000) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:8], "little")) % (2**31 - 1))


def augmented_graph_views(
    graph: GraphData,
    adjacency: Tensor,
    views: int,
    edge_drop_ratio: float,
    feature_drop_ratio: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Create paired, deterministic DShield-style graph augmentations on CPU."""
    if views < 1:
        raise ValueError("views must be positive")
    if not 0.0 <= edge_drop_ratio <= 1.0:
        raise ValueError("edge drop ratio must lie in [0, 1]")
    if not 0.0 <= feature_drop_ratio <= 1.0:
        raise ValueError("feature drop ratio must lie in [0, 1]")
    adjacency = adjacency.detach().cpu().float()
    if adjacency.shape != (graph.num_nodes, graph.num_nodes):
        raise ValueError("adjacency shape does not match graph")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    # Draw for every potential pair, not only existing edges. Clean and
    # attacked graphs called with the same seed therefore share exactly the
    # same augmentation decision for every common edge.
    pair_random = torch.rand((views, graph.num_nodes, graph.num_nodes), generator=generator)
    upper = torch.triu(adjacency, diagonal=1).bool().unsqueeze(0)
    kept = upper & (pair_random >= edge_drop_ratio)
    adjacencies = kept.float() + kept.transpose(1, 2).float()

    features = graph.x.detach().cpu().float().unsqueeze(0).expand(views, -1, -1).clone()
    if feature_drop_ratio > 0.0:
        feature_random = torch.rand(features.shape, generator=generator)
        features[feature_random < feature_drop_ratio] = 0.0
    return adjacencies, features


@torch.no_grad()
def logits_for_augmented_views(
    model: GraphGPSNodeClassifier,
    graph: GraphData,
    adjacencies: Tensor,
    features: Tensor,
    device: torch.device,
    batch_size: int,
    telemetry: BatchTelemetry | None = None,
) -> Tensor:
    if adjacencies.ndim != 3 or features.ndim != 3:
        raise ValueError("adjacencies and features must be batched")
    if adjacencies.size(0) != features.size(0) or adjacencies.size(1) != graph.num_nodes:
        raise ValueError("view batches do not match")
    model.eval()
    outputs: list[Tensor] = []
    cursor = 0
    resolved = max(1, int(batch_size))
    if telemetry is not None:
        telemetry.requested_batch_size = int(batch_size)
        telemetry.minimum_resolved_batch_size = min(telemetry.minimum_resolved_batch_size, int(batch_size))
    while cursor < adjacencies.size(0):
        size = min(resolved, adjacencies.size(0) - cursor)
        try:
            adjacency = adjacencies[cursor : cursor + size].to(device=device, dtype=torch.float32)
            view_x = features[cursor : cursor + size].to(device=device, dtype=torch.float32)
            count, nodes = adjacency.size(0), adjacency.size(1)
            pe = rwse_from_adjacency(adjacency, model.walk_length).reshape(count * nodes, model.walk_length)
            edge_index = batched_edge_index(adjacency)
            batch = torch.arange(count, device=device).repeat_interleave(nodes)
            logits = model(view_x.reshape(count * nodes, graph.num_features), pe, edge_index, batch)
            outputs.append(logits.reshape(count, nodes, graph.num_classes).detach().cpu())
            cursor += count
            if telemetry is not None:
                telemetry.graphs += count
                telemetry.forwards += 1
                telemetry.maximum_graphs_per_forward = max(telemetry.maximum_graphs_per_forward, count)
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or size == 1:
                raise
            adjacency = view_x = None
            torch.cuda.empty_cache()
            resolved = max(1, size // 2)
            if telemetry is not None:
                telemetry.minimum_resolved_batch_size = min(telemetry.minimum_resolved_batch_size, resolved)
    return torch.cat(outputs, dim=0)


def view_statistics(logits: Tensor, target: int, true_label: int) -> ViewStatistics:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [views, nodes, classes]")
    probabilities = torch.softmax(logits[:, target].double(), dim=-1)
    mean_probability = probabilities.mean(0)
    entropy_mean = -(mean_probability.clamp_min(1e-12) * mean_probability.clamp_min(1e-12).log()).sum()
    view_entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(-1).mean()
    return ViewStatistics(
        prediction=int(mean_probability.argmax()),
        target_probability=float(mean_probability[true_label]),
        disagreement=max(0.0, float(entropy_mean - view_entropy)),
    )
