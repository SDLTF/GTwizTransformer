from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .model import ModelTrace


LOCALIZERS = ("random", "final_attention", "layer_attention", "full_dynamics")
TRACE_CHANNELS = (
    "attention",
    "entropy",
    "inter_layer_js",
    "value_contribution",
    "hidden_step",
    "probability_step",
    "hidden_cosine",
)


@dataclass(frozen=True)
class EdgeFeatures:
    pairs: Tensor
    groups: dict[str, Tensor]
    per_layer: Tensor


@dataclass(frozen=True)
class GaussianProfile:
    """Numerical normal-distribution parameters; no calibration edge IDs."""

    mean: Tensor
    precision: Tensor

    @classmethod
    def fit(cls, features: Tensor, shrinkage: float = 0.20) -> "GaussianProfile":
        x = features.detach().cpu().double()
        if x.ndim != 2 or x.size(0) < 2:
            raise ValueError("profile fitting needs a 2D matrix with at least two rows")
        mean = x.mean(0)
        centered = x - mean
        cov = centered.T @ centered / max(1, x.size(0) - 1)
        diagonal = torch.diag(torch.diag(cov))
        cov = (1.0 - shrinkage) * cov + shrinkage * diagonal
        scale = float(torch.diag(cov).mean().clamp_min(1e-12).item())
        cov = cov + torch.eye(cov.size(0), dtype=cov.dtype) * (1e-6 * scale + 1e-12)
        return cls(mean=mean, precision=torch.linalg.pinv(cov, hermitian=True))

    def score(self, features: Tensor) -> Tensor:
        centered = features.detach().cpu().double() - self.mean
        return torch.einsum("ni,ij,nj->n", centered, self.precision, centered).float()


def _endpoint_mean(values: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Convert [L, N] node values to [E, L] symmetric endpoint values."""
    return (0.5 * (values[:, u] + values[:, v])).T.contiguous()


def edge_features(trace: ModelTrace, pairs: Tensor) -> EdgeFeatures:
    pairs = pairs.detach().cpu().long()
    u, v = pairs[0], pairs[1]
    attention = trace.attentions.float()  # [L, H, N, N]
    layers, _, nodes, _ = attention.shape

    a_uv = attention[:, :, u, v].permute(2, 0, 1)  # [E, L, H]
    a_vu = attention[:, :, v, u].permute(2, 0, 1)
    symmetric_attention = 0.5 * (a_uv + a_vu).mean(-1)

    probability = attention.clamp_min(1e-12)
    entropy = -(probability * probability.log()).sum(-1) / max(math.log(nodes), 1.0)
    endpoint_entropy = _endpoint_mean(entropy.mean(1), u, v)

    js = torch.zeros_like(entropy)
    for layer in range(1, layers):
        left = probability[layer]
        right = probability[layer - 1]
        mixture = 0.5 * (left + right)
        js[layer] = 0.5 * (
            (left * (left.log() - mixture.log())).sum(-1)
            + (right * (right.log() - mixture.log())).sum(-1)
        )
    endpoint_js = _endpoint_mean(js.mean(1), u, v)

    value_norm = trace.values.float().norm(dim=-1)  # [L, N, H]
    c_uv = a_uv * value_norm[:, v, :].permute(1, 0, 2)
    c_vu = a_vu * value_norm[:, u, :].permute(1, 0, 2)
    contribution = 0.5 * (c_uv + c_vu).mean(-1)

    hidden_delta = (trace.hidden[1:] - trace.hidden[:-1]).float().norm(dim=-1)
    endpoint_hidden_delta = _endpoint_mean(hidden_delta, u, v)

    layer_probability = torch.softmax(trace.layer_logits.float(), dim=-1)
    probability_delta = 0.5 * (layer_probability[1:] - layer_probability[:-1]).abs().sum(-1)
    endpoint_probability_delta = _endpoint_mean(probability_delta, u, v)

    normalized_hidden = torch.nn.functional.normalize(trace.hidden[1:].float(), dim=-1)
    hidden_cosine = (
        normalized_hidden[:, u, :] * normalized_hidden[:, v, :]
    ).sum(-1).T.contiguous()

    per_layer = torch.stack(
        (
            symmetric_attention,
            endpoint_entropy,
            endpoint_js,
            contribution,
            endpoint_hidden_delta,
            endpoint_probability_delta,
            hidden_cosine,
        ),
        dim=-1,
    ).double()
    groups = {
        "final_attention": symmetric_attention[:, -1:].double(),
        "layer_attention": torch.cat(
            (symmetric_attention, endpoint_entropy, endpoint_js), dim=1
        ).double(),
        "full_dynamics": per_layer.flatten(1).double(),
    }
    return EdgeFeatures(pairs=pairs, groups=groups, per_layer=per_layer)


def fit_profiles(clean: EdgeFeatures) -> dict[str, GaussianProfile]:
    return {
        name: GaussianProfile.fit(features)
        for name, features in clean.groups.items()
    }


def feature_views(features: EdgeFeatures) -> dict[str, Tensor]:
    """Expose Phase-1 layer/channel ablations from the shared trace tensor."""
    values = features.per_layer
    if values.ndim != 3 or values.size(-1) != len(TRACE_CHANNELS):
        raise ValueError("unexpected per-layer trace layout")
    if values.size(1) < 2:
        temporal = torch.zeros((values.size(0), 1), dtype=values.dtype)
    else:
        temporal = (values[:, 1:, :] - values[:, :-1, :]).flatten(1)
    return {
        "first_layer_full": values[:, 0, :],
        "last_layer_full": values[:, -1, :],
        "all_layer_full": values.flatten(1),
        "temporal_residual": temporal,
        "attention_trajectory": values[:, :, :3].flatten(1),
        "value_trajectory": values[:, :, 3:4].flatten(1),
        "hidden_logit_trajectory": values[:, :, 4:].flatten(1),
        "all_layer_no_attention": values[:, :, 3:].flatten(1),
    }


def fit_view_profiles(clean: EdgeFeatures) -> dict[str, GaussianProfile]:
    return {
        name: GaussianProfile.fit(values)
        for name, values in feature_views(clean).items()
    }


def fit_layer_profiles(clean: EdgeFeatures) -> tuple[list[GaussianProfile], Tensor]:
    profiles: list[GaussianProfile] = []
    thresholds: list[float] = []
    for layer in range(clean.per_layer.size(1)):
        profile = GaussianProfile.fit(clean.per_layer[:, layer, :])
        scores = profile.score(clean.per_layer[:, layer, :])
        profiles.append(profile)
        thresholds.append(float(torch.quantile(scores, 0.95).item()))
    return profiles, torch.tensor(thresholds)


def score_layers(features: EdgeFeatures, profiles: list[GaussianProfile]) -> Tensor:
    return torch.stack(
        [
            profile.score(features.per_layer[:, layer, :])
            for layer, profile in enumerate(profiles)
        ],
        dim=1,
    )


def _canonical(pair: tuple[int, int]) -> tuple[int, int]:
    return tuple(sorted((int(pair[0]), int(pair[1]))))


def positive_mask(pairs: Tensor, positives: tuple[tuple[int, int], ...]) -> Tensor:
    wanted = {_canonical(pair) for pair in positives}
    return torch.tensor(
        [_canonical((int(u), int(v))) in wanted for u, v in pairs.T.tolist()],
        dtype=torch.bool,
    )


def ranking_metrics(pairs: Tensor, scores: Tensor, positives: tuple[tuple[int, int], ...], budget: int) -> dict[str, float]:
    labels = positive_mask(pairs, positives)
    total_positive = int(labels.sum())
    if total_positive == 0:
        return {key: float("nan") for key in (
            "edge_auprc", "recall_at_b", "iou_at_b", "first_positive_rank", "prevalence", "auprc_lift"
        )}
    order = torch.argsort(scores.detach().cpu(), descending=True)
    ranked = labels[order].float()
    cumulative = ranked.cumsum(0)
    precision = cumulative / torch.arange(1, ranked.numel() + 1)
    average_precision = float((precision * ranked).sum().item() / total_positive)
    k = min(int(budget), ranked.numel())
    hits = int(ranked[:k].sum().item())
    recall = hits / total_positive
    union = k + total_positive - hits
    first = int(torch.nonzero(ranked, as_tuple=False)[0].item()) + 1
    prevalence = total_positive / max(1, ranked.numel())
    return {
        "edge_auprc": average_precision,
        "recall_at_b": recall,
        "iou_at_b": hits / union if union else 0.0,
        "first_positive_rank": float(first),
        "prevalence": prevalence,
        "auprc_lift": average_precision / prevalence if prevalence else float("nan"),
    }


def top_pairs(pairs: Tensor, scores: Tensor, count: int) -> list[tuple[int, int]]:
    order = torch.argsort(scores.detach().cpu(), descending=True)[:count]
    return [tuple(map(int, pairs[:, index].tolist())) for index in order]


def first_divergence(
    features: EdgeFeatures,
    layer_profiles: list[GaussianProfile],
    thresholds: Tensor,
    positives: tuple[tuple[int, int], ...],
) -> dict[str, float]:
    mask = positive_mask(features.pairs, positives)
    if not mask.any():
        return {"first_divergence_layer": float("nan"), "divergence_detected_fraction": 0.0}
    scores = score_layers(features, layer_profiles)[mask]
    first_layers: list[int] = []
    for row in scores:
        crossed = torch.nonzero(row > thresholds, as_tuple=False)
        if crossed.numel():
            first_layers.append(int(crossed[0].item()) + 1)
    return {
        "first_divergence_layer": float(np.median(first_layers)) if first_layers else float("nan"),
        "divergence_detected_fraction": len(first_layers) / int(mask.sum()),
    }
