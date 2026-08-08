from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .data import adjacency_from_edge_index


PE_NAMES = ("heat", "rw")


@dataclass(frozen=True)
class PEBundle:
    name: str
    values: Tensor


def _laplacian(adj: Tensor) -> Tensor:
    return torch.diag(adj.sum(1)) - adj


def _heat(adj: Tensor, dim: int) -> Tensor:
    values, vectors = torch.linalg.eigh(_laplacian(adj))
    times = torch.logspace(np.log10(0.05), np.log10(3.2), steps=dim, dtype=adj.dtype)
    channels = []
    for time in times:
        weights = torch.exp(-time * values)
        channels.append(((vectors * weights.unsqueeze(0)) @ vectors.T).float())
    return torch.stack(channels, dim=-1)


def _rw(adj: Tensor, dim: int) -> Tensor:
    n = adj.size(0)
    transition = adj / adj.sum(1).clamp_min(1.0).unsqueeze(1)
    lazy = 0.5 * (torch.eye(n, dtype=adj.dtype) + transition)
    power = torch.eye(n, dtype=adj.dtype)
    channels = []
    for _ in range(dim):
        power = power @ lazy
        channels.append(power.float())
    return torch.stack(channels, dim=-1)


def build_pe(name: str, edge_index: Tensor, num_nodes: int, dim: int) -> PEBundle:
    normalized = name.lower().strip()
    if normalized not in PE_NAMES:
        raise ValueError(f"unknown PE {name!r}; choose from {PE_NAMES}")
    if dim < 2:
        raise ValueError("pe dimension must be at least 2")
    adj = adjacency_from_edge_index(edge_index, num_nodes)
    values = _heat(adj, dim) if normalized == "heat" else _rw(adj, dim)
    return PEBundle(normalized, values)


def rms_scale(pe: Tensor) -> float:
    return max(float(pe.square().mean().sqrt().item()), 1.0 / 32.0)

