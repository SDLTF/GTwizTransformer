from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .data import GraphData, set_seed


@dataclass(frozen=True)
class ModelTrace:
    logits: Tensor
    attentions: Tensor
    values: Tensor
    hidden: Tensor
    layer_logits: Tensor


class DenseGraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, pe_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden dimension must be divisible by number of heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.pe_to_bias = nn.Linear(pe_dim, heads, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, pe: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n = hidden.size(0)
        qkv = self.qkv(self.norm_attn(hidden)).view(n, 3, self.heads, self.head_dim)
        query, key, value = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        content = torch.einsum("ihd,jhd->hij", query, key) / math.sqrt(self.head_dim)
        bias = self.pe_to_bias(pe).permute(2, 0, 1)
        attention = torch.softmax(content + bias, dim=-1)
        mixed = torch.einsum("hij,jhd->ihd", attention, value).reshape(n, -1)
        hidden = hidden + self.dropout(self.out(mixed))
        hidden = hidden + self.dropout(self.ffn(self.norm_ffn(hidden)))
        return hidden, attention, value


class DenseGraphTransformer(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, pe_dim: int, hidden_dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            DenseGraphTransformerLayer(hidden_dim, heads, pe_dim, dropout)
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _classify(self, hidden: Tensor) -> Tensor:
        return self.classifier(self.final_norm(hidden))

    def forward(self, x: Tensor, pe: Tensor, return_trace: bool = False):
        hidden = self.input_projection(x)
        hidden_states = [hidden]
        attentions: list[Tensor] = []
        values: list[Tensor] = []
        for layer in self.layers:
            hidden, attention, value = layer(hidden, pe)
            hidden_states.append(hidden)
            attentions.append(attention)
            values.append(value)
        logits = self._classify(hidden)
        if not return_trace:
            return logits
        layer_logits = torch.stack([self._classify(item) for item in hidden_states], dim=0)
        return ModelTrace(
            logits=logits,
            attentions=torch.stack(attentions, dim=0),
            values=torch.stack(values, dim=0),
            hidden=torch.stack(hidden_states, dim=0),
            layer_logits=layer_logits,
        )


@dataclass(frozen=True)
class TrainResult:
    model: DenseGraphTransformer
    best_epoch: int
    best_val_accuracy: float
    train_accuracy: float
    test_accuracy: float


@torch.no_grad()
def accuracy(logits: Tensor, labels: Tensor, indices: Tensor) -> float:
    if not indices.numel():
        return float("nan")
    return float((logits[indices].argmax(1) == labels[indices]).float().mean().item())


def train_model(graph: GraphData, pe: Tensor, device: torch.device, seed: int, hidden_dim: int, heads: int, layers: int, dropout: float, epochs: int, patience: int, learning_rate: float, weight_decay: float) -> TrainResult:
    set_seed(seed)
    model = DenseGraphTransformer(
        graph.num_features,
        graph.num_classes,
        pe.size(-1),
        hidden_dim,
        heads,
        layers,
        dropout,
    ).to(device)
    x, y, pe_device = graph.x.to(device), graph.y.to(device), pe.to(device)
    train_idx, val_idx, test_idx = (v.to(device) for v in (graph.train_idx, graph.val_idx, graph.test_idx))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val, best_val_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, pe_device)
        loss = torch.nn.functional.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(x, pe_device)
            val_acc = accuracy(validation_logits, y, val_idx)
            val_loss = float(torch.nn.functional.cross_entropy(validation_logits[val_idx], y[val_idx]).item())
        improved = val_acc > best_val + 1e-8 or (abs(val_acc - best_val) <= 1e-8 and val_loss < best_val_loss - 1e-6)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_val, best_val_loss, best_epoch, stale = val_acc, val_loss, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, pe_device)
    return TrainResult(model, best_epoch, best_val, accuracy(logits, y, train_idx), accuracy(logits, y, test_idx))


@torch.no_grad()
def evaluate_trace(model: DenseGraphTransformer, graph: GraphData, pe: Tensor, device: torch.device) -> ModelTrace:
    model.eval()
    trace = model(graph.x.to(device), pe.to(device), return_trace=True)
    return ModelTrace(*(value.detach().cpu() for value in (
        trace.logits,
        trace.attentions,
        trace.values,
        trace.hidden,
        trace.layer_logits,
    )))


def true_margin(logits: Tensor, label: int) -> float:
    row = logits.detach().cpu().flatten()
    mask = torch.ones_like(row, dtype=torch.bool)
    mask[int(label)] = False
    return float((row[int(label)] - row[mask].max()).item())

