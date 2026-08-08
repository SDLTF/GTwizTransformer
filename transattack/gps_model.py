from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GINConv, GPSConv
from torch_geometric.utils import to_dense_batch

from .data import GraphData, adjacency_from_edge_index, set_seed
from .model import ModelTrace, accuracy


def rwse_from_adjacency(adjacency: Tensor, walk_length: int) -> Tensor:
    """Exact random-walk return probabilities for one or more dense graphs."""
    if walk_length < 1:
        raise ValueError("walk_length must be positive")
    squeeze = adjacency.ndim == 2
    if squeeze:
        adjacency = adjacency.unsqueeze(0)
    if adjacency.ndim != 3 or adjacency.size(-1) != adjacency.size(-2):
        raise ValueError("adjacency must have shape [N,N] or [B,N,N]")
    adjacency = adjacency.float()
    degree = adjacency.sum(-1).clamp_min(1.0)
    transition = adjacency / degree.unsqueeze(-1)
    power = transition
    diagonals: list[Tensor] = []
    for step in range(walk_length):
        diagonals.append(torch.diagonal(power, dim1=-2, dim2=-1))
        if step + 1 < walk_length:
            power = torch.bmm(power, transition)
    result = torch.stack(diagonals, dim=-1)
    return result.squeeze(0) if squeeze else result


def batched_edge_index(adjacencies: Tensor) -> Tensor:
    if adjacencies.ndim != 3 or adjacencies.size(-1) != adjacencies.size(-2):
        raise ValueError("adjacencies must have shape [B,N,N]")
    graph_id, source, target = torch.where(adjacencies > 0)
    nodes = adjacencies.size(1)
    return torch.stack((graph_id * nodes + source, graph_id * nodes + target), dim=0).long()


class TraceableGPSConv(GPSConv):
    """PyG GPSConv with an opt-in trace path for exact MHA weights/values."""

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
        return_trace: bool = False,
        **kwargs,
    ):
        hs: list[Tensor] = []
        if self.conv is not None:
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                h = self.norm1(h, batch=batch) if self.norm_with_batch else self.norm1(h)
            hs.append(h)

        dense, mask = to_dense_batch(x, batch)
        attention: Tensor | None = None
        values: Tensor | None = None
        padding_mask = None if bool(mask.all()) else ~mask
        if not isinstance(self.attn, torch.nn.MultiheadAttention):
            raise TypeError("Phase 3 traceable GPS requires exact multi-head attention")
        if return_trace:
            if self.attn.in_proj_weight is None:
                raise TypeError("separate MHA projection weights are not supported")
            width = self.attn.embed_dim
            value_weight = self.attn.in_proj_weight[2 * width :]
            value_bias = None if self.attn.in_proj_bias is None else self.attn.in_proj_bias[2 * width :]
            projected = F.linear(dense, value_weight, value_bias)
            values = projected.view(projected.size(0), projected.size(1), self.heads, width // self.heads)
            dense, attention = self.attn(
                dense,
                dense,
                dense,
                key_padding_mask=padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            dense, _ = self.attn(
                dense,
                dense,
                dense,
                key_padding_mask=padding_mask,
                need_weights=False,
            )

        h = dense[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x
        if self.norm2 is not None:
            h = self.norm2(h, batch=batch) if self.norm_with_batch else self.norm2(h)
        hs.append(h)
        out = sum(hs)
        out = out + self.mlp(out)
        if self.norm3 is not None:
            out = self.norm3(out, batch=batch) if self.norm_with_batch else self.norm3(out)
        if return_trace:
            assert attention is not None and values is not None
            return out, attention, values
        return out


class GraphGPSNodeClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        channels: int = 96,
        pe_channels: int = 16,
        walk_length: int = 8,
        layers: int = 4,
        heads: int = 8,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if channels <= pe_channels:
            raise ValueError("channels must exceed PE channels")
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.pe_channels = pe_channels
        self.walk_length = walk_length
        self.heads = heads
        self.node_projection = nn.Linear(input_dim, channels - pe_channels)
        self.pe_norm = nn.BatchNorm1d(walk_length)
        self.pe_projection = nn.Linear(walk_length, pe_channels)
        blocks: list[TraceableGPSConv] = []
        for _ in range(layers):
            local_mlp = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Linear(channels, channels),
            )
            blocks.append(
                TraceableGPSConv(
                    channels,
                    GINConv(local_mlp),
                    heads=heads,
                    dropout=dropout,
                    attn_type="multihead",
                    attn_kwargs={"dropout": dropout},
                )
            )
        self.convs = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(channels)
        self.classifier = nn.Linear(channels, num_classes)

    def _classify(self, hidden: Tensor) -> Tensor:
        return self.classifier(self.final_norm(hidden))

    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
        return_trace: bool = False,
    ):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        hidden = torch.cat(
            (self.node_projection(x), self.pe_projection(self.pe_norm(pe))),
            dim=-1,
        )
        hidden_states = [hidden]
        attentions: list[Tensor] = []
        values: list[Tensor] = []
        for conv in self.convs:
            if return_trace:
                hidden, attention, value = conv(hidden, edge_index, batch, return_trace=True)
                attentions.append(attention)
                values.append(value)
            else:
                hidden = conv(hidden, edge_index, batch)
            hidden_states.append(hidden)
        logits = self._classify(hidden)
        if not return_trace:
            return logits
        graph_count = int(batch.max().item()) + 1
        nodes = logits.size(0) // graph_count
        if graph_count == 1:
            return ModelTrace(
                logits=logits,
                attentions=torch.stack([item.squeeze(0) for item in attentions], dim=0),
                values=torch.stack([item.squeeze(0) for item in values], dim=0),
                hidden=torch.stack(hidden_states, dim=0),
                layer_logits=torch.stack([self._classify(item) for item in hidden_states], dim=0),
            )
        hidden_batch = torch.stack(hidden_states, dim=1).reshape(graph_count, nodes, len(hidden_states), self.channels)
        layer_logits = torch.stack([self._classify(item) for item in hidden_states], dim=1)
        return GPSBatchTrace(
            logits=logits.reshape(graph_count, nodes, -1),
            attentions=torch.stack(attentions, dim=1),
            values=torch.stack(values, dim=1),
            hidden=hidden_batch.permute(0, 2, 1, 3).contiguous(),
            layer_logits=layer_logits.reshape(graph_count, nodes, len(hidden_states), -1).permute(0, 2, 1, 3).contiguous(),
        )


@dataclass(frozen=True)
class GPSTrainResult:
    model: GraphGPSNodeClassifier
    best_epoch: int
    best_val_accuracy: float
    train_accuracy: float
    test_accuracy: float


@dataclass
class BatchTelemetry:
    requested_batch_size: int
    minimum_resolved_batch_size: int
    maximum_graphs_per_forward: int = 0
    graphs: int = 0
    forwards: int = 0


@dataclass(frozen=True)
class GPSBatchTrace:
    logits: Tensor
    attentions: Tensor
    values: Tensor
    hidden: Tensor
    layer_logits: Tensor


@dataclass(frozen=True)
class CandidateTraceViews:
    logits: Tensor
    all_layer_full: Tensor
    temporal_residual: Tensor


def _single_inputs(graph: GraphData, edge_index: Tensor, walk_length: int, device: torch.device):
    adjacency = adjacency_from_edge_index(edge_index, graph.num_nodes).float().to(device)
    pe = rwse_from_adjacency(adjacency, walk_length)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    return graph.x.to(device), pe, edge_index.to(device), batch


def train_gps_model(
    graph: GraphData,
    device: torch.device,
    seed: int,
    channels: int,
    pe_channels: int,
    walk_length: int,
    layers: int,
    heads: int,
    dropout: float,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> GPSTrainResult:
    set_seed(seed)
    model = GraphGPSNodeClassifier(
        graph.num_features,
        graph.num_classes,
        channels,
        pe_channels,
        walk_length,
        layers,
        heads,
        dropout,
    ).to(device)
    x, pe, edge_index, batch = _single_inputs(graph, graph.edge_index, walk_length, device)
    y = graph.y.to(device)
    train_idx, val_idx, test_idx = (item.to(device) for item in (graph.train_idx, graph.val_idx, graph.test_idx))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, pe, edge_index, batch)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = model(x, pe, edge_index, batch)
            val_acc = accuracy(validation, y, val_idx)
            val_loss = float(F.cross_entropy(validation[val_idx], y[val_idx]).item())
        improved = val_acc > best_val + 1e-8 or (abs(val_acc - best_val) <= 1e-8 and val_loss < best_loss - 1e-6)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_val, best_loss, best_epoch, stale = val_acc, val_loss, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, pe, edge_index, batch)
    return GPSTrainResult(
        model,
        best_epoch,
        best_val,
        accuracy(logits, y, train_idx),
        accuracy(logits, y, test_idx),
    )


@torch.no_grad()
def evaluate_gps_trace(
    model: GraphGPSNodeClassifier,
    graph: GraphData,
    edge_index: Tensor,
    device: torch.device,
) -> ModelTrace:
    model.eval()
    x, pe, edges, batch = _single_inputs(graph, edge_index, model.walk_length, device)
    trace = model(x, pe, edges, batch, return_trace=True)
    return ModelTrace(*(value.detach().cpu() for value in (
        trace.logits,
        trace.attentions,
        trace.values,
        trace.hidden,
        trace.layer_logits,
    )))


@torch.no_grad()
def logits_for_adjacencies(
    model: GraphGPSNodeClassifier,
    graph: GraphData,
    adjacencies: Tensor,
    device: torch.device,
    batch_size: int,
    telemetry: BatchTelemetry | None = None,
) -> Tensor:
    """Evaluate graph variants in CUDA batches, halving only on CUDA OOM."""
    if adjacencies.ndim == 2:
        adjacencies = adjacencies.unsqueeze(0)
    if adjacencies.ndim != 3:
        raise ValueError("adjacencies must have shape [B,N,N]")
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
            count, nodes = adjacency.size(0), adjacency.size(1)
            pe = rwse_from_adjacency(adjacency, model.walk_length).reshape(count * nodes, model.walk_length)
            edge_index = batched_edge_index(adjacency)
            x = graph.x.to(device).unsqueeze(0).expand(count, -1, -1).reshape(count * nodes, graph.num_features)
            batch = torch.arange(count, device=device).repeat_interleave(nodes)
            logits = model(x, pe, edge_index, batch).reshape(count, nodes, graph.num_classes)
            outputs.append(logits.detach().cpu())
            cursor += count
            if telemetry is not None:
                telemetry.graphs += count
                telemetry.forwards += 1
                telemetry.maximum_graphs_per_forward = max(telemetry.maximum_graphs_per_forward, count)
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or size == 1:
                raise
            adjacency = None
            torch.cuda.empty_cache()
            resolved = max(1, size // 2)
            if telemetry is not None:
                telemetry.minimum_resolved_batch_size = min(telemetry.minimum_resolved_batch_size, resolved)
    return torch.cat(outputs, dim=0)


def _endpoint(values: Tensor, nodes: Tensor) -> Tensor:
    index = nodes[:, None, None].expand(-1, values.size(1), 1)
    return torch.gather(values, 2, index).squeeze(2)


def _candidate_edge_views(trace: GPSBatchTrace, candidates: list[tuple[int, int]]) -> tuple[Tensor, Tensor]:
    """Extract the Phase-3 edge views for one designated edge per graph."""
    device = trace.logits.device
    graph_ids = torch.arange(len(candidates), device=device)
    u = torch.tensor([pair[0] for pair in candidates], device=device, dtype=torch.long)
    v = torch.tensor([pair[1] for pair in candidates], device=device, dtype=torch.long)
    attention = trace.attentions.float()
    layers, heads, nodes = attention.size(1), attention.size(2), attention.size(3)
    a_uv = attention[graph_ids, :, :, u, v]
    a_vu = attention[graph_ids, :, :, v, u]
    symmetric_attention = 0.5 * (a_uv + a_vu).mean(-1)

    probability = attention.clamp_min(1e-12)
    entropy = -(probability * probability.log()).sum(-1) / max(float(torch.log(torch.tensor(float(nodes)))), 1.0)
    entropy = entropy.mean(2)
    endpoint_entropy = 0.5 * (_endpoint(entropy, u) + _endpoint(entropy, v))

    js = torch.zeros_like(entropy)
    if layers > 1:
        left = probability[:, 1:]
        right = probability[:, :-1]
        mixture = 0.5 * (left + right)
        layer_js = 0.5 * (
            (left * (left.log() - mixture.log())).sum(-1)
            + (right * (right.log() - mixture.log())).sum(-1)
        ).mean(2)
        js[:, 1:] = layer_js
    endpoint_js = 0.5 * (_endpoint(js, u) + _endpoint(js, v))

    value_norm = trace.values.float().norm(dim=-1)
    value_u = torch.gather(value_norm, 2, u[:, None, None, None].expand(-1, layers, 1, heads)).squeeze(2)
    value_v = torch.gather(value_norm, 2, v[:, None, None, None].expand(-1, layers, 1, heads)).squeeze(2)
    contribution = 0.5 * (a_uv * value_v + a_vu * value_u).mean(-1)

    hidden_delta = (trace.hidden[:, 1:] - trace.hidden[:, :-1]).float().norm(dim=-1)
    endpoint_hidden_delta = 0.5 * (_endpoint(hidden_delta, u) + _endpoint(hidden_delta, v))

    layer_probability = torch.softmax(trace.layer_logits.float(), dim=-1)
    probability_delta = 0.5 * (layer_probability[:, 1:] - layer_probability[:, :-1]).abs().sum(-1)
    endpoint_probability_delta = 0.5 * (_endpoint(probability_delta, u) + _endpoint(probability_delta, v))

    normalized_hidden = F.normalize(trace.hidden[:, 1:].float(), dim=-1)
    hidden_u = torch.gather(
        normalized_hidden,
        2,
        u[:, None, None, None].expand(-1, layers, 1, normalized_hidden.size(-1)),
    ).squeeze(2)
    hidden_v = torch.gather(
        normalized_hidden,
        2,
        v[:, None, None, None].expand(-1, layers, 1, normalized_hidden.size(-1)),
    ).squeeze(2)
    hidden_cosine = (hidden_u * hidden_v).sum(-1)

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
    )
    temporal = per_layer[:, 1:] - per_layer[:, :-1]
    return per_layer.flatten(1).double(), temporal.flatten(1).double()


@torch.no_grad()
def candidate_trace_views_for_adjacencies(
    model: GraphGPSNodeClassifier,
    graph: GraphData,
    adjacencies: Tensor,
    candidates: list[tuple[int, int]],
    device: torch.device,
    batch_size: int,
    telemetry: BatchTelemetry | None = None,
) -> CandidateTraceViews:
    """Evaluate variants in batches and retain only each candidate edge's trace views."""
    if adjacencies.ndim != 3 or len(candidates) != adjacencies.size(0):
        raise ValueError("one candidate edge is required for every adjacency")
    model.eval()
    logits_output: list[Tensor] = []
    all_output: list[Tensor] = []
    temporal_output: list[Tensor] = []
    cursor = 0
    resolved = max(1, int(batch_size))
    if telemetry is not None:
        telemetry.requested_batch_size = int(batch_size)
        telemetry.minimum_resolved_batch_size = min(telemetry.minimum_resolved_batch_size, int(batch_size))
    while cursor < adjacencies.size(0):
        size = min(resolved, adjacencies.size(0) - cursor)
        try:
            adjacency = adjacencies[cursor : cursor + size].to(device=device, dtype=torch.float32)
            count, nodes = adjacency.size(0), adjacency.size(1)
            pe = rwse_from_adjacency(adjacency, model.walk_length).reshape(count * nodes, model.walk_length)
            edge_index = batched_edge_index(adjacency)
            x = graph.x.to(device).unsqueeze(0).expand(count, -1, -1).reshape(count * nodes, graph.num_features)
            batch = torch.arange(count, device=device).repeat_interleave(nodes)
            trace = model(x, pe, edge_index, batch, return_trace=True)
            if isinstance(trace, ModelTrace):
                trace = GPSBatchTrace(
                    logits=trace.logits.unsqueeze(0),
                    attentions=trace.attentions.unsqueeze(0),
                    values=trace.values.unsqueeze(0),
                    hidden=trace.hidden.unsqueeze(0),
                    layer_logits=trace.layer_logits.unsqueeze(0),
                )
            all_view, temporal_view = _candidate_edge_views(trace, candidates[cursor : cursor + count])
            logits_output.append(trace.logits.detach().cpu())
            all_output.append(all_view.detach().cpu())
            temporal_output.append(temporal_view.detach().cpu())
            cursor += count
            if telemetry is not None:
                telemetry.graphs += count
                telemetry.forwards += 1
                telemetry.maximum_graphs_per_forward = max(telemetry.maximum_graphs_per_forward, count)
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or size == 1:
                raise
            adjacency = None
            torch.cuda.empty_cache()
            resolved = max(1, size // 2)
            if telemetry is not None:
                telemetry.minimum_resolved_batch_size = min(telemetry.minimum_resolved_batch_size, resolved)
    return CandidateTraceViews(
        logits=torch.cat(logits_output),
        all_layer_full=torch.cat(all_output),
        temporal_residual=torch.cat(temporal_output),
    )
