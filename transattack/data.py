from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch import Tensor


@dataclass(frozen=True)
class GraphData:
    name: str
    x: Tensor
    y: Tensor
    edge_index: Tensor
    train_idx: Tensor
    val_idx: Tensor
    test_idx: Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_features(self) -> int:
        return int(self.x.size(1))

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adjacency_from_edge_index(edge_index: Tensor, num_nodes: int) -> Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    if edge_index.numel():
        edges = edge_index.detach().cpu().long()
        adj[edges[0], edges[1]] = 1.0
    adj.fill_diagonal_(0.0)
    return torch.maximum(adj, adj.T)


def undirected_pairs(edge_index: Tensor) -> Tensor:
    edges = edge_index.detach().cpu().long()
    if not edges.numel():
        return torch.empty((2, 0), dtype=torch.long)
    u = torch.minimum(edges[0], edges[1])
    v = torch.maximum(edges[0], edges[1])
    keep = u != v
    pairs = torch.stack((u[keep], v[keep]), dim=0)
    return torch.unique(pairs, dim=1, sorted=True)


def pairs_to_edge_index(pairs: Tensor) -> Tensor:
    pairs = pairs.detach().cpu().long()
    if not pairs.numel():
        return torch.empty((2, 0), dtype=torch.long)
    return torch.cat((pairs, pairs.flip(0)), dim=1)


def edge_index_from_adjacency(adj: Tensor) -> Tensor:
    rows, cols = torch.where(torch.triu(adj.detach().cpu(), diagonal=1) > 0)
    return pairs_to_edge_index(torch.stack((rows, cols), dim=0))


def add_edge(edge_index: Tensor, num_nodes: int, u: int, v: int) -> Tensor:
    if u == v:
        raise ValueError("self-loops are outside the threat model")
    u, v = sorted((int(u), int(v)))
    adj = adjacency_from_edge_index(edge_index, num_nodes)
    if adj[u, v] > 0:
        raise ValueError(f"edge ({u}, {v}) already exists")
    adj[u, v] = adj[v, u] = 1.0
    return edge_index_from_adjacency(adj)


def remove_edges(edge_index: Tensor, num_nodes: int, pairs: list[tuple[int, int]]) -> Tensor:
    adj = adjacency_from_edge_index(edge_index, num_nodes)
    for u, v in pairs:
        u, v = sorted((int(u), int(v)))
        adj[u, v] = adj[v, u] = 0.0
    return edge_index_from_adjacency(adj)


def count_edges(edge_index: Tensor) -> int:
    return int(undirected_pairs(edge_index).size(1))


def _split(y: Tensor, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    labels = y.cpu().numpy()
    for cls in np.unique(labels):
        ids = np.flatnonzero(labels == cls)
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(round(0.60 * n)))
        n_val = max(1, int(round(0.20 * n)))
        if n_train + n_val >= n:
            n_train, n_val = max(1, n - 2), 1
        train.extend(ids[:n_train].tolist())
        val.extend(ids[n_train : n_train + n_val].tolist())
        test.extend(ids[n_train + n_val :].tolist())
    return tuple(torch.tensor(sorted(v), dtype=torch.long) for v in (train, val, test))


def make_sbm(num_nodes: int, seed: int, num_classes: int = 3) -> GraphData:
    if num_nodes < 24:
        raise ValueError("SBM smoke graph needs at least 24 nodes")
    rng = np.random.default_rng(seed)
    labels = np.arange(num_nodes, dtype=np.int64) % num_classes
    rng.shuffle(labels)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = 0.24 if labels[i] == labels[j] else 0.025
            if rng.random() < p:
                adj[i, j] = adj[j, i] = 1.0
    for i in range(num_nodes):
        if adj[i].sum() == 0:
            choices = np.flatnonzero(labels == labels[i])
            choices = choices[choices != i]
            j = int(rng.choice(choices))
            adj[i, j] = adj[j, i] = 1.0
    centers = rng.normal(size=(num_classes, 24))
    x = centers[labels] + 1.10 * rng.normal(size=(num_nodes, 24))
    x = (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)
    y = torch.from_numpy(labels).long()
    train, val, test = _split(y, seed + 17)
    return GraphData(
        "SBM",
        torch.from_numpy(x).float(),
        y,
        edge_index_from_adjacency(torch.from_numpy(adj)),
        train,
        val,
        test,
    )


def _bfs_subset(edge_index: Tensor, y: Tensor, max_nodes: int, seed: int) -> Tensor:
    n = int(y.numel())
    if max_nodes <= 0 or max_nodes >= n:
        return torch.arange(n)
    adj = adjacency_from_edge_index(edge_index, n).numpy()
    count, component = connected_components(csr_matrix(adj), directed=False)
    lcc = np.arange(n) if count == 1 else np.flatnonzero(component == np.bincount(component).argmax())
    lcc_set = set(int(v) for v in lcc)
    neighbors = [np.flatnonzero(adj[i]).astype(int).tolist() for i in range(n)]
    rng = np.random.default_rng(seed)
    for row in neighbors:
        rng.shuffle(row)
    labels = y.cpu().numpy()
    seeds: list[int] = []
    for cls in np.unique(labels[lcc]):
        candidates = [int(i) for i in lcc if labels[i] == cls]
        seeds.append(max(candidates, key=lambda i: len(neighbors[i])))
    queue = list(dict.fromkeys(seeds))
    seen = set(queue)
    selected: list[int] = []
    cursor = 0
    while cursor < len(queue) and len(selected) < max_nodes:
        node = queue[cursor]
        cursor += 1
        selected.append(node)
        for other in neighbors[node]:
            if other in lcc_set and other not in seen:
                seen.add(other)
                queue.append(other)
    if len(selected) < max_nodes:
        remainder = [int(i) for i in lcc if int(i) not in seen]
        remainder.sort(key=lambda i: len(neighbors[i]), reverse=True)
        selected.extend(remainder[: max_nodes - len(selected)])
    return torch.tensor(sorted(selected[:max_nodes]), dtype=torch.long)


def _induced(name: str, x: Tensor, y: Tensor, edge_index: Tensor, max_nodes: int, seed: int) -> GraphData:
    selected = _bfs_subset(edge_index, y, max_nodes, seed)
    mapping = torch.full((x.size(0),), -1, dtype=torch.long)
    mapping[selected] = torch.arange(selected.numel())
    pairs = undirected_pairs(edge_index)
    keep = (mapping[pairs[0]] >= 0) & (mapping[pairs[1]] >= 0)
    local_pairs = mapping[pairs[:, keep]]
    local_x = x[selected].float()
    local_x = local_x / local_x.abs().sum(1, keepdim=True).clamp_min(1.0)
    local_y = y[selected].long()
    train, val, test = _split(local_y, seed + 23)
    return GraphData(name, local_x, local_y, pairs_to_edge_index(local_pairs), train, val, test)


def load_graph(name: str, root: Path, max_nodes: int, seed: int) -> GraphData:
    normalized = name.lower().strip()
    if normalized in {"sbm", "toy", "synthetic"}:
        return make_sbm(max_nodes, seed)
    if normalized not in {"cora", "citeseer"}:
        raise ValueError(f"unsupported dataset: {name}")
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import to_undirected

    canonical = "Cora" if normalized == "cora" else "CiteSeer"
    item = Planetoid(root=str(root / "Planetoid"), name=canonical)[0]
    edges = to_undirected(item.edge_index.cpu(), num_nodes=item.num_nodes)
    return _induced(canonical, item.x.cpu(), item.y.cpu(), edges, max_nodes, seed)

