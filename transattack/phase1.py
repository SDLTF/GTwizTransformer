from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .data import add_edge, load_graph, remove_edges, undirected_pairs
from .localize import (
    GaussianProfile,
    edge_features,
    feature_views,
    fit_view_profiles,
    positive_mask,
    ranking_metrics,
    top_pairs,
)
from .model import DenseGraphTransformer, evaluate_trace, true_margin
from .pe import build_pe


ABLATIONS = (
    "first_layer_full",
    "last_layer_full",
    "all_layer_full",
    "temporal_residual",
    "attention_trajectory",
    "value_trajectory",
    "hidden_logit_trajectory",
    "all_layer_no_attention",
)

RERANKERS = (
    "random",
    "anomaly_base",
    "cf_pred_label_free",
    "cf_anomaly_global",
    "cf_anomaly_target",
    "hybrid_label_free",
    "oracle_true_margin",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--combined-metrics", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--candidate-multiplier", type=int, default=5)
    parser.add_argument("--attack-limit", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--strict-cuda", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _device(requested: str, strict: bool) -> torch.device:
    available = torch.cuda.is_available()
    if strict and not available:
        raise RuntimeError("strict CUDA requested but CUDA is unavailable")
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if available else "cpu")


def _safe(value: object):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_pairs(text: str) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for token in str(text).split(";"):
        if not token:
            continue
        left, right = token.split("-", maxsplit=1)
        pairs.append(tuple(sorted((int(left), int(right)))))
    return tuple(pairs)


def _format_pairs(pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> str:
    return ";".join(f"{u}-{v}" for u, v in pairs)


def _top_mean(scores: Tensor, pairs: Tensor, count: int, target: int | None = None) -> float:
    values = scores.detach().cpu().float()
    if target is not None:
        mask = (pairs[0] == target) | (pairs[1] == target)
        values = values[mask]
    if not values.numel():
        return 0.0
    k = min(max(1, int(count)), values.numel())
    return float(torch.topk(values, k=k).values.mean().item())


def _rank_promoted(base_scores: Tensor, candidate_indices: Tensor, candidate_values: Tensor) -> Tensor:
    """Rank all candidates first by reranker, then retain base order outside K."""
    base = base_scores.detach().cpu().float()
    base_order = torch.argsort(base)
    base_rank = torch.empty_like(base)
    base_rank[base_order] = torch.arange(base.numel(), dtype=torch.float32)
    base_rank /= max(1, base.numel() - 1)
    result = base_rank
    candidate_order = torch.argsort(candidate_values.detach().cpu().float())
    candidate_rank = torch.empty(candidate_indices.numel(), dtype=torch.float32)
    candidate_rank[candidate_order] = torch.arange(candidate_indices.numel(), dtype=torch.float32)
    candidate_rank /= max(1, candidate_indices.numel() - 1)
    result[candidate_indices] = 2.0 + candidate_rank
    return result


def _robust_z(values: Tensor) -> Tensor:
    x = values.detach().cpu().float()
    median = x.median()
    mad = (x - median).abs().median()
    if float(mad) > 1e-8:
        return (x - median) / (1.4826 * mad)
    std = x.std(unbiased=False)
    return (x - x.mean()) / std if float(std) > 1e-8 else torch.zeros_like(x)


@torch.no_grad()
def _trace_for_edges(model, graph, edge_index, pe_name: str, pe_dim: int, pe_scale: float, device: torch.device):
    pe = build_pe(pe_name, edge_index, graph.num_nodes, pe_dim)
    return evaluate_trace(model, graph, pe.values / pe_scale, device)


def _repair_outcome(model, graph, attacked_edges, selected, pe_name: str, pe_dim: int, pe_scale: float, target: int, clean_margin: float, attacked_margin: float, device: torch.device) -> dict[str, Any]:
    repaired_edges = remove_edges(attacked_edges, graph.num_nodes, selected)
    trace = _trace_for_edges(model, graph, repaired_edges, pe_name, pe_dim, pe_scale, device)
    label = int(graph.y[target])
    margin = true_margin(trace.logits[target], label)
    denominator = clean_margin - attacked_margin
    return {
        "repaired_margin": margin,
        "margin_recovery": (margin - attacked_margin) / denominator if denominator > 1e-12 else float("nan"),
        "repair_restored": int(trace.logits[target].argmax()) == label,
        "repaired_prediction": int(trace.logits[target].argmax()),
    }


def _selection_metrics(selected: list[tuple[int, int]], positives: tuple[tuple[int, int], ...]) -> dict[str, float]:
    selected_set = {tuple(sorted(pair)) for pair in selected}
    positive_set = {tuple(sorted(pair)) for pair in positives}
    hits = len(selected_set & positive_set)
    union = len(selected_set | positive_set)
    return {
        "selected_recall": hits / len(positive_set),
        "selected_iou": hits / union if union else 0.0,
        "selected_hits": float(hits),
    }


def _bootstrap(values: np.ndarray, repetitions: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if not n:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "clusters": 0}
    indices = rng.integers(0, n, size=(repetitions, n))
    estimates = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "clusters": int(n),
    }


def _paired_intervals(frame: pd.DataFrame, method_column: str, primary: str, baseline: str, metrics: list[str], repetitions: int, seed: int) -> dict[str, dict[str, float]]:
    identity = ["attack_id", "dataset", "pe", "seed", "target"]
    left = frame[frame[method_column] == primary][identity + metrics]
    right = frame[frame[method_column] == baseline][["attack_id"] + metrics]
    paired = left.merge(right, on="attack_id", suffixes=("_primary", "_baseline"))
    cluster_keys = ["dataset", "pe", "seed", "target"]
    output: dict[str, dict[str, float]] = {}
    for index, metric in enumerate(metrics):
        column = f"delta_{metric}"
        paired[column] = paired[f"{metric}_primary"].astype(float) - paired[f"{metric}_baseline"].astype(float)
        clustered = paired.groupby(cluster_keys)[column].mean().to_numpy(dtype=float)
        output[metric] = _bootstrap(clustered, repetitions, seed + index)
    return output


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            text = f"{value:.6f}" if isinstance(value, float) else str(value)
            rendered.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _load_model(graph, config: dict[str, Any], checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = DenseGraphTransformer(
        graph.num_features,
        graph.num_classes,
        int(config["pe_dim"]),
        int(config["hidden_dim"]),
        int(config["heads"]),
        int(config["layers"]),
        float(config["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, float(payload["pe_scale"])


def main() -> int:
    args = _args()
    if args.candidate_multiplier < 1:
        raise ValueError("candidate multiplier must be positive")
    device = _device(args.device, args.strict_cuda)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output_dir or Path("results") / f"phase1_{timestamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_map = {run.resolve().name: run.resolve() for run in args.phase0_runs}
    run_configs = {
        name: json.loads((path / "config.json").read_text(encoding="utf-8"))
        for name, path in run_map.items()
    }
    attacks = pd.read_csv(args.combined_metrics.resolve()).drop_duplicates("attack_id")
    attacks = attacks[attacks.attack_success].sort_values(["dataset", "pe", "seed", "target", "budget"])
    if args.attack_limit > 0:
        attacks = attacks.head(args.attack_limit)
    config_out = vars(args).copy()
    config_out.update(
        device_resolved=str(device),
        output_dir=output,
        phase0_successful_snapshots=int(len(attacks)),
        primary_dynamics_comparison="all_layer_full_vs_first_layer_full",
        primary_causal_method="hybrid_label_free",
        candidate_k=f"{args.candidate_multiplier}B",
        deployable_forbidden_inputs=["paired_clean_graph", "true_attack_edges", "true_label"],
        oracle_only_input="true_label",
    )
    _write_json(output / "config.json", config_out)
    _write_json(output / "environment.json", {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    print(f"[phase1] device={device} output={output} attacks={len(attacks)}", flush=True)
    ablation_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    model_keys = ["source_run", "dataset", "pe", "seed"]
    for key, group in attacks.groupby(model_keys, sort=True):
        source_run, dataset, pe_name, seed = key
        source_run = str(source_run)
        dataset = str(dataset)
        pe_name = str(pe_name)
        seed = int(seed)
        if source_run not in run_map:
            raise KeyError(f"missing Phase-0 run directory for {source_run}")
        config = run_configs[source_run]
        graph = load_graph(dataset, args.data_root.resolve(), int(config["nodes"]), seed)
        checkpoint = run_map[source_run] / "checkpoints" / f"{dataset.lower()}_{pe_name}_seed{seed}.pt"
        model, pe_scale = _load_model(graph, config, checkpoint, device)
        clean_trace = _trace_for_edges(model, graph, graph.edge_index, pe_name, int(config["pe_dim"]), pe_scale, device)
        clean_pairs = undirected_pairs(graph.edge_index)
        clean_features = edge_features(clean_trace, clean_pairs)
        view_profiles = fit_view_profiles(clean_features)
        clean_views = feature_views(clean_features)
        print(f"[phase1] model={dataset}/{pe_name}/seed{seed} attacks={len(group)}", flush=True)

        for _, attack in group.iterrows():
            target = int(attack.target)
            budget = int(attack.budget)
            label = int(graph.y[target])
            positives = _parse_pairs(attack.true_added_edges)
            attacked_edges = graph.edge_index
            for u, v in positives:
                attacked_edges = add_edge(attacked_edges, graph.num_nodes, u, v)
            attacked_trace = _trace_for_edges(model, graph, attacked_edges, pe_name, int(config["pe_dim"]), pe_scale, device)
            attacked_margin = true_margin(attacked_trace.logits[target], label)
            clean_margin = true_margin(clean_trace.logits[target], label)
            if int(attacked_trace.logits[target].argmax()) == label:
                raise AssertionError(f"frozen successful attack no longer succeeds: {attack.attack_id}")
            if abs(attacked_margin - float(attack.attacked_margin)) > 2e-4:
                raise AssertionError(f"frozen margin mismatch for {attack.attack_id}")
            attacked_pairs = undirected_pairs(attacked_edges)
            attacked_features = edge_features(attacked_trace, attacked_pairs)
            attacked_views = feature_views(attacked_features)
            repair_cache: dict[str, dict[str, Any]] = {}

            def repair(selected: list[tuple[int, int]]) -> dict[str, Any]:
                cache_key = _format_pairs(sorted(tuple(sorted(pair)) for pair in selected))
                if cache_key not in repair_cache:
                    repair_cache[cache_key] = _repair_outcome(
                        model, graph, attacked_edges, selected, pe_name,
                        int(config["pe_dim"]), pe_scale, target,
                        clean_margin, attacked_margin, device,
                    )
                return repair_cache[cache_key]

            identity = {
                "attack_id": str(attack.attack_id),
                "dataset": dataset,
                "pe": pe_name,
                "seed": seed,
                "target": target,
                "budget": budget,
                "cluster": f"{dataset}|{pe_name}|{seed}|{target}",
                "true_added_edges": _format_pairs(positives),
                "clean_margin": clean_margin,
                "attacked_margin": attacked_margin,
            }
            for view_name in ABLATIONS:
                scores = view_profiles[view_name].score(attacked_views[view_name])
                selected = top_pairs(attacked_pairs, scores, budget)
                ablation_rows.append({
                    **identity,
                    "view": view_name,
                    "selected_edges": _format_pairs(selected),
                    **ranking_metrics(attacked_pairs, scores, positives, budget),
                    **_selection_metrics(selected, positives),
                    **repair(selected),
                })

            base_scores = view_profiles["all_layer_full"].score(attacked_views["all_layer_full"])
            candidate_count = min(attacked_pairs.size(1), args.candidate_multiplier * budget)
            candidate_indices = torch.argsort(base_scores, descending=True)[:candidate_count]
            candidate_pairs = [tuple(map(int, attacked_pairs[:, index].tolist())) for index in candidate_indices]
            candidate_mask = positive_mask(attacked_pairs[:, candidate_indices], positives)
            candidate_recall = float(candidate_mask.sum().item() / len(positives))
            baseline_global = _top_mean(base_scores, attacked_pairs, budget)
            baseline_target = _top_mean(base_scores, attacked_pairs, budget, target)
            attacked_prediction = int(attacked_trace.logits[target].argmax())
            attacked_probability = float(torch.softmax(attacked_trace.logits[target], dim=0)[attacked_prediction])
            candidate_base: list[float] = []
            pred_drop: list[float] = []
            anomaly_global_drop: list[float] = []
            anomaly_target_drop: list[float] = []
            oracle_margin_gain: list[float] = []
            for rank, (edge_index_position, pair) in enumerate(zip(candidate_indices.tolist(), candidate_pairs), start=1):
                counterfactual_edges = remove_edges(attacked_edges, graph.num_nodes, [pair])
                counterfactual_trace = _trace_for_edges(
                    model, graph, counterfactual_edges, pe_name,
                    int(config["pe_dim"]), pe_scale, device,
                )
                counterfactual_pairs = undirected_pairs(counterfactual_edges)
                counterfactual_features = edge_features(counterfactual_trace, counterfactual_pairs)
                counterfactual_full = feature_views(counterfactual_features)["all_layer_full"]
                counterfactual_scores = view_profiles["all_layer_full"].score(counterfactual_full)
                probability = float(torch.softmax(counterfactual_trace.logits[target], dim=0)[attacked_prediction])
                cf_margin = true_margin(counterfactual_trace.logits[target], label)
                base_value = float(base_scores[edge_index_position])
                pred_value = attacked_probability - probability
                global_value = baseline_global - _top_mean(counterfactual_scores, counterfactual_pairs, budget)
                target_value = baseline_target - _top_mean(counterfactual_scores, counterfactual_pairs, budget, target)
                oracle_value = cf_margin - attacked_margin
                candidate_base.append(base_value)
                pred_drop.append(pred_value)
                anomaly_global_drop.append(global_value)
                anomaly_target_drop.append(target_value)
                oracle_margin_gain.append(oracle_value)
                candidate_rows.append({
                    **identity,
                    "candidate_rank": rank,
                    "candidate_edge": _format_pairs([pair]),
                    "is_true_attack_edge": tuple(sorted(pair)) in set(positives),
                    "base_anomaly": base_value,
                    "predicted_class_probability_drop": pred_value,
                    "global_anomaly_drop": global_value,
                    "target_anomaly_drop": target_value,
                    "oracle_true_margin_gain": oracle_value,
                })

            candidate_base_tensor = torch.tensor(candidate_base)
            pred_tensor = torch.tensor(pred_drop)
            global_tensor = torch.tensor(anomaly_global_drop)
            target_tensor = torch.tensor(anomaly_target_drop)
            oracle_tensor = torch.tensor(oracle_margin_gain)
            generator = torch.Generator().manual_seed(seed * 1000003 + target * 101 + budget + 17)
            method_scores = {
                "random": torch.rand(attacked_pairs.size(1), generator=generator),
                "anomaly_base": base_scores,
                "cf_pred_label_free": _rank_promoted(base_scores, candidate_indices, pred_tensor),
                "cf_anomaly_global": _rank_promoted(base_scores, candidate_indices, global_tensor),
                "cf_anomaly_target": _rank_promoted(base_scores, candidate_indices, target_tensor),
                "hybrid_label_free": _rank_promoted(
                    base_scores, candidate_indices,
                    _robust_z(candidate_base_tensor) + _robust_z(pred_tensor),
                ),
                "oracle_true_margin": _rank_promoted(base_scores, candidate_indices, oracle_tensor),
            }
            for method in RERANKERS:
                scores = method_scores[method]
                selected = top_pairs(attacked_pairs, scores, budget)
                causal_rows.append({
                    **identity,
                    "reranker": method,
                    "deployable_without_true_label": method != "oracle_true_margin",
                    "candidate_count": candidate_count,
                    "candidate_recall_at_5b": candidate_recall,
                    "selected_edges": _format_pairs(selected),
                    **ranking_metrics(attacked_pairs, scores, positives, budget),
                    **_selection_metrics(selected, positives),
                    **repair(selected),
                })
            pd.DataFrame(ablation_rows).to_csv(output / "ablation_metrics.csv", index=False)
            pd.DataFrame(causal_rows).to_csv(output / "causal_metrics.csv", index=False)
            pd.DataFrame(candidate_rows).to_csv(output / "candidate_scores.csv", index=False)
            print(
                f"[phase1] {attack.attack_id} K={candidate_count} coverage={candidate_recall:.3f} "
                f"base_R={causal_rows[-6]['selected_recall']:.3f} hybrid_R={causal_rows[-2]['selected_recall']:.3f}",
                flush=True,
            )

    ablations = pd.DataFrame(ablation_rows)
    causals = pd.DataFrame(causal_rows)
    ablation_summary = ablations.groupby("view", as_index=False).agg(
        snapshots=("attack_id", "count"),
        auprc=("edge_auprc", "mean"),
        recall_at_b=("recall_at_b", "mean"),
        repair_rate=("repair_restored", "mean"),
        margin_recovery=("margin_recovery", "mean"),
    )
    causal_summary = causals.groupby("reranker", as_index=False).agg(
        snapshots=("attack_id", "count"),
        candidate_recall=("candidate_recall_at_5b", "mean"),
        auprc=("edge_auprc", "mean"),
        recall_at_b=("recall_at_b", "mean"),
        repair_rate=("repair_restored", "mean"),
        margin_recovery=("margin_recovery", "mean"),
    )
    dynamics = _paired_intervals(
        ablations, "view", "all_layer_full", "first_layer_full",
        ["edge_auprc", "recall_at_b", "margin_recovery", "repair_restored"],
        args.bootstrap_repetitions, 5101,
    )
    causal_primary = _paired_intervals(
        causals, "reranker", "hybrid_label_free", "anomaly_base",
        ["edge_auprc", "recall_at_b", "margin_recovery", "repair_restored"],
        args.bootstrap_repetitions, 6101,
    )
    temporal_secondary = _paired_intervals(
        ablations, "view", "temporal_residual", "first_layer_full",
        ["edge_auprc", "recall_at_b", "margin_recovery", "repair_restored"],
        args.bootstrap_repetitions, 8101,
    )
    causal_pred_secondary = _paired_intervals(
        causals, "reranker", "cf_pred_label_free", "anomaly_base",
        ["edge_auprc", "recall_at_b", "margin_recovery", "repair_restored"],
        args.bootstrap_repetitions, 9101,
    )
    causal_anomaly_secondary = _paired_intervals(
        causals, "reranker", "cf_anomaly_target", "anomaly_base",
        ["edge_auprc", "recall_at_b", "margin_recovery", "repair_restored"],
        args.bootstrap_repetitions, 10101,
    )
    oracle = _paired_intervals(
        causals, "reranker", "oracle_true_margin", "anomaly_base",
        ["edge_auprc", "recall_at_b", "margin_recovery", "repair_restored"],
        args.bootstrap_repetitions, 7101,
    )
    if args.attack_limit > 0:
        status = "smoke_only"
    else:
        depth_supported = dynamics["edge_auprc"]["ci95_low"] > 0 and dynamics["recall_at_b"]["ci95_low"] > 0
        causal_supported = causal_primary["margin_recovery"]["ci95_low"] > 0 and causal_primary["repair_restored"]["ci95_low"] > 0
        status = (
            "depth_and_causal_supported" if depth_supported and causal_supported
            else "depth_supported_causal_inconclusive" if depth_supported
            else "causal_supported_depth_inconclusive" if causal_supported
            else "both_inconclusive"
        )
    decision = {
        "status": status,
        "successful_attack_snapshots": int(ablations.attack_id.nunique()),
        "model_target_clusters": int(ablations.cluster.nunique()),
        "mean_top5b_candidate_recall": float(causals.candidate_recall_at_5b.mean()),
        "all_layer_full_minus_first_layer_full": dynamics,
        "hybrid_label_free_minus_anomaly_base": causal_primary,
        "secondary_temporal_residual_minus_first_layer_full": temporal_secondary,
        "secondary_cf_pred_label_free_minus_anomaly_base": causal_pred_secondary,
        "secondary_cf_anomaly_target_minus_anomaly_base": causal_anomaly_secondary,
        "oracle_true_margin_minus_anomaly_base": oracle,
    }
    _write_json(output / "decision.json", decision)
    summary = "\n".join([
        "# GraphTransAttack Phase-1 summary",
        "",
        f"Decision: **{status}**.",
        "",
        "All comparisons reuse the same Phase-0 checkpoints and successful attack edges. "
        "True labels enter only evaluation and the explicitly marked oracle.",
        "",
        "## Layer/channel ablation",
        "",
        _table(ablation_summary),
        "",
        "## Top-5B causal reranking",
        "",
        _table(causal_summary),
        "",
        "## Cluster-bootstrap paired intervals",
        "",
        "```json",
        json.dumps(_safe({
            "dynamics_primary": dynamics,
            "causal_primary": causal_primary,
            "temporal_secondary": temporal_secondary,
            "cf_pred_secondary": causal_pred_secondary,
            "cf_anomaly_secondary": causal_anomaly_secondary,
            "oracle": oracle,
        }), ensure_ascii=False, indent=2),
        "```",
        "",
    ])
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(f"[phase1] complete status={status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
