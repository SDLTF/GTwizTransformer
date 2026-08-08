from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .attack import select_attack_targets
from .data import GraphData, adjacency_from_edge_index, edge_index_from_adjacency, load_graph, undirected_pairs
from .gps_attack import ATTACK_OBJECTIVES, ATTACK_TYPES, adaptive_gps_attack
from .gps_model import BatchTelemetry, GPSTrainResult, GraphGPSNodeClassifier, evaluate_gps_trace, logits_for_adjacencies, train_gps_model
from .localize import edge_features, feature_views, fit_view_profiles, positive_mask, ranking_metrics, top_pairs
from .model import true_margin
from .phase1 import _bootstrap, _format_pairs, _rank_promoted, _robust_z, _safe, _selection_metrics, _table, _write_json
from .phase2 import _incident_first_order, _ordered, _round_robin


BASE_METHODS = ("random", "attention_trajectory", "anomaly_base", "temporal_anomaly")
CANDIDATE_GENERATORS = ("all_layer", "temporal_residual", "global_union", "target_incident")
RERANKERS = ("cf_pred_label_free", "hybrid_label_free", "oracle_true_margin")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[4510, 4511, 4512])
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[4510, 4511, 4512])
    parser.add_argument("--attack-types", nargs="+", choices=ATTACK_TYPES, default=list(ATTACK_TYPES))
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--nodes", type=int, default=192)
    parser.add_argument("--targets", type=int, default=6)
    parser.add_argument("--candidate-pool", type=int, default=64)
    parser.add_argument("--candidate-multiplier", type=int, default=10)
    parser.add_argument("--graph-batch-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--pe-channels", type=int, default=16)
    parser.add_argument("--walk-length", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--attack-objective", choices=ATTACK_OBJECTIVES, default="classification_only")
    parser.add_argument("--adaptive-stealth-strength", type=float, default=1.0)
    parser.add_argument("--classification-retention-ratio", type=float, default=0.85)
    parser.add_argument("--minimum-remote-clusters", type=int, default=12)
    parser.add_argument("--minimum-remote-seeds", type=int, default=2)
    parser.add_argument("--minimum-remote-datasets", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--strict-cuda", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--reference-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
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


def _pairs_text(pairs: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> str:
    return _format_pairs(list(pairs))


def _candidate_rankings(
    pairs: Tensor,
    all_scores: Tensor,
    temporal_scores: Tensor,
    target: int,
) -> dict[str, list[int]]:
    all_order = _ordered(all_scores)
    temporal_order = _ordered(temporal_scores)
    combined = _robust_z(all_scores) + _robust_z(temporal_scores)
    incident_order = _incident_first_order(pairs, combined, target)
    return {
        "all_layer": all_order,
        "temporal_residual": temporal_order,
        "global_union": _round_robin((all_order, temporal_order), pairs.size(1)),
        "target_incident": incident_order,
    }


def _remove_pair_variants(adjacency: Tensor, positions: list[int], pairs: Tensor) -> Tensor:
    variants = adjacency.unsqueeze(0).repeat(len(positions), 1, 1)
    for row, position in enumerate(positions):
        u, v = (int(value) for value in pairs[:, position].tolist())
        variants[row, u, v] = 0.0
        variants[row, v, u] = 0.0
    return variants


def _remove_selections(adjacency: Tensor, selections: list[list[tuple[int, int]]]) -> Tensor:
    variants = adjacency.unsqueeze(0).repeat(len(selections), 1, 1)
    for row, selected in enumerate(selections):
        for u, v in selected:
            variants[row, u, v] = 0.0
            variants[row, v, u] = 0.0
    return variants


def _paired_intervals(
    frame: pd.DataFrame,
    attack_type: str,
    primary: str,
    baseline: str,
    metrics: list[str],
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    identity = ["attack_id", "dataset", "seed", "attack_type", "target"]
    left = frame[(frame.attack_type == attack_type) & (frame.method == primary)][identity + metrics]
    right = frame[(frame.attack_type == attack_type) & (frame.method == baseline)][["attack_id"] + metrics]
    paired = left.merge(right, on="attack_id", suffixes=("_primary", "_baseline"))
    if paired.attack_id.duplicated().any():
        raise AssertionError("method filters do not identify one row per attack")
    cluster_keys = ["dataset", "seed", "attack_type", "target"]
    output: dict[str, dict[str, float]] = {}
    for offset, metric in enumerate(metrics):
        name = f"delta_{metric}"
        paired[name] = paired[f"{metric}_primary"].astype(float) - paired[f"{metric}_baseline"].astype(float)
        values = paired.groupby(cluster_keys)[name].mean().to_numpy(dtype=float)
        output[metric] = _bootstrap(values, repetitions, seed + offset)
    return output


def _checkpoint_payload(model: GraphGPSNodeClassifier, graph: GraphData, seed: int, trained, config: dict[str, Any]):
    return {
        "state_dict": model.state_dict(),
        "dataset": graph.name,
        "seed": seed,
        "best_epoch": trained.best_epoch,
        "model_config": {
            key: config[key]
            for key in ("channels", "pe_channels", "walk_length", "layers", "heads", "dropout")
        },
        "architecture": "torch_geometric.nn.GPSConv compatible trace subclass with GIN local branch",
    }


def _write_frames(output: Path, attacks: list[dict[str, Any]], coverage: list[dict[str, Any]], metrics: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    if attacks:
        pd.DataFrame(attacks).to_csv(output / "attack_metrics.csv", index=False)
    if coverage:
        pd.DataFrame(coverage).to_csv(output / "candidate_coverage.csv", index=False)
    if metrics:
        pd.DataFrame(metrics).to_csv(output / "localization_metrics.csv", index=False)
    if candidates:
        pd.DataFrame(candidates).to_csv(output / "counterfactual_scores.csv", index=False)


def main() -> int:
    args = _args()
    if any(value < 1 for value in args.budgets):
        raise ValueError("budgets must be positive")
    if args.channels % args.heads:
        raise ValueError("channels must be divisible by heads")
    if min(args.minimum_remote_clusters, args.minimum_remote_seeds, args.minimum_remote_datasets) < 1:
        raise ValueError("remote viability thresholds must be positive")
    if args.adaptive_stealth_strength < 0:
        raise ValueError("adaptive stealth strength must be nonnegative")
    if not 0 < args.classification_retention_ratio <= 1:
        raise ValueError("classification retention ratio must lie in (0, 1]")
    if not args.smoke and sorted(set(args.seeds)) != sorted(set(args.expected_seeds)):
        raise ValueError("canonical Phase-3 seeds do not match the frozen holdout seeds")
    device = _device(args.device, args.strict_cuda)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output_dir or Path("results") / f"phase3_{timestamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    telemetry = BatchTelemetry(args.graph_batch_size, args.graph_batch_size)
    config = vars(args).copy()
    config.update(
        output_dir=output,
        device_resolved=str(device),
        architecture="PyG GPSConv: RWSE + GIN local branch + exact global MHA",
        primary_remote_candidate_generator="global_union",
        primary_remote_reranker="hybrid_label_free",
        forbidden_localizer_inputs=["paired_clean_graph", "edge_diff", "true_attack_edges", "true_label"],
        oracle_only_input="true_label",
    )
    _write_json(output / "config.json", config)
    _write_json(output / "environment.json", {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": __import__("torch_geometric").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    print(f"[phase3] device={device} output={output}", flush=True)

    attack_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    reference_attacks: pd.DataFrame | None = None
    reference_dir: Path | None = None
    if args.reference_run_dir is not None:
        reference_dir = args.reference_run_dir.resolve()
        reference_attacks = pd.read_csv(reference_dir / "attack_metrics.csv")
        reference_config = json.loads((reference_dir / "config.json").read_text(encoding="utf-8"))
        for key in ("datasets", "seeds", "budgets", "nodes", "targets", "channels", "pe_channels", "walk_length", "layers", "heads", "dropout"):
            current = config[key]
            if reference_config[key] != current:
                raise ValueError(f"reference run configuration mismatch for {key}")
        print(f"[phase3] loading paired checkpoints and targets from {reference_dir}", flush=True)

    for dataset_name in args.datasets:
        for seed in args.seeds:
            model_started = time.perf_counter()
            graph = load_graph(dataset_name, args.data_root.resolve(), args.nodes, int(seed))
            if reference_dir is None:
                trained = train_gps_model(
                    graph,
                    device,
                    int(seed),
                    args.channels,
                    args.pe_channels,
                    args.walk_length,
                    args.layers,
                    args.heads,
                    args.dropout,
                    args.epochs,
                    args.patience,
                    args.learning_rate,
                    args.weight_decay,
                )
            else:
                checkpoint_path = reference_dir / "checkpoints" / f"{graph.name.lower()}_rwse_seed{seed}.pt"
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                model = GraphGPSNodeClassifier(
                    graph.num_features,
                    graph.num_classes,
                    args.channels,
                    args.pe_channels,
                    args.walk_length,
                    args.layers,
                    args.heads,
                    args.dropout,
                ).to(device)
                model.load_state_dict(checkpoint["state_dict"])
                assert reference_attacks is not None
                reference_slice = reference_attacks[
                    (reference_attacks.dataset.str.lower() == graph.name.lower())
                    & (reference_attacks.seed == int(seed))
                ]
                if reference_slice.empty:
                    raise ValueError(f"reference attacks missing for {graph.name}/seed{seed}")
                trained = GPSTrainResult(
                    model=model,
                    best_epoch=int(reference_slice.best_epoch.iloc[0]),
                    best_val_accuracy=float(reference_slice.best_val_accuracy.iloc[0]),
                    train_accuracy=float("nan"),
                    test_accuracy=float(reference_slice.clean_test_accuracy.iloc[0]),
                )
            model = trained.model
            clean_trace = evaluate_gps_trace(model, graph, graph.edge_index, device)
            if reference_attacks is None:
                targets = select_attack_targets(clean_trace, graph, args.targets)
            else:
                reference_slice = reference_attacks[
                    (reference_attacks.dataset.str.lower() == graph.name.lower())
                    & (reference_attacks.seed == int(seed))
                ]
                targets = reference_slice.drop_duplicates("target").target.astype(int).tolist()
                if not targets or len(targets) > args.targets:
                    raise ValueError(f"invalid reference target count for {graph.name}/seed{seed}")
                reference_margins = reference_slice.groupby("target").clean_margin.first().to_dict()
                margin_differences: list[float] = []
                for target in targets:
                    label = int(graph.y[target])
                    margin = true_margin(clean_trace.logits[target], label)
                    difference = abs(margin - float(reference_margins[target]))
                    margin_differences.append(difference)
                    if difference > 3e-4:
                        raise ValueError(
                            f"reference clean margin mismatch for {graph.name}/seed{seed}/target{target}: "
                            f"difference={difference:.9g}"
                        )
                print(
                    f"[phase3] reference_margin_max_abs_diff={max(margin_differences):.3g} "
                    f"for {graph.name}/seed{seed}",
                    flush=True,
                )
            clean_pairs = undirected_pairs(graph.edge_index)
            clean_features = edge_features(clean_trace, clean_pairs)
            profiles = fit_view_profiles(clean_features)
            torch.save(
                _checkpoint_payload(model, graph, int(seed), trained, config),
                checkpoints / f"{graph.name.lower()}_rwse_seed{seed}.pt",
            )
            print(
                f"[phase3] model={graph.name}/seed{seed} nodes={graph.num_nodes} "
                f"test={trained.test_accuracy:.4f} targets={targets}",
                flush=True,
            )
            for attack_type in args.attack_types:
                for target in targets:
                    snapshots = adaptive_gps_attack(
                        model,
                        graph,
                        clean_trace,
                        target,
                        sorted(set(args.budgets)),
                        args.candidate_pool,
                        attack_type,
                        device,
                        args.graph_batch_size,
                        telemetry,
                        args.attack_objective,
                        profiles,
                        args.adaptive_stealth_strength,
                        args.classification_retention_ratio,
                    )
                    label = int(graph.y[target])
                    clean_margin = true_margin(clean_trace.logits[target], label)
                    for budget, snapshot in sorted(snapshots.items()):
                        attack_id = f"{graph.name}-gps-rwse-{seed}-{attack_type}-t{target}-b{budget}"
                        attack_identity = {
                            "attack_id": attack_id,
                            "dataset": graph.name,
                            "seed": int(seed),
                            "attack_type": attack_type,
                            "target": int(target),
                            "budget": int(budget),
                            "cluster": f"{graph.name}|{seed}|{attack_type}|{target}",
                            "true_added_edges": _pairs_text(snapshot.added_edges),
                            "clean_margin": clean_margin,
                            "attacked_margin": snapshot.margin,
                            "attack_objective": args.attack_objective,
                            "adaptive_stealth_strength": args.adaptive_stealth_strength,
                            "classification_retention_ratio": args.classification_retention_ratio,
                        }
                        attack_rows.append({
                            **attack_identity,
                            "attack_success": snapshot.success,
                            "target_loss": snapshot.target_loss,
                            "attacked_prediction": snapshot.prediction,
                            "clean_test_accuracy": trained.test_accuracy,
                            "best_val_accuracy": trained.best_val_accuracy,
                            "best_epoch": trained.best_epoch,
                            "selected_gain_ratio": snapshot.selected_gain_ratio,
                            "minimum_selected_gain_ratio": snapshot.minimum_selected_gain_ratio,
                            "eligible_candidates": snapshot.eligible_candidates,
                            "mean_eligible_candidates": snapshot.mean_eligible_candidates,
                        })
                        if not snapshot.success:
                            print(f"[phase3] {attack_id} success=False", flush=True)
                            _write_frames(output, attack_rows, coverage_rows, metric_rows, candidate_rows)
                            continue

                        attacked_trace = evaluate_gps_trace(model, graph, snapshot.edge_index, device)
                        attacked_prediction = int(attacked_trace.logits[target].argmax())
                        attacked_margin = true_margin(attacked_trace.logits[target], label)
                        if attacked_prediction != snapshot.prediction or abs(attacked_margin - snapshot.margin) > 3e-4:
                            raise AssertionError(f"attack reconstruction mismatch: {attack_id}")
                        pairs = undirected_pairs(snapshot.edge_index)
                        views = feature_views(edge_features(attacked_trace, pairs))
                        all_scores = profiles["all_layer_full"].score(views["all_layer_full"])
                        temporal_scores = profiles["temporal_residual"].score(views["temporal_residual"])
                        attention_scores = profiles["attention_trajectory"].score(views["attention_trajectory"])
                        rankings = _candidate_rankings(pairs, all_scores, temporal_scores, target)
                        candidate_count = min(pairs.size(1), args.candidate_multiplier * budget)
                        required = sorted({position for ranking in rankings.values() for position in ranking[:candidate_count]})
                        counterfactual_adjacencies = _remove_pair_variants(snapshot.adjacency, required, pairs)
                        counterfactual_logits = logits_for_adjacencies(
                            model,
                            graph,
                            counterfactual_adjacencies,
                            device,
                            args.graph_batch_size,
                            telemetry,
                        )
                        attacked_probability = float(torch.softmax(attacked_trace.logits[target], dim=-1)[attacked_prediction])
                        cf_by_position: dict[int, tuple[float, float]] = {}
                        positive_set = {tuple(sorted(pair)) for pair in snapshot.added_edges}
                        for row, position in enumerate(required):
                            logits = counterfactual_logits[row, target]
                            probability = float(torch.softmax(logits, dim=-1)[attacked_prediction])
                            margin = true_margin(logits, label)
                            cf_by_position[position] = (attacked_probability - probability, margin - attacked_margin)
                            pair = tuple(int(value) for value in pairs[:, position].tolist())
                            candidate_rows.append({
                                **attack_identity,
                                "edge_position": position,
                                "candidate_edge": _pairs_text([pair]),
                                "is_true_attack_edge": tuple(sorted(pair)) in positive_set,
                                "all_layer_anomaly": float(all_scores[position]),
                                "temporal_anomaly": float(temporal_scores[position]),
                                "predicted_class_probability_drop": cf_by_position[position][0],
                                "oracle_true_margin_gain": cf_by_position[position][1],
                            })

                        generator = torch.Generator().manual_seed(
                            int(seed) * 1000003 + target * 101 + budget * 7 + (0 if attack_type == "incident" else 1)
                        )
                        method_scores: dict[str, Tensor] = {
                            "random": torch.rand(pairs.size(1), generator=generator),
                            "attention_trajectory": attention_scores,
                            "anomaly_base": all_scores,
                            "temporal_anomaly": temporal_scores,
                        }
                        for candidate_generator in CANDIDATE_GENERATORS:
                            indices = torch.tensor(rankings[candidate_generator][:candidate_count], dtype=torch.long)
                            recall = float(positive_mask(pairs[:, indices], snapshot.added_edges).sum().item() / len(snapshot.added_edges))
                            coverage_rows.append({
                                **attack_identity,
                                "generator": candidate_generator,
                                "candidate_count": candidate_count,
                                "candidate_recall": recall,
                            })
                            pred_values = torch.tensor([cf_by_position[int(index)][0] for index in indices.tolist()])
                            oracle_values = torch.tensor([cf_by_position[int(index)][1] for index in indices.tolist()])
                            base_values = all_scores[indices]
                            reranked = {
                                "cf_pred_label_free": pred_values,
                                "hybrid_label_free": _robust_z(base_values) + _robust_z(pred_values),
                                "oracle_true_margin": oracle_values,
                            }
                            for reranker in RERANKERS:
                                method = f"{candidate_generator}+{reranker}"
                                method_scores[method] = _rank_promoted(all_scores, indices, reranked[reranker])

                        selected_by_method = {
                            method: top_pairs(pairs, scores, budget)
                            for method, scores in method_scores.items()
                        }
                        unique_keys: list[str] = []
                        unique_selections: list[list[tuple[int, int]]] = []
                        method_to_key: dict[str, str] = {}
                        for method, selected in selected_by_method.items():
                            key = _pairs_text(sorted(tuple(sorted(pair)) for pair in selected))
                            method_to_key[method] = key
                            if key not in unique_keys:
                                unique_keys.append(key)
                                unique_selections.append(selected)
                        repair_logits = logits_for_adjacencies(
                            model,
                            graph,
                            _remove_selections(snapshot.adjacency, unique_selections),
                            device,
                            args.graph_batch_size,
                            telemetry,
                        )
                        repair_by_key: dict[str, dict[str, Any]] = {}
                        denominator = clean_margin - attacked_margin
                        for key, logits in zip(unique_keys, repair_logits[:, target, :]):
                            repaired_margin = true_margin(logits, label)
                            repaired_prediction = int(logits.argmax())
                            repair_by_key[key] = {
                                "repaired_margin": repaired_margin,
                                "repaired_margin_gain": repaired_margin - attacked_margin,
                                "margin_recovery": (repaired_margin - attacked_margin) / denominator if denominator > 1e-12 else float("nan"),
                                "repair_restored": repaired_prediction == label,
                                "repaired_prediction": repaired_prediction,
                            }
                        for method, scores in method_scores.items():
                            selected = selected_by_method[method]
                            if "+" in method:
                                candidate_generator, reranker = method.split("+", maxsplit=1)
                                deployable = reranker != "oracle_true_margin"
                            else:
                                candidate_generator, reranker, deployable = "none", method, True
                            metric_rows.append({
                                **attack_identity,
                                "method": method,
                                "generator": candidate_generator,
                                "reranker": reranker,
                                "candidate_count": candidate_count if candidate_generator != "none" else 0,
                                "deployable_without_true_label": deployable,
                                "selected_edges": _pairs_text(selected),
                                **ranking_metrics(pairs, scores, snapshot.added_edges, budget),
                                **_selection_metrics(selected, snapshot.added_edges),
                                **repair_by_key[method_to_key[method]],
                            })
                        print(
                            f"[phase3] {attack_id} success=True remote={attack_type == 'remote'} "
                            f"global_cov={coverage_rows[-2]['candidate_recall']:.3f}",
                            flush=True,
                        )
                        _write_frames(output, attack_rows, coverage_rows, metric_rows, candidate_rows)
            print(f"[phase3] model_complete={graph.name}/seed{seed} seconds={time.perf_counter()-model_started:.1f}", flush=True)

    attacks = pd.DataFrame(attack_rows)
    successful_metrics = pd.DataFrame(metric_rows)
    coverage = pd.DataFrame(coverage_rows)
    attack_summary = attacks.groupby(["dataset", "seed", "attack_type", "budget"], as_index=False).agg(
        attempts=("attack_id", "count"), success_rate=("attack_success", "mean")
    )
    if successful_metrics.empty:
        method_summary = pd.DataFrame()
    else:
        method_summary = successful_metrics.groupby(["attack_type", "method"], as_index=False).agg(
            snapshots=("attack_id", "count"),
            auprc=("edge_auprc", "mean"),
            recall_at_b=("recall_at_b", "mean"),
            repair_rate=("repair_restored", "mean"),
            repaired_margin_gain=("repaired_margin_gain", "mean"),
            margin_recovery=("margin_recovery", "mean"),
        )

    remote_success = attacks[(attacks.attack_type == "remote") & attacks.attack_success]
    remote_clusters = int(remote_success.cluster.nunique())
    remote_seeds = int(remote_success.seed.nunique())
    remote_datasets = int(remote_success.dataset.nunique())
    viability = (
        remote_clusters >= args.minimum_remote_clusters
        and remote_seeds >= args.minimum_remote_seeds
        and remote_datasets >= args.minimum_remote_datasets
    )
    if successful_metrics.empty or remote_success.empty:
        fingerprint = {metric: {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "clusters": 0} for metric in ("edge_auprc", "recall_at_b")}
        causal = {metric: {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "clusters": 0} for metric in ("recall_at_b", "repaired_margin_gain", "margin_recovery", "repair_restored")}
    else:
        fingerprint = _paired_intervals(
            successful_metrics,
            "remote",
            "anomaly_base",
            "random",
            ["edge_auprc", "recall_at_b"],
            args.bootstrap_repetitions,
            21001,
        )
        causal = _paired_intervals(
            successful_metrics,
            "remote",
            "global_union+hybrid_label_free",
            "anomaly_base",
            ["recall_at_b", "repaired_margin_gain", "margin_recovery", "repair_restored"],
            args.bootstrap_repetitions,
            22001,
        )
    fingerprint_supported = viability and fingerprint["edge_auprc"]["ci95_low"] > 0 and fingerprint["recall_at_b"]["ci95_low"] > 0
    causal_supported = viability and causal["recall_at_b"]["ci95_low"] > 0 and causal["repaired_margin_gain"]["ci95_low"] > 0
    if args.smoke:
        status = "smoke_only"
    elif not viability:
        status = "remote_attack_underpowered"
    elif fingerprint_supported and causal_supported:
        status = "formal_graphgps_fingerprint_and_causal_localization_supported"
    elif fingerprint_supported:
        status = "formal_graphgps_fingerprint_supported_causal_inconclusive"
    elif causal_supported:
        status = "formal_graphgps_causal_supported_fingerprint_inconclusive"
    else:
        status = "formal_graphgps_primary_tests_inconclusive_or_negative"

    elapsed = time.perf_counter() - started
    gpu_stats = {
        "requested_graph_batch_size": telemetry.requested_batch_size,
        "minimum_resolved_graph_batch_size": telemetry.minimum_resolved_batch_size,
        "maximum_graphs_per_forward": telemetry.maximum_graphs_per_forward,
        "batched_graph_evaluations": telemetry.graphs,
        "batched_forward_calls": telemetry.forwards,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "elapsed_seconds": elapsed,
    }
    decision = {
        "status": status,
        "remote_successful_snapshots": int(remote_success.attack_id.nunique()),
        "remote_successful_clusters": remote_clusters,
        "remote_successful_seeds": remote_seeds,
        "remote_successful_datasets": remote_datasets,
        "remote_viability_requirements": {
            "minimum_clusters": args.minimum_remote_clusters,
            "minimum_seeds": args.minimum_remote_seeds,
            "minimum_datasets": args.minimum_remote_datasets,
        },
        "remote_viability_supported": viability,
        "primary_remote_anomaly_minus_random": fingerprint,
        "primary_remote_global_union_hybrid_minus_anomaly": causal,
        "gpu_telemetry": gpu_stats,
        "architecture": "PyG GPSConv compatible; RWSE + GIN + exact global MHA",
    }
    _write_json(output / "decision.json", decision)
    _write_json(output / "gpu_telemetry.json", gpu_stats)
    summary = "\n".join([
        "# GraphTransAttack Phase-3 summary",
        "",
        f"Decision: **{status}**.",
        "",
        "Phase 3 uses a PyG GPSConv-compatible GraphGPS model with RWSE, a GIN local branch, and exact global multi-head attention. Incident and remote attacks are never pooled into a primary claim.",
        "",
        "## Attack success",
        "",
        _table(attack_summary),
        "",
        "## Successful-attack localization",
        "",
        _table(method_summary) if not method_summary.empty else "No successful attacks were available for localization.",
        "",
        "## Predeclared remote tests",
        "",
        "```json",
        json.dumps(_safe({"viability": {
            "clusters": remote_clusters,
            "seeds": remote_seeds,
            "datasets": remote_datasets,
            "minimum_clusters": args.minimum_remote_clusters,
            "minimum_seeds": args.minimum_remote_seeds,
            "minimum_datasets": args.minimum_remote_datasets,
            "supported": viability,
        }, "fingerprint": fingerprint, "causal": causal}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## GPU telemetry",
        "",
        "```json",
        json.dumps(_safe(gpu_stats), ensure_ascii=False, indent=2),
        "```",
        "",
    ])
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(f"[phase3] complete status={status} elapsed={elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
