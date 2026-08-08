from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch import Tensor

from .data import add_edge, load_graph, remove_edges, undirected_pairs
from .localize import edge_features, feature_views, fit_view_profiles, positive_mask, ranking_metrics, top_pairs
from .model import true_margin
from .phase1 import (
    _bootstrap,
    _format_pairs,
    _load_model,
    _parse_pairs,
    _rank_promoted,
    _repair_outcome,
    _robust_z,
    _safe,
    _selection_metrics,
    _table,
    _trace_for_edges,
    _write_json,
)


GENERATORS = ("all_layer", "temporal_residual", "target_incident", "multiview_union")
RERANKERS = ("cf_pred_label_free", "hybrid_label_free", "oracle_true_margin")
METRICS = ["edge_auprc", "recall_at_b", "margin_recovery", "repair_restored"]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--combined-metrics", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--candidate-multipliers", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[3410, 3411, 3412])
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


def _ordered(scores: Tensor) -> list[int]:
    return [int(value) for value in torch.argsort(scores.detach().cpu(), descending=True).tolist()]


def _incident_first_order(pairs: Tensor, combined_scores: Tensor, target: int) -> list[int]:
    order = _ordered(combined_scores)
    incident = [index for index in order if int(pairs[0, index]) == target or int(pairs[1, index]) == target]
    incident_set = set(incident)
    return incident + [index for index in order if index not in incident_set]


def _round_robin(rankings: Iterable[list[int]], count: int) -> list[int]:
    lists = [list(ranking) for ranking in rankings]
    cursors = [0] * len(lists)
    selected: list[int] = []
    seen: set[int] = set()
    while len(selected) < count:
        progressed = False
        for slot, ranking in enumerate(lists):
            while cursors[slot] < len(ranking) and ranking[cursors[slot]] in seen:
                cursors[slot] += 1
            if cursors[slot] >= len(ranking):
                continue
            value = ranking[cursors[slot]]
            cursors[slot] += 1
            selected.append(value)
            seen.add(value)
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def candidate_rankings(pairs: Tensor, all_scores: Tensor, temporal_scores: Tensor, target: int) -> dict[str, list[int]]:
    """Build deterministic, label-free candidate rankings for Phase 2."""
    if pairs.ndim != 2 or pairs.size(0) != 2:
        raise ValueError("pairs must have shape [2, E]")
    if all_scores.numel() != pairs.size(1) or temporal_scores.numel() != pairs.size(1):
        raise ValueError("score and edge counts must match")
    all_order = _ordered(all_scores)
    temporal_order = _ordered(temporal_scores)
    combined = _robust_z(all_scores) + _robust_z(temporal_scores)
    incident_order = _incident_first_order(pairs, combined, int(target))
    return {
        "all_layer": all_order,
        "temporal_residual": temporal_order,
        "target_incident": incident_order,
        "multiview_union": _round_robin((all_order, temporal_order, incident_order), pairs.size(1)),
    }


def _paired(
    frame: pd.DataFrame,
    left_filter: dict[str, object],
    right_filter: dict[str, object],
    metrics: list[str],
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    identity = ["attack_id", "dataset", "pe", "seed", "target"]
    left, right = frame.copy(), frame.copy()
    for column, value in left_filter.items():
        left = left[left[column] == value]
    for column, value in right_filter.items():
        right = right[right[column] == value]
    paired = left[identity + metrics].merge(
        right[["attack_id"] + metrics], on="attack_id", suffixes=("_left", "_right")
    )
    if paired.attack_id.duplicated().any():
        raise AssertionError("comparison filters do not identify one row per attack")
    output: dict[str, dict[str, float]] = {}
    cluster_keys = ["dataset", "pe", "seed", "target"]
    for offset, metric in enumerate(metrics):
        name = f"delta_{metric}"
        paired[name] = paired[f"{metric}_left"].astype(float) - paired[f"{metric}_right"].astype(float)
        values = paired.groupby(cluster_keys)[name].mean().to_numpy(dtype=float)
        output[metric] = _bootstrap(values, repetitions, seed + offset)
    return output


def _coverage_paired(frame: pd.DataFrame, multiplier: int, repetitions: int, seed: int) -> dict[str, float]:
    columns = ["attack_id", "dataset", "pe", "seed", "target", "candidate_recall"]
    left = frame[(frame.generator == "multiview_union") & (frame.k_multiplier == multiplier)][columns]
    right = frame[(frame.generator == "all_layer") & (frame.k_multiplier == multiplier)][["attack_id", "candidate_recall"]]
    paired = left.merge(right, on="attack_id", suffixes=("_union", "_base"))
    paired["delta"] = paired.candidate_recall_union - paired.candidate_recall_base
    values = paired.groupby(["dataset", "pe", "seed", "target"]).delta.mean().to_numpy(dtype=float)
    return _bootstrap(values, repetitions, seed)


def main() -> int:
    args = _args()
    multipliers = sorted(set(int(value) for value in args.candidate_multipliers))
    if not multipliers or any(value < 1 for value in multipliers):
        raise ValueError("candidate multipliers must be positive")
    if 5 not in multipliers or 10 not in multipliers:
        raise ValueError("predeclared primary multipliers 5 and 10 are required")
    device = _device(args.device, args.strict_cuda)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output_dir or Path("results") / f"phase2_{timestamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)

    run_map = {run.resolve().name: run.resolve() for run in args.phase0_runs}
    run_configs = {
        name: json.loads((path / "config.json").read_text(encoding="utf-8"))
        for name, path in run_map.items()
    }
    attacks_all = pd.read_csv(args.combined_metrics.resolve()).drop_duplicates("attack_id")
    observed_seeds = sorted(int(value) for value in attacks_all.seed.unique())
    if args.attack_limit <= 0 and observed_seeds != sorted(args.expected_seeds):
        raise ValueError(f"holdout seed mismatch: expected {sorted(args.expected_seeds)}, observed {observed_seeds}")
    attacks = attacks_all[attacks_all.attack_success].sort_values(["dataset", "pe", "seed", "target", "budget"])
    if args.attack_limit > 0:
        attacks = attacks.head(args.attack_limit)
    if attacks.empty:
        raise RuntimeError("no successful holdout attacks to analyze")

    config_out = vars(args).copy()
    config_out.update(
        device_resolved=str(device),
        output_dir=output,
        observed_seeds=observed_seeds,
        phase0_successful_snapshots=int(len(attacks)),
        primary_replication="all_layer_cf_pred_5B_vs_anomaly_base",
        primary_candidate_comparison="multiview_union_vs_all_layer_at_10B",
        primary_end_to_end="multiview_union_cf_pred_vs_all_layer_cf_pred_at_10B",
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
    print(f"[phase2] device={device} output={output} attacks={len(attacks)} seeds={observed_seeds}", flush=True)

    coverage_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    model_keys = ["source_run", "dataset", "pe", "seed"]

    for key, group in attacks.groupby(model_keys, sort=True):
        source_run, dataset, pe_name, seed = str(key[0]), str(key[1]), str(key[2]), int(key[3])
        if source_run not in run_map:
            raise KeyError(f"missing Phase-0 run directory for {source_run}")
        config = run_configs[source_run]
        graph = load_graph(dataset, args.data_root.resolve(), int(config["nodes"]), seed)
        checkpoint = run_map[source_run] / "checkpoints" / f"{dataset.lower()}_{pe_name}_seed{seed}.pt"
        model, pe_scale = _load_model(graph, config, checkpoint, device)
        clean_trace = _trace_for_edges(model, graph, graph.edge_index, pe_name, int(config["pe_dim"]), pe_scale, device)
        clean_features = edge_features(clean_trace, undirected_pairs(graph.edge_index))
        view_profiles = fit_view_profiles(clean_features)
        print(f"[phase2] model={dataset}/{pe_name}/seed{seed} attacks={len(group)}", flush=True)

        for _, attack in group.iterrows():
            target, budget = int(attack.target), int(attack.budget)
            label = int(graph.y[target])
            positives = _parse_pairs(attack.true_added_edges)
            attacked_edges = graph.edge_index
            for u, v in positives:
                attacked_edges = add_edge(attacked_edges, graph.num_nodes, u, v)
            attacked_trace = _trace_for_edges(model, graph, attacked_edges, pe_name, int(config["pe_dim"]), pe_scale, device)
            clean_margin = true_margin(clean_trace.logits[target], label)
            attacked_margin = true_margin(attacked_trace.logits[target], label)
            if int(attacked_trace.logits[target].argmax()) == label:
                raise AssertionError(f"frozen successful attack no longer succeeds: {attack.attack_id}")
            if abs(attacked_margin - float(attack.attacked_margin)) > 2e-4:
                raise AssertionError(f"frozen margin mismatch for {attack.attack_id}")

            attacked_pairs = undirected_pairs(attacked_edges)
            attacked_views = feature_views(edge_features(attacked_trace, attacked_pairs))
            all_scores = view_profiles["all_layer_full"].score(attacked_views["all_layer_full"])
            temporal_scores = view_profiles["temporal_residual"].score(attacked_views["temporal_residual"])
            rankings = candidate_rankings(attacked_pairs, all_scores, temporal_scores, target)
            max_count = min(attacked_pairs.size(1), max(multipliers) * budget)
            required_indices = sorted({index for ranking in rankings.values() for index in ranking[:max_count]})
            attacked_prediction = int(attacked_trace.logits[target].argmax())
            attacked_probability = float(torch.softmax(attacked_trace.logits[target], dim=0)[attacked_prediction])
            positive_set = {tuple(sorted(pair)) for pair in positives}
            cf_values: dict[int, tuple[float, float]] = {}

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
            for edge_position in required_indices:
                pair = tuple(map(int, attacked_pairs[:, edge_position].tolist()))
                cf_edges = remove_edges(attacked_edges, graph.num_nodes, [pair])
                cf_trace = _trace_for_edges(model, graph, cf_edges, pe_name, int(config["pe_dim"]), pe_scale, device)
                probability = float(torch.softmax(cf_trace.logits[target], dim=0)[attacked_prediction])
                cf_margin = true_margin(cf_trace.logits[target], label)
                cf_values[edge_position] = (attacked_probability - probability, cf_margin - attacked_margin)
                candidate_rows.append({
                    **identity,
                    "edge_position": edge_position,
                    "candidate_edge": _format_pairs([pair]),
                    "is_true_attack_edge": tuple(sorted(pair)) in positive_set,
                    "base_anomaly": float(all_scores[edge_position]),
                    "temporal_anomaly": float(temporal_scores[edge_position]),
                    "predicted_class_probability_drop": cf_values[edge_position][0],
                    "oracle_true_margin_gain": cf_values[edge_position][1],
                })

            repair_cache: dict[str, dict[str, Any]] = {}

            def repair(selected: list[tuple[int, int]]) -> dict[str, Any]:
                cache_key = _format_pairs(sorted(tuple(sorted(pair)) for pair in selected))
                if cache_key not in repair_cache:
                    repair_cache[cache_key] = _repair_outcome(
                        model, graph, attacked_edges, selected, pe_name, int(config["pe_dim"]),
                        pe_scale, target, clean_margin, attacked_margin, device,
                    )
                return repair_cache[cache_key]

            baseline_selected = top_pairs(attacked_pairs, all_scores, budget)
            metric_rows.append({
                **identity,
                "generator": "none",
                "k_multiplier": 0,
                "reranker": "anomaly_base",
                "candidate_count": 0,
                "candidate_recall": float("nan"),
                "deployable_without_true_label": True,
                "selected_edges": _format_pairs(baseline_selected),
                **ranking_metrics(attacked_pairs, all_scores, positives, budget),
                **_selection_metrics(baseline_selected, positives),
                **repair(baseline_selected),
            })

            attack_coverage_start = len(coverage_rows)
            for generator in GENERATORS:
                full_order = rankings[generator]
                for multiplier in multipliers:
                    count = min(attacked_pairs.size(1), multiplier * budget)
                    indices = torch.tensor(full_order[:count], dtype=torch.long)
                    candidate_pairs = attacked_pairs[:, indices]
                    candidate_recall = float(positive_mask(candidate_pairs, positives).sum().item() / len(positives))
                    coverage_rows.append({
                        **identity,
                        "generator": generator,
                        "k_multiplier": multiplier,
                        "candidate_count": count,
                        "candidate_recall": candidate_recall,
                    })
                    pred = torch.tensor([cf_values[int(index)][0] for index in indices.tolist()])
                    oracle = torch.tensor([cf_values[int(index)][1] for index in indices.tolist()])
                    base = all_scores[indices]
                    reranker_values = {
                        "cf_pred_label_free": pred,
                        "hybrid_label_free": _robust_z(base) + _robust_z(pred),
                        "oracle_true_margin": oracle,
                    }
                    for reranker in RERANKERS:
                        scores = _rank_promoted(all_scores, indices, reranker_values[reranker])
                        selected = top_pairs(attacked_pairs, scores, budget)
                        metric_rows.append({
                            **identity,
                            "generator": generator,
                            "k_multiplier": multiplier,
                            "reranker": reranker,
                            "candidate_count": count,
                            "candidate_recall": candidate_recall,
                            "deployable_without_true_label": reranker != "oracle_true_margin",
                            "selected_edges": _format_pairs(selected),
                            **ranking_metrics(attacked_pairs, scores, positives, budget),
                            **_selection_metrics(selected, positives),
                            **repair(selected),
                        })
            pd.DataFrame(coverage_rows).to_csv(output / "candidate_coverage.csv", index=False)
            pd.DataFrame(metric_rows).to_csv(output / "localization_metrics.csv", index=False)
            pd.DataFrame(candidate_rows).to_csv(output / "counterfactual_scores.csv", index=False)
            current_coverage = coverage_rows[attack_coverage_start:]
            union10 = next(row for row in current_coverage if row["generator"] == "multiview_union" and row["k_multiplier"] == 10)
            print(f"[phase2] {attack.attack_id} eval={len(required_indices)} union10_coverage={union10['candidate_recall']:.3f}", flush=True)

    coverage = pd.DataFrame(coverage_rows)
    metrics = pd.DataFrame(metric_rows)
    coverage_summary = coverage.groupby(["generator", "k_multiplier"], as_index=False).agg(
        snapshots=("attack_id", "count"),
        candidate_count=("candidate_count", "mean"),
        candidate_recall=("candidate_recall", "mean"),
    )
    localization_summary = metrics.groupby(
        ["generator", "k_multiplier", "reranker"], as_index=False, dropna=False
    ).agg(
        snapshots=("attack_id", "count"),
        candidate_recall=("candidate_recall", "mean"),
        auprc=("edge_auprc", "mean"),
        recall_at_b=("recall_at_b", "mean"),
        repair_rate=("repair_restored", "mean"),
        margin_recovery=("margin_recovery", "mean"),
    )

    replication = _paired(
        metrics,
        {"generator": "all_layer", "k_multiplier": 5, "reranker": "cf_pred_label_free"},
        {"generator": "none", "k_multiplier": 0, "reranker": "anomaly_base"},
        METRICS, args.bootstrap_repetitions, 12001,
    )
    coverage_primary = _coverage_paired(coverage, 10, args.bootstrap_repetitions, 13001)
    end_to_end = _paired(
        metrics,
        {"generator": "multiview_union", "k_multiplier": 10, "reranker": "cf_pred_label_free"},
        {"generator": "all_layer", "k_multiplier": 10, "reranker": "cf_pred_label_free"},
        METRICS, args.bootstrap_repetitions, 14001,
    )
    hybrid_comparison = _paired(
        metrics,
        {"generator": "multiview_union", "k_multiplier": 10, "reranker": "cf_pred_label_free"},
        {"generator": "multiview_union", "k_multiplier": 10, "reranker": "hybrid_label_free"},
        METRICS, args.bootstrap_repetitions, 15001,
    )
    secondary_coverage = {
        str(multiplier): _coverage_paired(coverage, multiplier, args.bootstrap_repetitions, 16001 + multiplier)
        for multiplier in multipliers
    }

    replication_supported = replication["recall_at_b"]["ci95_low"] > 0 and replication["margin_recovery"]["ci95_low"] > 0
    coverage_supported = coverage_primary["ci95_low"] > 0
    end_to_end_supported = end_to_end["recall_at_b"]["ci95_low"] > 0 and end_to_end["margin_recovery"]["ci95_low"] > 0
    if args.attack_limit > 0:
        status = "smoke_only"
    elif replication_supported and coverage_supported and end_to_end_supported:
        status = "replicated_and_candidate_bottleneck_reduced"
    elif replication_supported and coverage_supported:
        status = "replicated_and_coverage_improved_end_to_end_inconclusive"
    elif replication_supported:
        status = "reranker_replicated_candidate_strategy_inconclusive"
    else:
        status = "holdout_replication_failed_or_inconclusive"
    decision = {
        "status": status,
        "holdout_seeds": observed_seeds,
        "successful_attack_snapshots": int(metrics.attack_id.nunique()),
        "model_target_clusters": int(metrics.cluster.nunique()),
        "primary_holdout_replication_cf_pred_5b_minus_anomaly": replication,
        "primary_union_minus_all_layer_candidate_recall_at_10b": coverage_primary,
        "primary_union_cf_pred_minus_all_layer_cf_pred_at_10b": end_to_end,
        "secondary_cf_pred_minus_hybrid_union_10b": hybrid_comparison,
        "secondary_union_minus_all_layer_coverage_by_k": secondary_coverage,
        "architecture_scope": "custom_dense_graph_transformer_mechanism_proxy",
        "target_incident_caveat": "all generated attack edges are incident to the known target in the present threat model",
    }
    _write_json(output / "decision.json", decision)
    summary = "\n".join([
        "# GraphTransAttack Phase-2 summary",
        "",
        f"Decision: **{status}**.",
        "",
        "This is a new-seed holdout test on the existing dense graph-transformer mechanism proxy. "
        "It is not yet a Graphormer/GraphGPS validation. True labels enter only evaluation and the marked oracle.",
        "",
        "## Candidate coverage curve",
        "",
        _table(coverage_summary),
        "",
        "## Localization and causal repair",
        "",
        _table(localization_summary),
        "",
        "## Predeclared tests",
        "",
        "```json",
        json.dumps(_safe({
            "holdout_replication": replication,
            "candidate_coverage_at_10b": coverage_primary,
            "end_to_end_at_10b": end_to_end,
            "cf_pred_minus_hybrid_secondary": hybrid_comparison,
            "coverage_curve_secondary": secondary_coverage,
        }), ensure_ascii=False, indent=2),
        "```",
        "",
        "Target-incident rankings exploit a known-target assumption that exactly matches the current attack generator. "
        "They do not establish localization of remote adversarial subgraphs.",
        "",
    ])
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(f"[phase2] complete status={status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

