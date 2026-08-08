from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import adjacency_from_edge_index, load_graph
from .dshield_aug import (
    augmented_graph_views,
    logits_for_augmented_views,
    stable_view_seed,
    view_statistics,
)
from .gps_model import BatchTelemetry, GraphGPSNodeClassifier


PRIMARY_METHOD = "global_union+hybrid_label_free"
ARMS = {
    "edge_aug": (0.20, 0.00),
    "feature_aug": (0.00, 0.20),
    "dshield_aug": (0.20, 0.20),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--views", type=int, default=64)
    parser.add_argument("--graph-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--minimum-success-clusters", type=int, default=18)
    parser.add_argument("--minimum-success-seeds", type=int, default=4)
    parser.add_argument("--minimum-clean-correctness", type=float, default=0.95)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--strict-cuda", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True, default=str) + "\n",
        encoding="utf-8",
    )


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def _bootstrap(values: np.ndarray, repetitions: int, seed: int) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "clusters": 0}
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "clusters": int(len(values)),
    }


def _cluster_interval(
    frame: pd.DataFrame,
    column: str,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    keys = ["dataset", "seed", "target"]
    if frame.empty:
        empty = pd.DataFrame(columns=keys + [column])
        return _bootstrap(np.array([], dtype=float), repetitions, seed), empty
    clustered = frame.groupby(keys, as_index=False)[column].mean().sort_values(keys)
    return _bootstrap(clustered[column].to_numpy(float), repetitions, seed), clustered


def _edge_set(text: str) -> set[tuple[int, int]]:
    if not isinstance(text, str) or not text:
        return set()
    return {tuple(sorted(map(int, item.split("-")))) for item in text.split(";")}


def _attacked_adjacency(clean: torch.Tensor, text: str) -> torch.Tensor:
    attacked = clean.clone()
    for u, v in _edge_set(text):
        attacked[u, v] = attacked[v, u] = 1.0
    return attacked


def _load_model(checkpoint_path: Path, graph, device: torch.device) -> GraphGPSNodeClassifier:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload["model_config"]
    model = GraphGPSNodeClassifier(
        graph.num_features,
        graph.num_classes,
        channels=int(cfg["channels"]),
        pe_channels=int(cfg["pe_channels"]),
        walk_length=int(cfg["walk_length"]),
        layers=int(cfg["layers"]),
        heads=int(cfg["heads"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def main() -> int:
    args = _args()
    if args.views < 2 or args.graph_batch_size < 1 or args.bootstrap_repetitions < 1:
        raise ValueError("views, graph batch size, and bootstrap repetitions must be positive")
    if args.strict_cuda and not torch.cuda.is_available():
        raise RuntimeError("strict CUDA requested but CUDA is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    attack_dir = args.attack_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    config = json.loads((attack_dir / "config.json").read_text(encoding="utf-8"))
    if [str(item).lower() for item in config["datasets"]] != ["pubmed"]:
        raise ValueError("Phase 6 requires the PubMed transfer run")
    if config["seeds"] != args.expected_seeds:
        raise ValueError("attack seeds differ from the Phase-6 protocol")
    if config["attack_types"] != ["remote"]:
        raise ValueError("Phase 6 requires remote attacks only")
    if config["attack_objective"] != "classification_constrained_stealth":
        raise ValueError("Phase 6 defense requires the constrained adaptive attack")
    if abs(float(config["classification_retention_ratio"]) - 0.95) > 1e-12:
        raise ValueError("Phase 6 freezes rho at 0.95")

    attacks = pd.read_csv(attack_dir / "attack_metrics.csv")
    attacks["attack_success"] = _as_bool(attacks.attack_success)
    if attacks.attack_id.duplicated().any():
        raise ValueError("duplicate attack IDs")
    telemetry = BatchTelemetry(args.graph_batch_size, args.graph_batch_size)
    rows: list[dict[str, Any]] = []
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        data_root = Path.cwd() / data_root

    for (dataset, run_seed), group in attacks.groupby(["dataset", "seed"], sort=True):
        graph = load_graph(str(dataset), data_root, int(config["nodes"]), int(run_seed))
        checkpoint = attack_dir / "checkpoints" / f"{str(dataset).lower()}_rwse_seed{int(run_seed)}.pt"
        model = _load_model(checkpoint, graph, device)
        clean_adjacency = adjacency_from_edge_index(graph.edge_index, graph.num_nodes).float()
        clean_cache: dict[tuple[str, str], Any] = {}
        for _, attack in group.sort_values(["target", "budget"]).iterrows():
            target = int(attack.target)
            true_label = int(graph.y[target])
            attacked_adjacency = _attacked_adjacency(clean_adjacency, str(attack.true_added_edges))
            for arm, (edge_drop, feature_drop) in ARMS.items():
                cache_key = (str(attack.cluster), arm)
                view_seed = stable_view_seed(f"{attack.cluster}|{arm}")
                if cache_key not in clean_cache:
                    clean_adj_views, clean_x_views = augmented_graph_views(
                        graph, clean_adjacency, args.views, edge_drop, feature_drop, view_seed
                    )
                    clean_logits = logits_for_augmented_views(
                        model, graph, clean_adj_views, clean_x_views, device, args.graph_batch_size, telemetry
                    )
                    clean_cache[cache_key] = view_statistics(clean_logits, target, true_label)
                clean_stats = clean_cache[cache_key]
                attacked_adj_views, attacked_x_views = augmented_graph_views(
                    graph, attacked_adjacency, args.views, edge_drop, feature_drop, view_seed
                )
                attacked_logits = logits_for_augmented_views(
                    model, graph, attacked_adj_views, attacked_x_views, device, args.graph_batch_size, telemetry
                )
                attacked_stats = view_statistics(attacked_logits, target, true_label)
                raw_success = bool(attack.attack_success)
                rows.append({
                    "attack_id": str(attack.attack_id),
                    "dataset": str(dataset),
                    "seed": int(run_seed),
                    "target": target,
                    "budget": int(attack.budget),
                    "cluster": str(attack.cluster),
                    "arm": arm,
                    "views": int(args.views),
                    "edge_drop_ratio": edge_drop,
                    "feature_drop_ratio": feature_drop,
                    "raw_attack_success": raw_success,
                    "raw_true_probability": float(math.exp(-float(attack.target_loss))),
                    "clean_prediction": clean_stats.prediction,
                    "clean_correct": clean_stats.prediction == true_label,
                    "clean_true_probability": clean_stats.target_probability,
                    "clean_disagreement": clean_stats.disagreement,
                    "defended_prediction": attacked_stats.prediction,
                    "defended_correct": attacked_stats.prediction == true_label,
                    "recovered": raw_success and attacked_stats.prediction == true_label,
                    "defended_true_probability": attacked_stats.target_probability,
                    "attacked_disagreement": attacked_stats.disagreement,
                    "delta_disagreement": attacked_stats.disagreement - clean_stats.disagreement,
                    "true_probability_gain": attacked_stats.target_probability - math.exp(-float(attack.target_loss)),
                })
        pd.DataFrame(rows).to_csv(output / "view_metrics.csv", index=False)
        print(f"[phase6] model_complete={dataset}/seed{run_seed} rows={len(rows)}", flush=True)

    metrics = pd.DataFrame(rows)
    successful_ids = set(attacks[attacks.attack_success].attack_id.astype(str))
    successful = metrics[metrics.attack_id.isin(successful_ids)].copy()
    joint_clusters = int(successful.cluster.nunique())
    joint_seeds = int(successful.seed.nunique())
    viability = joint_clusters >= args.minimum_success_clusters and joint_seeds >= args.minimum_success_seeds

    arm_results: dict[str, Any] = {}
    arm_clusters: dict[str, pd.DataFrame] = {}
    for offset, arm in enumerate(ARMS):
        arm_all = metrics[metrics.arm == arm].copy()
        arm_success = successful[successful.arm == arm].copy()
        clean_unique = arm_all.sort_values("budget").drop_duplicates("cluster")
        clean_unique["clean_utility_delta"] = clean_unique.clean_correct.astype(float) - 1.0
        clean_correctness, _ = _cluster_interval(
            clean_unique, "clean_correct", args.bootstrap_repetitions, 41001 + offset
        )
        clean_delta, _ = _cluster_interval(
            clean_unique, "clean_utility_delta", args.bootstrap_repetitions, 41101 + offset
        )
        recovery, clustered = _cluster_interval(
            arm_success, "recovered", args.bootstrap_repetitions, 41201 + offset
        )
        disagreement, _ = _cluster_interval(
            arm_success, "delta_disagreement", args.bootstrap_repetitions, 41301 + offset
        )
        probability_gain, _ = _cluster_interval(
            arm_success, "true_probability_gain", args.bootstrap_repetitions, 41401 + offset
        )
        arm_results[arm] = {
            "clean_correctness": clean_correctness,
            "clean_utility_delta": clean_delta,
            "attack_recovery": recovery,
            "attacked_minus_clean_disagreement": disagreement,
            "true_probability_gain": probability_gain,
        }
        arm_clusters[arm] = clustered.rename(columns={"recovered": f"recovered_{arm}"})

    controls: dict[str, Any] = {}
    primary_clustered = arm_clusters["dshield_aug"]
    for offset, control in enumerate(("edge_aug", "feature_aug")):
        paired = primary_clustered.merge(
            arm_clusters[control], on=["dataset", "seed", "target"], validate="one_to_one"
        )
        column = f"dshield_minus_{control}"
        paired[column] = paired.recovered_dshield_aug - paired[f"recovered_{control}"]
        controls[column] = _bootstrap(
            paired[column].to_numpy(float), args.bootstrap_repetitions, 42001 + offset
        )

    local_metrics = pd.read_csv(attack_dir / "localization_metrics.csv")
    local_metrics["repair_restored"] = _as_bool(local_metrics.repair_restored)
    causal = local_metrics[
        (local_metrics.method == PRIMARY_METHOD) & local_metrics.attack_id.astype(str).isin(successful_ids)
    ].copy()
    causal_repair, _ = _cluster_interval(causal, "repair_restored", args.bootstrap_repetitions, 43001)
    causal_margin, _ = _cluster_interval(causal, "repaired_margin_gain", args.bootstrap_repetitions, 43002)

    primary = arm_results["dshield_aug"]
    clean_supported = float(primary["clean_correctness"]["ci95_low"]) >= args.minimum_clean_correctness
    recovery_supported = float(primary["attack_recovery"]["ci95_low"]) > 0.0
    discrepancy_supported = float(primary["attacked_minus_clean_disagreement"]["ci95_low"]) > 0.0
    if args.smoke:
        status = "smoke_only"
    elif not viability:
        status = "pubmed_attack_underpowered_for_defense"
    elif clean_supported and recovery_supported:
        status = "dshield_aug_defense_supported"
    else:
        status = "dshield_aug_defense_inconclusive_or_negative"

    elapsed = time.perf_counter() - started
    gpu = {
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
        "defense_identity": "DShield-Aug adaptation; not official DShield",
        "threat_model": "oblivious-defense pilot against post-training remote edge additions",
        "viability": {
            "successful_snapshots": int(len(successful_ids)),
            "clusters": joint_clusters,
            "seeds": joint_seeds,
            "minimum_clusters": args.minimum_success_clusters,
            "minimum_seeds": args.minimum_success_seeds,
            "supported": viability,
        },
        "arms": arm_results,
        "primary_clean_supported": clean_supported,
        "primary_recovery_supported": recovery_supported,
        "primary_discrepancy_supported": discrepancy_supported,
        "primary_minus_controls": controls,
        "causal_top_b_benchmark": {
            "repair_restored": causal_repair,
            "repaired_margin_gain": causal_margin,
        },
        "gpu": gpu,
    }
    _write_json(output / "decision.json", decision)
    _write_json(output / "config.json", {
        **vars(args),
        "attack_dir": str(attack_dir),
        "output_dir": str(output),
        "arms": ARMS,
        "primary_arm": "dshield_aug",
    })
    summary = "\n".join([
        "# GraphTransAttack Phase-6 defense summary",
        "",
        f"Decision: **{status}**.",
        "",
        "DShield-Aug is an explicitly named test-time adaptation of DShield's augmentation/discrepancy idea; it is not the official training-node DShield defense.",
        "",
        "```json",
        json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=True),
        "```",
        "",
    ])
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(f"[phase6] complete status={status} clusters={joint_clusters} elapsed={elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
