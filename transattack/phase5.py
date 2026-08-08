from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_METHOD = "global_union+hybrid_label_free"
PRIMARY_METRICS = ("edge_auprc", "recall_at_b")
REPAIR_METRICS = ("repair_restored", "repaired_margin_gain")
FROZEN_CONFIG_KEYS = (
    "datasets",
    "seeds",
    "attack_types",
    "budgets",
    "nodes",
    "targets",
    "candidate_pool",
    "candidate_multiplier",
    "graph_batch_size",
    "channels",
    "pe_channels",
    "walk_length",
    "layers",
    "heads",
    "dropout",
    "epochs",
    "patience",
    "learning_rate",
    "weight_decay",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--expected-targets", type=int, required=True)
    parser.add_argument("--minimum-joint-clusters", type=int, default=24)
    parser.add_argument("--minimum-joint-seeds", type=int, default=4)
    parser.add_argument("--minimum-joint-datasets", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--success-noninferiority-margin", type=float, default=-0.05)
    parser.add_argument("--minimum-attack-progress-ratio", type=float, default=0.85)
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
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    estimates = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "clusters": int(len(values)),
    }


def _cluster_interval(
    frame: pd.DataFrame,
    column: str,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    keys = ["dataset", "seed", "target"]
    clustered = frame.groupby(keys, as_index=False)[column].mean().sort_values(keys)
    return _bootstrap(clustered[column].to_numpy(float), repetitions, seed), clustered


def _edge_set(text: str) -> set[tuple[int, int]]:
    if not isinstance(text, str) or not text:
        return set()
    return {tuple(sorted(map(int, item.split("-")))) for item in text.split(";")}


def _jaccard(left: str, right: str) -> float:
    left_edges = _edge_set(left)
    right_edges = _edge_set(right)
    union = left_edges | right_edges
    return len(left_edges & right_edges) / len(union) if union else 1.0


def _geometry(frame: pd.DataFrame) -> dict[str, int]:
    target_incidence = cardinality = nesting = 0
    for _, row in frame.iterrows():
        edges = _edge_set(row.true_added_edges)
        target_incidence += sum(int(row.target) in edge for edge in edges)
        cardinality += int(len(edges) != int(row.budget))
    for _, group in frame.groupby("cluster"):
        previous: set[tuple[int, int]] = set()
        for _, row in group.sort_values("budget").iterrows():
            current = _edge_set(row.true_added_edges)
            nesting += int(not previous.issubset(current))
            previous = current
    return {
        "target_incidence_violations": target_incidence,
        "budget_cardinality_violations": cardinality,
        "nested_budget_violations": nesting,
    }


def _method_pair(
    baseline: pd.DataFrame,
    adaptive: pd.DataFrame,
    joint_ids: set[str],
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    identity = ["attack_id", "dataset", "seed", "target", "budget", "cluster"]
    left = baseline[(baseline.method == PRIMARY_METHOD) & baseline.attack_id.isin(joint_ids)][identity + list(metrics)]
    right = adaptive[(adaptive.method == PRIMARY_METHOD) & adaptive.attack_id.isin(joint_ids)][["attack_id"] + list(metrics)]
    paired = left.merge(right, on="attack_id", suffixes=("_baseline", "_adaptive"), validate="one_to_one")
    for metric in metrics:
        paired[f"delta_{metric}"] = paired[f"{metric}_adaptive"].astype(float) - paired[f"{metric}_baseline"].astype(float)
    return paired


def main() -> int:
    args = _args()
    baseline_dir = args.baseline_dir.resolve()
    adaptive_dir = args.adaptive_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    baseline_config = json.loads((baseline_dir / "config.json").read_text(encoding="utf-8"))
    adaptive_config = json.loads((adaptive_dir / "config.json").read_text(encoding="utf-8"))
    for key in FROZEN_CONFIG_KEYS:
        if baseline_config[key] != adaptive_config[key]:
            raise ValueError(f"paired run configuration mismatch for {key}")
    if baseline_config["seeds"] != args.expected_seeds:
        raise ValueError("seeds differ from the comparison protocol")
    if baseline_config["targets"] != args.expected_targets:
        raise ValueError("target count differs from the comparison protocol")
    if baseline_config["attack_types"] != ["remote"]:
        raise ValueError("Phase 5 requires remote attacks only")
    if baseline_config["attack_objective"] != "classification_only":
        raise ValueError("baseline objective mismatch")
    if adaptive_config["attack_objective"] != "classification_constrained_stealth":
        raise ValueError("adaptive objective mismatch")
    retention_ratio = float(adaptive_config["classification_retention_ratio"])

    baseline = pd.read_csv(baseline_dir / "attack_metrics.csv")
    adaptive = pd.read_csv(adaptive_dir / "attack_metrics.csv")
    baseline["attack_success"] = _as_bool(baseline.attack_success)
    adaptive["attack_success"] = _as_bool(adaptive.attack_success)
    identity = ["attack_id", "dataset", "seed", "target", "budget", "cluster"]
    columns = identity + ["clean_margin", "attacked_margin", "target_loss", "attack_success", "true_added_edges"]
    paired = baseline[columns].merge(
        adaptive[["attack_id", "clean_margin", "attacked_margin", "target_loss", "attack_success", "true_added_edges",
                  "minimum_selected_gain_ratio", "mean_eligible_candidates"]],
        on="attack_id",
        suffixes=("_baseline", "_adaptive"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or set(baseline.attack_id) != set(adaptive.attack_id):
        raise ValueError("paired attack IDs do not match")
    if float((paired.clean_margin_baseline - paired.clean_margin_adaptive).abs().max()) > 3e-4:
        raise ValueError("paired clean margins differ")
    if float(adaptive.minimum_selected_gain_ratio.min()) + 1e-6 < retention_ratio:
        raise ValueError("classification-retention constraint was violated")
    if _geometry(baseline) != {k: 0 for k in _geometry(baseline)} or _geometry(adaptive) != {k: 0 for k in _geometry(adaptive)}:
        raise ValueError("attack geometry violation")

    paired["success_delta"] = paired.attack_success_adaptive.astype(float) - paired.attack_success_baseline.astype(float)
    utility, utility_clusters = _cluster_interval(paired, "success_delta", args.bootstrap_repetitions, 31001)
    baseline_success = set(baseline[baseline.attack_success].attack_id)
    adaptive_success = set(adaptive[adaptive.attack_success].attack_id)
    joint_ids = baseline_success & adaptive_success
    joint_attacks = paired[paired.attack_id.isin(joint_ids)].copy()
    joint_attacks["baseline_attack_progress"] = joint_attacks.clean_margin_baseline - joint_attacks.attacked_margin_baseline
    joint_attacks["adaptive_attack_progress"] = joint_attacks.clean_margin_adaptive - joint_attacks.attacked_margin_adaptive
    joint_attacks["attack_progress_ratio"] = (
        joint_attacks.adaptive_attack_progress / joint_attacks.baseline_attack_progress.clip(lower=1e-12)
    )
    joint_attacks["delta_attacked_margin"] = joint_attacks.attacked_margin_adaptive - joint_attacks.attacked_margin_baseline
    joint_attacks["delta_target_loss"] = joint_attacks.target_loss_adaptive - joint_attacks.target_loss_baseline
    joint_attacks["edge_set_jaccard"] = [
        _jaccard(left, right)
        for left, right in zip(joint_attacks.true_added_edges_baseline, joint_attacks.true_added_edges_adaptive)
    ]
    progress, progress_clusters = _cluster_interval(
        joint_attacks, "attack_progress_ratio", args.bootstrap_repetitions, 31002
    )
    severity_margin, _ = _cluster_interval(
        joint_attacks, "delta_attacked_margin", args.bootstrap_repetitions, 31003
    )
    severity_loss, _ = _cluster_interval(
        joint_attacks, "delta_target_loss", args.bootstrap_repetitions, 31004
    )
    jaccard, _ = _cluster_interval(joint_attacks, "edge_set_jaccard", args.bootstrap_repetitions, 31005)

    metric_names = PRIMARY_METRICS + REPAIR_METRICS + ("margin_recovery",)
    effects: dict[str, dict[str, float | int]] = {}
    effect_clusters: list[pd.DataFrame] = []
    if joint_ids:
        baseline_metrics = pd.read_csv(baseline_dir / "localization_metrics.csv")
        adaptive_metrics = pd.read_csv(adaptive_dir / "localization_metrics.csv")
        metric_pair = _method_pair(baseline_metrics, adaptive_metrics, joint_ids, metric_names)
        if len(metric_pair) != len(joint_ids):
            raise ValueError("missing primary localization rows for joint successes")
        for offset, metric in enumerate(metric_names):
            interval, clustered = _cluster_interval(
                metric_pair, f"delta_{metric}", args.bootstrap_repetitions, 32001 + offset
            )
            effects[metric] = interval
            effect_clusters.append(clustered)

        baseline_coverage = pd.read_csv(baseline_dir / "candidate_coverage.csv")
        adaptive_coverage = pd.read_csv(adaptive_dir / "candidate_coverage.csv")
        left_coverage = baseline_coverage[
            (baseline_coverage.generator == "global_union") & baseline_coverage.attack_id.isin(joint_ids)
        ][identity + ["candidate_recall"]]
        right_coverage = adaptive_coverage[
            (adaptive_coverage.generator == "global_union") & adaptive_coverage.attack_id.isin(joint_ids)
        ][["attack_id", "candidate_recall"]]
        coverage_pair = left_coverage.merge(
            right_coverage, on="attack_id", suffixes=("_baseline", "_adaptive"), validate="one_to_one"
        )
        coverage_pair["delta_candidate_recall"] = (
            coverage_pair.candidate_recall_adaptive - coverage_pair.candidate_recall_baseline
        )
        coverage, coverage_clusters = _cluster_interval(
            coverage_pair, "delta_candidate_recall", args.bootstrap_repetitions, 33001
        )
    else:
        baseline_metrics = pd.DataFrame(columns=["method", "attack_id", *metric_names])
        adaptive_metrics = pd.DataFrame(columns=["method", "attack_id", *metric_names])
        metric_pair = pd.DataFrame(columns=[*identity, *[f"delta_{metric}" for metric in metric_names]])
        empty = np.array([], dtype=float)
        effects = {metric: _bootstrap(empty, args.bootstrap_repetitions, 32001) for metric in metric_names}
        coverage_pair = pd.DataFrame(columns=[*identity, "delta_candidate_recall"])
        coverage, coverage_clusters = _cluster_interval(
            coverage_pair, "delta_candidate_recall", args.bootstrap_repetitions, 33001
        )

    joint_clusters = int(joint_attacks.cluster.nunique())
    joint_seeds = int(joint_attacks.seed.nunique())
    joint_datasets = int(joint_attacks.dataset.nunique())
    viable = (
        joint_clusters >= args.minimum_joint_clusters
        and joint_seeds >= args.minimum_joint_seeds
        and joint_datasets >= args.minimum_joint_datasets
    )
    success_noninferior = utility["ci95_low"] > args.success_noninferiority_margin
    severity_noninferior = progress["ci95_low"] > args.minimum_attack_progress_ratio
    utility_retained = success_noninferior and severity_noninferior
    localization_evasion = all(effects[metric]["ci95_high"] < 0 for metric in PRIMARY_METRICS)
    repair_evasion = all(effects[metric]["ci95_high"] < 0 for metric in REPAIR_METRICS)
    if not viable:
        status = "classification_constrained_comparison_underpowered"
    elif localization_evasion and utility_retained and repair_evasion:
        status = "strong_end_to_end_adaptive_evasion_supported"
    elif localization_evasion and utility_retained:
        status = "adaptive_localization_evasion_supported"
    elif localization_evasion:
        status = "adaptive_evasion_with_attack_utility_cost"
    elif not utility_retained:
        status = "classification_constraint_did_not_retain_end_to_end_utility"
    else:
        status = "adaptive_evasion_not_supported"

    cluster_effects = utility_clusters.rename(columns={"success_delta": "success_delta"})
    cluster_effects = cluster_effects.merge(progress_clusters, on=["dataset", "seed", "target"], how="left")
    for table in effect_clusters:
        cluster_effects = cluster_effects.merge(table, on=["dataset", "seed", "target"], how="left")
    cluster_effects = cluster_effects.merge(coverage_clusters, on=["dataset", "seed", "target"], how="left")
    paired.to_csv(output / "paired_attack_utility.csv", index=False)
    joint_attacks.to_csv(output / "paired_joint_success_severity.csv", index=False)
    metric_pair.to_csv(output / "paired_primary_localization.csv", index=False)
    cluster_effects.to_csv(output / "cluster_effects.csv", index=False)

    method_columns = list(metric_names)
    absolute = {
        "baseline": baseline_metrics[(baseline_metrics.method == PRIMARY_METHOD) & baseline_metrics.attack_id.isin(joint_ids)][method_columns].astype(float).mean().to_dict(),
        "adaptive": adaptive_metrics[(adaptive_metrics.method == PRIMARY_METHOD) & adaptive_metrics.attack_id.isin(joint_ids)][method_columns].astype(float).mean().to_dict(),
    }
    decision = {
        "status": status,
        "retention_ratio": retention_ratio,
        "joint_viability": {
            "snapshots": len(joint_ids),
            "clusters": joint_clusters,
            "seeds": joint_seeds,
            "datasets": joint_datasets,
            "minimum_clusters": args.minimum_joint_clusters,
            "minimum_seeds": args.minimum_joint_seeds,
            "minimum_datasets": args.minimum_joint_datasets,
            "supported": viable,
        },
        "attack_utility": {
            "baseline_successes": len(baseline_success),
            "adaptive_successes": len(adaptive_success),
            "adaptive_minus_baseline_success": utility,
            "success_noninferiority_margin": args.success_noninferiority_margin,
            "success_noninferior": success_noninferior,
            "adaptive_over_baseline_attack_progress": progress,
            "minimum_attack_progress_ratio": args.minimum_attack_progress_ratio,
            "severity_noninferior": severity_noninferior,
            "retained": utility_retained,
        },
        "severity_diagnostics": {
            "adaptive_minus_baseline_attacked_margin": severity_margin,
            "adaptive_minus_baseline_target_loss": severity_loss,
            "attack_edge_set_jaccard": jaccard,
        },
        "primary_localization_adaptive_minus_baseline": effects,
        "localization_evasion_supported": localization_evasion,
        "repair_evasion_supported": repair_evasion,
        "global_union_candidate_recall_adaptive_minus_baseline": coverage,
        "absolute_primary_on_joint_successes": absolute,
        "constraint_diagnostics": {
            "minimum_observed_selected_gain_ratio": float(adaptive.minimum_selected_gain_ratio.min()),
            "mean_eligible_candidates": float(adaptive.mean_eligible_candidates.mean()),
        },
        "geometry": {"baseline": _geometry(baseline), "adaptive": _geometry(adaptive)},
        "gpu": {
            "baseline": json.loads((baseline_dir / "gpu_telemetry.json").read_text(encoding="utf-8")),
            "adaptive": json.loads((adaptive_dir / "gpu_telemetry.json").read_text(encoding="utf-8")),
        },
    }
    _write_json(output / "decision.json", decision)
    _write_json(output / "config.json", vars(args) | {
        "baseline_dir": str(baseline_dir),
        "adaptive_dir": str(adaptive_dir),
        "primary_method": PRIMARY_METHOD,
    })
    (output / "summary.md").write_text(
        "\n".join([
            "# GraphTransAttack Phase-5 comparison",
            "",
            f"Decision: **{status}**.",
            "",
            f"Classification-retention ratio: `{retention_ratio}`.",
            "",
            "```json",
            json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=True),
            "```",
            "",
        ]),
        encoding="utf-8",
    )
    print(
        f"[phase5] status={status} ratio={retention_ratio:.2f} "
        f"joint_clusters={joint_clusters} utility_retained={utility_retained} "
        f"localization_evasion={localization_evasion}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
