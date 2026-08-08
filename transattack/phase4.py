from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_METHOD = "global_union+hybrid_label_free"
PRIMARY_METRICS = ["edge_auprc", "recall_at_b"]
REPAIR_METRICS = ["repair_restored", "repaired_margin_gain"]
FROZEN_CONFIG_KEYS = [
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
]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--minimum-joint-clusters", type=int, default=18)
    parser.add_argument("--minimum-joint-seeds", type=int, default=3)
    parser.add_argument("--minimum-joint-datasets", type=int, default=2)
    parser.add_argument("--utility-noninferiority-margin", type=float, default=-0.10)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[4540, 4541, 4542])
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def _bootstrap(values: np.ndarray, repetitions: int, seed: int) -> dict[str, float | int]:
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


def _cluster_interval(frame: pd.DataFrame, column: str, repetitions: int, seed: int) -> tuple[dict[str, float | int], pd.DataFrame]:
    keys = ["dataset", "seed", "target"]
    clustered = frame.groupby(keys, as_index=False)[column].mean().sort_values(keys)
    return _bootstrap(clustered[column].to_numpy(dtype=float), repetitions, seed), clustered


def _edge_set(text: str) -> set[tuple[int, int]]:
    if not isinstance(text, str) or not text:
        return set()
    return {tuple(sorted(map(int, item.split("-")))) for item in text.split(";")}


def _geometry(frame: pd.DataFrame) -> dict[str, int]:
    target_incidence = 0
    cardinality = 0
    nesting = 0
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
    method: str,
    metrics: list[str],
) -> pd.DataFrame:
    identity = ["attack_id", "dataset", "seed", "target", "budget", "cluster"]
    left = baseline[(baseline.method == method) & baseline.attack_id.isin(joint_ids)][identity + metrics]
    right = adaptive[(adaptive.method == method) & adaptive.attack_id.isin(joint_ids)][["attack_id"] + metrics]
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
    if baseline_config["seeds"] != args.expected_seeds or baseline_config["targets"] != 15:
        raise ValueError("Phase-4 seeds or target count differ from the frozen protocol")
    if baseline_config["attack_types"] != ["remote"]:
        raise ValueError("Phase 4 requires remote attacks only")
    if baseline_config["attack_objective"] != "classification_only":
        raise ValueError("baseline objective mismatch")
    if adaptive_config["attack_objective"] != "adaptive_stealth" or adaptive_config["adaptive_stealth_strength"] != 1.0:
        raise ValueError("adaptive objective mismatch")

    baseline_attacks = pd.read_csv(baseline_dir / "attack_metrics.csv")
    adaptive_attacks = pd.read_csv(adaptive_dir / "attack_metrics.csv")
    baseline_attacks["attack_success"] = _as_bool(baseline_attacks.attack_success)
    adaptive_attacks["attack_success"] = _as_bool(adaptive_attacks.attack_success)
    if len(baseline_attacks) != len(adaptive_attacks):
        raise ValueError("paired attackers produced different snapshot counts")
    if any(set(group.budget) != {1, 2, 4, 8} for _, group in baseline_attacks.groupby("cluster")):
        raise ValueError("every attacked target must have all four frozen budgets")
    if set(baseline_attacks.attack_id) != set(adaptive_attacks.attack_id):
        raise ValueError("paired attacker IDs do not match")

    attack_identity = ["attack_id", "dataset", "seed", "target", "budget", "cluster", "clean_margin"]
    paired_attacks = baseline_attacks[attack_identity + ["attack_success", "attacked_margin"]].merge(
        adaptive_attacks[["attack_id", "clean_margin", "attack_success", "attacked_margin"]],
        on="attack_id",
        suffixes=("_baseline", "_adaptive"),
        validate="one_to_one",
    )
    if float((paired_attacks.clean_margin_baseline - paired_attacks.clean_margin_adaptive).abs().max()) > 3e-4:
        raise ValueError("clean margins differ between paired deterministic runs")
    paired_attacks["success_delta"] = paired_attacks.attack_success_adaptive.astype(float) - paired_attacks.attack_success_baseline.astype(float)
    utility, utility_clusters = _cluster_interval(
        paired_attacks,
        "success_delta",
        args.bootstrap_repetitions,
        24001,
    )

    baseline_success_ids = set(baseline_attacks[baseline_attacks.attack_success].attack_id)
    adaptive_success_ids = set(adaptive_attacks[adaptive_attacks.attack_success].attack_id)
    joint_ids = baseline_success_ids & adaptive_success_ids
    joint_attacks = paired_attacks[paired_attacks.attack_id.isin(joint_ids)]
    joint_clusters = int(joint_attacks.cluster.nunique())
    joint_seeds = int(joint_attacks.seed.nunique())
    joint_datasets = int(joint_attacks.dataset.nunique())
    viability = (
        joint_clusters >= args.minimum_joint_clusters
        and joint_seeds >= args.minimum_joint_seeds
        and joint_datasets >= args.minimum_joint_datasets
    )

    baseline_metrics = pd.read_csv(baseline_dir / "localization_metrics.csv")
    adaptive_metrics = pd.read_csv(adaptive_dir / "localization_metrics.csv")
    primary_pair = _method_pair(
        baseline_metrics,
        adaptive_metrics,
        joint_ids,
        PRIMARY_METHOD,
        PRIMARY_METRICS + REPAIR_METRICS + ["margin_recovery"],
    )
    if len(primary_pair) != len(joint_ids):
        raise ValueError("missing primary-method rows for joint successful attacks")

    effects: dict[str, dict[str, float | int]] = {}
    cluster_effect_tables: list[pd.DataFrame] = []
    for offset, metric in enumerate(PRIMARY_METRICS + REPAIR_METRICS + ["margin_recovery"]):
        column = f"delta_{metric}"
        interval, clustered = _cluster_interval(primary_pair, column, args.bootstrap_repetitions, 25001 + offset)
        effects[metric] = interval
        cluster_effect_tables.append(clustered.rename(columns={column: f"primary_{column}"}))

    anomaly_pair = _method_pair(
        baseline_metrics,
        adaptive_metrics,
        joint_ids,
        "anomaly_base",
        PRIMARY_METRICS,
    )
    anomaly_effects: dict[str, dict[str, float | int]] = {}
    for offset, metric in enumerate(PRIMARY_METRICS):
        anomaly_effects[metric], _ = _cluster_interval(
            anomaly_pair,
            f"delta_{metric}",
            args.bootstrap_repetitions,
            26001 + offset,
        )

    baseline_coverage = pd.read_csv(baseline_dir / "candidate_coverage.csv")
    adaptive_coverage = pd.read_csv(adaptive_dir / "candidate_coverage.csv")
    coverage_identity = ["attack_id", "dataset", "seed", "target", "budget", "cluster"]
    left_coverage = baseline_coverage[(baseline_coverage.generator == "global_union") & baseline_coverage.attack_id.isin(joint_ids)][coverage_identity + ["candidate_recall"]]
    right_coverage = adaptive_coverage[(adaptive_coverage.generator == "global_union") & adaptive_coverage.attack_id.isin(joint_ids)][["attack_id", "candidate_recall"]]
    coverage_pair = left_coverage.merge(right_coverage, on="attack_id", suffixes=("_baseline", "_adaptive"), validate="one_to_one")
    coverage_pair["delta_candidate_recall"] = coverage_pair.candidate_recall_adaptive - coverage_pair.candidate_recall_baseline
    coverage_effect, coverage_clusters = _cluster_interval(
        coverage_pair,
        "delta_candidate_recall",
        args.bootstrap_repetitions,
        27001,
    )

    utility_noninferior = utility["ci95_low"] > args.utility_noninferiority_margin
    primary_evasion = all(effects[metric]["ci95_high"] < 0 for metric in PRIMARY_METRICS)
    strong_repair_evasion = all(effects[metric]["ci95_high"] < 0 for metric in REPAIR_METRICS)
    if not viability:
        status = "adaptive_comparison_underpowered"
    elif primary_evasion and utility_noninferior:
        status = "adaptive_localization_evasion_supported"
    elif primary_evasion:
        status = "adaptive_evasion_with_attack_utility_cost"
    else:
        status = "adaptive_evasion_not_supported"

    cluster_effects = utility_clusters.rename(columns={"success_delta": "utility_success_delta"})
    for table in cluster_effect_tables:
        cluster_effects = cluster_effects.merge(table, on=["dataset", "seed", "target"], how="left", validate="one_to_one")
    cluster_effects = cluster_effects.merge(
        coverage_clusters.rename(columns={"delta_candidate_recall": "global_union_coverage_delta"}),
        on=["dataset", "seed", "target"],
        how="left",
        validate="one_to_one",
    )

    primary_pair.to_csv(output / "paired_primary_snapshots.csv", index=False)
    paired_attacks.to_csv(output / "paired_attack_utility.csv", index=False)
    cluster_effects.to_csv(output / "cluster_effects.csv", index=False)

    method_columns = ["edge_auprc", "recall_at_b", "repair_restored", "repaired_margin_gain", "margin_recovery"]
    absolute = {
        "baseline": baseline_metrics[(baseline_metrics.method == PRIMARY_METHOD) & baseline_metrics.attack_id.isin(joint_ids)][method_columns].astype(float).mean().to_dict(),
        "adaptive": adaptive_metrics[(adaptive_metrics.method == PRIMARY_METHOD) & adaptive_metrics.attack_id.isin(joint_ids)][method_columns].astype(float).mean().to_dict(),
    }
    attack_rates = {
        "baseline": baseline_attacks.groupby("budget").attack_success.mean().to_dict(),
        "adaptive": adaptive_attacks.groupby("budget").attack_success.mean().to_dict(),
    }
    decision = {
        "status": status,
        "joint_viability": {
            "successful_snapshots": len(joint_ids),
            "clusters": joint_clusters,
            "seeds": joint_seeds,
            "datasets": joint_datasets,
            "minimum_clusters": args.minimum_joint_clusters,
            "minimum_seeds": args.minimum_joint_seeds,
            "minimum_datasets": args.minimum_joint_datasets,
            "supported": viability,
        },
        "attack_success": {
            "baseline_snapshots": len(baseline_success_ids),
            "adaptive_snapshots": len(adaptive_success_ids),
            "baseline_clusters": int(baseline_attacks[baseline_attacks.attack_success].cluster.nunique()),
            "adaptive_clusters": int(adaptive_attacks[adaptive_attacks.attack_success].cluster.nunique()),
            "rates_by_budget": attack_rates,
            "adaptive_minus_baseline": utility,
            "noninferiority_margin": args.utility_noninferiority_margin,
            "noninferior": utility_noninferior,
        },
        "primary_global_union_hybrid_adaptive_minus_baseline": effects,
        "primary_evasion_supported": primary_evasion,
        "strong_repair_evasion_supported": strong_repair_evasion,
        "anomaly_base_adaptive_minus_baseline": anomaly_effects,
        "global_union_candidate_recall_adaptive_minus_baseline": coverage_effect,
        "joint_success_absolute_primary": absolute,
        "geometry": {
            "baseline": _geometry(baseline_attacks),
            "adaptive": _geometry(adaptive_attacks),
        },
        "gpu": {
            "baseline": json.loads((baseline_dir / "gpu_telemetry.json").read_text(encoding="utf-8")),
            "adaptive": json.loads((adaptive_dir / "gpu_telemetry.json").read_text(encoding="utf-8")),
        },
    }
    _write_json(output / "decision.json", decision)
    _write_json(output / "config.json", {
        "baseline_dir": str(baseline_dir),
        "adaptive_dir": str(adaptive_dir),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "minimum_joint_clusters": args.minimum_joint_clusters,
        "minimum_joint_seeds": args.minimum_joint_seeds,
        "minimum_joint_datasets": args.minimum_joint_datasets,
        "utility_noninferiority_margin": args.utility_noninferiority_margin,
        "primary_method": PRIMARY_METHOD,
    })

    summary = "\n".join([
        "# GraphTransAttack Phase-4 summary",
        "",
        f"Decision: **{status}**.",
        "",
        f"Joint successful comparison: {len(joint_ids)} snapshots from {joint_clusters} clusters, {joint_seeds} seeds, and {joint_datasets} datasets.",
        "",
        "## Attack utility",
        "",
        "```json",
        json.dumps(decision["attack_success"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Frozen primary detector: adaptive minus baseline",
        "",
        "```json",
        json.dumps(effects, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Absolute primary metrics on joint successes",
        "",
        "```json",
        json.dumps(absolute, ensure_ascii=False, indent=2),
        "```",
        "",
    ])
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(f"[phase4] status={status} joint_clusters={joint_clusters} utility_noninferior={utility_noninferior} primary_evasion={primary_evasion}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
