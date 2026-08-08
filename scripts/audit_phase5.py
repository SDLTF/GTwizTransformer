from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY = "global_union+hybrid_label_free"
METRICS = ("edge_auprc", "recall_at_b", "repair_restored", "repaired_margin_gain", "margin_recovery")


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def _bootstrap(values: np.ndarray, seed: int, repetitions: int = 5000) -> dict[str, float | int]:
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


def _interval(frame: pd.DataFrame, column: str, seed: int) -> tuple[dict[str, float | int], pd.DataFrame]:
    keys = ["dataset", "seed", "target"]
    clustered = frame.groupby(keys, as_index=False)[column].mean().sort_values(keys)
    return _bootstrap(clustered[column].to_numpy(float), seed), clustered


def _edges(text: str) -> set[tuple[int, int]]:
    if not isinstance(text, str) or not text:
        return set()
    return {tuple(sorted(map(int, item.split("-")))) for item in text.split(";")}


def _jaccard(left: str, right: str) -> float:
    a, b = _edges(left), _edges(right)
    return len(a & b) / len(a | b) if a | b else 1.0


def _geometry(frame: pd.DataFrame) -> dict[str, int]:
    incidence = cardinality = nesting = 0
    for _, row in frame.iterrows():
        edge_set = _edges(row.true_added_edges)
        incidence += sum(int(row.target) in edge for edge in edge_set)
        cardinality += int(len(edge_set) != int(row.budget))
    for _, group in frame.groupby("cluster"):
        previous: set[tuple[int, int]] = set()
        for _, row in group.sort_values("budget").iterrows():
            current = _edges(row.true_added_edges)
            nesting += int(not previous.issubset(current))
            previous = current
    return {"target_incidence": incidence, "budget_cardinality": cardinality, "nested_budget": nesting}


def _signs(clustered: pd.DataFrame, column: str) -> dict[str, int]:
    values = clustered[column].to_numpy(float)
    return {
        "negative": int((values < -1e-12).sum()),
        "zero": int((np.abs(values) <= 1e-12).sum()),
        "positive": int((values > 1e-12).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--write-json", type=Path, required=True)
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_dir / "attack_metrics.csv")
    adaptive = pd.read_csv(args.adaptive_dir / "attack_metrics.csv")
    baseline["attack_success"] = _bool(baseline.attack_success)
    adaptive["attack_success"] = _bool(adaptive.attack_success)
    decision = json.loads((args.comparison_dir / "decision.json").read_text(encoding="utf-8"))
    identity = ["attack_id", "dataset", "seed", "target", "budget", "cluster"]
    paired = baseline[identity + ["clean_margin", "attacked_margin", "target_loss", "attack_success", "true_added_edges"]].merge(
        adaptive[["attack_id", "clean_margin", "attacked_margin", "target_loss", "attack_success", "true_added_edges",
                  "minimum_selected_gain_ratio", "mean_eligible_candidates"]],
        on="attack_id", suffixes=("_baseline", "_adaptive"), validate="one_to_one",
    )
    paired["success_delta"] = paired.attack_success_adaptive.astype(float) - paired.attack_success_baseline.astype(float)
    utility, utility_clusters = _interval(paired, "success_delta", 31001)
    joint_ids = set(baseline[baseline.attack_success].attack_id) & set(adaptive[adaptive.attack_success].attack_id)
    joint = paired[paired.attack_id.isin(joint_ids)].copy()
    joint["attack_progress_ratio"] = (
        (joint.clean_margin_adaptive - joint.attacked_margin_adaptive)
        / (joint.clean_margin_baseline - joint.attacked_margin_baseline).clip(lower=1e-12)
    )
    joint["delta_attacked_margin"] = joint.attacked_margin_adaptive - joint.attacked_margin_baseline
    joint["delta_target_loss"] = joint.target_loss_adaptive - joint.target_loss_baseline
    joint["edge_set_jaccard"] = [
        _jaccard(left, right) for left, right in zip(joint.true_added_edges_baseline, joint.true_added_edges_adaptive)
    ]
    severity = {}
    severity_clusters = {}
    for offset, name in enumerate(("attack_progress_ratio", "delta_attacked_margin", "delta_target_loss", "edge_set_jaccard")):
        severity[name], clustered = _interval(joint, name, 31002 + offset)
        severity_clusters[name] = clustered

    baseline_metrics = pd.read_csv(args.baseline_dir / "localization_metrics.csv")
    adaptive_metrics = pd.read_csv(args.adaptive_dir / "localization_metrics.csv")
    left = baseline_metrics[(baseline_metrics.method == PRIMARY) & baseline_metrics.attack_id.isin(joint_ids)][identity + list(METRICS)]
    right = adaptive_metrics[(adaptive_metrics.method == PRIMARY) & adaptive_metrics.attack_id.isin(joint_ids)][["attack_id"] + list(METRICS)]
    metric_pair = left.merge(right, on="attack_id", suffixes=("_baseline", "_adaptive"), validate="one_to_one")
    effects = {}
    signs = {}
    slices = {}
    budget_means = {}
    for offset, metric in enumerate(METRICS):
        column = f"delta_{metric}"
        # ``repair_restored`` is persisted as a boolean while the remaining
        # metrics are floats.  Cast both sides explicitly so every paired
        # effect uses the same adaptive-minus-baseline arithmetic.
        metric_pair[column] = (
            metric_pair[f"{metric}_adaptive"].astype(float)
            - metric_pair[f"{metric}_baseline"].astype(float)
        )
        effects[metric], clustered = _interval(metric_pair, column, 32001 + offset)
        signs[metric] = _signs(clustered, column)
        slices[metric] = {
            f"{dataset}|{int(seed)}": float(group[column].mean())
            for (dataset, seed), group in clustered.groupby(["dataset", "seed"])
        }
        budget_means[metric] = {
            str(int(budget)): float(group[column].mean()) for budget, group in metric_pair.groupby("budget")
        }

    baseline_coverage = pd.read_csv(args.baseline_dir / "candidate_coverage.csv")
    adaptive_coverage = pd.read_csv(args.adaptive_dir / "candidate_coverage.csv")
    left_cov = baseline_coverage[(baseline_coverage.generator == "global_union") & baseline_coverage.attack_id.isin(joint_ids)][identity + ["candidate_recall"]]
    right_cov = adaptive_coverage[(adaptive_coverage.generator == "global_union") & adaptive_coverage.attack_id.isin(joint_ids)][["attack_id", "candidate_recall"]]
    coverage_pair = left_cov.merge(right_cov, on="attack_id", suffixes=("_baseline", "_adaptive"), validate="one_to_one")
    coverage_pair["delta"] = coverage_pair.candidate_recall_adaptive - coverage_pair.candidate_recall_baseline
    coverage, coverage_clustered = _interval(coverage_pair, "delta", 33001)

    expected_effects = decision["primary_localization_adaptive_minus_baseline"]
    exact = {
        "utility": all(abs(float(utility[key]) - float(decision["attack_utility"]["adaptive_minus_baseline_success"][key])) < 1e-12 for key in ("mean", "ci95_low", "ci95_high")),
        "severity": all(abs(float(severity["attack_progress_ratio"][key]) - float(decision["attack_utility"]["adaptive_over_baseline_attack_progress"][key])) < 1e-12 for key in ("mean", "ci95_low", "ci95_high")),
        "localization": all(abs(float(effects[metric][key]) - float(expected_effects[metric][key])) < 1e-12 for metric in METRICS for key in ("mean", "ci95_low", "ci95_high")),
        "coverage": all(abs(float(coverage[key]) - float(decision["global_union_candidate_recall_adaptive_minus_baseline"][key])) < 1e-12 for key in ("mean", "ci95_low", "ci95_high")),
    }
    audit = {
        "attack_ids_equal": set(baseline.attack_id) == set(adaptive.attack_id),
        "duplicates": {"baseline": int(baseline.attack_id.duplicated().sum()), "adaptive": int(adaptive.attack_id.duplicated().sum())},
        "attempted_snapshots": len(paired),
        "attempted_clusters": int(paired.cluster.nunique()),
        "joint_snapshots": len(joint_ids),
        "joint_clusters": int(joint.cluster.nunique()),
        "joint_seeds": int(joint.seed.nunique()),
        "joint_datasets": int(joint.dataset.nunique()),
        "clean_margin_max_abs_diff": float((paired.clean_margin_baseline - paired.clean_margin_adaptive).abs().max()),
        "geometry": {"baseline": _geometry(baseline), "adaptive": _geometry(adaptive)},
        "constraint": {
            "minimum_selected_gain_ratio": float(adaptive.minimum_selected_gain_ratio.min()),
            "violations_below_0p95": int((adaptive.minimum_selected_gain_ratio < 0.95 - 1e-6).sum()),
            "mean_eligible_candidates_by_budget": {
                str(int(budget)): float(group.mean_eligible_candidates.mean()) for budget, group in adaptive.groupby("budget")
            },
        },
        "success_rates_by_budget": {
            "baseline": {str(int(k)): float(v) for k, v in baseline.groupby("budget").attack_success.mean().items()},
            "adaptive": {str(int(k)): float(v) for k, v in adaptive.groupby("budget").attack_success.mean().items()},
        },
        "utility": utility,
        "utility_cluster_signs": _signs(utility_clusters, "success_delta"),
        "severity": severity,
        "primary_effects": effects,
        "primary_cluster_signs": signs,
        "primary_dataset_seed_means": slices,
        "primary_budget_means": budget_means,
        "coverage": coverage,
        "coverage_cluster_signs": _signs(coverage_clustered, "delta"),
        "independent_bootstrap_matches_decision": exact,
    }
    args.write_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
