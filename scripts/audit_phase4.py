from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY = "global_union+hybrid_label_free"


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def _bootstrap(values: np.ndarray, repetitions: int, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    estimates = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "clusters": int(len(values)),
    }


def _interval(frame: pd.DataFrame, column: str, seed: int) -> tuple[dict[str, float | int], pd.DataFrame]:
    keys = ["dataset", "seed", "target"]
    clustered = frame.groupby(keys, as_index=False)[column].mean().sort_values(keys)
    return _bootstrap(clustered[column].to_numpy(float), 5000, seed), clustered


def _edges(text: str) -> set[tuple[int, int]]:
    return {tuple(sorted(map(int, item.split("-")))) for item in text.split(";")} if isinstance(text, str) and text else set()


def _geometry(frame: pd.DataFrame) -> dict[str, int]:
    incidence = cardinality = nesting = 0
    for _, row in frame.iterrows():
        edges = _edges(row.true_added_edges)
        incidence += sum(int(row.target) in edge for edge in edges)
        cardinality += int(len(edges) != int(row.budget))
    for _, group in frame.groupby("cluster"):
        previous: set[tuple[int, int]] = set()
        for _, row in group.sort_values("budget").iterrows():
            current = _edges(row.true_added_edges)
            nesting += int(not previous.issubset(current))
            previous = current
    return {"target_incidence": incidence, "cardinality": cardinality, "nesting": nesting}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--write-json", type=Path)
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_dir / "attack_metrics.csv")
    adaptive = pd.read_csv(args.adaptive_dir / "attack_metrics.csv")
    baseline["attack_success"] = _bool(baseline.attack_success)
    adaptive["attack_success"] = _bool(adaptive.attack_success)
    decision = json.loads((args.comparison_dir / "decision.json").read_text(encoding="utf-8"))
    ids_equal = set(baseline.attack_id) == set(adaptive.attack_id)
    identity = ["attack_id", "dataset", "seed", "target", "budget", "cluster"]
    attacks = baseline[identity + ["clean_margin", "attack_success", "attacked_margin", "target_loss", "true_added_edges"]].merge(
        adaptive[["attack_id", "clean_margin", "attack_success", "attacked_margin", "target_loss", "true_added_edges"]],
        on="attack_id",
        suffixes=("_baseline", "_adaptive"),
        validate="one_to_one",
    )
    attacks["success_delta"] = attacks.attack_success_adaptive.astype(float) - attacks.attack_success_baseline.astype(float)
    utility, utility_clusters = _interval(attacks, "success_delta", 24001)

    baseline_success = set(baseline[baseline.attack_success].attack_id)
    adaptive_success = set(adaptive[adaptive.attack_success].attack_id)
    joint = baseline_success & adaptive_success
    baseline_metrics = pd.read_csv(args.baseline_dir / "localization_metrics.csv")
    adaptive_metrics = pd.read_csv(args.adaptive_dir / "localization_metrics.csv")
    metrics = ["edge_auprc", "recall_at_b", "repair_restored", "repaired_margin_gain", "margin_recovery"]
    left = baseline_metrics[(baseline_metrics.method == PRIMARY) & baseline_metrics.attack_id.isin(joint)][identity + metrics]
    right = adaptive_metrics[(adaptive_metrics.method == PRIMARY) & adaptive_metrics.attack_id.isin(joint)][["attack_id"] + metrics]
    paired = left.merge(right, on="attack_id", suffixes=("_baseline", "_adaptive"), validate="one_to_one")
    effects: dict[str, dict[str, float | int]] = {}
    signs: dict[str, dict[str, int]] = {}
    slices: dict[str, dict[str, float]] = {}
    for offset, metric in enumerate(metrics):
        column = f"delta_{metric}"
        paired[column] = paired[f"{metric}_adaptive"].astype(float) - paired[f"{metric}_baseline"].astype(float)
        effects[metric], clustered = _interval(paired, column, 25001 + offset)
        values = clustered[column].to_numpy(float)
        signs[metric] = {
            "negative": int((values < -1e-12).sum()),
            "zero": int((np.abs(values) <= 1e-12).sum()),
            "positive": int((values > 1e-12).sum()),
        }
        slices[metric] = {
            f"{dataset}|{int(seed)}": float(group[column].mean())
            for (dataset, seed), group in clustered.groupby(["dataset", "seed"])
        }

    base_cov = pd.read_csv(args.baseline_dir / "candidate_coverage.csv")
    adapt_cov = pd.read_csv(args.adaptive_dir / "candidate_coverage.csv")
    base_cov = base_cov[(base_cov.generator == "global_union") & base_cov.attack_id.isin(joint)][identity + ["candidate_recall"]]
    adapt_cov = adapt_cov[(adapt_cov.generator == "global_union") & adapt_cov.attack_id.isin(joint)][["attack_id", "candidate_recall"]]
    coverage = base_cov.merge(adapt_cov, on="attack_id", suffixes=("_baseline", "_adaptive"), validate="one_to_one")
    coverage["delta"] = coverage.candidate_recall_adaptive - coverage.candidate_recall_baseline
    coverage_effect, _ = _interval(coverage, "delta", 27001)

    exact = {
        "utility": all(abs(float(utility[k]) - float(decision["attack_success"]["adaptive_minus_baseline"][k])) < 1e-12 for k in ("mean", "ci95_low", "ci95_high")),
        "primary": all(abs(float(effects[m][k]) - float(decision["primary_global_union_hybrid_adaptive_minus_baseline"][m][k])) < 1e-12 for m in metrics for k in ("mean", "ci95_low", "ci95_high")),
        "coverage": all(abs(float(coverage_effect[k]) - float(decision["global_union_candidate_recall_adaptive_minus_baseline"][k])) < 1e-12 for k in ("mean", "ci95_low", "ci95_high")),
    }
    joint_frame = attacks[attacks.attack_id.isin(joint)]
    joint_frame = joint_frame.copy()
    joint_frame["delta_attacked_margin"] = joint_frame.attacked_margin_adaptive - joint_frame.attacked_margin_baseline
    joint_frame["delta_target_loss"] = joint_frame.target_loss_adaptive - joint_frame.target_loss_baseline
    severity_margin, _ = _interval(joint_frame, "delta_attacked_margin", 28001)
    severity_loss, _ = _interval(joint_frame, "delta_target_loss", 28002)
    edge_jaccards: list[float] = []
    for _, row in joint_frame.iterrows():
        baseline_edges = _edges(row.true_added_edges_baseline)
        adaptive_edges = _edges(row.true_added_edges_adaptive)
        edge_jaccards.append(len(baseline_edges & adaptive_edges) / len(baseline_edges | adaptive_edges))
    audit = {
        "integrity": {
            "baseline_attack_rows": len(baseline),
            "adaptive_attack_rows": len(adaptive),
            "attack_ids_equal": ids_equal,
            "baseline_duplicate_ids": int(baseline.attack_id.duplicated().sum()),
            "adaptive_duplicate_ids": int(adaptive.attack_id.duplicated().sum()),
            "max_clean_margin_difference": float((attacks.clean_margin_baseline - attacks.clean_margin_adaptive).abs().max()),
            "baseline_success_snapshots": len(baseline_success),
            "adaptive_success_snapshots": len(adaptive_success),
            "adaptive_success_not_baseline": len(adaptive_success - baseline_success),
            "joint_snapshots": len(joint),
            "joint_clusters": int(joint_frame.cluster.nunique()),
            "joint_seeds": int(joint_frame.seed.nunique()),
            "joint_datasets": int(joint_frame.dataset.nunique()),
            "baseline_localization_rows": len(baseline_metrics),
            "baseline_expected_localization_rows": len(baseline_success) * baseline_metrics.method.nunique(),
            "adaptive_localization_rows": len(adaptive_metrics),
            "adaptive_expected_localization_rows": len(adaptive_success) * adaptive_metrics.method.nunique(),
            "paired_primary_rows": len(paired),
            "baseline_checkpoints": len(list((args.baseline_dir / "checkpoints").glob("*.pt"))),
            "adaptive_checkpoints": len(list((args.adaptive_dir / "checkpoints").glob("*.pt"))),
        },
        "geometry": {"baseline": _geometry(baseline), "adaptive": _geometry(adaptive)},
        "success_by_dataset_seed": {
            "baseline": baseline.groupby(["dataset", "seed"]).attack_success.agg(["count", "sum"]).reset_index().to_dict(orient="records"),
            "adaptive": adaptive.groupby(["dataset", "seed"]).attack_success.agg(["count", "sum"]).reset_index().to_dict(orient="records"),
        },
        "joint_by_budget": joint_frame.groupby("budget").size().to_dict(),
        "joint_attack_severity": {
            "adaptive_minus_baseline_attacked_margin": severity_margin,
            "adaptive_minus_baseline_target_loss": severity_loss,
            "snapshot_mean_edge_set_jaccard": float(np.mean(edge_jaccards)),
        },
        "utility": utility,
        "utility_cluster_signs": {
            "negative": int((utility_clusters.success_delta < -1e-12).sum()),
            "zero": int((utility_clusters.success_delta.abs() <= 1e-12).sum()),
            "positive": int((utility_clusters.success_delta > 1e-12).sum()),
        },
        "primary_effects": effects,
        "primary_cluster_signs": signs,
        "primary_dataset_seed_means": slices,
        "coverage_effect": coverage_effect,
        "independent_bootstrap_matches_decision": exact,
    }
    rendered = json.dumps(audit, ensure_ascii=False, indent=2, default=lambda value: int(value))
    if args.write_json:
        args.write_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
