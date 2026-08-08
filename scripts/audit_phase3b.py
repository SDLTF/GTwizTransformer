from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_METHOD = "global_union+hybrid_label_free"


def _edge_set(text: str) -> set[tuple[int, int]]:
    if not isinstance(text, str) or not text:
        return set()
    return {tuple(sorted(map(int, edge.split("-")))) for edge in text.split(";")}


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


def _paired_cluster_effects(
    frame: pd.DataFrame,
    primary: str,
    baseline: str,
    metrics: list[str],
    seed: int,
    repetitions: int,
) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    identity = ["attack_id", "dataset", "seed", "attack_type", "target"]
    left = frame[frame.method == primary][identity + metrics]
    right = frame[frame.method == baseline][["attack_id"] + metrics]
    paired = left.merge(right, on="attack_id", suffixes=("_primary", "_baseline"))
    cluster_keys = ["dataset", "seed", "attack_type", "target"]
    intervals: dict[str, dict[str, float | int]] = {}
    signs: dict[str, dict[str, int]] = {}
    slices: dict[str, dict[str, float]] = {}
    for offset, metric in enumerate(metrics):
        delta = f"delta_{metric}"
        paired[delta] = paired[f"{metric}_primary"].astype(float) - paired[f"{metric}_baseline"].astype(float)
        clustered = paired.groupby(cluster_keys, as_index=False)[delta].mean()
        values = clustered[delta].to_numpy(dtype=float)
        intervals[metric] = _bootstrap(values, repetitions, seed + offset)
        signs[metric] = {
            "positive": int((values > 1e-12).sum()),
            "zero": int((np.abs(values) <= 1e-12).sum()),
            "negative": int((values < -1e-12).sum()),
        }
        slices[metric] = {
            f"{dataset}|{int(run_seed)}": float(group[delta].mean())
            for (dataset, run_seed), group in clustered.groupby(["dataset", "seed"])
        }
    return intervals, signs, slices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--phase3-dir", type=Path)
    parser.add_argument("--write-json", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()

    attacks = pd.read_csv(output / "attack_metrics.csv")
    localization = pd.read_csv(output / "localization_metrics.csv")
    coverage = pd.read_csv(output / "candidate_coverage.csv")
    candidates = pd.read_csv(output / "counterfactual_scores.csv")
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    attacks["attack_success"] = attacks.attack_success.astype(bool)

    successful = attacks[attacks.attack_success]
    remote = attacks[attacks.attack_type == "remote"]
    remote_success = remote[remote.attack_success]
    successful_ids = set(successful.attack_id)
    remote_success_ids = set(remote_success.attack_id)

    remote_geometry = {
        "successful_added_edges": 0,
        "target_incidence_violations": 0,
        "budget_cardinality_violations": 0,
        "nested_budget_violations": 0,
    }
    for _, row in remote_success.iterrows():
        edges = _edge_set(row.true_added_edges)
        remote_geometry["successful_added_edges"] += len(edges)
        remote_geometry["target_incidence_violations"] += sum(int(row.target) in edge for edge in edges)
        remote_geometry["budget_cardinality_violations"] += int(len(edges) != int(row.budget))
    for _, group in remote.groupby("cluster"):
        previous: set[tuple[int, int]] = set()
        for _, row in group.sort_values("budget").iterrows():
            current = _edge_set(row.true_added_edges)
            remote_geometry["nested_budget_violations"] += int(not previous.issubset(current))
            previous = current

    loc_success = localization[localization.attack_id.isin(successful_ids)]
    remote_loc = localization[localization.attack_id.isin(remote_success_ids)]
    fingerprint, fingerprint_signs, fingerprint_slices = _paired_cluster_effects(
        remote_loc,
        "anomaly_base",
        "random",
        ["edge_auprc", "recall_at_b"],
        21001,
        int(config["bootstrap_repetitions"]),
    )
    causal, causal_signs, causal_slices = _paired_cluster_effects(
        remote_loc,
        PRIMARY_METHOD,
        "anomaly_base",
        ["recall_at_b", "repaired_margin_gain", "margin_recovery", "repair_restored"],
        22001,
        int(config["bootstrap_repetitions"]),
    )

    method_names = [
        "random",
        "anomaly_base",
        "temporal_anomaly",
        "all_layer+hybrid_label_free",
        "temporal_residual+hybrid_label_free",
        PRIMARY_METHOD,
        "target_incident+hybrid_label_free",
        "global_union+oracle_true_margin",
    ]
    metric_columns = ["edge_auprc", "recall_at_b", "repair_restored", "repaired_margin_gain", "margin_recovery"]
    absolute = (
        remote_loc[remote_loc.method.isin(method_names)]
        .groupby("method")[metric_columns]
        .mean()
        .reindex(method_names)
        .round(9)
        .to_dict(orient="index")
    )
    remote_coverage = (
        coverage[coverage.attack_id.isin(remote_success_ids)]
        .groupby("generator")
        .agg(snapshots=("attack_id", "count"), candidate_recall=("candidate_recall", "mean"))
        .round(9)
        .to_dict(orient="index")
    )

    attack_slices = (
        remote.groupby(["dataset", "seed"])
        .agg(
            attempts=("attack_id", "count"),
            successful_snapshots=("attack_success", "sum"),
            successful_clusters=("target", lambda values: int(remote_success[remote_success.index.isin(values.index)].target.nunique())),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    remote_budgets = (
        remote.groupby("budget").attack_success.agg(["count", "sum"]).rename(columns={"count": "attempts", "sum": "successful_snapshots"}).to_dict(orient="index")
    )

    required_methods = int(localization.method.nunique())
    integrity = {
        "attack_rows": int(len(attacks)),
        "unique_attack_ids": int(attacks.attack_id.nunique()),
        "attack_duplicate_ids": int(attacks.attack_id.duplicated().sum()),
        "successful_snapshots": int(len(successful)),
        "successful_by_type": successful.groupby("attack_type").size().to_dict(),
        "successful_clusters_by_type": successful.groupby("attack_type").cluster.nunique().to_dict(),
        "localization_methods": required_methods,
        "localization_rows": int(len(localization)),
        "expected_localization_rows": int(len(successful) * required_methods),
        "localization_duplicate_attack_method": int(localization.duplicated(["attack_id", "method"]).sum()),
        "localization_non_success_rows": int((~localization.attack_id.isin(successful_ids)).sum()),
        "coverage_rows": int(len(coverage)),
        "expected_coverage_rows": int(len(successful) * coverage.generator.nunique()),
        "coverage_duplicate_attack_generator": int(coverage.duplicated(["attack_id", "generator"]).sum()),
        "counterfactual_rows": int(len(candidates)),
        "counterfactual_duplicate_attack_edge": int(candidates.duplicated(["attack_id", "edge_position"]).sum()),
        "clean_margin_nonpositive": int((attacks.clean_margin <= 0).sum()),
        "success_margin_mismatch": int((attacks.attack_success != (attacks.attacked_margin < 0)).sum()),
        "primary_deployability_violations": int((~remote_loc[remote_loc.method == PRIMARY_METHOD].deployable_without_true_label.astype(bool)).sum()),
        "checkpoints": len(list((output / "checkpoints").glob("*.pt"))),
    }

    independent_bootstrap_matches_decision = {
        "fingerprint": bool(all(abs(fingerprint[m][k] - decision["primary_remote_anomaly_minus_random"][m][k]) < 1e-12 for m in fingerprint for k in ("mean", "ci95_low", "ci95_high"))),
        "causal": bool(all(abs(causal[m][k] - decision["primary_remote_global_union_hybrid_minus_anomaly"][m][k]) < 1e-12 for m in causal for k in ("mean", "ci95_low", "ci95_high"))),
    }

    comparison = None
    if args.phase3_dir:
        old = json.loads((args.phase3_dir.resolve() / "decision.json").read_text(encoding="utf-8"))
        comparison = {
            "phase3_status": old["status"],
            "phase3b_status": decision["status"],
            "remote_clusters": [old["remote_successful_clusters"], decision["remote_successful_clusters"]],
            "remote_snapshots": [old["remote_successful_snapshots"], decision["remote_successful_snapshots"]],
            "fingerprint_auprc_delta": [old["primary_remote_anomaly_minus_random"]["edge_auprc"]["mean"], fingerprint["edge_auprc"]["mean"]],
            "fingerprint_recall_delta": [old["primary_remote_anomaly_minus_random"]["recall_at_b"]["mean"], fingerprint["recall_at_b"]["mean"]],
            "causal_recall_delta": [old["primary_remote_global_union_hybrid_minus_anomaly"]["recall_at_b"]["mean"], causal["recall_at_b"]["mean"]],
            "causal_margin_gain_delta": [old["primary_remote_global_union_hybrid_minus_anomaly"]["repaired_margin_gain"]["mean"], causal["repaired_margin_gain"]["mean"]],
        }

    audit = {
        "integrity": integrity,
        "remote_geometry": remote_geometry,
        "remote_slices": attack_slices,
        "remote_budgets": remote_budgets,
        "remote_absolute_methods": absolute,
        "remote_candidate_coverage": remote_coverage,
        "independent_fingerprint": fingerprint,
        "fingerprint_cluster_signs": fingerprint_signs,
        "fingerprint_dataset_seed_means": fingerprint_slices,
        "independent_causal": causal,
        "causal_cluster_signs": causal_signs,
        "causal_dataset_seed_means": causal_slices,
        "independent_bootstrap_matches_decision": independent_bootstrap_matches_decision,
        "phase3_to_phase3b": comparison,
    }
    rendered = json.dumps(audit, ensure_ascii=False, indent=2, default=lambda value: int(value))
    if args.write_json:
        args.write_json.resolve().write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
