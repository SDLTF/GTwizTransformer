from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("edge_aug", "feature_aug", "dshield_aug")


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


def _interval(frame: pd.DataFrame, column: str, seed: int) -> dict[str, float | int]:
    clustered = frame.groupby(["dataset", "seed", "target"], as_index=False)[column].mean()
    return _bootstrap(clustered[column].to_numpy(float), seed)


def _exact(left: dict, right: dict) -> bool:
    return all(abs(float(left[key]) - float(right[key])) < 1e-12 for key in ("mean", "ci95_low", "ci95_high"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--defense-dir", type=Path, required=True)
    parser.add_argument("--write-json", type=Path, required=True)
    args = parser.parse_args()

    attacks = pd.read_csv(args.attack_dir / "attack_metrics.csv")
    attacks["attack_success"] = _bool(attacks.attack_success)
    views = pd.read_csv(args.defense_dir / "view_metrics.csv")
    for column in ("raw_attack_success", "clean_correct", "defended_correct", "recovered"):
        views[column] = _bool(views[column])
    defense = json.loads((args.defense_dir / "decision.json").read_text(encoding="utf-8"))
    comparison = json.loads((args.comparison_dir / "decision.json").read_text(encoding="utf-8"))

    arm_counts = views.groupby("attack_id").arm.nunique()
    duplicates = int(views.duplicated(["attack_id", "arm"]).sum())
    raw = views[["attack_id", "raw_attack_success"]].drop_duplicates()
    attack_check = attacks[["attack_id", "attack_success"]].merge(raw, on="attack_id", validate="one_to_one")
    clean_variation = views.groupby(["cluster", "arm"]).agg(
        predictions=("clean_prediction", "nunique"),
        correctness=("clean_correct", "nunique"),
        disagreement=("clean_disagreement", "nunique"),
    )

    successful_ids = set(attacks[attacks.attack_success].attack_id)
    successful = views[views.attack_id.isin(successful_ids)].copy()
    recomputed: dict[str, dict] = {}
    exact: dict[str, dict] = {}
    clean_errors: dict[str, list[dict]] = {}
    for offset, arm in enumerate(ARMS):
        all_arm = views[views.arm == arm].copy()
        success_arm = successful[successful.arm == arm].copy()
        clean_unique = all_arm.sort_values("budget").drop_duplicates("cluster").copy()
        clean_unique["clean_utility_delta"] = clean_unique.clean_correct.astype(float) - 1.0
        result = {
            "clean_correctness": _interval(clean_unique, "clean_correct", 41001 + offset),
            "clean_utility_delta": _interval(clean_unique, "clean_utility_delta", 41101 + offset),
            "attack_recovery": _interval(success_arm, "recovered", 41201 + offset),
            "attacked_minus_clean_disagreement": _interval(success_arm, "delta_disagreement", 41301 + offset),
            "true_probability_gain": _interval(success_arm, "true_probability_gain", 41401 + offset),
        }
        recomputed[arm] = result
        exact[arm] = {key: _exact(value, defense["arms"][arm][key]) for key, value in result.items()}
        clean_errors[arm] = clean_unique[~clean_unique.clean_correct][
            ["dataset", "seed", "target", "cluster", "clean_prediction", "clean_disagreement"]
        ].to_dict(orient="records")

    paired_predictions = views.pivot(index="attack_id", columns="arm", values="defended_prediction")
    paired_recovery = views.pivot(index="attack_id", columns="arm", values="recovered").astype(bool)
    success_rows = paired_recovery.loc[sorted(successful_ids)] if successful_ids else paired_recovery.iloc[:0]
    success_by_seed_budget = (
        attacks.groupby(["seed", "budget"]).attack_success.agg(["count", "sum"]).reset_index().to_dict(orient="records")
    )
    recovery_by_arm_budget = (
        successful.groupby(["arm", "budget"]).recovered.agg(["count", "sum"]).reset_index().to_dict(orient="records")
    )

    audit = {
        "attack_snapshots": int(len(attacks)),
        "view_rows": int(len(views)),
        "expected_view_rows": int(len(attacks) * len(ARMS)),
        "duplicate_attack_arm_rows": duplicates,
        "all_attack_ids_have_three_arms": bool((arm_counts == len(ARMS)).all()),
        "attack_ids_equal": set(attacks.attack_id) == set(views.attack_id),
        "raw_success_matches_attack_file": bool((attack_check.attack_success == attack_check.raw_attack_success).all()),
        "clean_values_constant_across_budgets": bool((clean_variation == 1).all().all()),
        "successful_snapshots": int(len(successful_ids)),
        "successful_clusters": int(attacks[attacks.attack_success].cluster.nunique()),
        "successful_seeds": int(attacks[attacks.attack_success].seed.nunique()),
        "success_by_seed_budget": success_by_seed_budget,
        "recovery_by_arm_budget": recovery_by_arm_budget,
        "clean_errors": clean_errors,
        "edge_and_dshield_prediction_equal_all_snapshots": int(
            (paired_predictions.edge_aug == paired_predictions.dshield_aug).sum()
        ),
        "edge_and_dshield_prediction_total": int(len(paired_predictions)),
        "edge_and_dshield_recovery_equal_on_successes": bool(
            (success_rows.edge_aug == success_rows.dshield_aug).all()
        ),
        "feature_and_dshield_recovery_differences_on_successes": int(
            (success_rows.feature_aug != success_rows.dshield_aug).sum()
        ),
        "recomputed": recomputed,
        "independent_bootstrap_matches_decision": exact,
        "comparison_status": comparison["status"],
        "defense_status": defense["status"],
        "constraint_minimum_gain_ratio": comparison["constraint_diagnostics"]["minimum_observed_selected_gain_ratio"],
        "geometry": comparison["geometry"],
    }
    args.write_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
