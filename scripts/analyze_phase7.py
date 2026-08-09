from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WINNER = "obj=normalized_margin|beam=8|cand=multi_rival|pool=adaptive|policy=within_budget"
CONFIGS = {
    "winner": WINNER,
    "legacy_greedy": "obj=cross_entropy|beam=1|cand=single_rival|pool=fixed|policy=exact",
    "greedy_within_budget": "obj=cross_entropy|beam=1|cand=single_rival|pool=fixed|policy=within_budget",
    "replace_margin_with_ce": "obj=cross_entropy|beam=8|cand=multi_rival|pool=adaptive|policy=within_budget",
    "replace_beam_with_greedy": "obj=normalized_margin|beam=1|cand=multi_rival|pool=adaptive|policy=within_budget",
    "replace_multi_with_single": "obj=normalized_margin|beam=8|cand=single_rival|pool=adaptive|policy=within_budget",
    "replace_adaptive_with_fixed": "obj=normalized_margin|beam=8|cand=multi_rival|pool=fixed|policy=within_budget",
    "replace_within_with_exact": "obj=normalized_margin|beam=8|cand=multi_rival|pool=adaptive|policy=exact",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    rows = pd.read_csv(run_dir / "search_results.csv")
    rows["success"] = rows.success.astype(str).str.lower().map({"true": True, "false": False})
    maximum_budget = int(rows.budget.max())

    selected = rows[rows.config_id.isin(CONFIGS.values())].copy()
    selected["comparison"] = selected.config_id.map({value: key for key, value in CONFIGS.items()})
    curves = selected.groupby(["comparison", "dataset", "budget"], as_index=False).agg(
        success_rate=("success", "mean"),
        mean_margin_progress=("margin_progress", "mean"),
        mean_used_edges=("used_edges", "mean"),
        mean_evaluated_candidates=("evaluated_candidates", "mean"),
        targets=("target", "count"),
    )
    curves.to_csv(run_dir / "budget_curves.csv", index=False)

    endpoint = selected[selected.budget == maximum_budget]
    by_dataset = endpoint.groupby(["comparison", "dataset"], as_index=False).agg(
        success_rate=("success", "mean"),
        mean_margin_progress=("margin_progress", "mean"),
        mean_used_edges=("used_edges", "mean"),
        mean_evaluated_candidates=("evaluated_candidates", "mean"),
    )
    macro = by_dataset.groupby("comparison", as_index=False).agg(
        worst_dataset_success=("success_rate", "min"),
        macro_success=("success_rate", "mean"),
        worst_dataset_margin_progress=("mean_margin_progress", "min"),
        macro_margin_progress=("mean_margin_progress", "mean"),
        mean_used_edges=("mean_used_edges", "mean"),
        mean_evaluated_candidates=("mean_evaluated_candidates", "mean"),
    )
    winner = macro[macro.comparison == "winner"].iloc[0]
    for column in (
        "worst_dataset_success", "macro_success", "worst_dataset_margin_progress",
        "macro_margin_progress", "mean_used_edges", "mean_evaluated_candidates",
    ):
        macro[f"delta_vs_winner_{column}"] = macro[column] - winner[column]
    macro.to_csv(run_dir / "component_ablation.csv", index=False)

    keys = ["dataset", "seed", "target", "budget"]
    winner_rows = selected[selected.comparison == "winner"].set_index(keys)
    baseline_rows = selected[selected.comparison == "legacy_greedy"].set_index(keys)
    paired = winner_rows[["success", "margin_progress", "used_edges"]].join(
        baseline_rows[["success", "margin_progress", "used_edges"]],
        lsuffix="_winner",
        rsuffix="_baseline",
        validate="one_to_one",
    ).reset_index()
    paired["margin_progress_delta"] = paired.margin_progress_winner - paired.margin_progress_baseline
    paired["new_success"] = paired.success_winner & ~paired.success_baseline
    paired["lost_success"] = ~paired.success_winner & paired.success_baseline
    paired.to_csv(run_dir / "winner_vs_legacy_pairs.csv", index=False)

    endpoint_pairs = paired[paired.budget == maximum_budget]

    def paired_interval(frame: pd.DataFrame, seed: int) -> dict[str, float]:
        values = frame.margin_progress_delta.to_numpy(float)
        rng = np.random.default_rng(seed)
        sampled = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
        success_delta = frame.success_winner.astype(float).to_numpy() - frame.success_baseline.astype(float).to_numpy()
        sampled_success = success_delta[
            rng.integers(0, len(success_delta), size=(5000, len(success_delta)))
        ].mean(axis=1)
        return {
            "margin_progress_delta_mean": float(values.mean()),
            "margin_progress_delta_ci95_low": float(np.quantile(sampled, 0.025)),
            "margin_progress_delta_ci95_high": float(np.quantile(sampled, 0.975)),
            "success_rate_delta": float(success_delta.mean()),
            "success_rate_delta_ci95_low": float(np.quantile(sampled_success, 0.025)),
            "success_rate_delta_ci95_high": float(np.quantile(sampled_success, 0.975)),
        }

    payload = {
        "winner": WINNER,
        "maximum_budget": maximum_budget,
        "targets": int(len(endpoint_pairs)),
        "new_successes": int(endpoint_pairs.new_success.sum()),
        "lost_successes": int(endpoint_pairs.lost_success.sum()),
        "winner_successes": int(endpoint_pairs.success_winner.sum()),
        "legacy_successes": int(endpoint_pairs.success_baseline.sum()),
        "targets_with_better_margin_progress": int((endpoint_pairs.margin_progress_delta > 1e-9).sum()),
        "targets_with_equal_margin_progress": int((endpoint_pairs.margin_progress_delta.abs() <= 1e-9).sum()),
        "mean_paired_margin_progress_delta": float(endpoint_pairs.margin_progress_delta.mean()),
        "paired_bootstrap_5000": paired_interval(endpoint_pairs, 47007),
        "mean_graph_evaluation_ratio_winner_over_legacy": float(
            endpoint[endpoint.comparison == "winner"].evaluated_candidates.mean()
            / endpoint[endpoint.comparison == "legacy_greedy"].evaluated_candidates.mean()
        ),
        "by_dataset": {},
    }
    for dataset, frame in endpoint_pairs.groupby("dataset"):
        payload["by_dataset"][dataset] = {
            "targets": int(len(frame)),
            "winner_successes": int(frame.success_winner.sum()),
            "legacy_successes": int(frame.success_baseline.sum()),
            "new_successes": int(frame.new_success.sum()),
            "lost_successes": int(frame.lost_success.sum()),
            "mean_paired_margin_progress_delta": float(frame.margin_progress_delta.mean()),
            "paired_bootstrap_5000": paired_interval(frame, 47100 + len(payload["by_dataset"])),
        }
    (run_dir / "paired_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
