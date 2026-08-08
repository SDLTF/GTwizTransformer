from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the Phase-5 retention ratio using development comparisons only")
    parser.add_argument("--comparisons", nargs="+", required=True, help="RATIO=COMPARISON_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-joint-clusters", type=int, default=8)
    parser.add_argument("--minimum-success-delta", type=float, default=-0.05)
    parser.add_argument("--minimum-progress-ratio", type=float, default=0.85)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    for specification in args.comparisons:
        ratio_text, directory_text = specification.split("=", maxsplit=1)
        ratio = float(ratio_text)
        directory = Path(directory_text).resolve()
        decision = json.loads((directory / "decision.json").read_text(encoding="utf-8"))
        success_delta = decision["attack_utility"]["adaptive_minus_baseline_success"]["mean"]
        progress_ratio = decision["attack_utility"]["adaptive_over_baseline_attack_progress"]["mean"]
        delta_auprc = decision["primary_localization_adaptive_minus_baseline"]["edge_auprc"]["mean"]
        delta_recall = decision["primary_localization_adaptive_minus_baseline"]["recall_at_b"]["mean"]
        joint_clusters = decision["joint_viability"]["clusters"]
        eligible = (
            joint_clusters >= args.minimum_joint_clusters
            and success_delta >= args.minimum_success_delta
            and progress_ratio >= args.minimum_progress_ratio
        )
        rows.append({
            "ratio": ratio,
            "comparison_dir": str(directory),
            "joint_clusters": joint_clusters,
            "success_delta": success_delta,
            "attack_progress_ratio": progress_ratio,
            "delta_auprc": delta_auprc,
            "delta_recall_at_b": delta_recall,
            "localization_score": delta_auprc + delta_recall,
            "development_utility_eligible": eligible,
        })

    eligible_rows = [row for row in rows if row["development_utility_eligible"]]
    if eligible_rows:
        selected = min(eligible_rows, key=lambda row: (row["localization_score"], -row["ratio"]))
        status = "retention_ratio_frozen"
        selected_ratio: float | None = float(selected["ratio"])
    else:
        selected = None
        status = "no_ratio_passed_development_utility_gate"
        selected_ratio = None
    payload = {
        "status": status,
        "selected_ratio": selected_ratio,
        "selection_rule": {
            "minimum_joint_clusters": args.minimum_joint_clusters,
            "minimum_success_delta": args.minimum_success_delta,
            "minimum_attack_progress_ratio": args.minimum_progress_ratio,
            "optimization": "minimize delta_auprc + delta_recall_at_b; ties prefer higher retention ratio",
        },
        "selected": selected,
        "candidates": sorted(rows, key=lambda row: row["ratio"]),
    }
    (output / "selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(
        "\n".join([
            "# Phase-5 development ratio selection",
            "",
            f"Decision: **{status}**.",
            "",
            f"Selected ratio: `{selected_ratio}`.",
            "",
            "```json",
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
            "```",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"[phase5-dev] status={status} selected_ratio={selected_ratio}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
