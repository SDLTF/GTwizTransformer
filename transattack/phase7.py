from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import load_graph
from .gps_model import BatchTelemetry, evaluate_gps_trace
from .heuristic_search import heuristic_remote_search
from .phase6 import _load_model


OBJECTIVES = ("cross_entropy", "normalized_margin")
BEAM_WIDTHS = (1, 8)
CANDIDATE_STRATEGIES = ("single_rival", "multi_rival")
POOL_MODES = ("fixed", "adaptive")
POLICIES = ("exact", "within_budget")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-7 generalized heuristic search factorial")
    parser.add_argument("--source-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True, help="Dataset:seed entries")
    parser.add_argument("--targets-per-model", type=int, default=6)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--candidate-pool", type=int, default=128)
    parser.add_argument("--maximum-candidate-pool", type=int, default=512)
    parser.add_argument("--graph-batch-size", type=int, default=512)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--strict-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True, default=str) + "\n",
        encoding="utf-8",
    )


def _parse_models(values: list[str]) -> list[tuple[str, int]]:
    parsed: list[tuple[str, int]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"invalid model entry: {value}")
        dataset, seed = value.rsplit(":", 1)
        parsed.append((dataset.strip().lower(), int(seed)))
    if len(parsed) != len(set(parsed)):
        raise ValueError("duplicate model entries")
    return parsed


def _source_index(source_runs: list[Path]) -> tuple[dict[tuple[str, int], Path], dict[Path, dict[str, Any]]]:
    index: dict[tuple[str, int], Path] = {}
    configs: dict[Path, dict[str, Any]] = {}
    for source in source_runs:
        source = source.resolve()
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        configs[source] = config
        for dataset in config["datasets"]:
            for seed in config["seeds"]:
                key = (str(dataset).lower(), int(seed))
                if key in index:
                    raise ValueError(f"duplicate source model: {key}")
                index[key] = source
    return index, configs


def _even_margin_targets(metrics: pd.DataFrame, count: int, maximum_budget: int) -> list[int]:
    candidates = metrics[metrics.budget == maximum_budget][["target", "clean_margin"]].drop_duplicates("target")
    candidates = candidates[np.isfinite(candidates.clean_margin) & (candidates.clean_margin > 0)].copy()
    candidates = candidates.sort_values(["clean_margin", "target"]).reset_index(drop=True)
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} clean-correct targets available; need {count}")
    positions = np.rint(np.linspace(0, len(candidates) - 1, count)).astype(int)
    if len(set(positions.tolist())) != count:
        raise ValueError("target quantile selection produced duplicates")
    return [int(candidates.iloc[position].target) for position in positions]


def _edge_text(edges: tuple[tuple[int, int], ...]) -> str:
    return ";".join(f"{u}-{v}" for u, v in edges)


def _execution_id(objective: str, beam_width: int, strategy: str, pool_mode: str) -> str:
    return f"obj={objective}|beam={beam_width}|cand={strategy}|pool={pool_mode}"


def _config_id(execution_id: str, policy: str) -> str:
    return f"{execution_id}|policy={policy}"


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _aggregate(rows: pd.DataFrame, maximum_budget: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    endpoint = rows[rows.budget == maximum_budget].copy()
    grouped = endpoint.groupby(["config_id", "dataset"], as_index=False).agg(
        success_rate=("success", "mean"),
        mean_margin_progress=("margin_progress", "mean"),
        mean_evaluated_candidates=("evaluated_candidates", "mean"),
        targets=("target", "count"),
    )
    summary = grouped.groupby("config_id", as_index=False).agg(
        worst_dataset_success=("success_rate", "min"),
        macro_success=("success_rate", "mean"),
        worst_dataset_margin_progress=("mean_margin_progress", "min"),
        macro_margin_progress=("mean_margin_progress", "mean"),
        mean_evaluated_candidates=("mean_evaluated_candidates", "mean"),
        datasets=("dataset", "nunique"),
    )
    summary = summary.sort_values(
        [
            "worst_dataset_success",
            "macro_success",
            "worst_dataset_margin_progress",
            "macro_margin_progress",
            "mean_evaluated_candidates",
            "config_id",
        ],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    winner = summary.iloc[0]
    winner_id = str(winner.config_id)
    dataset_rows = grouped[grouped.config_id == winner_id].sort_values("dataset").to_dict("records")
    decision = {
        "status": "smoke_only" if rows.smoke.astype(bool).any() else "development_winner_selected",
        "selection_budget": int(maximum_budget),
        "selection_rule": [
            "maximize worst-dataset success rate",
            "maximize macro success rate",
            "maximize worst-dataset canonical margin progress",
            "maximize macro canonical margin progress",
            "minimize evaluated graph candidates",
            "lexicographic configuration ID",
        ],
        "winner": {key: _native(value) for key, value in winner.to_dict().items()},
        "winner_by_dataset": [
            {key: _native(value) for key, value in row.items()} for row in dataset_rows
        ],
        "configurations_compared": int(len(summary)),
    }
    return summary, decision


def main() -> int:
    args = _args()
    if args.targets_per_model < 1 or min(args.budgets) < 1:
        raise ValueError("targets and budgets must be positive")
    if args.maximum_candidate_pool < args.candidate_pool:
        raise ValueError("maximum candidate pool must cover the base pool")
    if args.beam_width < 2:
        raise ValueError("factorial beam width must be at least two")
    if args.strict_cuda and not torch.cuda.is_available():
        raise RuntimeError("strict CUDA requested but CUDA is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    models = _parse_models(args.models)
    source_index, source_configs = _source_index(args.source_runs)
    missing = [key for key in models if key not in source_index]
    if missing:
        raise ValueError(f"missing source checkpoints: {missing}")

    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"output exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    run_config = {
        **vars(args),
        "source_runs": [str(path.resolve()) for path in args.source_runs],
        "models": [f"{dataset}:{seed}" for dataset, seed in models],
        "output_dir": str(output),
        "objectives": OBJECTIVES,
        "beam_widths": [1, args.beam_width],
        "candidate_strategies": CANDIDATE_STRATEGIES,
        "pool_modes": POOL_MODES,
        "policies": POLICIES,
        "threat_model": "post-training remote edge additions; no candidate-node labels",
    }
    config_path = output / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "source_runs", "models", "targets_per_model", "budgets", "candidate_pool",
            "maximum_candidate_pool", "graph_batch_size", "beam_width", "device", "smoke",
            "validation_only",
        )
        for key in comparable_keys:
            if previous.get(key) != run_config.get(key):
                raise ValueError(f"resume configuration mismatch: {key}")
    else:
        _write_json(config_path, run_config)

    metrics_by_source = {
        source: pd.read_csv(source / "attack_metrics.csv") for source in source_configs
    }
    target_rows: list[dict[str, Any]] = []
    targets_by_model: dict[tuple[str, int], list[int]] = {}
    maximum_budget = max(args.budgets)
    for dataset, seed in models:
        source = source_index[(dataset, seed)]
        frame = metrics_by_source[source]
        subset = frame[(frame.dataset.astype(str).str.lower() == dataset) & (frame.seed == seed)]
        selected = _even_margin_targets(subset, args.targets_per_model, maximum_budget)
        targets_by_model[(dataset, seed)] = selected
        margins = subset.drop_duplicates("target").set_index("target").clean_margin
        for order, target in enumerate(selected):
            target_rows.append({
                "dataset": dataset,
                "seed": seed,
                "target": target,
                "quantile_order": order,
                "source_clean_margin": float(margins.loc[target]),
                "source_run": str(source),
            })
    selected_frame = pd.DataFrame(target_rows)
    selected_path = output / "selected_targets.csv"
    if selected_path.exists():
        prior_targets = pd.read_csv(selected_path)
        pd.testing.assert_frame_equal(prior_targets, selected_frame, check_dtype=False)
    else:
        selected_frame.to_csv(selected_path, index=False)

    results_path = output / "search_results.csv"
    existing = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    completed = set(existing.search_key.astype(str)) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []
    total_graphs = total_forwards = 0
    maximum_graphs = 0
    minimum_batch = args.graph_batch_size
    searches_completed = 0

    if args.validation_only:
        executions = [
            ("cross_entropy", 1, "single_rival", "fixed"),
            ("normalized_margin", args.beam_width, "multi_rival", "adaptive"),
        ]
    else:
        executions = list(itertools.product(
            OBJECTIVES, (1, args.beam_width), CANDIDATE_STRATEGIES, POOL_MODES
        ))
    for dataset, seed in models:
        source = source_index[(dataset, seed)]
        source_config = source_configs[source]
        data_root = args.data_root
        if str(data_root) == "data" and "data_root" in source_config:
            data_root = Path(source_config["data_root"])
        if not data_root.is_absolute():
            data_root = Path.cwd() / data_root
        graph = load_graph(dataset, data_root, int(source_config["nodes"]), seed)
        checkpoint = source / "checkpoints" / f"{dataset}_rwse_seed{seed}.pt"
        model = _load_model(checkpoint, graph, device)
        clean_trace = evaluate_gps_trace(model, graph, graph.edge_index, device)
        for target in targets_by_model[(dataset, seed)]:
            clean_row = clean_trace.logits[target].float()
            label = int(graph.y[target])
            rivals = clean_row.clone()
            rivals[label] = -float("inf")
            clean_margin = float(clean_row[label] - rivals.max())
            if clean_margin <= 0:
                raise RuntimeError(f"selected target is not clean-correct: {dataset}/{seed}/{target}")
            for objective, beam_width, strategy, pool_mode in executions:
                execution_id = _execution_id(objective, beam_width, strategy, pool_mode)
                search_key = f"{dataset}|{seed}|{target}|{execution_id}"
                if search_key in completed:
                    continue
                telemetry = BatchTelemetry(args.graph_batch_size, args.graph_batch_size)
                search_started = time.perf_counter()
                result = heuristic_remote_search(
                    model,
                    graph,
                    clean_trace,
                    target,
                    args.budgets,
                    objective,
                    beam_width,
                    strategy,
                    pool_mode,
                    args.candidate_pool,
                    args.maximum_candidate_pool,
                    device,
                    args.graph_batch_size,
                    telemetry,
                )
                elapsed = time.perf_counter() - search_started
                total_graphs += telemetry.graphs
                total_forwards += telemetry.forwards
                maximum_graphs = max(maximum_graphs, telemetry.maximum_graphs_per_forward)
                minimum_batch = min(minimum_batch, telemetry.minimum_resolved_batch_size)
                for policy, snapshots in (("exact", result.exact), ("within_budget", result.within_budget)):
                    for budget in args.budgets:
                        snapshot = snapshots.get(budget)
                        if snapshot is None:
                            raise RuntimeError(f"search stopped before budget {budget}: {search_key}")
                        rows.append({
                            "search_key": search_key,
                            "execution_id": execution_id,
                            "config_id": _config_id(execution_id, policy),
                            "dataset": dataset,
                            "seed": seed,
                            "target": target,
                            "budget": budget,
                            "objective": objective,
                            "beam_width": beam_width,
                            "candidate_strategy": strategy,
                            "pool_mode": pool_mode,
                            "policy": policy,
                            "candidate_pool": args.candidate_pool,
                            "maximum_candidate_pool": args.maximum_candidate_pool,
                            "clean_margin": clean_margin,
                            "attacked_margin": snapshot.margin,
                            "margin_progress": clean_margin - snapshot.margin,
                            "attack_score": snapshot.attack_score,
                            "objective_score": snapshot.objective_score,
                            "target_loss": snapshot.target_loss,
                            "prediction": snapshot.prediction,
                            "true_label": label,
                            "success": snapshot.success,
                            "used_edges": snapshot.used_edges,
                            "added_edges": _edge_text(snapshot.added_edges),
                            "expanded_states": result.expanded_states,
                            "evaluated_candidates": result.evaluated_candidates,
                            "adaptive_expansions": result.adaptive_expansions,
                            "requested_graph_batch_size": args.graph_batch_size,
                            "minimum_resolved_graph_batch_size": telemetry.minimum_resolved_batch_size,
                            "maximum_graphs_per_forward": telemetry.maximum_graphs_per_forward,
                            "batched_forward_calls": telemetry.forwards,
                            "search_elapsed_seconds": elapsed,
                            "smoke": args.smoke,
                        })
                pd.DataFrame(rows).to_csv(results_path, index=False)
                completed.add(search_key)
                searches_completed += 1
                print(
                    f"[phase7] {dataset}/seed{seed}/t{target} {execution_id} "
                    f"graphs={telemetry.graphs} batch={telemetry.maximum_graphs_per_forward} "
                    f"elapsed={elapsed:.2f}s",
                    flush=True,
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[phase7] model_complete={dataset}/seed{seed}", flush=True)

    results = pd.DataFrame(rows)
    expected_searches = len(models) * args.targets_per_model * len(executions)
    unique_searches = int(results.search_key.nunique())
    if unique_searches != expected_searches:
        raise RuntimeError(f"incomplete search grid: {unique_searches}/{expected_searches}")
    summary, decision = _aggregate(results, maximum_budget)
    if args.validation_only:
        decision["status"] = "frozen_validation_complete"
    summary.to_csv(output / "configuration_summary.csv", index=False)
    elapsed = time.perf_counter() - started
    gpu = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "requested_graph_batch_size": args.graph_batch_size,
        "minimum_resolved_graph_batch_size_this_invocation": minimum_batch,
        "maximum_graphs_per_forward_this_invocation": maximum_graphs,
        "batched_graph_evaluations_this_invocation": total_graphs,
        "batched_forward_calls_this_invocation": total_forwards,
        "searches_completed_this_invocation": searches_completed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "elapsed_seconds_this_invocation": elapsed,
    }
    decision["grid"] = {
        "models": len(models),
        "targets": len(target_rows),
        "search_executions": unique_searches,
        "reported_configurations": int(summary.config_id.nunique()),
        "rows": len(results),
    }
    decision["gpu"] = gpu
    _write_json(output / "decision.json", decision)
    _write_json(output / "gpu_telemetry.json", gpu)
    _write_json(output / "environment.json", {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    })
    (output / "summary.md").write_text("\n".join([
        "# GraphTransAttack Phase-7 heuristic factorial",
        "",
        f"Decision: **{decision['status']}**.",
        "",
        f"Winner: `{decision['winner']['config_id']}`.",
        "",
        "This is a frozen development comparison, not a confirmatory result.",
        "",
        "```json",
        json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=True),
        "```",
        "",
    ]), encoding="utf-8")
    print(
        f"[phase7] complete winner={decision['winner']['config_id']} "
        f"configs={len(summary)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
