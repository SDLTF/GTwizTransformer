from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .attack import adaptive_addition_attack, select_attack_targets
from .data import count_edges, load_graph, remove_edges, set_seed, undirected_pairs
from .localize import (
    LOCALIZERS,
    edge_features,
    first_divergence,
    fit_layer_profiles,
    fit_profiles,
    ranking_metrics,
    top_pairs,
)
from .model import evaluate_trace, train_model, true_margin
from .pe import PE_NAMES, build_pe, rms_scale


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cora"])
    parser.add_argument("--pes", nargs="+", choices=PE_NAMES, default=list(PE_NAMES))
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407])
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--targets", type=int, default=6)
    parser.add_argument("--candidate-pool", type=int, default=24)
    parser.add_argument("--pe-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--strict-cuda", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _device(requested: str, strict: bool) -> torch.device:
    available = torch.cuda.is_available()
    if strict and not available:
        raise RuntimeError("strict CUDA requested but CUDA is unavailable")
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if available else "cpu")


def _safe(value: object):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


@torch.no_grad()
def _repair_margin(model, graph, edge_index, pe_name: str, pe_dim: int, pe_scale: float, target: int, device: torch.device) -> tuple[float, int]:
    pe = build_pe(pe_name, edge_index, graph.num_nodes, pe_dim).values / pe_scale
    model.eval()
    logits = model(graph.x.to(device), pe.to(device))[target].detach().cpu()
    label = int(graph.y[target])
    return true_margin(logits, label), int(logits.argmax())


def _format_pairs(pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> str:
    return ";".join(f"{u}-{v}" for u, v in pairs)


def _decision(frame: pd.DataFrame) -> dict[str, Any]:
    successful = frame[frame["attack_success"]]
    full = successful[successful["localizer"] == "full_dynamics"]
    random = successful[successful["localizer"] == "random"]
    result: dict[str, Any] = {
        "successful_attack_snapshots": int(full.shape[0]),
        "status": "inconclusive",
        "reason": "fewer than three successful attack snapshots",
    }
    if len(full) < 3 or len(random) < 3:
        return result
    columns = ["attack_id", "edge_auprc", "recall_at_b", "margin_recovery", "repair_restored"]
    paired = full[columns].merge(random[columns], on="attack_id", suffixes=("_full", "_random"))
    deltas = {
        "auprc_delta_vs_random": float((paired.edge_auprc_full - paired.edge_auprc_random).mean()),
        "recall_delta_vs_random": float((paired.recall_at_b_full - paired.recall_at_b_random).mean()),
        "margin_recovery_delta_vs_random": float((paired.margin_recovery_full - paired.margin_recovery_random).mean()),
        "repair_rate_delta_vs_random": float((paired.repair_restored_full.astype(float) - paired.repair_restored_random.astype(float)).mean()),
    }
    result.update(deltas)
    positive = deltas["auprc_delta_vs_random"] > 0 and deltas["recall_delta_vs_random"] >= 0.10 and deltas["margin_recovery_delta_vs_random"] > 0.05
    negative = all(value <= 0 for value in deltas.values())
    if positive:
        result.update(status="positive_signal", reason="full dynamics beats random ranking and random repair on matched successful attacks")
    elif negative:
        result.update(status="negative_signal", reason="full dynamics does not beat random on any primary matched criterion")
    else:
        result.update(status="mixed_signal", reason="localization and counterfactual criteria do not agree")
    return result


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, float):
                text = f"{value:.6f}"
            else:
                text = str(value)
            rendered.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _summary(frame: pd.DataFrame, decision: dict[str, Any]) -> str:
    attack_models = (
        frame.drop_duplicates("attack_id")
        .groupby(["dataset", "pe", "budget"], as_index=False)
        .agg(attacks=("attack_id", "count"), success_rate=("attack_success", "mean"))
    )
    successful = frame[frame.attack_success]
    if successful.empty:
        localization = pd.DataFrame()
    else:
        localization = (
            successful.groupby(["dataset", "pe", "localizer"], as_index=False)
            .agg(
                snapshots=("attack_id", "count"),
                auprc=("edge_auprc", "mean"),
                auprc_lift=("auprc_lift", "mean"),
                recall_at_b=("recall_at_b", "mean"),
                repair_rate=("repair_restored", "mean"),
                margin_recovery=("margin_recovery", "mean"),
            )
        )
    return "\n".join(
        [
            "# GraphTransAttack Phase-0 summary",
            "",
            f"Decision: **{decision['status']}** - {decision['reason']}.",
            "",
            "The localizer never receives a paired clean graph or edge diff. Metrics below use true added edges only as labels.",
            "",
            "## Attack success",
            "",
            _markdown_table(attack_models),
            "",
            "## Successful-attack localization",
            "",
            _markdown_table(localization) if not localization.empty else "No successful attacks; localization hypothesis is not testable in this run.",
            "",
            "## Machine-readable decision",
            "",
            "```json",
            json.dumps(_safe(decision), ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def main() -> int:
    args = _args()
    if any(value <= 0 for value in args.budgets):
        raise ValueError("budgets must be positive")
    if args.hidden_dim % args.heads:
        raise ValueError("hidden dimension must be divisible by heads")
    device = _device(args.device, args.strict_cuda)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output_dir or Path("results") / f"phase0_{timestamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        device_resolved=str(device),
        output_dir=output,
        data_root=data_root,
        detector_inputs=["current_graph_trace", "clean_distribution_parameters"],
        detector_forbidden_inputs=["paired_clean_graph", "edge_symmetric_difference", "true_attack_edges"],
    )
    _write_json(output / "config.json", config)
    _write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    )
    print(f"[transattack] device={device} output={output}", flush=True)
    rows: list[dict[str, Any]] = []

    for dataset_name in args.datasets:
        for seed in args.seeds:
            set_seed(seed)
            graph = load_graph(dataset_name, data_root, args.nodes, seed)
            print(f"[transattack] dataset={graph.name} seed={seed} nodes={graph.num_nodes} edges={count_edges(graph.edge_index)}", flush=True)
            for pe_name in args.pes:
                clean_pe = build_pe(pe_name, graph.edge_index, graph.num_nodes, args.pe_dim)
                pe_scale = rms_scale(clean_pe.values)
                trained = train_model(
                    graph,
                    clean_pe.values / pe_scale,
                    device,
                    seed,
                    args.hidden_dim,
                    args.heads,
                    args.layers,
                    args.dropout,
                    args.epochs,
                    args.patience,
                    args.learning_rate,
                    args.weight_decay,
                )
                clean_trace = evaluate_trace(trained.model, graph, clean_pe.values / pe_scale, device)
                clean_pairs = undirected_pairs(graph.edge_index)
                clean_features = edge_features(clean_trace, clean_pairs)
                profiles = fit_profiles(clean_features)
                layer_profiles, layer_thresholds = fit_layer_profiles(clean_features)
                targets = select_attack_targets(clean_trace, graph, args.targets)
                torch.save(
                    {
                        "state_dict": trained.model.state_dict(),
                        "dataset": graph.name,
                        "pe": pe_name,
                        "seed": seed,
                        "pe_scale": pe_scale,
                        "best_epoch": trained.best_epoch,
                        "profile_parameters": {
                            name: {"mean": profile.mean, "precision": profile.precision}
                            for name, profile in profiles.items()
                        },
                    },
                    checkpoints / f"{graph.name.lower()}_{pe_name}_seed{seed}.pt",
                )
                print(f"[transattack] pe={pe_name} clean_test={trained.test_accuracy:.4f} targets={targets}", flush=True)
                for target in targets:
                    label = int(graph.y[target])
                    clean_prediction = int(clean_trace.logits[target].argmax())
                    clean_margin = true_margin(clean_trace.logits[target], label)
                    attacks = adaptive_addition_attack(
                        trained.model,
                        graph,
                        clean_trace,
                        pe_name,
                        args.pe_dim,
                        pe_scale,
                        target,
                        sorted(set(args.budgets)),
                        args.candidate_pool,
                        device,
                    )
                    for budget, snapshot in sorted(attacks.items()):
                        attack_id = f"{graph.name}-{pe_name}-{seed}-t{target}-b{budget}"
                        attacked_trace = evaluate_trace(
                            trained.model,
                            graph,
                            snapshot.pe.values / pe_scale,
                            device,
                        )
                        attacked_pairs = undirected_pairs(snapshot.edge_index)
                        attacked_features = edge_features(attacked_trace, attacked_pairs)
                        scores = {
                            name: profiles[name].score(attacked_features.groups[name])
                            for name in profiles
                        }
                        generator = torch.Generator().manual_seed(seed * 1000003 + target * 101 + budget)
                        scores["random"] = torch.rand(attacked_pairs.size(1), generator=generator)
                        divergence = first_divergence(
                            attacked_features,
                            layer_profiles,
                            layer_thresholds,
                            snapshot.added_edges,
                        )
                        oracle_margin = clean_margin
                        denominator = clean_margin - snapshot.margin
                        for localizer in LOCALIZERS:
                            local_scores = scores[localizer]
                            rank = ranking_metrics(attacked_pairs, local_scores, snapshot.added_edges, budget)
                            selected = top_pairs(attacked_pairs, local_scores, budget)
                            repaired_edges = remove_edges(snapshot.edge_index, graph.num_nodes, selected)
                            repaired_margin, repaired_prediction = _repair_margin(
                                trained.model,
                                graph,
                                repaired_edges,
                                pe_name,
                                args.pe_dim,
                                pe_scale,
                                target,
                                device,
                            )
                            recovery = (repaired_margin - snapshot.margin) / denominator if denominator > 1e-12 else float("nan")
                            row = {
                                "attack_id": attack_id,
                                "dataset": graph.name,
                                "pe": pe_name,
                                "seed": seed,
                                "target": target,
                                "target_label": label,
                                "budget": budget,
                                "localizer": localizer,
                                "num_nodes": graph.num_nodes,
                                "clean_edges": count_edges(graph.edge_index),
                                "attacked_edges": count_edges(snapshot.edge_index),
                                "train_accuracy": trained.train_accuracy,
                                "clean_test_accuracy": trained.test_accuracy,
                                "best_val_accuracy": trained.best_val_accuracy,
                                "best_epoch": trained.best_epoch,
                                "clean_prediction": clean_prediction,
                                "attacked_prediction": snapshot.prediction,
                                "clean_margin": clean_margin,
                                "attacked_margin": snapshot.margin,
                                "attack_target_loss": snapshot.target_loss,
                                "attack_success": snapshot.success,
                                "true_added_edges": _format_pairs(snapshot.added_edges),
                                "selected_edges": _format_pairs(selected),
                                "repaired_prediction": repaired_prediction,
                                "repaired_margin": repaired_margin,
                                "margin_recovery": recovery,
                                "repair_restored": repaired_prediction == label,
                                "oracle_margin": oracle_margin,
                                "oracle_restored": True,
                                **rank,
                                **divergence,
                            }
                            rows.append(row)
                        pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
                        print(
                            f"[transattack] pe={pe_name} target={target} B={budget} success={snapshot.success} "
                            f"full_R@B={rows[-1]['recall_at_b']:.3f} full_AP={rows[-1]['edge_auprc']:.3f}",
                            flush=True,
                        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no attack snapshots were produced")
    decision = _decision(frame)
    _write_json(output / "decision.json", decision)
    (output / "summary.md").write_text(_summary(frame, decision), encoding="utf-8")
    print(f"[transattack] complete rows={len(frame)} decision={decision['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
