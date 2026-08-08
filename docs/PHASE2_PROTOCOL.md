# Phase-2 protocol: holdout replication and candidate coverage

This protocol is frozen before any Phase-2 holdout attack is generated or
scored. Phase 2 uses seeds 3410, 3411, and 3412, which were not used to choose
the Phase-1 reranker or candidate generators.

## Scope and architecture boundary

Phase 2 continues with the existing four-layer dense graph-transformer
mechanism proxy so that the new-seed result is directly comparable with Phases
0 and 1. It is not evidence about a standard Graphormer or GraphGPS
implementation. A positive Phase-2 result is a gate for, not a substitute for,
formal-architecture validation.

The attack is targeted and addition-only. Its current implementation adds only
edges incident to the known target node. The target identity is therefore a
legal detector input, but the true target label, clean counterpart graph, and
true added-edge set are forbidden. Target-incident performance cannot be
generalized to remote attacked subgraphs.

## Holdout data generation

- datasets: Cora and CiteSeer induced 128-node subgraphs;
- positional encodings: heat and random-walk;
- seeds: 3410, 3411, 3412;
- budgets: 1, 2, 4;
- six boundary-near, correctly classified targets per model;
- the Phase-0 training and greedy attack configuration is unchanged.

Only successful attack snapshots enter localization analysis. Budgets from the
same model/target are clustered together for bootstrap inference.

## Candidate generators

Every generator returns a nested deterministic ranking. Candidate sets are
prefixes of this ranking at K in {5B, 10B, 20B}.

1. `all_layer`: global all-layer full-channel anomaly ranking (Phase-1
   baseline).
2. `temporal_residual`: global inter-layer residual anomaly ranking.
3. `target_incident`: target-incident edges first, ordered by the sum of robust
   z-scores from the all-layer and temporal-residual views, followed by the
   remaining edges in the same combined order.
4. `multiview_union`: deterministic round-robin union of the preceding three
   ranked lists, skipping duplicates.

The union uses no attack labels. Because one branch uses target incidence, its
benefit is reported separately from the two fully global branches.

## Rerankers

Each candidate is removed once from the currently observed graph. PE and model
traces are recomputed. Candidates are reranked by:

- `cf_pred_label_free`: drop in the probability of the current attacked
  prediction (predeclared primary reranker);
- `hybrid_label_free`: robust-z base anomaly plus robust-z predicted-class drop
  (the Phase-1 primary, retained as a holdout comparator);
- `oracle_true_margin`: gain in true-label margin (evaluation-only ceiling).

The anomaly baseline is also evaluated without a candidate shortlist.

## Predeclared primary tests

1. **Holdout replication:** at `all_layer`, K=5B, `cf_pred_label_free` must
   improve both Recall@B and margin recovery over the anomaly baseline with
   cluster-bootstrap 95% confidence-interval lower bounds above zero.
2. **Candidate coverage:** at K=10B, `multiview_union` must improve candidate
   recall over `all_layer` with a cluster-bootstrap 95% lower bound above zero.
3. **End-to-end utility:** at K=10B, multiview-union plus
   `cf_pred_label_free` must improve both Recall@B and margin recovery over
   all-layer candidates plus the same reranker, again with lower bounds above
   zero.

The 5B and 20B union comparisons, target-incident-only results, AUPRC, repair
rate, hybrid comparison, dataset/PE slices, and oracle results are secondary.
No threshold or method will be changed after inspecting holdout outcomes.

## Leakage contract

Deployable scores may use the attacked graph, the known target node, the
current predicted class, current model traces, and distribution parameters fit
on clean calibration traces. They may not use the paired clean graph, an edge
diff, true attack edges, or the true class. The true class is used only for
evaluation and the explicitly marked oracle.

