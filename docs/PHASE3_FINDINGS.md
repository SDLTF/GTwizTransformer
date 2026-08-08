# Phase-3 findings: formal GraphGPS and remote structural attacks

Canonical result directory: `results/phase3_graphgps_holdout_20260808/`.

## Predeclared decision

**Remote attack viability is underpowered by the frozen rule, while every
measured primary localization effect is positive.**

Phase 3 produced 19 successful remote snapshots from 11 independent
model-target clusters across all three seeds. The protocol required at least
12 clusters across at least two seeds before a formal localization claim could
be made. The run therefore remains `remote_attack_underpowered`; the positive
intervals below are promising evidence, not a passed Phase-3 result.

## Architecture validation

The model is no longer the custom dense proxy used in Phases 0-2. It is a
traceable subclass of PyTorch Geometric 2.8 `GPSConv` with:

- random-walk structural encodings embedded alongside node features;
- a GIN local message-passing branch;
- exact global eight-head self-attention;
- four GPS processing layers with width 96.

Trace mode only exposes per-head attention and projected values. Unit tests
show that trace and ordinary logits agree and that batched graph logits match
single-graph evaluation.

## Attack viability and geometry

There were 288 attempted attack snapshots: two datasets, three seeds, two
attack types, six targets, and four budgets. Incident attacks succeeded on 69
snapshots. Remote attacks succeeded on 19 snapshots from 11 clusters.

| Remote slice | Successful clusters | Successful snapshots |
| --- | ---: | ---: |
| Cora, seed 4510 | 2 | 2 |
| Cora, seed 4511 | 1 | 1 |
| Cora, seed 4512 | 3 | 3 |
| CiteSeer, seed 4510 | 2 | 8 |
| CiteSeer, seed 4511 | 3 | 5 |

Remote success is budget-heavy: B=1/2/4/8 contribute 2/2/4/11 successful
snapshots. No successful remote snapshot contains an edge incident to the
target. Every remote edge instead joins a current one-hop target neighbor to a
different node, giving a non-incident two-hop perturbation path.

## Primary remote results

All intervals resample dataset/seed/attack-type/target clusters, averaging
nested budgets within each cluster.

| Predeclared comparison | Metric | Mean delta | Cluster-bootstrap 95% CI |
| --- | --- | ---: | ---: |
| all-layer anomaly minus random | AUPRC | **0.240** | **[0.167, 0.303]** |
|  | Recall@B | **0.151** | **[0.085, 0.222]** |
| global-union hybrid minus anomaly | Recall@B | **0.205** | **[0.068, 0.352]** |
|  | Repaired true-margin gain | **1.405** | **[0.225, 2.910]** |
|  | Normalized recovery | **0.619** | **[0.299, 0.941]** |
|  | Repair rate | 0.136 | [0.000, 0.364] |

The all-layer fingerprint has positive descriptive AUPRC and Recall deltas in
every one of the five dataset/seed slices that contains a successful remote
attack. The causal repaired-margin cluster delta is positive in nine of 11
clusters and negative in two. The effect is heterogeneous but not attributable
to a single dataset, seed, or outlier.

## Absolute remote localization

| Method | AUPRC | Recall@B | Repair rate | Repaired margin gain | Recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.025 | 0.013 | 0.053 | -0.018 | -0.036 |
| All-layer anomaly | 0.229 | 0.132 | 0.316 | 0.013 | 0.124 |
| Attention trajectory | 0.234 | 0.164 | 0.368 | 0.244 | 0.178 |
| Temporal anomaly | 0.303 | 0.211 | 0.526 | 0.302 | 0.294 |
| All-layer hybrid | 0.367 | 0.336 | 0.421 | 0.975 | 0.469 |
| Temporal-residual hybrid | **0.397** | **0.362** | 0.421 | 1.899 | 0.474 |
| Global-union hybrid (primary) | 0.341 | 0.309 | **0.474** | **1.931** | **0.613** |
| Global-union true-margin oracle | 0.419 | 0.414 | 0.684 | 2.872 | 1.016 |

Temporal residuals are the strongest deployable ranking view in this holdout,
but this is a secondary observation. The predeclared equal-weight global union
has candidate recall 0.684, below all-layer 0.743 and temporal residual 0.796.
Its causal margin remains strong because counterfactual reranking can recover
useful edges from an imperfect shortlist. A later development phase may alter
union weighting, but the present holdout cannot be reused to validate that
change.

The oracle gap remains material: the primary hybrid reaches Recall 0.309 and
repair rate 0.474, versus 0.414 and 0.684 for the true-margin oracle. Causal
reranking is still improvable even when a candidate edge is available.

## Incident-prior control works as intended

For incident attacks, target-incident hybrid achieves Recall 0.580 and repaired
margin gain 3.417, compared with 0.384 and 2.225 for the global-union hybrid.
This confirms the structural prior is powerful when its assumption matches the
attack.

For remote attacks, the target-incident hybrid has the same aggregate Recall as
the global union (0.309) but repair rate 0.105, repaired margin gain -1.019, and
normalized recovery -2.867. It tends to remove influential clean target edges
instead of the non-incident perturbation. The positive remote causal result is
therefore not a replay of the target-incidence shortcut exposed in Phase 2.

## GPU execution

The run evaluated 42,946 complete graph variants in 786 GraphGPS forward calls
with requested and realized graph batch size 64. No OOM fallback occurred.
Wall time was 46.6 seconds. Peak CUDA allocation was 429,673,472 bytes (about
410 MiB) and peak reservation was 1,082,130,432 bytes (about 1.01 GiB).

The low peak memory reflects the 192-node graph size and inference without
gradients, not a return to serial CPU execution. Attack candidates, RWSE powers,
and counterfactual graph variants were all processed in CUDA batches. Larger
batches could use more memory, but the current run is already throughput-bound
enough to finish all 42,946 graph evaluations in under one minute.

## Limitations

- The viability threshold was missed by one cluster, so the protocol blocks a
  formal positive claim.
- Remote attacks are two-hop and budget-heavy; this is not arbitrary remote
  subgraph manipulation.
- Clean test accuracy ranges from 0.400 to 0.842 across induced graphs. The
  GraphGPS hyperparameters were frozen rather than tuned per dataset; Cora seed
  4511 is particularly weak.
- Only RWSE and GraphGPS are tested. Graphormer, heat/diffusion encodings, and
  certificates remain untested in this phase.

## Next experiment

The next defensible step is a fresh Phase-3 replication, not adding one
convenient seed to this run. Freeze the present model, attacker, primary
global-union hybrid, and statistics; use three new seeds and increase the
number of boundary targets from six to ten to obtain adequate independent
remote clusters. Temporal-residual hybrid should remain a declared secondary
method because this holdout revealed its advantage.

If that replication passes, the project will have its first support for both a
layer-wise fingerprint and label-free causal localization in a standard graph
Transformer under non-incident attack. Only then should certificate or
adaptive anti-localization experiments begin.

