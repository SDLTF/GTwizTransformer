# Phase-3b protocol: fresh formal-GraphGPS replication

This protocol is frozen before generating any Phase-3b result. Phase 3b is an
independent replication of the Phase-3 non-incident result, not an extension
of the original holdout and not a method-development run.

## Frozen model and attack

Phase 3b keeps the complete Phase-3 implementation unchanged:

- PyTorch Geometric `GPSConv`-compatible trace model;
- RWSE length 8 projected to 16 channels;
- GIN local branch and exact global eight-head attention;
- four layers, hidden width 96, and 192-node induced graphs;
- incident and non-incident two-hop remote attacks reported separately;
- attack budgets B in {1, 2, 4, 8}, candidate pool 64;
- CUDA graph batching of 64;
- candidate shortlist size K=10B.

The fresh seeds are **4520, 4521, and 4522** on both Cora and CiteSeer. Ten
correctly classified boundary-near targets are attacked per model, increased
from six only to raise the number of independent model-target clusters. Model
training remains 100 epochs with patience 20. No Phase-3 checkpoint, target,
or successful attack is reused.

## Frozen primary methods

The fingerprint test remains all-layer anomaly versus deterministic random.
The causal test remains `global_union + hybrid_label_free` versus all-layer
anomaly. The temporal-residual hybrid, although descriptively strongest in
Phase 3, remains secondary and cannot determine the Phase-3b conclusion.

The localizer receives the attacked graph, known target, current predicted
class, current traces, and clean calibration-distribution parameters. It may
not receive the paired clean graph, graph difference, added-edge identities,
or true target label. True labels and true added edges are used only for
evaluation; the true-margin method remains an explicitly marked oracle.

## Predeclared viability and primary decisions

Only successful remote attack snapshots enter the primary localization tests.
Budgets are averaged within dataset/seed/attack-type/target clusters before
cluster bootstrap resampling.

Remote viability requires at least **18 successful remote model-target
clusters**, with successful clusters represented in **all three seeds** and
in **both datasets**. Failure of any part of this rule makes the replication
underpowered regardless of effect estimates.

Subject to viability:

1. the fingerprint is supported only if the 95% confidence-interval lower
   bounds for both AUPRC and Recall@B improvements over random are above zero;
2. causal localization is supported only if the lower bounds for both Recall@B
   and repaired true-margin-gain improvements over anomaly are above zero;
3. the joint Phase-3b claim passes only if both tests pass.

Normalized recovery and repair rate are secondary endpoints. Incident attacks,
temporal-residual results, and the true-label oracle are descriptive controls
and cannot rescue a failed primary decision.

## Interpretation

A pass is a fresh-seed replication for this GraphGPS architecture and the
specified two-hop remote attack. It does not establish a Graphormer result,
architecture independence, arbitrary remote-subgraph detection, robustness
certification, or resistance to an adaptive anti-localization attacker.
