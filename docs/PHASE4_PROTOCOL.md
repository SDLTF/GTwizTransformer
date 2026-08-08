# Phase-4 protocol: adaptive anti-localization stress test

This protocol is frozen before any Phase-4 Cora or CiteSeer result is generated.
Phase 4 is a falsification test of the Phase-3b detector, not a detector-tuning
run. Engineering smoke tests may use synthetic graphs but cannot change the
objective, weights, seeds, endpoints, or decision rules below.

## Frozen model, data, and detector

Phase 4 keeps the formal GraphGPS configuration from Phase 3b:

- PyG `GPSConv`-compatible trace model;
- RWSE length 8 projected to 16 channels;
- GIN local branch, exact global eight-head attention;
- four layers, width 96, 192-node induced graphs;
- training for at most 100 epochs with patience 20;
- candidate pool 64 and CUDA graph batch size 64;
- remote two-hop edge additions only, with B in {1, 2, 4, 8}.

The formal fresh seeds are **4540, 4541, and 4542** on Cora and CiteSeer. Up to
fifteen correctly classified boundary-near targets are selected per model;
fewer are allowed only when the frozen model has fewer than fifteen correctly
classified eligible nodes. The frozen
detector is `global_union + hybrid_label_free`, with shortlist K=10B. It may
not be retrained or reweighted after viewing Phase-4 results.

## Paired attackers

Both attackers receive the same model, target true label, current graph,
candidate pool, target set, and budgets. Each baseline model is trained once;
the adaptive run reloads that exact checkpoint and target list. Clean margins,
target identities, and model-training metadata must match. Repeated CUDA
forwards may differ numerically, so clean margins use the same absolute
`3e-4` reconstruction tolerance already enforced for Phase-3 attack snapshots.

### Classification-only baseline

At each greedy step, the baseline adds the candidate with the largest target
cross-entropy loss, exactly as in Phase 3b.

### Adaptive anti-localization attacker

The adaptive attacker knows the detector and evaluates all candidate additions
with batched GraphGPS traces. For every candidate it computes:

- target cross-entropy loss `L`;
- clean-profile all-layer anomaly `A` for the proposed edge;
- clean-profile temporal-residual anomaly `T`;
- positive predicted-class counterfactual signal `C`: the post-addition
  predicted-class probability minus that class's probability before the edge
  is added, clipped below at zero.

Each vector is robustly standardized across the current candidate pool with
median and MAD. The frozen adaptive score is

`z(L) - 1.0 * mean(z(A), z(T), z(C))`.

The coefficient is fixed at 1.0 because the four terms are on the same robust-z
scale. No coefficient sweep is allowed on Cora or CiteSeer. A strength-zero
unit test must reproduce the classification-only greedy choices.

The attacker may use the target true label for its attack loss, as did the
baseline. It may use clean calibration-distribution parameters because this is
a white-box detector-aware stress test. The deployable detector still cannot
use the true label, paired clean graph, graph difference, or attack-edge IDs.

## GPU trace contract

Adaptive candidate traces must be evaluated in batches of up to 64 complete
graphs. Attention, value, hidden, and layer-logit tensors are computed on CUDA;
only the designated candidate edge's feature vector is transferred for
profile scoring. Serial per-candidate trace evaluation is forbidden except as
a unit-test reference. OOM fallback may reduce batch size without changing
candidate sets or scores.

## Statistical units and viability

Attack IDs are paired by dataset/seed/target/budget. Attack-utility effects use
all attempted snapshots and average the adaptive-minus-baseline success
indicator over budgets within each dataset/seed/target cluster.

Localization effects use only snapshots on which **both** attackers succeed.
Metrics are paired by attack ID and then averaged over budgets within the same
cluster. The adaptive comparison is viable only if joint successes cover at
least **18 model-target clusters**, all three seeds, and both datasets.

All intervals use 5,000 cluster-bootstrap resamples. Nested budgets are never
treated as independent experimental units.

## Predeclared decisions

### A. Attack-utility retention

The adaptive attack is noninferior if the 95% confidence-interval lower bound
for adaptive-minus-baseline attack success is above **-0.10**. This is an
absolute ten-percentage-point noninferiority margin.

### B. Primary localization evasion

On joint successful snapshots, adaptive evasion is supported only if the 95%
confidence-interval **upper bounds are below zero** for both adaptive-minus-
baseline AUPRC and Recall@B of the frozen global-union hybrid detector.

### C. End-to-end repair degradation

Repair-restoration rate and repaired true-margin gain are secondary. Strong
end-to-end evasion requires both upper bounds below zero; otherwise any passed
primary evasion claim is limited to localization quality.

### D. Result labels

- `adaptive_localization_evasion_supported`: viability, attack-utility
  noninferiority, and both primary localization reductions pass;
- `adaptive_evasion_with_attack_utility_cost`: both localization reductions
  pass but utility noninferiority fails;
- `adaptive_comparison_underpowered`: joint-success viability fails;
- `adaptive_evasion_not_supported`: the comparison is viable but both primary
  localization reductions do not pass.

Failure to support this particular adaptive attacker is not proof of detector
robustness. Likewise, a successful evasion result is specific to this white-box
objective and two-hop structural threat model.

## Execution amendment after an invalid engineering run

An initial 4530-4532 execution trained the nominally paired models twice.
Small CUDA nondeterminism changed several boundary target identities in the
second run. The mismatch was detected from live target lists before the run
completed; execution was stopped and those directories are excluded from all
Phase-4 statistics. No objective weight, endpoint, confidence rule, or
noninferiority margin was changed.

To remove this technical ambiguity, the formal run uses the next untouched
sequential seeds 4540-4542 and reloads the baseline checkpoint and targets for
the adaptive arm. The invalid run is documented separately in
`docs/PHASE4_INVALID_RUN.md` and remains on disk for auditability.
