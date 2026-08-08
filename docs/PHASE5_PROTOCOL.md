# Phase-5 protocol: classification-constrained adaptive evasion

This protocol is frozen before inspecting any Phase-5 development result. Phase
5 asks whether a detector-aware attacker can suppress internal-trajectory edge
localization without achieving apparent stealth merely by weakening the
underlying classification attack.

## Attack objective

The paired baseline greedily selects the candidate edge with maximum target
cross-entropy loss. The adaptive arm uses the same model, target, candidate
pool, graph and budget. At every greedy step it:

1. computes the target-loss gain of every candidate relative to the current
   graph;
2. retains candidates whose positive gain is at least a fixed fraction `rho`
   of the best positive gain; when no candidate has positive gain, only the
   maximum-loss candidate is retained;
3. among retained candidates, minimizes the equal-weight robust-z mean of the
   all-layer anomaly, temporal-residual anomaly and positive predicted-class
   counterfactual signal.

At `rho=1`, the implementation must reproduce the classification-only greedy
edge sequence exactly. Every saved snapshot records the minimum realized gain
ratio and eligible-candidate count so the constraint can be audited.

## Development-only ratio selection

The development seeds are 4550, 4551 and 4552 on Cora and CiteSeer, with up to
20 correctly classified targets per model. The candidate pool and graph batch
size are both 128. The tested ratios are 0.50, 0.70, 0.85 and 0.95.

A ratio is development-eligible only when paired joint successes cover at
least eight model-target clusters, adaptive-minus-baseline attack success is at
least -0.05, and mean adaptive/baseline attack progress is at least 0.85.
Among eligible ratios, freeze the one minimizing
`delta(AUPRC) + delta(Recall@B)` for the frozen
`global_union+hybrid_label_free` localizer. Exact ties prefer the higher ratio.
If no ratio passes, no formal Phase-5 run is performed.

Development data select only `rho`; they cannot support the final claim.

## Fresh formal comparison

If a ratio is frozen, the formal seeds are 4560 through 4564 on both datasets,
with up to 30 targets per model. All other architecture and threat-model
settings remain GraphGPS with RWSE length 8, GIN local processing, exact global
eight-head attention, four layers of width 96, 192 nodes, remote two-hop edge
additions and budgets 1, 2, 4 and 8.

The formal comparison is viable with at least 24 joint-success model-target
clusters, at least four seeds and both datasets. Cluster bootstrap intervals
use 5,000 resamples; budgets from one target are never independent units.

Attack utility is retained only if the lower 95% bound of the paired attack-
success difference exceeds -0.05 and the lower bound of the joint-success
adaptive/baseline attack-progress ratio exceeds 0.85.

Primary localization evasion requires upper 95% bounds below zero for both
AUPRC and Recall@B. Strong end-to-end evasion additionally requires upper
bounds below zero for repair-restoration rate and repaired true-margin gain.

Failure of this attacker is not proof of robustness; success is specific to
the stated white-box, remote structural threat model.
