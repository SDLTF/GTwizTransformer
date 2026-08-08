# Phase-4 findings: adaptive anti-localization stress test

Canonical valid directories:

- baseline: `results/phase4_paired_baseline_20260808/`;
- adaptive: `results/phase4_paired_adaptive_v2_20260808/`;
- paired comparison: `results/phase4_paired_comparison_v2_20260808/`.

The earlier 4530-4532 and partial first checkpoint-reuse directories are
excluded and documented in `docs/PHASE4_INVALID_RUN.md`.

## Predeclared decision

**The formal result is `adaptive_comparison_underpowered`.**

Both attackers succeed on 22 paired snapshots from 11 model-target clusters.
The clusters cover all three fresh seeds and both datasets, but the frozen
viability gate requires at least 18 clusters. The experiment therefore cannot
support or reject a formal adaptive-evasion claim.

The descriptive pattern is nevertheless informative: the adaptive objective
reliably reduces ranking precision and global-union candidate coverage, but it
does not reliably reduce Top-B recall or repair success. It also produces
weaker successful attacks. This implementation exposes a precision/utility
trade-off rather than a demonstrated end-to-end break.

## Frozen decision components

All effects are adaptive minus classification-only baseline. Attack utility is
clustered over all attempted targets; localization is paired only where both
attackers succeed.

| Gate or endpoint | Mean delta | Cluster-bootstrap 95% CI | Result |
| --- | ---: | ---: | --- |
| Attack success | -0.031 | [-0.054, -0.011] | Noninferior to the frozen -0.10 margin |
| Primary AUPRC | **-0.101** | **[-0.153, -0.059]** | Decreases |
| Primary Recall@B | -0.017 | [-0.091, 0.055] | Does not pass evasion criterion |
| Repair-restoration rate | +0.091 | [-0.197, 0.318] | No degradation |
| Repaired true-margin gain | -0.424 | [-1.687, 0.803] | Inconclusive |
| Normalized recovery | +0.051 | [-0.226, 0.304] | Inconclusive |

The attack-success interval is entirely below zero: the adaptive objective is
statistically less successful, but the loss is only 3.1 percentage points on
average and remains within the predeclared ten-point noninferiority margin.
Noninferiority does not mean equal utility.

The primary evasion rule requires the upper bounds for both AUPRC and Recall@B
to be below zero. Only AUPRC passes. Strong repair evasion also fails.

## Mechanism: what the adaptive attacker changes

The anti-localization objective is active rather than a no-op:

- global-union candidate recall decreases by **0.136**, 95% CI
  **[-0.239, -0.045]**;
- the frozen primary AUPRC decreases in all 11 joint-success clusters;
- even the raw all-layer anomaly AUPRC decreases by 0.061, 95% CI
  [-0.104, -0.032];
- the adaptive and baseline added-edge sets have mean snapshot Jaccard only
  0.209, so the two attackers choose materially different perturbations.

However, lower candidate coverage and AUPRC do not translate into consistent
Top-B failure. Primary Recall@B decreases in three clusters, is unchanged in
five, and increases in three. Repair restoration decreases in two, ties in
four, and increases in five. The detector often still places enough of the
remaining true edges inside Top-B to repair the prediction.

## Absolute performance on joint successes

These are snapshot-weighted descriptive means on the same 22 paired successful
snapshots.

| Frozen global-union hybrid | Baseline attack | Adaptive attack |
| --- | ---: | ---: |
| AUPRC | 0.336 | 0.254 |
| Recall@B | 0.273 | 0.256 |
| Repair rate | 0.273 | 0.409 |
| Repaired margin gain | 1.828 | 1.233 |
| Normalized recovery | 0.520 | 0.554 |

The apparent increase in repair rate is not evidence that adaptive attacks are
beneficial. A post-hoc severity audit shows that, even conditional on both
attacks succeeding, the adaptive attacked margin is 0.472 higher (closer to
the decision boundary), 95% CI [0.279, 0.648], and target loss is 0.300 lower,
95% CI [-0.411, -0.174]. The adaptive objective trades away attack severity to
hide edge signals, making its successes easier to reverse.

## Attack viability

There are 352 paired attempted snapshots, representing 88 candidate-viable
model-target clusters and four budgets. The classification baseline succeeds
on 33 snapshots from 12 clusters; the adaptive attacker succeeds on 22 from
11 clusters. Every adaptive success is also a baseline success.

| Budget | Baseline success rate | Adaptive success rate | Joint successes |
| ---: | ---: | ---: | ---: |
| 1 | 5.7% | 0.0% | 0 |
| 2 | 8.0% | 3.4% | 3 |
| 4 | 10.2% | 9.1% | 8 |
| 8 | 13.6% | 12.5% | 11 |

Successful snapshots are concentrated in four dataset-seed slices. Cora/4540
and CiteSeer/4541 produce no successful attack under either objective. This is
the direct reason the 18-cluster viability target is missed despite requesting
up to 15 targets per model.

## Pairing and integrity audit

The adaptive arm reloads the exact six baseline checkpoints and the target
lists derived from the baseline run. Both arms contain 352 identical attack
IDs with no duplicates. Maximum repeated-forward clean-margin difference is
1.91e-6, far below the 3e-4 reconstruction tolerance.

The baseline contains 528 localization rows (=33 successes x 16 methods) and
the adaptive arm contains 352 (=22 x 16), with 22 one-to-one primary-method
pairs. Independent code exactly reproduces the utility, localization, and
candidate-coverage bootstrap intervals. Both attackers have zero target-
incidence, budget-cardinality, and nested-budget violations.

## GPU execution

The classification baseline evaluates 47,417 graph variants in 783 forward
calls and takes 34.7 seconds. The adaptive arm evaluates 46,940 variants in
763 forward calls and takes 49.0 seconds. Both request and realize batch size
64 with no OOM fallback.

The adaptive batched trace path raises peak CUDA allocation from about 424 MiB
to **1.67 GiB** and reservation from about 880 MiB to **2.30 GiB**. This uses
substantially more of the GPU while keeping complete attention/value/hidden/
logit trace extraction batched rather than serial.

## Interpretation and next experiment

Phase 4 does not overturn Phase 3b. It shows a real weakness—ranking precision
and candidate coverage can be reduced by a detector-aware attacker—but the
frozen detector's Top-B recall and repair behavior are not broken by this
equal-weight objective. Because only 11 joint clusters are available, this is
not a positive robustness result either.

The next defensible development step is a **classification-constrained stealth
attacker**: first retain candidates within a fixed fraction of the best target-
loss gain, then minimize the stealth penalty only inside that near-optimal set.
This directly addresses the measured severity trade-off. Its constraint must
be developed on synthetic or designated development seeds and then frozen for
another fresh-seed comparison; the present 4540-4542 results cannot validate a
new coefficient or constraint.
