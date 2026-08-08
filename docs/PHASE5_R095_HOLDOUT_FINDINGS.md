# Phase 5: rho=0.95 Fresh-Seed Holdout Findings

## Scope and evidence status

This experiment is a user-directed, fresh-seed holdout for the
classification-constrained stealth attacker at `rho=0.95`.  It is separate
from the Phase-5 development selection rule, which selected `rho=0.50` for the
eventual primary formal run.  The holdout protocol was frozen before running
seeds 4570--4574 and is documented in
`docs/PHASE5_R095_HOLDOUT_PROTOCOL.md`.

The evidence directories are:

- baseline: `results/phase5_r0p95_holdout_baseline_20260809/`;
- adaptive: `results/phase5_r0p95_holdout_adaptive_20260809/`;
- paired comparison: `results/phase5_r0p95_holdout_comparison_20260809/`.

The paired decision is
`adaptive_localization_evasion_supported`: the adaptive attack retains attack
utility while both preregistered localization metrics decrease.  It does not
support repair evasion.

## Integrity and viability

The run contains 960 paired attack snapshots (240 dataset-seed-target
clusters, four budgets each).  The baseline and adaptive attack IDs are
identical and have no duplicates.  Their joint-success subset contains 139
snapshots from 53 model-target clusters, all five seeds, and both datasets,
exceeding the frozen viability thresholds of 24 clusters, four seeds, and two
datasets.

Both arms have zero target-incidence, budget-cardinality, and nested-budget
violations.  The adaptive arm has zero classification-retention violations:
the minimum observed selected-gain ratio is 0.950023.  Its mean eligible
candidate count is 2.487, increasing from 1.729 at budget 1 to 3.114 at budget
8.

An independent audit reconstructed the paired samples and all cluster
bootstraps.  Its utility, severity, localization, and candidate-coverage
intervals exactly match `decision.json`; see `audit.json` in the comparison
directory.

## Primary results

All intervals below are 95% cluster-bootstrap intervals.  Localization and
repair effects are adaptive minus classification-only baseline on joint
successes.

| Quantity | Estimate | 95% CI | Interpretation |
|---|---:|---:|---|
| Attack-success difference | -0.00104 | [-0.00313, 0.00000] | Noninferior to the -0.05 margin |
| Attack-progress ratio | 0.99347 | [0.99083, 0.99581] | Almost all classification progress retained |
| Edge-set Jaccard | 0.73329 | [0.66492, 0.79960] | Adaptive path changes some, not most, selected edges |
| Edge AUPRC difference | -0.01302 | [-0.01975, -0.00660] | Significant localization evasion |
| Recall@B difference | -0.03007 | [-0.04756, -0.01376] | Significant localization evasion |
| Repair-restored difference | +0.02358 | [0.00000, 0.06604] | No repair evasion; point estimate improves |
| Repaired-margin-gain difference | +0.06191 | [-0.10391, 0.30559] | No resolved degradation |
| Margin-recovery difference | +0.02077 | [-0.02248, 0.08291] | No resolved degradation |
| Global-union candidate-recall difference | -0.01985 | [-0.03754, -0.00590] | Some evasion occurs before reranking |

The baseline has 140 successful snapshots and the adaptive arm has 139.  Only
one of 240 attempted clusters has a negative success-rate difference; 239 have
zero difference.  The progress ratio nevertheless shows a small real utility
cost: adaptive attacked margins are 0.01537 higher and target losses are
0.01109 lower on average.  At about 0.65% lost attack progress, that cost is
well inside the frozen noninferiority threshold.

For descriptive context, snapshot-weighted absolute primary scores on the
joint-success subset are:

| Metric | Baseline | rho=0.95 adaptive |
|---|---:|---:|
| Edge AUPRC | 0.64743 | 0.63657 |
| Recall@B | 0.59802 | 0.57554 |
| Repair restored | 0.80576 | 0.82014 |

The absolute detector therefore remains useful.  The supported claim is a
small deterioration in attack-edge ranking, not detector collapse.

## Where the effect occurs

The cluster signs show that the AUPRC reduction is broad but not universal:
29 of 53 clusters decrease, 17 are unchanged, and seven increase.  Recall@B
decreases in 15 clusters, is unchanged in 36, and increases in two.  Candidate
coverage decreases in only seven clusters and is unchanged in 46, suggesting
that a small number of candidate-set changes can propagate into a larger
Top-B ranking effect.

The budget slices identify the mechanism more clearly:

| Budget | Delta AUPRC | Delta Recall@B |
|---:|---:|---:|
| 1 | 0.00000 | 0.00000 |
| 2 | +0.00003 | 0.00000 |
| 4 | -0.01230 | -0.04348 |
| 8 | -0.01782 | -0.02123 |

The evasion appears only after the attacker has enough steps to choose among
near-optimal classification edges.  This agrees with the eligible-candidate
diagnostic, which grows with budget.

Dataset-seed slices are heterogeneous.  All four available CiteSeer slices
have negative AUPRC effects (-0.0095 to -0.0280), while the four Cora slices
are smaller (-0.0004 to -0.0128).  CiteSeer Recall@B decreases in three slices
and is unchanged in one; Cora decreases in two and is unchanged in two.  Two
dataset-seed combinations have no joint-success cluster and therefore do not
appear in this descriptive slice analysis.

## Relation to development results

The development run at `rho=0.95` retained similar utility (progress ratio
0.99133) but did not support joint localization evasion: its AUPRC difference
was -0.00753 and Recall@B difference was +0.00217 over 24 clusters.  The fresh
holdout changes Recall@B's direction and has more than twice as many joint
clusters.  This is evidence that the effect is small and sensitive to the
model/data realization even though the new holdout interval excludes zero.

Accordingly, the defensible paper-level conclusion is:

> Under this GraphGPS model and remote two-hop edge-addition attack, a
> classification-constrained (`rho=0.95`) white-box attacker can slightly but
> measurably reduce attack-edge localization while preserving nearly all
> attack progress.  It does not defeat causal edge-removal repair, and the
> localization effect is dataset- and budget-dependent.

The result should be replicated on an additional dataset or model family
before claiming general adaptive vulnerability.  The predeclared Phase-5
formal run at the development-selected `rho=0.50` also remains a separate,
unrun experiment.

## Compute

Both arms use a requested and realized graph batch size of 128 without
fallback.  Baseline execution evaluated 240,172 graphs in 2,205 batched
forwards, peaked at 0.573 GB allocated GPU memory, and took 163.2 seconds.  The
adaptive arm evaluated 240,096 graphs in 2,204 forwards, peaked at 3.537 GB
allocated (4.943 GB reserved), and took 291.8 seconds.
