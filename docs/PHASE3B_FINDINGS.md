# Phase-3b findings: fresh formal-GraphGPS replication

Canonical result directory:
`results/phase3b_graphgps_replication_20260808/`.

## Predeclared decision

**Phase 3b passes its frozen joint decision rule.** The result status is
`formal_graphgps_fingerprint_and_causal_localization_supported`.

The run produced 48 successful remote snapshots from 20 independent
model-target clusters, represented in all three fresh seeds and both datasets.
This exceeds the preregistered viability requirement of 18 clusters, three
seeds, and two datasets. Conditional on that viability gate, both metrics in
the dynamics-fingerprint test and both primary metrics in the label-free
causal-localization test have cluster-bootstrap 95% confidence-interval lower
bounds above zero.

This is the first fresh-seed replication in this project that supports both
claims in a standard GraphGPS architecture under the specified non-incident
two-hop structural attack. It is not a claim about every graph Transformer or
an arbitrary remote attacker.

## Primary remote tests

Budgets are averaged within dataset/seed/attack-type/target clusters before
bootstrap resampling, so four nested budget snapshots are not counted as four
independent observations.

| Predeclared comparison | Metric | Mean cluster delta | Cluster-bootstrap 95% CI |
| --- | --- | ---: | ---: |
| all-layer anomaly minus random | AUPRC | **0.329** | **[0.228, 0.435]** |
|  | Recall@B | **0.301** | **[0.220, 0.384]** |
| global-union hybrid minus anomaly | Recall@B | **0.364** | **[0.235, 0.481]** |
|  | Repaired true-margin gain | **0.669** | **[0.309, 1.005]** |
|  | Normalized recovery (secondary) | 0.266 | [-0.252, 0.653] |
|  | Repair-rate change (secondary) | 0.321 | [0.096, 0.525] |

The normalized-recovery interval crosses zero, but this does not contradict
the frozen decision. Phase 3 made unnormalized repaired margin primary because
the recovery ratio is unstable when the clean-to-attacked margin denominator
is small. The unnormalized causal endpoint and Recall@B both replicate.

## Absolute remote performance

The following are snapshot-weighted descriptive means; they are not substitutes
for the cluster-weighted primary intervals above.

| Method | AUPRC | Recall@B | Repair rate | Repaired margin gain | Recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.034 | 0.018 | 0.104 | 0.087 | 0.050 |
| All-layer anomaly | 0.430 | 0.357 | 0.542 | 0.606 | 0.557 |
| Temporal anomaly | 0.438 | 0.378 | 0.521 | 0.597 | 0.592 |
| All-layer hybrid | 0.725 | 0.664 | 0.854 | 1.146 | 0.653 |
| Temporal-residual hybrid (secondary) | 0.758 | 0.724 | 0.812 | 1.016 | 0.041 |
| Global-union hybrid (primary) | **0.779** | **0.729** | **0.896** | **1.223** | **0.900** |
| Target-incident hybrid (control) | 0.616 | 0.568 | 0.417 | -0.140 | -3.208 |
| Global-union true-margin oracle | 0.795 | 0.758 | 0.958 | 1.388 | 1.208 |

The deployable primary method closes most of the descriptive gap to the
true-label oracle: Recall@B is 0.729 versus 0.758 and repair rate is 0.896
versus 0.958. This comparison is descriptive and was not given a noninferiority
test.

The target-incident control again behaves diagnostically. Its candidate list
eventually contains many true remote edges at large K, but its hybrid selection
frequently removes influential clean target edges: repaired margin gain is
negative and recovery is strongly negative. The primary result is therefore
not explained by the target-incidence shortcut found in Phase 2.

## Attack viability, geometry, and candidate coverage

There were 480 attempted snapshots: two datasets, three seeds, two threat
models, ten targets, and four budgets. Of these, 106 incident and 48 remote
snapshots succeeded. The 48 remote successes contain 241 added edges; there
are zero target-incidence violations, zero budget-cardinality violations, and
zero nesting violations across budgets.

| Remote slice | Successful clusters | Successful snapshots |
| --- | ---: | ---: |
| Cora, seed 4520 | 3 | 12 |
| Cora, seed 4521 | 5 | 9 |
| Cora, seed 4522 | 0 | 0 |
| CiteSeer, seed 4520 | 0 | 0 |
| CiteSeer, seed 4521 | 5 | 11 |
| CiteSeer, seed 4522 | 7 | 16 |

Successful remote snapshots by B=1/2/4/8 are 5/8/15/20 out of 60 attempts at
each budget. The attack remains budget-dependent, although it is less
concentrated at B=8 than in Phase 3.

On successful remote snapshots, candidate recall is 0.846 for all-layer,
0.885 for temporal residual, and **0.906 for the frozen global union**. This
reverses the candidate-coverage weakness seen in Phase 3 and helps explain the
primary method's much higher absolute Recall@B.

## Replication strength and heterogeneity

Relative to the original Phase-3 holdout, successful remote clusters increase
from 11 to 20 and snapshots from 19 to 48. The cluster-mean effects remain in
the same direction:

| Primary effect | Phase 3 | Phase 3b |
| --- | ---: | ---: |
| Fingerprint AUPRC delta | 0.240 | 0.329 |
| Fingerprint Recall@B delta | 0.151 | 0.301 |
| Causal Recall@B delta | 0.205 | 0.364 |
| Causal repaired-margin delta | 1.405 | 0.669 |

The causal margin effect is smaller than in Phase 3 but has a positive lower
confidence bound on fresh seeds. This is stronger replication evidence than
simply appending a convenient seed to the first run.

The result is nevertheless heterogeneous. Fingerprint AUPRC improves in 19 of
20 clusters and fingerprint Recall@B improves in 18, with two ties. Causal
Recall@B improves in 17 clusters, ties in one, and decreases in two; repaired
margin improves in 18 and decreases in two. Both negative causal clusters are
concentrated enough that the Cora/4521 dataset-seed slice has negative mean
Recall and margin deltas, while the other three contributing slices are
positive. Also, Cora/4522 and CiteSeer/4520 have no successful remote attacks.
The preregistered rule requires representation across every seed and dataset,
not success in all six combinations, so the formal pass is valid, but it should
not be presented as uniform behavior.

Clean test accuracy also varies substantially across the six induced graphs:
Cora is 0.475/0.750/0.385 and CiteSeer is 0.605/0.718/0.897 for seeds
4520/4521/4522. The experiment deliberately freezes training rather than
tuning away this variance.

## GPU execution and integrity audit

The CUDA run evaluated 72,771 complete graph variants in 1,336 batched forward
calls. Requested and realized graph batch size are both 64, with no OOM
fallback. Wall time is 65.2 seconds; peak allocation is about 416 MiB and peak
reservation about 1.62 GiB. GPU occupancy is still bounded by the 192-node
graphs, but the workload is genuinely batched and the whole replication
finishes in just over one minute.

An independent CSV audit reproduces every reported bootstrap estimate exactly.
It verifies 480 unique attack rows, 2,464 localization rows (=154 successful
snapshots x 16 methods), 616 candidate-coverage rows (=154 x 4 generators),
10,065 unique counterfactual edge rows, six checkpoints, no duplicate keys,
no localization rows for failed attacks, and no primary-method deployability
flag violations. All selected targets have positive clean true margin, and
the recorded attack-success flags agree with attacked-margin sign.

## What the result establishes—and what it does not

The experiment now supports the core idea that a non-incident GraphGPS attack
leaves a repeatable layer-wise edge fingerprint and that label-free
counterfactual reranking can identify removals that causally undo the attack.
The evidence is specific to RWSE + GIN + exact-attention GraphGPS on 192-node
Cora/CiteSeer induced graphs and this two-hop greedy attacker.

The next high-value falsification experiment should be an adaptive
anti-localization attacker. It should jointly maximize target loss while
penalizing the same layer-wise anomaly and predicted-class counterfactual
signals used by the detector, with a clean classification-only attacker kept
as the baseline. Passing that test would say more about robustness than adding
more ordinary seeds. Certification, Graphormer replication, and larger-graph
scaling remain separate later gates.
