# Phase-2 findings: holdout replication and candidate coverage

Canonical result directory: `results/phase2_holdout_20260808/`.

## Predeclared decision

**The holdout reranker replication failed or was inconclusive. Candidate
coverage improved, but the predeclared end-to-end criterion remained
inconclusive.**

Phase 2 trained and attacked new seeds 3410, 3411, and 3412 after the protocol
was frozen. There were 34 successful snapshots from 14 model-target clusters
out of 216 attempted snapshots. All 34 successes used RWPE; 30 were on Cora
and four on CiteSeer. HeatPE therefore has no successful holdout attack on
which localization can be evaluated.

## Primary tests

Intervals below resample dataset/PE/seed/target clusters rather than treating
nested budgets as independent.

| Predeclared comparison | Metric | Mean delta | Cluster-bootstrap 95% CI | Result |
| --- | --- | ---: | ---: | --- |
| all-layer `cf_pred@5B` minus anomaly base | AUPRC | -0.002 | [-0.123, 0.124] | inconclusive |
|  | Recall@B | 0.024 | [-0.119, 0.179] | fail |
|  | Margin recovery | -0.407 | [-1.519, 0.274] | fail |
|  | Repair rate | 0.048 | [-0.167, 0.238] | inconclusive |
| union minus all-layer candidates at 10B | Candidate recall | **0.250** | **[0.071, 0.464]** | pass |
| union `cf_pred@10B` minus all-layer `cf_pred@10B` | AUPRC | **0.150** | **[0.002, 0.320]** | pass (secondary metric) |
|  | Recall@B | 0.134 | [-0.018, 0.315] | fail |
|  | Margin recovery | 0.292 | [-0.013, 0.742] | fail |
|  | Repair rate | 0.155 | [-0.024, 0.369] | inconclusive |

The end-to-end Recall and recovery effects are positive and their lower bounds
are close to zero, but the protocol required both bounds to be above zero.
They cannot be reported as confirmed.

## Candidate coverage was improved under the targeted threat model

| Generator | 5B | 10B | 20B |
| --- | ---: | ---: | ---: |
| All-layer full | 0.618 | 0.721 | 0.831 |
| Temporal residual | 0.640 | 0.787 | 0.831 |
| Multiview union | 0.912 | **1.000** | **1.000** |
| Target-incident | **1.000** | **1.000** | **1.000** |

This is not a general subgraph-localization result. The attack generator adds
every adversarial edge directly to the known target node. The target-incident
branch alone therefore contains every true attack edge at 5B. The union's
coverage gain is largely attributable to this legal but highly specific
structural prior, rather than evidence that transformer dynamics alone found
remote adversarial structure.

## Absolute end-to-end results

| Method | Candidate recall | AUPRC | Recall@B | Repair rate | Margin recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Anomaly base | n/a | 0.441 | 0.368 | 0.324 | 0.095 |
| All-layer `cf_pred@5B` | 0.618 | 0.456 | 0.397 | 0.412 | 0.043 |
| Union `cf_pred@5B` | 0.912 | 0.610 | 0.559 | 0.559 | 0.307 |
| Union `cf_pred@10B` | 1.000 | 0.638 | 0.581 | 0.618 | 0.433 |
| Union hybrid@10B | 1.000 | 0.655 | 0.588 | 0.647 | 0.450 |
| Union true-margin oracle@10B | 1.000 | 0.798 | 0.735 | 0.912 | 1.094 |

Increasing union candidates from 10B to 20B leaves coverage at 1.0 but does
not improve Recall@B (0.581 to 0.574) and reduces AUPRC (0.638 to 0.613). Once
coverage is complete, candidate count is no longer the limiting factor. The
large gap to the true-margin oracle shows that causal scoring/reranking is now
the bottleneck under this targeted setting.

The holdout also does not support replacing the Phase-1 hybrid with pure
predicted-class drop. At union-10B, pure `cf_pred` has 0.0145 lower normalized
margin recovery than the hybrid in the paired cluster analysis, with a 95%
interval of [-0.0403, -0.00002]. This was a secondary comparison, but it argues
against treating the earlier post-hoc `cf_pred` winner as stable.

## Why the Phase-1 reranker did not replicate

Restricting Phase 1 to RWPE does not remove seed heterogeneity. The
`cf_pred@5B` Recall delta versus anomaly base was -0.083 on seed 3407, +0.250
on 3408, and +0.232 on 3409. The holdout repeats this instability: snapshot
weighted Recall deltas are -0.212 on 3410, +0.208 on 3411, and +0.139 on 3412.

The Phase-2 anomaly baseline is also stronger than in Phase 1's RW subset:
AUPRC/Recall are 0.441/0.368 instead of 0.379/0.311. The holdout `cf_pred`
absolute values remain similar (0.456/0.397 versus 0.486/0.432), so most of the
previous relative advantage disappears.

Normalized margin recovery is noisy when the attack-induced clean-to-attacked
margin gap is small. Cora seed 3412 has a median denominator of 0.060, and
`Cora-rw-3410-t65-b4` contributes a normalized paired delta of -7.03. A
post-hoc sensitivity analysis using unnormalized repaired-margin delta finds
union-10B minus all-layer-10B at +0.766 with a cluster-bootstrap interval of
[0.010, 1.582]. This supports a real margin improvement, but it does not
override the preregistered decision because normalized recovery and Recall@B
were the primary endpoints.

## Interpretation and next gate

Phase 2 establishes three narrower points:

1. the Phase-1 pure predicted-class-drop improvement is not robust across new
   seeds;
2. a target-aware candidate prior can remove the shortlist coverage bottleneck
   for this exact incident-edge attack;
3. after coverage reaches 1.0, label-free causal reranking still leaves a large
   oracle gap.

The current proxy should not be tuned further on these holdout seeds. The next
credible experiment should move to a standard Graphormer or GraphGPS model,
separate method-development seeds from a new final holdout, and include an
attack capable of adding remote/non-incident edges. Results should be reported
both with and without known-target incidence so that structural prior and
transformer-dynamics evidence are identifiable.

