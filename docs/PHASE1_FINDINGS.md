# Phase-1 findings: depth versus causality

Canonical result directory: `results/phase1_3seed_20260808/`.

## Decision

**Causal reranking is supported; the necessity of full-depth dynamics remains
inconclusive under the predeclared criterion.**

Phase 1 reused all 45 successful Phase-0 snapshots from 20 model-target
clusters. No model was retrained and no attack was regenerated.

## Depth and channel ablation

| Trace view | AUPRC | Recall@B | Repair rate | Margin recovery |
| --- | ---: | ---: | ---: | ---: |
| First-layer full | 0.196 | 0.106 | 0.044 | -0.012 |
| Last-layer full | 0.191 | 0.083 | 0.022 | -0.016 |
| Attention trajectory | 0.180 | 0.117 | 0.089 | 0.034 |
| Hidden/logit trajectory | 0.178 | 0.089 | 0.022 | -0.018 |
| Value trajectory | 0.027 | 0.006 | 0.022 | -0.038 |
| Temporal residual | 0.307 | 0.194 | 0.200 | 0.083 |
| All-layer full | 0.304 | 0.228 | 0.156 | 0.093 |

All-layer full improves over first-layer full in cluster-bootstrap AUPRC, but
the Recall@B interval crosses zero. Therefore the strict primary depth claim is
inconclusive. Temporal residuals have positive AUPRC and repair-rate intervals
relative to first-layer full, but this was a secondary comparison and the
effect is concentrated in Cora-RW.

The ablation does establish a useful negative result: final/first-layer state,
attention alone, value alone, or hidden/logit alone cannot explain the full
score. The useful signal is a multichannel interaction, with some evidence for
inter-layer changes.

## Label-free causal reranking

The unsupervised all-layer score selected Top-5B candidates. Removing each
candidate and measuring the drop in the currently predicted class requires the
target node but not its true label or a clean counterpart.

| Reranker | AUPRC | Recall@B | Repair rate | Margin recovery |
| --- | ---: | ---: | ---: | ---: |
| Base anomaly | 0.304 | 0.228 | 0.156 | 0.093 |
| Target-anomaly reduction | 0.401 | 0.339 | 0.244 | 0.221 |
| Predicted-class drop | **0.425** | **0.372** | **0.378** | **0.328** |
| Predeclared hybrid | 0.381 | 0.322 | 0.289 | 0.213 |
| True-label oracle | 0.455 | 0.411 | 0.422 | 0.405 |

The predeclared hybrid improves over the base on every primary metric with
cluster-bootstrap intervals above zero. Predicted-class drop is an even
stronger secondary result: it achieves about 90.5% of oracle Recall@B and 81.2%
of oracle margin recovery, but it degrades relative to the base in seed 3407,
so it should not replace the predeclared primary result post hoc.

## New bottleneck

Mean true-edge coverage of the Top-5B shortlist is only 0.433. The oracle
achieves Recall@B 0.411, using about 94.9% of the available candidate recall.
Thus causal reranking is no longer the main bottleneck. Candidate generation is.

Recommended Phase 2:

1. form the candidate set from the union of all-layer full, temporal residual,
   and target-aware node/incident-edge rankings;
2. report coverage-versus-compute curves for K in {5B, 10B, 20B};
3. retain predicted-class-drop reranking, but compare it with the predeclared
   hybrid on new seeds;
4. move to Graphormer or GraphGPS before making an architecture-level claim;
5. only after that introduce an adaptive anti-localization attacker.

