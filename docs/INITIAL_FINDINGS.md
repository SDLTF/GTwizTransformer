# Initial findings (three-seed Phase 0)

Canonical aggregate: `results/phase0_3seed_20260808/`.

## Bottom line

The updated idea is **partly supported**. A clean-distribution-only localizer can
rank adversarial edges from current-graph internal traces, especially for RWPE.
The stronger claim that the ranked edges are reliably causal is not yet
supported because counterfactual margin recovery is unstable across seeds.

This should be described as:

> Layer-wise internal features contain an adversarial-edge fingerprint; robust
> causal subgraph localization remains open.

It should not yet be described as recovery of the true attack history or as a
reliable repair method.

## Evidence

Across Cora/CiteSeer, Heat/RW, and seeds 3407/3408/3409 there were 216 attack
snapshots, of which 45 were successful. Primary localization statistics use
only those 45 snapshots. A model-target cluster bootstrap avoids treating
nested B=1/2/4 snapshots as independent.

| Matched comparison | Mean delta | Cluster-bootstrap 95% CI |
| --- | ---: | ---: |
| Full dynamics vs random, edge AUPRC | +0.273 | [+0.156, +0.403] |
| Full dynamics vs random, Recall@B | +0.200 | [+0.067, +0.350] |
| Full dynamics vs layer-attention, edge AUPRC | +0.134 | [+0.055, +0.233] |
| Full dynamics vs layer-attention, Recall@B | +0.108 | [+0.013, +0.221] |
| Full dynamics vs random, margin recovery | +0.047 | [-0.086, +0.196] |

Thus the ranking result is positive, including against the stronger
layer-attention baseline, while the causal margin result is inconclusive.

## PE-dependent behavior

- Cora-RW: mean AUPRC 0.389, Recall@B 0.329, repair rate 0.263.
- CiteSeer-RW: mean AUPRC 0.364, Recall@B 0.286, repair rate 0.071.
- Cora-Heat and CiteSeer-Heat: Recall@B was 0 on successful attacks.
- Final-layer attention was close to chance in every dataset/PE group.

This is consistent with the proposed stable-for-prediction versus
sensitive-for-forensics tension: Heat is harder to attack but also supplies a
weaker localization signal; RW is easier to attack and much more forensic.
This is a Phase-0 observation, not a general theorem.

## Layer timing

For true added edges that crossed the clean 95th-percentile anomaly threshold,
the mean first detected layer was 1.07 on Cora-RW and 1.00 on CiteSeer-RW.
Therefore the present results do not prove that a multi-layer propagation
trajectory is necessary. They may instead reflect a strong first-layer
attention/value/hidden anomaly that persists through the stack.

## Limitations that materially affect the claim

1. Added edges are target-incident; nonlocal causal subgraphs are not tested.
2. Deleted edges are excluded because they are absent from the observed-edge
   ranking universe.
3. Clean calibration comes from the same transductive graph, although only
   distribution parameters (not edge identities) reach the detector.
4. The graphs are 128-node induced Planetoid subgraphs and model accuracy is
   below full-dataset state of the art.
5. Successful snapshots are only 45 observations from 20 model-target
   clusters.
6. The attack does not yet optimize an anti-localization objective.

## Recommended Phase 1

1. Add first-layer-only full-channel, last-layer-only full-channel, and temporal
   residual ablations. This tests whether dynamics, rather than a static early
   anomaly, carries the signal.
2. Use the unsupervised score only as a top-K prefilter, then perform one-edge
   leave-one-out causal re-ranking. This directly optimizes counterfactual
   margin recovery without using the clean counterpart.
3. Fit degree/class-conditional or mixture normal profiles to reduce seed
   sensitivity.
4. Expand to non-target-incident additions, then rank a controlled non-edge set
   for deletion attacks.
5. Introduce the adaptive objective `L_attack - lambda * L_localization`.
6. Repeat on larger/full graphs and report target-cluster confidence intervals.

