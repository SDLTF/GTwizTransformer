# Phase-0 protocol: layer-wise forensic localization

## Hypothesis

Successful graph attacks create a localized, repeatable deviation in the
layer-wise computation trajectory. A detector fitted only to a clean normal
distribution can use this deviation to rank adversarial edges without seeing a
paired clean graph at inference.

## Threat model

- White-box, test-time, target-node structural attack.
- Undirected, addition-only edge budget `B in {1, 2, 4}` for v0.1.
- The target starts correctly classified; attack success means it becomes
  misclassified.
- The attack adaptively recomputes the positional encoding and maximizes target
  cross-entropy after every addition.
- Added edges are incident to the target in v0.1. This creates a controlled
  localization test; nonlocal and deletion attacks are follow-up stress tests.

Addition-only is deliberate: a deleted edge is absent from the observed graph
and cannot be ranked as an observed edge without expanding the candidate space
to all non-edges.

## No-leakage contract

The localizer may use a mean/covariance normal profile fitted on clean traces.
The fitted object stores numerical distribution parameters only. It does not
store edge identities. At attacked inference it sees only the attacked graph
and its trace. Clean/attacked edge differencing is forbidden.

The true added-edge set is used only after scoring, to compute metrics and the
oracle repair control.

## Trace channels

For each layer, head, node, and observed edge the harness records or derives:

- attention strength;
- normalized attention entropy;
- inter-layer Jensen-Shannon divergence;
- `alpha_ij * ||V_j||` value-contribution magnitude;
- endpoint hidden-state step norm;
- endpoint probability-trajectory step;
- endpoint hidden-state cosine.

Gaussian profiles with shrinkage covariance define unsupervised Mahalanobis
anomaly scores. No learned attack labels are used.

## Metrics and controls

- edge AUPRC;
- Recall@B and top-B IoU;
- AUPRC lift over class prevalence;
- first layer whose true-edge anomaly exceeds the clean 95th percentile;
- top-B counterfactual target-margin recovery and label restoration;
- oracle and random repair controls.

Primary analysis is restricted to successful attacks. All attempted attacks
remain in `metrics.csv` so selection effects are visible.

## Phase-0 interpretation

The idea has an initial positive signal only if full dynamics beats random
ranking on successful attacks and its top-B removal recovers more target margin
than random removal. This is evidence of a fingerprint, not proof of universal
attack-history recovery or a robustness certificate.

