# Phase 7: generalized heuristic-search repair

## Scope

Phase 7 tested general repairs to the post-training **remote edge-addition**
attack. Candidate-node labels were never used. No branch, threshold, or feature
was specific to PubMed.

The frozen development factorial compared 32 configurations formed from:

- cross-entropy versus normalized true-to-best-rival margin;
- greedy search versus beam search with width 8;
- single-rival versus multi-rival candidate generation;
- a fixed Top-128 pool versus conditional expansion to Top-512;
- an exact-budget output versus the best state within the budget.

The development checkpoints were Cora/CiteSeer seeds 4570 and 4572 and PubMed
seeds 4580 and 4582. Six clean-correct targets per checkpoint were selected at
evenly spaced clean-margin ranks. Budgets were 1, 2, 4, and 8.

## Development selection

The predeclared minimax rule selected:

`normalized_margin + beam8 + multi_rival + adaptive_pool + within_budget`.

At budget 8 it attacked 9/36 targets, compared with 8/36 for the legacy
cross-entropy greedy path. It improved canonical margin progress on 27/36
targets and raised mean progress by 0.1319. The only additional success was on
PubMed; no legacy success was lost.

The success change was small. Beam search and multi-rival candidates were the
only individual replacements whose removal reduced the worst-dataset success
rate from 1/6 to 1/12. The other three modules affected margin-based
tie-breaking but did not add a development success by themselves.

## Frozen validation

The selected configuration and the legacy greedy attack were then evaluated on
all nine remaining checkpoints: Cora/CiteSeer seeds 4571, 4573, and 4574 and
PubMed seeds 4581, 4583, and 4584. These checkpoints did not participate in
configuration selection. The target-selection rule and all search parameters
were frozen.

At budget 8 both methods attacked exactly 10/54 targets. There were no discordant
successes in either direction. The selected configuration improved mean margin
progress by 0.0766 overall, but the effect was heterogeneous:

| Dataset | Selected successes | Legacy successes | Paired margin-progress change |
|---|---:|---:|---:|
| Cora | 7/18 | 7/18 | +0.2160 |
| CiteSeer | 2/18 | 2/18 | +0.0245 |
| PubMed | 1/18 | 1/18 | -0.0108 |

Thus the development success improvement did **not** replicate, and the
PubMed margin improvement reversed sign. The heavy configuration evaluated
about nine times as many graph candidates as the legacy path.

Among the four outputs available in frozen validation, the minimax rule instead
ranked `cross_entropy + greedy + single_rival + fixed_pool + within_budget`
first. It had the same attack-success counts as both alternatives, improved the
worst-dataset mean margin progress relative to exact-budget legacy output, and
required no additional graph evaluations. This is evidence for an at-most-budget
output policy, not evidence that beam/multi-rival search generally improves
attack success.

## GPU execution

The development factorial performed 2,621,494 graph evaluations in 7,514 CUDA
forward calls. The requested and realized maximum batch size was 512 with no
OOM reduction. Peak CUDA allocation was about 3.04 GB on an NVIDIA GeForce RTX
5060 Laptop GPU. Frozen validation added 533,374 graph evaluations in 1,527
CUDA calls, also at a realized maximum batch size of 512.

## Decision

The heavy development winner is **not promoted** into the rho=0.95 adaptive
stealth attack because its success improvement failed frozen validation and it
was slightly worse on PubMed margin progress. The general repair retained for
the next experiment is the inexpensive best-within-budget policy. Beam search,
multi-rival candidates, and adaptive expansion remain diagnostic/optional
search modes rather than the new default.

This is a development-and-validation result, not a confirmatory claim about
population attack rates.
