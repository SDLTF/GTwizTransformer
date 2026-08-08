# Phase 6 findings: PubMed transfer and DShield-Aug

## Decision

The preregistered result is **underpowered** for both attack transfer and
defense support.

The `rho=0.95` attack produced 14 successful snapshots from eight target
clusters, but those clusters occur in only two of five PubMed seeds. This
misses the frozen viability requirement of 18 clusters and four seeds. The
DShield-Aug results are therefore exploratory and cannot be reported as a
formal defense success.

The experiment also shows that the official DShield name must not be used for
this defense arm. Official DShield detects poisoned training nodes and retrains
GCN/GAT/GraphSAGE models. **DShield-Aug** is our test-time adaptation of its
20% edge-removal, 20% feature-masking, and multi-view discrepancy ideas to a
post-training GraphGPS edge attack.

Evidence directories:

- classification baseline: `results/phase6_pubmed_baseline_20260809/`;
- `rho=0.95` adaptive attack: `results/phase6_pubmed_adaptive_r0p95_20260809/`;
- paired attack comparison: `results/phase6_pubmed_comparison_r0p95_20260809/`;
- DShield-Aug evaluation: `results/phase6_pubmed_dshield_aug_20260809/`.

## PubMed attack transfer

The five models supplied 87 correctly classified test targets and 348 attack
snapshots over budgets 1, 2, 4, and 8. Baseline and adaptive attacks each
succeeded on the same 14 snapshots. Success is highly concentrated:

| Budget | Successful targets | Attempted targets |
|---:|---:|---:|
| 1 | 0 | 87 |
| 2 | 3 | 87 |
| 4 | 3 | 87 |
| 8 | 8 | 87 |

Seeds 4580, 4581, and 4583 have no successful snapshot. Seed 4582 contributes
12 successful snapshots from six clusters, and seed 4584 contributes two
budget-8 snapshots from two clusters. This concentration is why the run is not
viable despite containing some clear attacks.

At budget 8, successful targets have substantially smaller clean margins than
failures: mean 0.247 versus 0.591 (median 0.289 versus 0.594). The current
attacker therefore transfers mainly to lower-margin PubMed targets rather than
across the normal target population.

Conditional on the eight joint-success clusters, the constrained attack keeps
almost all classification strength:

| Quantity | Estimate | 95% cluster-bootstrap CI |
|---|---:|---:|
| Adaptive/baseline attack progress | 0.99395 | [0.98834, 0.99801] |
| Attack-success difference | 0.00000 | [0.00000, 0.00000] |
| Attack-edge-set Jaccard | 0.80185 | [0.70741, 0.89074] |

Unlike the Cora/CiteSeer holdout, PubMed shows no localization evasion. The
adaptive-minus-baseline effects are +0.0221 AUPRC and +0.0156 Recall@B. On
joint successes, adaptive absolute AUPRC is 0.8830, Recall@B is 0.8393, and
causal Top-B repair restores every snapshot. These are descriptive results
because the viability gate failed.

## DShield-Aug defense

The defense used 64 deterministic views per arm. All uncertainty intervals
cluster by PubMed/seed/target.

| Defense arm | Clean correctness | Attack recovery |
|---|---:|---:|
| Edge augmentation | 0.9655 [0.9195, 1.0000] | 1.0000 [1.0000, 1.0000] |
| Feature augmentation | 1.0000 [1.0000, 1.0000] | 0.1250 [0.0000, 0.3750] |
| DShield-Aug | 0.9655 [0.9195, 1.0000] | 1.0000 [1.0000, 1.0000] |
| Causal Top-B repair | not a stochastic clean arm | 1.0000 [1.0000, 1.0000] |

DShield-Aug recovers all 14 successful snapshots, including all budgets 2, 4,
and 8. Its attacked-minus-clean view disagreement is +0.01018 with interval
[+0.00466, +0.01574], and its true-class probability gain is +0.2232 with
interval [+0.1301, +0.2932]. The augmentation ensemble therefore reacts
strongly to the observed successful attacks.

However, two limitations prevent a defense claim:

1. DShield-Aug misclassifies three of 87 clean targets. Its point clean
   correctness is 96.55%, but the lower confidence bound is only 91.95%, below
   the frozen 95% requirement. The three errors occur at seed/target pairs
   4580/188, 4582/99, and 4582/61. Their clean margins are 0.0200, 0.2655, and
   0.0786, respectively.
2. DShield-Aug and edge-only augmentation produce identical final predictions
   on all 348 attack snapshots. They also recover exactly the same successful
   snapshots. Feature masking adds no observed classification benefit; the
   recovery is entirely explained by random structural edge removal.

Thus the experiment supports a narrower statement: **randomized edge smoothing
is a promising defense against the small set of successful PubMed attacks, but
the present evidence does not establish a DShield-specific benefit or a
clean-utility-safe defense.**

The existing causal Top-B repair remains the stronger result. It also restores
every successful PubMed snapshot, removes selected suspicious edges rather
than 20% of the graph, and does not rely on a stochastic prediction ensemble.

## Integrity audit

The independent audit verifies:

- exactly 348 attack IDs and 1,044 defense rows (three arms per attack);
- no duplicate attack/arm rows and no missing arm;
- raw-success flags exactly match the attack file;
- clean view results are constant across nested budgets;
- every independently recomputed bootstrap matches `decision.json`;
- minimum observed classification-gain ratio is 0.950184;
- zero target-incidence, budget-cardinality, and nested-budget violations.

## Next experiment

The next step should strengthen the PubMed attack before spending more evidence
budget on defense. Use the present seeds only as development data to compare
budgets 8, 16, and 32 and possibly a larger induced graph. Freeze the first
configuration that produces successful attacks across at least four seeds,
then validate it on new seeds. After viability is established, attack through
the stochastic edge-smoothing expectation rather than evaluating only an
oblivious attack. That experiment can distinguish genuine robustness from
gradient/optimization mismatch.
