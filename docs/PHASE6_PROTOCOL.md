# Phase 6 protocol: PubMed transfer and DShield-style defense

This protocol is frozen before any Phase-6 attack or defense result is
generated. The experiment remains a local, offline graph-robustness study.

## Questions

1. Does the `rho=0.95` classification-constrained structural attacker transfer
   from Cora/CiteSeer to an independently induced PubMed graph?
2. Can a DShield-style multi-view discrepancy defense recover predictions
   changed by that attack without materially damaging clean predictions?

The official DShield implementation is a training-time backdoor defense. It
identifies suspicious *training nodes* and retrains GCN/GAT/GraphSAGE victims.
Our threat is a post-training GraphGPS edge-addition attack. Consequently the
defense below is named **DShield-Aug**, not official DShield: it reuses
DShield's default 20% edge-removal and 20% feature-masking augmentations and
its supervised/self-supervised discrepancy motivation as a test-time
multi-view consistency defense. This distinction must be retained in every
report.

## Frozen attack transfer

- dataset: PubMed, loaded entirely from the local DShield cache;
- induced nodes: 192;
- seeds: 4580, 4581, 4582, 4583, 4584;
- victim: the frozen Phase-5 GraphGPS architecture (RWSE + GIN + exact global
  MHA, 4 layers, width 96, 8 heads);
- targets requested per seed: 30 correctly classified test nodes;
- remote two-hop addition budgets: 1, 2, 4, 8;
- candidate pool and GPU graph batch size: 128;
- paired objectives: classification-only baseline and
  classification-constrained stealth with `rho=0.95`;
- attack-transfer viability: at least 18 joint-success model-target clusters,
  at least four seeds, and PubMed represented.

The Phase-5 paired utility and localization criteria are unchanged. Failure of
the viability gate is reported as underpowered.

## Frozen defense arms

The attack is generated before applying a defense. This is an oblivious-
defense pilot; a positive result must later be challenged by an attacker that
optimizes through the defense.

For every adaptive attack snapshot, use 64 deterministic Monte Carlo views:

- `edge_aug`: independently drop 20% of existing undirected edges;
- `feature_aug`: independently mask 20% of feature entries;
- `dshield_aug` (primary): apply both operations using paired randomness.

Each arm averages target-class probabilities across views. Clean and attacked
graphs use the same view randomness. The Jensen-Shannon-style prediction
disagreement is entropy of the mean probability minus mean entropy.

The existing `global_union+hybrid_label_free` Top-B causal edge-removal repair
is a separate strong benchmark, not part of DShield-Aug.

## Defense endpoints

All uncertainty intervals cluster by PubMed/seed/target and use 5,000 bootstrap
repetitions.

Primary DShield-Aug endpoints on snapshots that the adaptive attack changed:

1. clean correctness across unique target clusters;
2. attack recovery rate;
3. attacked-minus-clean view disagreement;
4. recovery differences against `edge_aug` and `feature_aug`;
5. comparison with causal Top-B repair.

DShield-Aug defense is supported only if the attack-transfer viability gate
passes, the lower 95% confidence bound for clean correctness is at least 0.95,
and the lower bound for attack recovery is strictly above zero. Discrepancy
separation is a mechanistic endpoint, not an additional success requirement.

All attack snapshots, view metrics, decisions, and audits must remain under
`D:\Code\GraphTransAttack`.
