# GraphTransAttack

Phase-0 implementation of the updated **Graph Attack** idea: localize an
adversarial subgraph from a graph transformer's layer-wise internal dynamics.

The detector receives only:

- the graph currently presented at inference time;
- the model's current attention, value-contribution, hidden-state, and logit
  trajectories;
- distribution parameters fitted on clean calibration traces.

It never receives the clean counterpart graph, a clean/attacked edge diff, or
the identities of clean calibration edges. True added edges are retained only
as evaluation labels.

## Phase-0 question

After a successful addition-only structural attack, do the added edges have a
repeatable layer-wise fingerprint that ranks them above ordinary observed
edges, and does removing the localized top-B set undo the attack?

The experiment compares:

1. random ranking;
2. final-layer attention only;
3. all-layer attention/entropy/JS dynamics;
4. full attention + value contribution + hidden/logit trajectory.

See `docs/PHASE0_PROTOCOL.md` for the threat model and leakage contract.

## Run

Use the CUDA-enabled Python environment already available on this machine:

```powershell
& 'D:\Code\GSDD\GSDD-Bench-v1.0.0-DShield-Integration\GSDD-Bench-v1.0.0\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& .\scripts\run_smoke.ps1
& .\scripts\run_phase0.ps1
```

Every run writes a timestamped directory below `results/` containing the exact
configuration, environment, checkpoint, per-attack metrics, aggregate summary,
and a machine-readable decision.

The completed three-seed Phase-0 conclusion is documented in
`docs/INITIAL_FINDINGS.md`: ranking fingerprints are supported, while causal
localization remains inconclusive.

Phase 1 reuses the frozen Phase-0 attacks for layer/channel ablations and
label-free leave-one-out causal reranking. Its conclusion is documented in
`docs/PHASE1_FINDINGS.md`: causal reranking is supported, while the necessity
of full-depth dynamics remains inconclusive.

Phase 2 is a preregistered holdout test on new seeds. It tests whether the
label-free predicted-class counterfactual score replicates and whether a
multiview candidate union improves the candidate-coverage bottleneck at fixed
compute budgets. See `docs/PHASE2_PROTOCOL.md`. This stage still uses the
custom dense graph-transformer mechanism proxy; formal Graphormer/GraphGPS
validation remains a separate gate.

The completed holdout conclusion is documented in
`docs/PHASE2_FINDINGS.md`: candidate coverage improves under the known-target
incident-edge threat model, but the Phase-1 pure counterfactual reranker does
not replicate and the predeclared end-to-end criterion remains inconclusive.

Phase 3 replaces the mechanism proxy with a traceable PyTorch Geometric
`GPSConv` model using RWSE, a GIN local branch, and exact global multi-head
attention. It evaluates incident and non-incident two-hop attacks separately
and batches attack/counterfactual graph variants on CUDA. See
`docs/PHASE3_PROTOCOL.md`.

The first formal-GraphGPS holdout is documented in
`docs/PHASE3_FINDINGS.md`. All measured remote localization intervals are
positive, but the frozen remote-viability threshold is missed by one
model-target cluster, so the run is reported as underpowered rather than
positive.

Phase 3b is the preregistered fresh-seed replication of that result. It freezes
the Phase-3 model, attacker, primary methods, and statistics; uses seeds
4520-4522; increases targets per model from six to ten; and requires successful
remote clusters in all three seeds and both datasets. See
`docs/PHASE3B_PROTOCOL.md`.

The completed replication is documented in `docs/PHASE3B_FINDINGS.md`. It
passes the frozen viability, dynamics-fingerprint, and label-free causal
localization criteria on 20 successful remote model-target clusters. The
effect is not uniform across dataset-seed slices, so the documented conclusion
is deliberately limited to this GraphGPS architecture and two-hop attacker.

Phase 4 stress-tests that detector with a white-box adaptive attacker that
jointly penalizes all-layer anomaly, temporal residual anomaly, and predicted-
class counterfactual signal. See `docs/PHASE4_PROTOCOL.md` and
`docs/PHASE4_FINDINGS.md`. The fresh paired run is underpowered at 11 joint
clusters: adaptive AUPRC and candidate coverage decrease, but Top-B recall and
repair do not. Invalid preliminary execution directories are explicitly
excluded in `docs/PHASE4_INVALID_RUN.md`.

Phase 5 replaces the equal-weight tradeoff with a classification-constrained
stealth objective: at each attack step it minimizes trajectory/causal evidence
only among edges retaining at least `rho` of the best positive classification
gain. See `docs/PHASE5_PROTOCOL.md`. A separate user-directed fresh-seed
holdout at `rho=0.95` is documented in
`docs/PHASE5_R095_HOLDOUT_PROTOCOL.md` and
`docs/PHASE5_R095_HOLDOUT_FINDINGS.md`. Across 53 joint-success clusters it
retains 99.35% attack progress while slightly reducing AUPRC and Recall@B; it
does not reduce causal repair. This holdout does not replace the still-unrun
formal experiment at the development-selected `rho=0.50`.
