# Invalid Phase-4 execution: independently retrained paired arms

The directories `results/phase4_baseline_20260808/` and
`results/phase4_adaptive_20260808/` are **not experimental evidence** and must
not be included in any Phase-4 analysis.

The baseline arm completed on seeds 4530-4532. During the adaptive arm, live
logs showed that Cora/4531 selected a slightly different tail of the boundary
target list despite the same rounded test accuracy and seed. This indicates
small CUDA training nondeterminism. Because the protocol requires exact
attack-ID pairing, the adaptive process was interrupted before completion.

The partial outputs are intentionally retained rather than overwritten or
deleted. The corrective formal execution uses untouched sequential seeds
4540-4542, trains each baseline model once, and reloads the exact checkpoint
and target list for the adaptive arm. No attack objective, strength, endpoint,
bootstrap rule, viability threshold, or noninferiority margin was changed.

The first checkpoint-reuse adaptive directory,
`results/phase4_paired_adaptive_20260808/`, is also partial and excluded. It
stopped when an initial `1e-6` clean-margin equality guard proved stricter than
CUDA repeat-forward numerical precision. The guard was aligned to the existing
`3e-4` Phase-3 reconstruction tolerance; target IDs and checkpoint weights
remain exact. The complete adaptive rerun uses a new `v2` directory.
