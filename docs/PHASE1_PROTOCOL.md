# Phase-1 protocol: dynamics ablation and causal reranking

Phase 1 reuses the exact successful attacks and trained checkpoints from the
three-seed Phase 0. It does not retrain models, regenerate attacks, or select a
new favorable target set.

## Question A: is depth dynamics necessary?

All anomaly profiles are fitted from clean edge traces and store distribution
parameters only. The following views are compared on the same attacked edges:

- first-layer full channels;
- last-layer full channels;
- all-layer full channels;
- temporal residuals between adjacent layers;
- attention/entropy/JS trajectory;
- value-contribution trajectory;
- hidden/logit trajectory;
- all-layer value + hidden/logit channels without attention.

The primary predeclared comparison is all-layer full versus first-layer full.
This distinguishes a genuinely depth-dependent signal from an early static
anomaly.

## Question B: can anomaly ranking become causal localization?

The all-layer unsupervised score first selects `K = 5B` candidate edges. Each
candidate is removed from the currently observed attacked graph, PE and model
traces are recomputed, and candidates are reranked by:

- drop in the current predicted-class probability (label-free);
- reduction in global internal anomaly (label-free);
- reduction in target-region internal anomaly (target-aware, label-free);
- anomaly + predicted-class-drop hybrid (primary deployable method);
- true-label margin gain (evaluation-only oracle).

The true class and true attack-edge set never enter deployable scores. They are
used only for final metrics and the oracle upper bound. No paired clean graph is
provided to a localizer.

## Statistical unit

Budgets 1/2/4 for one target are nested. Confidence intervals therefore
resample dataset/PE/seed/target clusters rather than individual snapshots.

