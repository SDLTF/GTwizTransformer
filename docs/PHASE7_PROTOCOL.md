# Phase 7 protocol: general heuristic-search factorial

This is a development experiment, frozen before inspecting any Phase-7 search
result. It is intended to repair dataset-independent attack-optimization
failures rather than tune specifically for PubMed.

## Development models

Reuse already trained GraphGPS checkpoints so every search arm sees identical
models and targets:

- Cora seeds 4570 and 4572;
- CiteSeer seeds 4570 and 4572;
- PubMed seeds 4580 and 4582.

For each model, select six correctly classified targets at evenly spaced clean-
margin ranks. Use remote, non-target-incident edge additions and budgets
1, 2, 4, and 8. No dataset receives a separate hyperparameter.

## Full factorial

The search modules are:

1. objective: cross-entropy or normalized true-vs-best-rival margin;
2. search: greedy (`beam=1`) or beam search (`beam=8`);
3. candidate semantics: current single-rival ranking or class-diverse
   multi-rival ranking;
4. candidate pool: fixed 128 or adaptive 128-to-512 expansion when the current
   state has no positive objective gain;
5. output policy: exact budget or best canonical attack margin found within the
   budget, including the no-op state.

The first four choices require 16 search executions; the two output policies
are derived from each execution without extra model queries, producing 32
reported configurations.

All candidate scoring remains label-free with respect to candidate nodes. The
attacker may use the attacked target's true label, as in earlier phases. The
threat model is not expanded to incident edges or deletion.

## Compute

Candidate variants from every live beam state are concatenated and evaluated
with a requested CUDA graph batch size of 512. CUDA OOM may only reduce the
realized batch size through the existing deterministic halving mechanism.
Every configuration reports graph evaluations, forward calls, realized batch
size, elapsed time, and peak CUDA memory.

## Frozen selection rule

At budget 8, select configurations lexicographically by:

1. maximum worst-dataset attack success rate;
2. maximum macro-average dataset success rate;
3. maximum worst-dataset mean canonical margin progress;
4. maximum macro-average canonical margin progress;
5. minimum graph evaluations;
6. deterministic configuration ID.

This minimax-first rule prevents a large Cora/CiteSeer gain from masking a
PubMed regression. Development selection is not confirmatory evidence.

## Follow-up

The winning classification search must be expanded to the remaining existing
development seeds before it is attached to a budget-level `rho=0.95` Pareto
stealth search. Only after that engineering validation may new seeds be used
for a fresh holdout.
