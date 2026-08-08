# Phase-5 user-specified rho=0.95 holdout

This is a local, offline robustness experiment on Cora and CiteSeer. It is a
user-specified independent holdout for `rho=0.95`, which development results
showed to preserve classification attack strength almost exactly but did not
support localization evasion.

To avoid reusing either the development seeds (4550-4552) or the seeds reserved
for the automatically selected Phase-5 ratio (4560-4564), this holdout uses
fresh seeds 4570-4574. It uses 30 targets per model, 192 nodes, remote two-hop
edge additions, budgets 1/2/4/8, candidate pool 128 and graph batch size 128.

The paired classification-only and constrained-stealth arms share exact model
checkpoints and target lists. Viability requires at least 24 joint-success
model-target clusters, four seeds and both datasets. The same Phase-5 utility,
localization and repair endpoints are used with 5,000 cluster bootstrap
resamples. This holdout is reported separately and does not retroactively alter
the development selection rule.
