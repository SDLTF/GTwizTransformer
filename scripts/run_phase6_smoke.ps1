. "$PSScriptRoot\common.ps1"

$Baseline = 'results\phase6_smoke_pubmed_baseline'
$Adaptive = 'results\phase6_smoke_pubmed_adaptive'
$Defense = 'results\phase6_smoke_pubmed_dshield_aug'

& $PythonExe -m transattack.phase3 --datasets pubmed --seeds 4579 --expected-seeds 4579 --attack-types remote --attack-objective classification_only --budgets 1 2 --nodes 96 --targets 4 --candidate-pool 32 --candidate-multiplier 4 --graph-batch-size 32 --channels 48 --pe-channels 8 --walk-length 4 --layers 2 --heads 4 --epochs 10 --patience 5 --minimum-remote-clusters 1 --minimum-remote-seeds 1 --minimum-remote-datasets 1 --device cuda --strict-cuda --data-root data --output-dir $Baseline --smoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonExe -m transattack.phase3 --datasets pubmed --seeds 4579 --expected-seeds 4579 --attack-types remote --attack-objective classification_constrained_stealth --classification-retention-ratio 0.95 --reference-run-dir $Baseline --budgets 1 2 --nodes 96 --targets 4 --candidate-pool 32 --candidate-multiplier 4 --graph-batch-size 32 --channels 48 --pe-channels 8 --walk-length 4 --layers 2 --heads 4 --epochs 10 --patience 5 --minimum-remote-clusters 1 --minimum-remote-seeds 1 --minimum-remote-datasets 1 --device cuda --strict-cuda --data-root data --output-dir $Adaptive --smoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonExe -m transattack.phase6 --attack-dir $Adaptive --output-dir $Defense --expected-seeds 4579 --views 8 --graph-batch-size 8 --minimum-success-clusters 1 --minimum-success-seeds 1 --device cuda --strict-cuda --smoke
exit $LASTEXITCODE
