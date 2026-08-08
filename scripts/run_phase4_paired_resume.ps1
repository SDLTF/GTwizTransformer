. "$PSScriptRoot\common.ps1"

$Baseline = 'results\phase4_paired_baseline_20260808'
$Adaptive = 'results\phase4_paired_adaptive_v2_20260808'
$Comparison = 'results\phase4_paired_comparison_v2_20260808'

& $PythonExe -m transattack.phase3 `
    --datasets cora citeseer `
    --seeds 4540 4541 4542 `
    --expected-seeds 4540 4541 4542 `
    --attack-types remote `
    --attack-objective adaptive_stealth `
    --adaptive-stealth-strength 1.0 `
    --reference-run-dir $Baseline `
    --budgets 1 2 4 8 `
    --nodes 192 `
    --targets 15 `
    --candidate-pool 64 `
    --candidate-multiplier 10 `
    --graph-batch-size 64 `
    --channels 96 `
    --pe-channels 16 `
    --walk-length 8 `
    --layers 4 `
    --heads 8 `
    --epochs 100 `
    --patience 20 `
    --minimum-remote-clusters 18 `
    --minimum-remote-seeds 3 `
    --minimum-remote-datasets 2 `
    --device cuda `
    --strict-cuda `
    --data-root data `
    --output-dir $Adaptive
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe -m transattack.phase4 `
    --baseline-dir $Baseline `
    --adaptive-dir $Adaptive `
    --output-dir $Comparison `
    --expected-seeds 4540 4541 4542
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
