. "$PSScriptRoot\common.ps1"

$Baseline = 'results\phase5_r0p95_holdout_baseline_20260809'
$Adaptive = 'results\phase5_r0p95_holdout_adaptive_20260809'
$Comparison = 'results\phase5_r0p95_holdout_comparison_20260809'

if (-not (Test-Path -LiteralPath "$Baseline\decision.json")) {
    if (Test-Path -LiteralPath $Baseline) {
        throw "Incomplete baseline directory already exists: $Baseline"
    }
    & $PythonExe -m transattack.phase3 `
        --datasets cora citeseer `
        --seeds 4570 4571 4572 4573 4574 `
        --expected-seeds 4570 4571 4572 4573 4574 `
        --attack-types remote `
        --attack-objective classification_only `
        --budgets 1 2 4 8 `
        --nodes 192 `
        --targets 30 `
        --candidate-pool 128 `
        --candidate-multiplier 10 `
        --graph-batch-size 128 `
        --channels 96 `
        --pe-channels 16 `
        --walk-length 8 `
        --layers 4 `
        --heads 8 `
        --epochs 100 `
        --patience 20 `
        --minimum-remote-clusters 24 `
        --minimum-remote-seeds 4 `
        --minimum-remote-datasets 2 `
        --device cuda `
        --strict-cuda `
        --data-root data `
        --output-dir $Baseline
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path -LiteralPath "$Adaptive\decision.json")) {
    if (Test-Path -LiteralPath $Adaptive) {
        throw "Incomplete adaptive directory already exists: $Adaptive"
    }
    & $PythonExe -m transattack.phase3 `
        --datasets cora citeseer `
        --seeds 4570 4571 4572 4573 4574 `
        --expected-seeds 4570 4571 4572 4573 4574 `
        --attack-types remote `
        --attack-objective classification_constrained_stealth `
        --classification-retention-ratio 0.95 `
        --reference-run-dir $Baseline `
        --budgets 1 2 4 8 `
        --nodes 192 `
        --targets 30 `
        --candidate-pool 128 `
        --candidate-multiplier 10 `
        --graph-batch-size 128 `
        --channels 96 `
        --pe-channels 16 `
        --walk-length 8 `
        --layers 4 `
        --heads 8 `
        --epochs 100 `
        --patience 20 `
        --minimum-remote-clusters 24 `
        --minimum-remote-seeds 4 `
        --minimum-remote-datasets 2 `
        --device cuda `
        --strict-cuda `
        --data-root data `
        --output-dir $Adaptive
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path -LiteralPath "$Comparison\decision.json")) {
    if (Test-Path -LiteralPath $Comparison) {
        throw "Incomplete comparison directory already exists: $Comparison"
    }
    & $PythonExe -m transattack.phase5 `
        --baseline-dir $Baseline `
        --adaptive-dir $Adaptive `
        --output-dir $Comparison `
        --expected-seeds 4570 4571 4572 4573 4574 `
        --expected-targets 30 `
        --minimum-joint-clusters 24 `
        --minimum-joint-seeds 4 `
        --minimum-joint-datasets 2
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
