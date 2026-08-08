. "$PSScriptRoot\common.ps1"

$Baseline = 'results\phase6_pubmed_baseline_20260809'
$Adaptive = 'results\phase6_pubmed_adaptive_r0p95_20260809'
$Comparison = 'results\phase6_pubmed_comparison_r0p95_20260809'
$Defense = 'results\phase6_pubmed_dshield_aug_20260809'

if (-not (Test-Path -LiteralPath "$Baseline\decision.json")) {
    if (Test-Path -LiteralPath $Baseline) { throw "Incomplete baseline directory already exists: $Baseline" }
    & $PythonExe -m transattack.phase3 `
        --datasets pubmed `
        --seeds 4580 4581 4582 4583 4584 `
        --expected-seeds 4580 4581 4582 4583 4584 `
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
        --minimum-remote-clusters 18 `
        --minimum-remote-seeds 4 `
        --minimum-remote-datasets 1 `
        --device cuda `
        --strict-cuda `
        --data-root data `
        --output-dir $Baseline
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path -LiteralPath "$Adaptive\decision.json")) {
    if (Test-Path -LiteralPath $Adaptive) { throw "Incomplete adaptive directory already exists: $Adaptive" }
    & $PythonExe -m transattack.phase3 `
        --datasets pubmed `
        --seeds 4580 4581 4582 4583 4584 `
        --expected-seeds 4580 4581 4582 4583 4584 `
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
        --minimum-remote-clusters 18 `
        --minimum-remote-seeds 4 `
        --minimum-remote-datasets 1 `
        --device cuda `
        --strict-cuda `
        --data-root data `
        --output-dir $Adaptive
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path -LiteralPath "$Comparison\decision.json")) {
    if (Test-Path -LiteralPath $Comparison) { throw "Incomplete comparison directory already exists: $Comparison" }
    & $PythonExe -m transattack.phase5 `
        --baseline-dir $Baseline `
        --adaptive-dir $Adaptive `
        --output-dir $Comparison `
        --expected-seeds 4580 4581 4582 4583 4584 `
        --expected-targets 30 `
        --minimum-joint-clusters 18 `
        --minimum-joint-seeds 4 `
        --minimum-joint-datasets 1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path -LiteralPath "$Defense\decision.json")) {
    if (Test-Path -LiteralPath $Defense) { throw "Incomplete defense directory already exists: $Defense" }
    & $PythonExe -m transattack.phase6 `
        --attack-dir $Adaptive `
        --output-dir $Defense `
        --expected-seeds 4580 4581 4582 4583 4584 `
        --views 64 `
        --graph-batch-size 64 `
        --minimum-success-clusters 18 `
        --minimum-success-seeds 4 `
        --minimum-clean-correctness 0.95 `
        --device cuda `
        --strict-cuda
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
