. "$PSScriptRoot\common.ps1"

$Baseline = 'results\phase5_dev_baseline_20260808'
$Selection = 'results\phase5_dev_selection_20260808'
$Ratios = @(0.50, 0.70, 0.85, 0.95)

if (-not (Test-Path -LiteralPath "$Baseline\decision.json")) {
    if (Test-Path -LiteralPath $Baseline) {
        throw "Incomplete baseline directory already exists: $Baseline"
    }
    & $PythonExe -m transattack.phase3 `
        --datasets cora citeseer `
        --seeds 4550 4551 4552 `
        --expected-seeds 4550 4551 4552 `
        --attack-types remote `
        --attack-objective classification_only `
        --budgets 1 2 4 8 `
        --nodes 192 `
        --targets 20 `
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
        --minimum-remote-clusters 8 `
        --minimum-remote-seeds 2 `
        --minimum-remote-datasets 2 `
        --device cuda `
        --strict-cuda `
        --data-root data `
        --output-dir $Baseline
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$Comparisons = @()
foreach ($Ratio in $Ratios) {
    $Tag = $Ratio.ToString('0.00', [System.Globalization.CultureInfo]::InvariantCulture).Replace('.', 'p')
    $Adaptive = "results\phase5_dev_adaptive_r${Tag}_20260808"
    $Comparison = "results\phase5_dev_comparison_r${Tag}_20260808"
    if (-not (Test-Path -LiteralPath "$Adaptive\decision.json")) {
        if (Test-Path -LiteralPath $Adaptive) {
            throw "Incomplete adaptive directory already exists: $Adaptive"
        }
        & $PythonExe -m transattack.phase3 `
            --datasets cora citeseer `
            --seeds 4550 4551 4552 `
            --expected-seeds 4550 4551 4552 `
            --attack-types remote `
            --attack-objective classification_constrained_stealth `
            --classification-retention-ratio $Ratio `
            --reference-run-dir $Baseline `
            --budgets 1 2 4 8 `
            --nodes 192 `
            --targets 20 `
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
            --minimum-remote-clusters 8 `
            --minimum-remote-seeds 2 `
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
            --expected-seeds 4550 4551 4552 `
            --expected-targets 20 `
            --minimum-joint-clusters 8 `
            --minimum-joint-seeds 2 `
            --minimum-joint-datasets 2
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    $Comparisons += "$Ratio=$Comparison"
}

if (-not (Test-Path -LiteralPath "$Selection\selection.json")) {
    if (Test-Path -LiteralPath $Selection) {
        throw "Incomplete selection directory already exists: $Selection"
    }
    & $PythonExe scripts\select_phase5_ratio.py `
        --comparisons $Comparisons `
        --output-dir $Selection
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
