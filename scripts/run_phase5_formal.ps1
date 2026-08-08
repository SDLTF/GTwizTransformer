. "$PSScriptRoot\common.ps1"

$Selection = 'results\phase5_dev_selection_20260808\selection.json'
if (-not (Test-Path -LiteralPath $Selection)) {
    throw "Run Phase-5 development selection first"
}
$SelectionPayload = Get-Content -Raw -LiteralPath $Selection | ConvertFrom-Json
if ($SelectionPayload.status -ne 'retention_ratio_frozen') {
    throw "Development did not freeze a viable retention ratio"
}
$Ratio = [double]$SelectionPayload.selected_ratio
$Tag = $Ratio.ToString('0.00', [System.Globalization.CultureInfo]::InvariantCulture).Replace('.', 'p')
$Baseline = 'results\phase5_formal_baseline_20260808'
$Adaptive = "results\phase5_formal_adaptive_r${Tag}_20260808"
$Comparison = "results\phase5_formal_comparison_r${Tag}_20260808"

& $PythonExe -m transattack.phase3 `
    --datasets cora citeseer `
    --seeds 4560 4561 4562 4563 4564 `
    --expected-seeds 4560 4561 4562 4563 4564 `
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

& $PythonExe -m transattack.phase3 `
    --datasets cora citeseer `
    --seeds 4560 4561 4562 4563 4564 `
    --expected-seeds 4560 4561 4562 4563 4564 `
    --attack-types remote `
    --attack-objective classification_constrained_stealth `
    --classification-retention-ratio $Ratio `
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

& $PythonExe -m transattack.phase5 `
    --baseline-dir $Baseline `
    --adaptive-dir $Adaptive `
    --output-dir $Comparison `
    --expected-seeds 4560 4561 4562 4563 4564 `
    --expected-targets 30 `
    --minimum-joint-clusters 24 `
    --minimum-joint-seeds 4 `
    --minimum-joint-datasets 2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
