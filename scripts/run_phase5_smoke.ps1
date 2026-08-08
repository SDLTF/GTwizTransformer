. "$PSScriptRoot\common.ps1"

$Baseline = 'results\phase5_smoke_baseline_20260808'
$Adaptive = 'results\phase5_smoke_adaptive_20260808'
$Comparison = 'results\phase5_smoke_comparison_20260808'

& $PythonExe -m transattack.phase3 `
    --datasets cora `
    --seeds 4599 `
    --expected-seeds 4599 `
    --attack-types remote `
    --attack-objective classification_only `
    --budgets 1 2 `
    --nodes 64 `
    --targets 3 `
    --candidate-pool 16 `
    --candidate-multiplier 5 `
    --graph-batch-size 16 `
    --channels 32 `
    --pe-channels 8 `
    --walk-length 4 `
    --layers 2 `
    --heads 4 `
    --epochs 4 `
    --patience 2 `
    --minimum-remote-clusters 1 `
    --minimum-remote-seeds 1 `
    --minimum-remote-datasets 1 `
    --device cuda `
    --strict-cuda `
    --data-root data `
    --output-dir $Baseline `
    --smoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe -m transattack.phase3 `
    --datasets cora `
    --seeds 4599 `
    --expected-seeds 4599 `
    --attack-types remote `
    --attack-objective classification_constrained_stealth `
    --classification-retention-ratio 0.85 `
    --reference-run-dir $Baseline `
    --budgets 1 2 `
    --nodes 64 `
    --targets 3 `
    --candidate-pool 16 `
    --candidate-multiplier 5 `
    --graph-batch-size 16 `
    --channels 32 `
    --pe-channels 8 `
    --walk-length 4 `
    --layers 2 `
    --heads 4 `
    --epochs 4 `
    --patience 2 `
    --minimum-remote-clusters 1 `
    --minimum-remote-seeds 1 `
    --minimum-remote-datasets 1 `
    --device cuda `
    --strict-cuda `
    --data-root data `
    --output-dir $Adaptive `
    --smoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe -m transattack.phase5 `
    --baseline-dir $Baseline `
    --adaptive-dir $Adaptive `
    --output-dir $Comparison `
    --expected-seeds 4599 `
    --expected-targets 3 `
    --minimum-joint-clusters 1 `
    --minimum-joint-seeds 1 `
    --minimum-joint-datasets 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
