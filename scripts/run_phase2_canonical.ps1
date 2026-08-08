. "$PSScriptRoot\common.ps1"
& $PythonExe -m transattack.phase2 `
    --phase0-runs 'results\phase0_phase2_holdout_20260808' `
    --combined-metrics 'results\phase0_phase2_holdout_aggregate_20260808\metrics_combined.csv' `
    --data-root 'data' `
    --candidate-multipliers 5 10 20 `
    --expected-seeds 3410 3411 3412 `
    --device cuda `
    --strict-cuda `
    --output-dir 'results\phase2_holdout_20260808'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

