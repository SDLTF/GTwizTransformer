. "$PSScriptRoot\common.ps1"
& $PythonExe -m transattack.phase1 `
    --phase0-runs 'results\phase0_20260808_195304' 'results\phase0_20260808_195431' `
    --combined-metrics 'results\phase0_3seed_20260808\metrics_combined.csv' `
    --data-root 'data' `
    --candidate-multiplier 5 `
    --device cuda `
    --strict-cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

