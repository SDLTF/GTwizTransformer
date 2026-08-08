. "$PSScriptRoot\common.ps1"
& $PythonExe -m transattack.aggregate `
    'results\phase0_phase2_holdout_20260808' `
    --output-dir 'results\phase0_phase2_holdout_aggregate_20260808'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

