. "$PSScriptRoot\common.ps1"
& $PythonExe -m transattack.aggregate `
    'results\phase0_20260808_195304' `
    'results\phase0_20260808_195431' `
    --output-dir 'results\phase0_3seed_20260808'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

