. "$PSScriptRoot\common.ps1"
& $PythonExe -m transattack.experiment `
    --datasets cora citeseer `
    --pes heat rw `
    --budgets 1 2 4 `
    --seeds 3410 3411 3412 `
    --nodes 128 `
    --targets 6 `
    --candidate-pool 24 `
    --pe-dim 8 `
    --hidden-dim 64 `
    --heads 4 `
    --layers 4 `
    --epochs 60 `
    --patience 15 `
    --device cuda `
    --strict-cuda `
    --output-dir 'results\phase0_phase2_holdout_20260808'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

