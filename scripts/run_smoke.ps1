. "$PSScriptRoot\common.ps1"
& $PythonExe -m transattack.experiment `
    --datasets sbm `
    --pes heat rw `
    --budgets 1 2 `
    --seeds 3407 `
    --nodes 48 `
    --targets 2 `
    --candidate-pool 4 `
    --pe-dim 4 `
    --hidden-dim 24 `
    --heads 4 `
    --layers 3 `
    --epochs 8 `
    --patience 5 `
    --device cuda `
    --strict-cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
