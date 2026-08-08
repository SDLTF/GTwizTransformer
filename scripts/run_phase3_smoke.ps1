. "$PSScriptRoot\common.ps1"
& $PythonExe -m transattack.phase3 `
    --datasets sbm `
    --seeds 73 `
    --attack-types incident remote `
    --budgets 1 2 4 8 `
    --nodes 64 `
    --targets 4 `
    --candidate-pool 32 `
    --candidate-multiplier 5 `
    --graph-batch-size 32 `
    --channels 32 `
    --pe-channels 8 `
    --walk-length 4 `
    --layers 2 `
    --heads 4 `
    --dropout 0.0 `
    --epochs 8 `
    --patience 8 `
    --device cuda `
    --strict-cuda `
    --smoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
