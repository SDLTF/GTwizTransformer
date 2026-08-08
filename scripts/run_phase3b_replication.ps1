. "$PSScriptRoot\common.ps1"
& $PythonExe -m transattack.phase3 `
    --datasets cora citeseer `
    --seeds 4520 4521 4522 `
    --expected-seeds 4520 4521 4522 `
    --attack-types incident remote `
    --budgets 1 2 4 8 `
    --nodes 192 `
    --targets 10 `
    --candidate-pool 64 `
    --candidate-multiplier 10 `
    --graph-batch-size 64 `
    --channels 96 `
    --pe-channels 16 `
    --walk-length 8 `
    --layers 4 `
    --heads 8 `
    --epochs 100 `
    --patience 20 `
    --minimum-remote-clusters 18 `
    --minimum-remote-seeds 3 `
    --minimum-remote-datasets 2 `
    --device cuda `
    --strict-cuda `
    --data-root data `
    --output-dir 'results\phase3b_graphgps_replication_20260808'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
