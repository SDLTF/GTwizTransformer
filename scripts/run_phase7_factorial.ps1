. $PSScriptRoot\common.ps1

$Output = Join-Path $ProjectRoot 'results\phase7_heuristic_factorial_20260809'
$Arguments = @(
    '-m', 'transattack.phase7',
    '--source-runs',
    'results\phase5_r0p95_holdout_baseline_20260809',
    'results\phase6_pubmed_baseline_20260809',
    '--models',
    'cora:4570', 'cora:4572',
    'citeseer:4570', 'citeseer:4572',
    'pubmed:4580', 'pubmed:4582',
    '--targets-per-model', '6',
    '--budgets', '1', '2', '4', '8',
    '--candidate-pool', '128',
    '--maximum-candidate-pool', '512',
    '--graph-batch-size', '512',
    '--beam-width', '8',
    '--device', 'cuda',
    '--strict-cuda',
    '--output-dir', $Output
)

if (Test-Path -LiteralPath (Join-Path $Output 'config.json')) {
    $Arguments += '--resume'
}

& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
