param(
    [Parameter(Mandatory=$true)][string]$Image,
    [ValidateSet('tracks','regions')][string]$Algorithm = 'tracks',
    [string]$Output = '',
    [switch]$RotateCcw,
    [switch]$FullDebug
)
$ErrorActionPreference = 'Stop'
$configName = if ($Algorithm -eq 'regions') { 'regions.yaml' } else { 'default.yaml' }
if (-not $Output) { $Output = Join-Path $PSScriptRoot "output\${Algorithm}_single" }
$counterArgs = @('run', '-n', 'pytorch', 'python',
    (Join-Path $PSScriptRoot 'main.py'), '--image', $Image,
    '--config', (Join-Path $PSScriptRoot "configs\$configName"), '--output', $Output)
if ($RotateCcw) { $counterArgs += '--rotate-ccw' }
if ($FullDebug) { $counterArgs += '--full-debug' }
& conda @counterArgs
if ($LASTEXITCODE -ne 0) { throw "Counter failed with exit code $LASTEXITCODE" }
