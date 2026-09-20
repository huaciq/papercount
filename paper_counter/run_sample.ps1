$ErrorActionPreference = 'Stop'
# This environment contains a legacy-encoded .pth file; disable forced UTF-8
# for this process only. No global environment or package changes are made.
$previousUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONUTF8 = '0'
    $counterArgs = @('run', '-n', 'pytorch', 'python', '-X', 'utf8=0',
        (Join-Path $PSScriptRoot 'main.py'),
        '--image', (Join-Path $PSScriptRoot '..\数据\1.png'),
        '--config', (Join-Path $PSScriptRoot 'configs\default.yaml'),
        '--output', (Join-Path $PSScriptRoot 'output'))
    & conda @counterArgs
    if ($LASTEXITCODE -ne 0) { throw "Counter failed with exit code $LASTEXITCODE" }
}
finally {
    $env:PYTHONUTF8 = $previousUtf8
}
