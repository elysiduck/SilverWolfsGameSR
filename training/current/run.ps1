[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Environment not found. Run setup.ps1 first."
}
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONIOENCODING = "utf-8"
if (-not $Command -or $Command.Count -eq 0) {
    & $Python (Join-Path $Root "check_env.py")
    exit $LASTEXITCODE
}
& $Python @Command
exit $LASTEXITCODE
