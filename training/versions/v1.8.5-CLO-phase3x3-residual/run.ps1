[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Resolver = Join-Path $Root "Find-GameSR-Python.ps1"
$Python = & $Resolver -Kind Training -StartDirectory $Root
if ($LASTEXITCODE -ne 0 -or -not $Python) { throw "Training environment not found. Run setup.ps1 first." }
$Python = [string]$Python
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONIOENCODING = "utf-8"
if (-not $Command -or $Command.Count -eq 0) {
    & $Python (Join-Path $Root "check_env.py")
    exit $LASTEXITCODE
}
& $Python @Command
exit $LASTEXITCODE
