[CmdletBinding()]
param([string[]]$SearchRoots = @($env:USERPROFILE, "C:\ProgramData"))

$ErrorActionPreference = "Continue"
$Results = [System.Collections.Generic.List[object]]::new()
$Candidates = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

foreach ($Name in @("CONDA_PREFIX", "VIRTUAL_ENV")) {
    $Value = [Environment]::GetEnvironmentVariable($Name)
    if ($Value) { [void]$Candidates.Add($Value) }
}
foreach ($Base in @(
    "$env:USERPROFILE\anaconda3\envs",
    "$env:USERPROFILE\miniconda3\envs",
    "$env:USERPROFILE\.conda\envs",
    "C:\ProgramData\anaconda3\envs",
    "C:\ProgramData\miniconda3\envs"
)) {
    if (Test-Path $Base) {
        Get-ChildItem $Base -Directory -ErrorAction SilentlyContinue | ForEach-Object { [void]$Candidates.Add($_.FullName) }
    }
}
foreach ($Root in $SearchRoots) {
    if (-not (Test-Path $Root)) { continue }
    Get-ChildItem $Root -Directory -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @(".venv", "venv") } |
        ForEach-Object { [void]$Candidates.Add($_.FullName) }
}

foreach ($EnvPath in $Candidates) {
    $Python = @("$EnvPath\Scripts\python.exe", "$EnvPath\python.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $Python) { continue }
    $Probe = 'import json,sys; r={"python":sys.version.split()[0]}; exec("try:\n import torch\n r.update(torch= torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())\nexcept Exception as e:\n r[\"torch\"]=\"not importable: \"+str(e)"); print(json.dumps(r))'
    $Info = & $Python -c $Probe 2>$null
    $Results.Add([pscustomobject]@{ Environment=$EnvPath; Python=$Python; Details=($Info -join " ") })
}

$Output = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "environment-audit.csv"
$Results | Sort-Object Environment | Export-Csv $Output -NoTypeInformation -Encoding utf8
$Results | Sort-Object Environment | Format-Table -AutoSize
Write-Host "Read-only audit saved to: $Output" -ForegroundColor Green
Write-Host "Nothing was deleted." -ForegroundColor Yellow

