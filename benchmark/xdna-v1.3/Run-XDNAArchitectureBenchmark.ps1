[CmdletBinding()]
param(
    [int]$Warmup = 20,
    [int]$Runs = 600,
    [switch]$ForceRecompile,
    [switch]$NoTaskManager
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$manifestPath = Join-Path $root 'architecture_manifest.json'
$coreRunner = Join-Path $root 'Core-Benchmark-v2.4.1.ps1'
$resultsDir = Join-Path $root 'results'
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

function Fail([string]$Message, [int]$Code = 2) {
    Write-Host "[FAIL] $Message" -ForegroundColor Red
    exit $Code
}

foreach ($required in @($manifestPath, $coreRunner)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        Fail "Required file is missing: $required"
    }
}
if ($Warmup -lt 1 -or $Runs -lt 1) { Fail 'Warmup and Runs must both be at least 1.' }

$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$probes = @($manifest.probes)
if ($probes.Count -eq 0) { Fail 'The architecture manifest contains no probes.' }

Write-Host ''
Write-Host 'GameSR XDNA Architecture Benchmark v1.3' -ForegroundColor Cyan
Write-Host '========================================' -ForegroundColor Cyan
Write-Host 'Contract : NCHW [1,3,720,1280] -> same-size RGB residual'
Write-Host 'Precision: AMD Quark XINT8/QDQ'
Write-Host 'Runner   : original GameSR Benchmark v2.4.1 perf_test + ORT profile'
Write-Host "Warmup   : $Warmup serial inferences per architecture"
Write-Host "Measure  : $Runs serial inferences per architecture"
Write-Host 'CPU test : disabled (architecture ranking is NPU-only)'
Write-Host ''
Write-Host 'The probes are exported by the exact GameSR v1.7.1 PyTorch model class.' -ForegroundColor Yellow
Write-Host 'A known-working Magpie SESR model and a CPU smoke test run first.' -ForegroundColor Yellow
Write-Host 'Task Manager utilization is useful supporting evidence, but the ranking uses stable measured latency.' -ForegroundColor Yellow
Write-Host 'Close games, video playback, training and other AI applications before continuing.' -ForegroundColor Yellow

if (-not $NoTaskManager) {
    try {
        Start-Process taskmgr.exe | Out-Null
        Write-Host ''
        Write-Host 'Task Manager was opened. Select Performance > NPU to watch utilization.' -ForegroundColor Green
    } catch {
        Write-Host '[INFO] Task Manager could not be opened automatically; the benchmark will continue.' -ForegroundColor Yellow
    }
}

Write-Host ''
Write-Host 'Press Enter to start all five tests, or Ctrl+C to cancel.' -ForegroundColor Cyan
[void](Read-Host)

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$diagnostics = [ordered]@{
    cpu_smoke_status = 'NOT_RUN'
    cpu_smoke_exit_code = $null
    control_npu_status = 'NOT_RUN'
    control_npu_exit_code = $null
    control_summary_json = $null
}

# First prove that the exact perf_test executable can load the new probe on CPU.
# This separates an invalid ONNX file from an NPU compiler/runtime failure.
$probeA = Join-Path $root ([string]$probes[0].model)
$cpuLog = Join-Path $resultsDir ("DIAGNOSTIC-CPU-A-$stamp.log")
$perfExe = Join-Path $root 'runtime\bin\onnxruntime_perf_test.exe'
Write-Host ''
Write-Host '[DIAGNOSTIC 1/2] Loading probe A with CPUExecutionProvider (one run)' -ForegroundColor Cyan
$savedErrorAction = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    $cpuOutput = @(& $perfExe '-I' '-m' 'times' '-c' '1' '-S' '42' '-e' 'cpu' '-r' '1' $probeA 2>&1)
    $cpuCode = [int]$LASTEXITCODE
} finally {
    $ErrorActionPreference = $savedErrorAction
}
$cpuOutput | Out-File -LiteralPath $cpuLog -Encoding utf8
foreach ($line in $cpuOutput) { Write-Host ([string]$line) }
$diagnostics.cpu_smoke_exit_code = $cpuCode
$diagnostics.cpu_smoke_status = if ($cpuCode -eq 0) { 'PASS' } else { 'FAIL' }

# Then run the original benchmark path against the exact SESR model already
# demonstrated to execute fully on VitisAI in Magpie. This distinguishes a
# perf_test/provider problem from a GameSR full-frame graph compiler problem.
$controlDir = Join-Path $root 'control'
Write-Host ''
Write-Host '[DIAGNOSTIC 2/2] Original v2.4.1 NPU path with known-working Magpie SESR model' -ForegroundColor Cyan
$controlArgs = @(
    '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass',
    '-File', $coreRunner,
    '-DeploymentPath', $controlDir,
    '-OnlyInt8', '-SkipCpu', '-Warmup', '2', '-Runs', '5'
)
& powershell.exe @controlArgs
$controlCode = [int]$LASTEXITCODE
$controlSummary = Get-ChildItem -LiteralPath $resultsDir -Filter 'GameSR-v24-Magpie-SESR-known-working-*-SUMMARY.json' -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$controlStatus = 'NO_SUMMARY'
if ($controlSummary) {
    $controlPayload = Get-Content -Raw -LiteralPath $controlSummary.FullName | ConvertFrom-Json
    $controlResult = @($controlPayload.results | Where-Object { $_.provider -eq 'vitisai' }) | Select-Object -First 1
    if ($controlResult) { $controlStatus = [string]$controlResult.status }
    $diagnostics.control_summary_json = $controlSummary.FullName
}
$diagnostics.control_npu_exit_code = $controlCode
$diagnostics.control_npu_status = $controlStatus
Write-Host ("[DIAGNOSTIC RESULT] CPU probe={0}; known-working NPU control={1}" -f $diagnostics.cpu_smoke_status, $diagnostics.control_npu_status) -ForegroundColor Yellow

$rows = @()
$index = 0
foreach ($probe in $probes) {
    $index++
    $modelPath = Join-Path $root ([string]$probe.model)
    $modelDir = Split-Path -Parent $modelPath
    if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
        $rows += [pscustomobject]@{
            rank=$null; architecture=[string]$probe.name; channels=[int]$probe.channels
            conv_layers=[int]$probe.conv_layers; parameters=[int64]$probe.parameters
            gmac_per_frame=[math]::Round(([double]$probe.macs_720p / 1.0e9), 4)
            status='MODEL_MISSING'; average_ms=$null; p50_ms=$null; p95_ms=$null
            pure_fps=$null; relative_speed_vs_A=$null; effective_gmac_s=$null
            npu_event_share_percent=$null; npu_duration_share_percent=$null
            npu_events=$null; cpu_events=$null; cache_present_before=$null
            warmup_wall_seconds=$null; summary_json=$null
        }
        continue
    }

    Write-Host ''
    Write-Host ("[{0}/5] {1}: C{2}, {3} convolution layers, {4:N0} parameters, {5:F3} GMAC/frame" -f `
        $index, $probe.name, $probe.channels, $probe.conv_layers, $probe.parameters, ([double]$probe.macs_720p / 1.0e9)) -ForegroundColor Cyan
    Write-Host 'Watch the NPU graph now. The measured phase follows warm-up/compilation.' -ForegroundColor Green
    Start-Sleep -Seconds 2

    $childArgs = @(
        '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', $coreRunner,
        '-DeploymentPath', $modelDir,
        '-OnlyInt8', '-SkipCpu',
        '-Warmup', [string]$Warmup,
        '-Runs', [string]$Runs
    )
    if ($ForceRecompile) { $childArgs += '-ForceRecompile' }
    & powershell.exe @childArgs
    $childCode = [int]$LASTEXITCODE

    $pattern = "GameSR-v24-$($probe.name)-*-SUMMARY.json"
    $summaryFile = Get-ChildItem -LiteralPath $resultsDir -Filter $pattern -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $summaryFile) {
        $rows += [pscustomobject]@{
            rank=$null; architecture=[string]$probe.name; channels=[int]$probe.channels
            conv_layers=[int]$probe.conv_layers; parameters=[int64]$probe.parameters
            gmac_per_frame=[math]::Round(([double]$probe.macs_720p / 1.0e9), 4)
            status=("NO_SUMMARY_EXIT_" + $childCode); average_ms=$null; p50_ms=$null; p95_ms=$null
            pure_fps=$null; relative_speed_vs_A=$null; effective_gmac_s=$null
            npu_event_share_percent=$null; npu_duration_share_percent=$null
            npu_events=$null; cpu_events=$null; cache_present_before=$null
            warmup_wall_seconds=$null; summary_json=$null
        }
        continue
    }
    $payload = Get-Content -Raw -LiteralPath $summaryFile.FullName | ConvertFrom-Json
    $result = @($payload.results | Where-Object { $_.provider -eq 'vitisai' }) | Select-Object -First 1
    $avg = if ($null -eq $result.average_ms) { $null } else { [double]$result.average_ms }
    $effective = if ($null -eq $avg -or $avg -le 0) { $null } else { [math]::Round(([double]$probe.macs_720p / 1.0e6 / $avg), 3) }
    $rows += [pscustomobject]@{
        rank=$null; architecture=[string]$probe.name; channels=[int]$probe.channels
        conv_layers=[int]$probe.conv_layers; parameters=[int64]$probe.parameters
        gmac_per_frame=[math]::Round(([double]$probe.macs_720p / 1.0e9), 4)
        status=[string]$result.status; average_ms=$avg; p50_ms=$result.p50_ms; p95_ms=$result.p95_ms
        pure_fps=$result.pure_fps; relative_speed_vs_A=$null; effective_gmac_s=$effective
        npu_event_share_percent=$result.npu_event_share_percent
        npu_duration_share_percent=$result.npu_duration_share_percent
        npu_events=$result.npu_events; cpu_events=$result.cpu_events
        cache_present_before=$result.cache_present_before
        warmup_wall_seconds=$result.warmup_wall_seconds
        summary_json=$summaryFile.FullName
    }
}

$baseline = @($rows | Where-Object { $_.architecture -eq 'A-C8-L8' -and $_.status -eq 'NPU_ONLY' -and $null -ne $_.average_ms }) | Select-Object -First 1
if ($baseline) {
    foreach ($row in $rows) {
        if ($row.status -eq 'NPU_ONLY' -and $null -ne $row.average_ms -and [double]$row.average_ms -gt 0) {
            $row.relative_speed_vs_A = [math]::Round(([double]$baseline.average_ms / [double]$row.average_ms), 4)
        }
    }
}
$valid = @($rows | Where-Object { $_.status -eq 'NPU_ONLY' -and $null -ne $_.average_ms } | Sort-Object {[double]$_.average_ms})
for ($position = 0; $position -lt $valid.Count; $position++) { $valid[$position].rank = $position + 1 }

$base = Join-Path $resultsDir ("XDNA-Architecture-$stamp")
$csvPath = $base + '-RANKING.csv'
$jsonPath = $base + '-FULL.json'
$textPath = $base + '-SUMMARY.txt'
$rows | Sort-Object @{Expression={if ($null -eq $_.rank) {999} else {[int]$_.rank}}} | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8
[ordered]@{
    schema_version=3; benchmark=[string]$manifest.benchmark; created=(Get-Date).ToString('o')
    contract=[string]$manifest.contract; warmup=$Warmup; runs=$Runs; force_recompile=[bool]$ForceRecompile
    execution_path='Original GameSR Benchmark v2.4.1 onnxruntime_perf_test plus ORT profile'
    diagnostics=$diagnostics; valid_ranking_count=$valid.Count; results=$rows
} | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

$lines = @(
    'GameSR XDNA Architecture Benchmark v1.3',
    '========================================',
    ('Finished: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
    ('Valid NPU-only results: ' + $valid.Count + '/' + $rows.Count),
    ('CPU model smoke test: ' + $diagnostics.cpu_smoke_status + ' (exit ' + $diagnostics.cpu_smoke_exit_code + ')'),
    ('Known-working NPU control: ' + $diagnostics.control_npu_status + ' (exit ' + $diagnostics.control_npu_exit_code + ')'),
    '',
    'RANKING (lower average_ms is better)',
    '------------------------------------'
)
foreach ($row in ($rows | Sort-Object @{Expression={if ($null -eq $_.rank) {999} else {[int]$_.rank}}})) {
    $rankText = if ($null -eq $row.rank) { '-' } else { [string]$row.rank }
    $avgText = if ($null -eq $row.average_ms) { 'N/A' } else { '{0:F4}' -f [double]$row.average_ms }
    $fpsText = if ($null -eq $row.pure_fps) { 'N/A' } else { '{0:F2}' -f [double]$row.pure_fps }
    $relativeText = if ($null -eq $row.relative_speed_vs_A) { 'N/A' } else { '{0:F3}x' -f [double]$row.relative_speed_vs_A }
    $lines += ("#{0} {1} | status={2} | avg={3} ms | FPS={4} | vs_A={5} | effective={6} GMAC/s" -f `
        $rankText, $row.architecture, $row.status, $avgText, $fpsText, $relativeText, $row.effective_gmac_s)
}
$lines += ''
$lines += 'Only NPU_ONLY rows are ranked. NPU_PARTIAL/CPU_FALLBACK results are invalid for architecture selection.'
$lines += 'npu_event_share_percent and npu_duration_share_percent are runtime profile shares, not Task Manager hardware utilization.'
$lines += 'Use the CSV to add any Task Manager utilization observations beside the measured latency.'
$lines += ('CSV : ' + $csvPath)
$lines += ('JSON: ' + $jsonPath)
$lines | Set-Content -LiteralPath $textPath -Encoding UTF8

Write-Host ''
Write-Host 'FINAL NPU-ONLY RANKING' -ForegroundColor Cyan
Write-Host '======================' -ForegroundColor Cyan
foreach ($row in $valid) {
    Write-Host ("#{0} {1}: {2:F4} ms, {3:F2} FPS, {4:F3}x versus A, {5:F1} GMAC/s" -f `
        $row.rank, $row.architecture, [double]$row.average_ms, [double]$row.pure_fps, [double]$row.relative_speed_vs_A, [double]$row.effective_gmac_s) -ForegroundColor Green
}
foreach ($row in @($rows | Where-Object { $_.status -ne 'NPU_ONLY' })) {
    Write-Host ("[INVALID] {0}: {1}" -f $row.architecture, $row.status) -ForegroundColor Red
}
Write-Host ''
Write-Host "Summary: $textPath"
Write-Host "CSV    : $csvPath"
Write-Host "JSON   : $jsonPath"

if ($valid.Count -ne $rows.Count) { exit 4 }
exit 0
