[CmdletBinding()]
param(
    [Parameter(Position=0)] [string]$PayloadPath = '',
    [int]$Warmup = 20,
    [int]$Runs = 600,
    [int]$CooldownSeconds = 2,
    [switch]$ForceRecompile,
    [switch]$SkipHashValidation
)

$ErrorActionPreference = 'Stop'
$runnerVersion = '1.0.2'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Fail([string]$Message, [int]$Code = 2) {
    Write-Host "[FAIL] $Message" -ForegroundColor Red
    exit $Code
}
function FullPath([string]$PathValue) { return [System.IO.Path]::GetFullPath($PathValue) }
function Is-UnderPath([string]$Child, [string]$Parent) {
    $c = (FullPath $Child).TrimEnd('\')
    $p = (FullPath $Parent).TrimEnd('\')
    if ($c.Equals($p, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    return $c.StartsWith($p + '\', [System.StringComparison]::OrdinalIgnoreCase)
}

if ($Warmup -lt 1) { Fail 'Warmup must be at least 1.' }
if ($Runs -lt 1) { Fail 'Runs must be at least 1.' }
if ($CooldownSeconds -lt 0) { Fail 'CooldownSeconds cannot be negative.' }

$benchmark = Join-Path $root 'Benchmark-GameSR-v24-Sweep.ps1'
$perfExe = Join-Path $root 'runtime\bin\onnxruntime_perf_test.exe'
$config = Join-Path $root 'runtime\vaip_config_stride.json'
$xclbin = Join-Path $root 'runtime\AMD_AIE2P_4x4_Overlay.xclbin'
foreach ($required in @($benchmark, $perfExe, $config, $xclbin)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        Fail "Required benchmark/runtime file is missing: $required"
    }
}

$resultsDir = Join-Path $root 'results'
$payloadCache = Join-Path $root '_ocp'
New-Item -ItemType Directory -Force -Path $resultsDir, $payloadCache | Out-Null

function Resolve-ManifestFromFolder([string]$Folder) {
    if (-not (Test-Path -LiteralPath $Folder -PathType Container)) { return $null }
    foreach ($name in @('output_channel_manifest.json','handheld_payload\output_channel_manifest.json')) {
        $p = Join-Path $Folder $name
        if (Test-Path -LiteralPath $p -PathType Leaf) { return (Get-Item -LiteralPath $p) }
    }
    $found = @(Get-ChildItem -LiteralPath $Folder -Filter 'output_channel_manifest.json' -File -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
    if ($found.Count -gt 0) { return $found[0] }
    return $null
}

function Resolve-PayloadManifest([string]$InputPath) {
    if (-not [string]::IsNullOrWhiteSpace($InputPath)) {
        $candidate = FullPath ($InputPath.Trim().Trim('"'))
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $ext = [System.IO.Path]::GetExtension($candidate).ToLowerInvariant()
            if ($ext -eq '.json') { return (Get-Item -LiteralPath $candidate) }
            if ($ext -eq '.zip') {
                # Keep the extraction path deliberately short. Windows PowerShell 5.1's
                # Microsoft.PowerShell.Archive module can fail near MAX_PATH and reports
                # a misleading Remove-Item "path does not exist" error while expanding.
                $zipHash = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant().Substring(0,16)
                $dest = Join-Path $payloadCache $zipHash
                $m = Resolve-ManifestFromFolder $dest
                if ($m) { return $m }

                # A prior interrupted extraction may have left a partial cache. Never
                # trust a directory merely because it exists; only a valid manifest counts.
                if (Test-Path -LiteralPath $dest) {
                    Write-Host "[PAYLOAD] Removing incomplete cache -> $dest" -ForegroundColor Yellow
                    Remove-Item -LiteralPath $dest -Recurse -Force -ErrorAction SilentlyContinue
                }

                $tmp = Join-Path $payloadCache ('.tmp-' + $zipHash + '-' + $PID)
                if (Test-Path -LiteralPath $tmp) {
                    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
                }
                New-Item -ItemType Directory -Force -Path $tmp | Out-Null
                try {
                    Write-Host "[PAYLOAD] Extracting -> $tmp" -ForegroundColor Cyan
                    # Destination is new and empty, so -Force is intentionally NOT used.
                    # This avoids the Archive module's overwrite/Remove-Item code path.
                    Expand-Archive -LiteralPath $candidate -DestinationPath $tmp
                    $tmpManifest = Resolve-ManifestFromFolder $tmp
                    if (-not $tmpManifest) {
                        Fail "ZIP has no output_channel_manifest.json after extraction: $candidate"
                    }
                    Move-Item -LiteralPath $tmp -Destination $dest
                } catch {
                    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
                    Fail ("Could not extract payload ZIP. " + $_.Exception.Message)
                }
                $m = Resolve-ManifestFromFolder $dest
                if ($m) { return $m }
                Fail "ZIP extraction completed but manifest is missing: $candidate"
            }
            Fail "Unsupported payload file: $candidate"
        }
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            $m = Resolve-ManifestFromFolder $candidate
            if ($m) { return $m }
            Fail "No output_channel_manifest.json under: $candidate"
        }
        Fail "Payload path not found: $candidate"
    }

    foreach ($folder in @((Join-Path $root 'handheld_payload'), $root)) {
        $m = Resolve-ManifestFromFolder $folder
        if ($m) { return $m }
    }
    $zips = @(Get-ChildItem -LiteralPath $root -Filter 'GameSR-NPU-Output-Channel-Probe-Models-*.zip' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
    if ($zips.Count -gt 0) { return Resolve-PayloadManifest $zips[0].FullName }
    Fail 'No output-channel probe payload found. Put the Builder ZIP beside this script or drag it onto RUN-GAMESR-OUTPUT-CHANNEL-PROBE.cmd.'
}

$manifestFile = Resolve-PayloadManifest $PayloadPath
$payloadRoot = $manifestFile.DirectoryName
try { $manifest = Get-Content -Raw -LiteralPath $manifestFile.FullName | ConvertFrom-Json }
catch { Fail "Invalid manifest: $($manifestFile.FullName). $($_.Exception.Message)" }

$models = @($manifest.models | Where-Object { [string]$_.status -eq 'READY' } | Sort-Object {[int]$_.output_channels})
if ($models.Count -eq 0) { Fail 'Manifest contains no READY models.' }

Write-Host ''
Write-Host "GameSR NPU Output Channel Probe Runner v$runnerVersion" -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host "Payload     : $payloadRoot"
Write-Host "Fixed body  : C$($manifest.fixed_channels)-L$($manifest.fixed_conv_layers)"
Write-Host "Outputs     : $(@($manifest.output_channels) -join ',')"
Write-Host "Input       : $(@($manifest.input_shape) -join 'x')"
Write-Host "Warmup/Runs : $Warmup / $Runs"
Write-Host "Cooldown    : $CooldownSeconds sec"
Write-Host 'Scope       : pure NPU graph/runtime only; NOT Magpie transfer/decoder timing' -ForegroundColor Yellow
Write-Host ''

$resolved = @()
foreach ($m in $models) {
    $rel = [string]$m.model_relpath
    $path = FullPath (Join-Path $payloadRoot ($rel -replace '/', '\'))
    if (-not (Is-UnderPath $path $payloadRoot)) { Fail "Unsafe model path: $rel" }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Fail "Model missing: $path" }
    if (-not $SkipHashValidation -and -not [string]::IsNullOrWhiteSpace([string]$m.sha256)) {
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne ([string]$m.sha256).ToLowerInvariant()) { Fail "SHA256 mismatch: $($m.architecture)" }
    }
    $resolved += [pscustomobject]@{ entry=$m; model_path=$path; model_dir=(Split-Path -Parent $path) }
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$rows = @()
$artifacts = New-Object 'System.Collections.Generic.List[string]'
$order = 0
foreach ($item in $resolved) {
    $order++
    $m = $item.entry
    $arch = [string]$m.architecture
    Write-Host ''
    Write-Host ("===== [{0}/{1}] {2} | O={3} | output x{4:N3} =====" -f $order,$resolved.Count,$arch,$m.output_channels,$m.output_ratio_vs_o3) -ForegroundColor Cyan

    $pattern = "GameSR-v24-$($arch -replace '[^A-Za-z0-9._-]', '_')-*-SUMMARY.json"
    $before = @{}
    foreach ($f in @(Get-ChildItem -LiteralPath $resultsDir -Filter $pattern -File -ErrorAction SilentlyContinue)) { $before[$f.FullName] = $true }

    $args = @('-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-File',$benchmark,$item.model_dir,'-Warmup',"$Warmup",'-Runs',"$Runs",'-OnlyInt8','-SkipCpu')
    if ($ForceRecompile) { $args += '-ForceRecompile' }
    $launchLog = Join-Path $resultsDir ("OutputChannelRunner-{0}-{1}-launch.log" -f $stamp, ($arch -replace '[^A-Za-z0-9._-]', '_'))
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $benchOutput = @(& powershell.exe @args 2>&1)
        $benchExit = [int]$LASTEXITCODE
    } finally { $ErrorActionPreference = $saved }
    $benchOutput | Out-File -LiteralPath $launchLog -Encoding utf8
    foreach ($line in $benchOutput) { Write-Host ([string]$line) }
    [void]$artifacts.Add($launchLog)

    $new = @(Get-ChildItem -LiteralPath $resultsDir -Filter $pattern -File -ErrorAction SilentlyContinue | Where-Object { -not $before.ContainsKey($_.FullName) } | Sort-Object LastWriteTime -Descending)
    if ($new.Count -eq 0) {
        $rows += [pscustomobject]@{
            test_order=$order; architecture=$arch; output_channels=[int]$m.output_channels; parameters=[int64]$m.parameters; gmac_per_frame=[double]$m.gmac_per_frame
            output_elements=[int64]$m.output_elements; output_fp32_mib=[double]$m.output_fp32_mib; output_ratio_vs_o3=[double]$m.output_ratio_vs_o3
            status='NO_SUMMARY'; benchmark_exit=$benchExit; average_ms=$null; p50_ms=$null; p95_ms=$null; pure_fps=$null; effective_gmac_s=$null
            npu_event_share_percent=$null; npu_duration_share_percent=$null; npu_events=$null; cpu_events=$null; summary_json=''
        }
        continue
    }

    $summaryFile = $new[0]
    [void]$artifacts.Add($summaryFile.FullName)
    try { $sum = Get-Content -Raw -LiteralPath $summaryFile.FullName | ConvertFrom-Json }
    catch { Fail "Could not parse SUMMARY.json: $($summaryFile.FullName)" }
    $r = @($sum.results | Where-Object { $_.variant -eq 'int8' -and $_.provider -eq 'vitisai' }) | Select-Object -First 1
    if ($null -eq $r) {
        $rows += [pscustomobject]@{
            test_order=$order; architecture=$arch; output_channels=[int]$m.output_channels; parameters=[int64]$m.parameters; gmac_per_frame=[double]$m.gmac_per_frame
            output_elements=[int64]$m.output_elements; output_fp32_mib=[double]$m.output_fp32_mib; output_ratio_vs_o3=[double]$m.output_ratio_vs_o3
            status='INVALID_SUMMARY'; benchmark_exit=$benchExit; average_ms=$null; p50_ms=$null; p95_ms=$null; pure_fps=$null; effective_gmac_s=$null
            npu_event_share_percent=$null; npu_duration_share_percent=$null; npu_events=$null; cpu_events=$null; summary_json=$summaryFile.FullName
        }
        continue
    }
    $eff = $null
    if ($null -ne $r.average_ms -and [double]$r.average_ms -gt 0) { $eff = [math]::Round(([double]$m.gmac_per_frame * 1000.0 / [double]$r.average_ms), 3) }
    $rows += [pscustomobject]@{
        test_order=$order; architecture=$arch; output_channels=[int]$m.output_channels; parameters=[int64]$m.parameters; gmac_per_frame=[double]$m.gmac_per_frame
        output_elements=[int64]$m.output_elements; output_fp32_mib=[double]$m.output_fp32_mib; output_ratio_vs_o3=[double]$m.output_ratio_vs_o3
        status=[string]$r.status; benchmark_exit=$benchExit; average_ms=[double]$r.average_ms; p50_ms=[double]$r.p50_ms; p95_ms=[double]$r.p95_ms; pure_fps=[double]$r.pure_fps
        effective_gmac_s=$eff; npu_event_share_percent=[double]$r.npu_event_share_percent; npu_duration_share_percent=[double]$r.npu_duration_share_percent
        npu_events=[int]$r.npu_events; cpu_events=[int]$r.cpu_events; summary_json=$summaryFile.FullName
    }
    if ($CooldownSeconds -gt 0 -and $order -lt $resolved.Count) { Start-Sleep -Seconds $CooldownSeconds }
}

$baseline = @($rows | Where-Object { $_.output_channels -eq 3 -and $_.status -eq 'NPU_ONLY' -and $null -ne $_.average_ms }) | Select-Object -First 1
foreach ($row in $rows) {
    $deltaMs = $null; $deltaPct = $null
    if ($baseline -and $null -ne $row.average_ms) {
        $deltaMs = [math]::Round([double]$row.average_ms - [double]$baseline.average_ms, 4)
        if ([double]$baseline.average_ms -gt 0) { $deltaPct = [math]::Round(100.0 * $deltaMs / [double]$baseline.average_ms, 3) }
    }
    Add-Member -InputObject $row -NotePropertyName delta_ms_vs_o3 -NotePropertyValue $deltaMs -Force
    Add-Member -InputObject $row -NotePropertyName delta_percent_vs_o3 -NotePropertyValue $deltaPct -Force
}

$base = Join-Path $resultsDir "GameSR-NPU-Output-Channel-Probe-$stamp"
$csv = $base + '-OUTPUT-CHANNELS.csv'
$full = $base + '-FULL.json'
$txt = $base + '-SUMMARY.txt'
$rows | Sort-Object output_channels | Select-Object test_order,architecture,output_channels,parameters,gmac_per_frame,output_elements,output_fp32_mib,output_ratio_vs_o3,status,average_ms,p50_ms,p95_ms,pure_fps,effective_gmac_s,delta_ms_vs_o3,delta_percent_vs_o3,npu_event_share_percent,npu_duration_share_percent,npu_events,cpu_events,benchmark_exit,summary_json | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8

[ordered]@{
    schema_version=1; probe='GameSR NPU Output Channel Probe'; runner_version=$runnerVersion; created=(Get-Date).ToString('o')
    source_manifest=$manifestFile.FullName; fixed_channels=$manifest.fixed_channels; fixed_conv_layers=$manifest.fixed_conv_layers
    input_shape=@($manifest.input_shape); warmup=$Warmup; runs=$Runs; cooldown_seconds=$CooldownSeconds; force_recompile=[bool]$ForceRecompile
    scope='pure_npu_graph_runtime_only'; baseline_output_channels=3; results=$rows
} | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $full -Encoding UTF8

$lines = @(
    "GameSR NPU Output Channel Probe v$runnerVersion",
    '============================================================',
    ("Fixed body: C{0}-L{1}" -f $manifest.fixed_channels,$manifest.fixed_conv_layers),
    ("Input: {0}" -f (@($manifest.input_shape) -join 'x')),
    ("Warmup/Runs: {0}/{1}" -f $Warmup,$Runs),
    '',
    'This test isolates the cost of changing ONLY the final output width.',
    'It does NOT measure Magpie output Map/stitch/upload or a learned GPU decoder.',
    '',
    'Results:'
)
foreach ($r in @($rows | Sort-Object output_channels)) {
    $avg = if ($null -eq $r.average_ms) { 'N/A' } else { '{0:N4}' -f $r.average_ms }
    $p95 = if ($null -eq $r.p95_ms) { 'N/A' } else { '{0:N4}' -f $r.p95_ms }
    $delta = if ($null -eq $r.delta_percent_vs_o3) { 'N/A' } else { '{0:+0.000;-0.000;0.000}%' -f $r.delta_percent_vs_o3 }
    $lines += ("  O{0,-2} status={1,-12} avg={2,9} ms  P95={3,9} ms  delta-vs-O3={4,10}  output={5:N3} MiB  x{6:N3}" -f $r.output_channels,$r.status,$avg,$p95,$delta,$r.output_fp32_mib,$r.output_ratio_vs_o3)
}
$lines += ''
$lines += 'Interpretation:'
$lines += '  - O4 near O3: NPU-side cost of the proposed C4 feature output is small.'
$lines += '  - O4 jumps sharply: final-channel shape changes NPU mapping/writeback and needs redesign.'
$lines += '  - O6/O8 are stress points to reveal whether the cost is roughly linear or has XDNA alignment cliffs.'
$lines += ''
$lines += "CSV : $csv"
$lines += "FULL: $full"
$lines | Set-Content -LiteralPath $txt -Encoding UTF8
[void]$artifacts.Add($csv); [void]$artifacts.Add($full); [void]$artifacts.Add($txt)

$compact = $base + '-COMPACT.zip'
$stage = Join-Path $resultsDir ("_output_channel_compact_" + $stamp)
New-Item -ItemType Directory -Force -Path $stage | Out-Null
try {
    $seen = @{}
    foreach ($a in $artifacts) {
        if ([string]::IsNullOrWhiteSpace($a) -or -not (Test-Path -LiteralPath $a -PathType Leaf)) { continue }
        if ($seen.ContainsKey($a)) { continue }
        $seen[$a] = $true
        Copy-Item -LiteralPath $a -Destination (Join-Path $stage ([System.IO.Path]::GetFileName($a))) -Force
    }
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $compact -CompressionLevel Optimal -Force
} finally { Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue }

Write-Host ''
Write-Host '================ OUTPUT CHANNELS ================' -ForegroundColor Green
$rows | Sort-Object output_channels | Select-Object output_channels,status,average_ms,p95_ms,delta_ms_vs_o3,delta_percent_vs_o3,output_fp32_mib,output_ratio_vs_o3 | Format-Table -AutoSize
Write-Host "[OK] CSV     : $csv" -ForegroundColor Green
Write-Host "[OK] Summary : $txt" -ForegroundColor Green
Write-Host "[OK] Compact : $compact" -ForegroundColor Green
Write-Host 'Upload the COMPACT.zip back to ChatGPT.' -ForegroundColor Cyan

$valid = @($rows | Where-Object { $_.status -eq 'NPU_ONLY' })
if ($valid.Count -eq 0) { exit 4 }
if ($valid.Count -lt $rows.Count) { exit 3 }
exit 0
