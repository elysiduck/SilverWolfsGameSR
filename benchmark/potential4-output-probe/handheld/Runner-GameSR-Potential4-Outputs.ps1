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
$runnerVersion = '1.1.0'
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
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { Fail "Required v1.3 benchmark/runtime file is missing: $required" }
}

$resultsDir = Join-Path $root 'results'
$payloadCache = Join-Path $root '_p4o'
New-Item -ItemType Directory -Force -Path $resultsDir, $payloadCache | Out-Null

function Resolve-ManifestFromFolder([string]$Folder) {
    if (-not (Test-Path -LiteralPath $Folder -PathType Container)) { return $null }
    foreach ($name in @('potential4_output_manifest.json','handheld_payload\potential4_output_manifest.json')) {
        $p = Join-Path $Folder $name
        if (Test-Path -LiteralPath $p -PathType Leaf) { return (Get-Item -LiteralPath $p) }
    }
    $found = @(Get-ChildItem -LiteralPath $Folder -Filter 'potential4_output_manifest.json' -File -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
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
                $zipHash = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant().Substring(0,16)
                $dest = Join-Path $payloadCache $zipHash
                $m = Resolve-ManifestFromFolder $dest
                if ($m) { return $m }
                if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Recurse -Force -ErrorAction SilentlyContinue }
                $tmp = Join-Path $payloadCache ('.tmp-' + $zipHash + '-' + $PID)
                if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
                New-Item -ItemType Directory -Force -Path $tmp | Out-Null
                try {
                    Write-Host "[PAYLOAD] Extracting -> $tmp" -ForegroundColor Cyan
                    Expand-Archive -LiteralPath $candidate -DestinationPath $tmp
                    $tmpManifest = Resolve-ManifestFromFolder $tmp
                    if (-not $tmpManifest) { Fail "ZIP has no potential4_output_manifest.json: $candidate" }
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
            Fail "No potential4_output_manifest.json under: $candidate"
        }
        Fail "Payload path not found: $candidate"
    }

    foreach ($folder in @((Join-Path $root 'handheld_payload'), $root)) {
        $m = Resolve-ManifestFromFolder $folder
        if ($m) { return $m }
    }
    $zips = @(Get-ChildItem -LiteralPath $root -Filter 'GameSR-NPU-Potential4-Output-Probe-Models-16-*.zip' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
    if ($zips.Count -gt 0) { return Resolve-PayloadManifest $zips[0].FullName }
    Fail 'No Potential-4 output payload found. Copy the PC-generated Models-16 ZIP beside this script or drag it onto the RUN cmd.'
}

$manifestFile = Resolve-PayloadManifest $PayloadPath
$payloadRoot = $manifestFile.DirectoryName
try { $manifest = Get-Content -Raw -LiteralPath $manifestFile.FullName | ConvertFrom-Json }
catch { Fail "Invalid manifest: $($manifestFile.FullName). $($_.Exception.Message)" }

$models = @($manifest.models | Where-Object { [string]$_.status -eq 'READY' } | Sort-Object @{Expression={[int]$_.candidate_order}}, @{Expression={[int]$_.output_channels}})
if ($models.Count -ne 16) { Fail "Expected 16 READY models (4 backbones x 4 outputs), found $($models.Count)." }

Write-Host ''
Write-Host "GameSR Potential-4 Output Probe Runner v$runnerVersion" -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host "Payload     : $payloadRoot"
Write-Host 'Backbones   : C8-L5 / C16-L4 / C24-L3 / C32-L3'
Write-Host "Outputs     : $(@($manifest.output_channels) -join ',')"
Write-Host "Input       : $(@($manifest.input_shape) -join 'x')"
Write-Host "Warmup/Runs : $Warmup / $Runs"
Write-Host "Cooldown    : $CooldownSeconds sec"
Write-Host 'Scope       : pure NPU graph/runtime only; NOT Magpie Map/stitch/upload/decoder' -ForegroundColor Yellow
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
    $launchLog = Join-Path $resultsDir ("Potential4Output-{0}-{1}-launch.log" -f $stamp, ($arch -replace '[^A-Za-z0-9._-]', '_'))
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
            test_order=$order; candidate_order=[int]$m.candidate_order; backbone=[string]$m.backbone; architecture=$arch; channels=[int]$m.channels; conv_layers=[int]$m.conv_layers; output_channels=[int]$m.output_channels
            parameters=[int64]$m.parameters; gmac_per_frame=[double]$m.gmac_per_frame; output_fp32_mib=[double]$m.output_fp32_mib; output_ratio_vs_o3=[double]$m.output_ratio_vs_o3
            source_o3_average_ms=[double]$m.source_o3_average_ms; status='NO_SUMMARY'; benchmark_exit=$benchExit; average_ms=$null; p50_ms=$null; p95_ms=$null; pure_fps=$null; effective_gmac_s=$null
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
            test_order=$order; candidate_order=[int]$m.candidate_order; backbone=[string]$m.backbone; architecture=$arch; channels=[int]$m.channels; conv_layers=[int]$m.conv_layers; output_channels=[int]$m.output_channels
            parameters=[int64]$m.parameters; gmac_per_frame=[double]$m.gmac_per_frame; output_fp32_mib=[double]$m.output_fp32_mib; output_ratio_vs_o3=[double]$m.output_ratio_vs_o3
            source_o3_average_ms=[double]$m.source_o3_average_ms; status='INVALID_SUMMARY'; benchmark_exit=$benchExit; average_ms=$null; p50_ms=$null; p95_ms=$null; pure_fps=$null; effective_gmac_s=$null
            npu_event_share_percent=$null; npu_duration_share_percent=$null; npu_events=$null; cpu_events=$null; summary_json=$summaryFile.FullName
        }
        continue
    }
    $eff = $null
    if ($null -ne $r.average_ms -and [double]$r.average_ms -gt 0) { $eff = [math]::Round(([double]$m.gmac_per_frame * 1000.0 / [double]$r.average_ms), 3) }
    $rows += [pscustomobject]@{
        test_order=$order; candidate_order=[int]$m.candidate_order; backbone=[string]$m.backbone; architecture=$arch; channels=[int]$m.channels; conv_layers=[int]$m.conv_layers; output_channels=[int]$m.output_channels
        parameters=[int64]$m.parameters; gmac_per_frame=[double]$m.gmac_per_frame; output_fp32_mib=[double]$m.output_fp32_mib; output_ratio_vs_o3=[double]$m.output_ratio_vs_o3
        source_o3_average_ms=[double]$m.source_o3_average_ms; status=[string]$r.status; benchmark_exit=$benchExit; average_ms=[double]$r.average_ms; p50_ms=[double]$r.p50_ms; p95_ms=[double]$r.p95_ms; pure_fps=[double]$r.pure_fps
        effective_gmac_s=$eff; npu_event_share_percent=[double]$r.npu_event_share_percent; npu_duration_share_percent=[double]$r.npu_duration_share_percent
        npu_events=[int]$r.npu_events; cpu_events=[int]$r.cpu_events; summary_json=$summaryFile.FullName
    }
    if ($CooldownSeconds -gt 0 -and $order -lt $resolved.Count) { Start-Sleep -Seconds $CooldownSeconds }
}

# Per-backbone O3 baseline. Never compare C8 O4 against C16 O3, etc.
foreach ($row in $rows) {
    $base = @($rows | Where-Object { $_.backbone -eq $row.backbone -and $_.output_channels -eq 3 -and $_.status -eq 'NPU_ONLY' -and $null -ne $_.average_ms }) | Select-Object -First 1
    $deltaMs = $null; $deltaPct = $null
    if ($base -and $null -ne $row.average_ms) {
        $deltaMs = [math]::Round([double]$row.average_ms - [double]$base.average_ms, 4)
        if ([double]$base.average_ms -gt 0) { $deltaPct = [math]::Round(100.0 * $deltaMs / [double]$base.average_ms, 3) }
    }
    Add-Member -InputObject $row -NotePropertyName delta_ms_vs_same_backbone_o3 -NotePropertyValue $deltaMs -Force
    Add-Member -InputObject $row -NotePropertyName delta_percent_vs_same_backbone_o3 -NotePropertyValue $deltaPct -Force
}

$baseName = Join-Path $resultsDir "GameSR-NPU-Potential4-Output-Probe-$stamp"
$csv = $baseName + '-ROWS.csv'
$matrixCsv = $baseName + '-LATENCY-MATRIX.csv'
$deltaCsv = $baseName + '-DELTA-MATRIX.csv'
$full = $baseName + '-FULL.json'
$txt = $baseName + '-SUMMARY.txt'

$rows | Sort-Object candidate_order,output_channels | Select-Object test_order,candidate_order,backbone,architecture,channels,conv_layers,output_channels,parameters,gmac_per_frame,output_fp32_mib,output_ratio_vs_o3,source_o3_average_ms,status,average_ms,p50_ms,p95_ms,pure_fps,effective_gmac_s,delta_ms_vs_same_backbone_o3,delta_percent_vs_same_backbone_o3,npu_event_share_percent,npu_duration_share_percent,npu_events,cpu_events,benchmark_exit,summary_json | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8

$backbones = @($manifest.candidate_backbones | Sort-Object {[int]$_.order})
$latMatrix = @()
$deltaMatrix = @()
foreach ($bb in $backbones) {
    $b = [string]$bb.name
    $lat = [ordered]@{ backbone=$b; channels=[int]$bb.channels; conv_layers=[int]$bb.conv_layers }
    $del = [ordered]@{ backbone=$b; channels=[int]$bb.channels; conv_layers=[int]$bb.conv_layers }
    foreach ($o in @(3,4,6,8)) {
        $r = @($rows | Where-Object { $_.backbone -eq $b -and $_.output_channels -eq $o }) | Select-Object -First 1
        $lat["O${o}_ms"] = if ($r) { $r.average_ms } else { $null }
        $del["O${o}_delta_pct"] = if ($r) { $r.delta_percent_vs_same_backbone_o3 } else { $null }
    }
    $latMatrix += [pscustomobject]$lat
    $deltaMatrix += [pscustomobject]$del
}
$latMatrix | Export-Csv -LiteralPath $matrixCsv -NoTypeInformation -Encoding UTF8
$deltaMatrix | Export-Csv -LiteralPath $deltaCsv -NoTypeInformation -Encoding UTF8

[ordered]@{
    schema_version=2; probe='GameSR Potential-4 Output Channel Probe'; runner_version=$runnerVersion; created=(Get-Date).ToString('o')
    source_manifest=$manifestFile.FullName; candidate_backbones=$manifest.candidate_backbones; output_channels=@($manifest.output_channels)
    input_shape=@($manifest.input_shape); warmup=$Warmup; runs=$Runs; cooldown_seconds=$CooldownSeconds; force_recompile=[bool]$ForceRecompile
    scope='pure_npu_graph_runtime_only'; baseline='O3 within each backbone'; results=$rows
} | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $full -Encoding UTF8

$lines = @(
    "GameSR Potential-4 Output Probe v$runnerVersion",
    '============================================================',
    'Backbones: C8-L5 / C16-L4 / C24-L3 / C32-L3',
    'Outputs: O3 / O4 / O6 / O8',
    ("Warmup/Runs: {0}/{1}" -f $Warmup,$Runs),
    '',
    'Each delta is against O3 of the SAME backbone.',
    'Scope: pure NPU runtime only; no Magpie Map/stitch/upload/learned decoder.',
    '',
    'Results:'
)
foreach ($bb in $backbones) {
    $b = [string]$bb.name
    $lines += ''
    $lines += ("[{0}] source sweep O3={1:N4} ms" -f $b,[double]$bb.source_average_ms)
    foreach ($r in @($rows | Where-Object { $_.backbone -eq $b } | Sort-Object output_channels)) {
        $avg = if ($null -eq $r.average_ms) { 'N/A' } else { '{0:N4}' -f $r.average_ms }
        $p95 = if ($null -eq $r.p95_ms) { 'N/A' } else { '{0:N4}' -f $r.p95_ms }
        $delta = if ($null -eq $r.delta_percent_vs_same_backbone_o3) { 'N/A' } else { '{0:+0.000;-0.000;0.000}%' -f $r.delta_percent_vs_same_backbone_o3 }
        $lines += ("  O{0,-2} {1,-12} avg={2,9} ms  P95={3,9} ms  delta={4,10}  output={5:N3} MiB" -f $r.output_channels,$r.status,$avg,$p95,$delta,$r.output_fp32_mib)
    }
}
$lines += ''
$lines += 'Interpretation targets:'
$lines += '  - Prefer O4/O6 when delta remains small and all nodes stay NPU_ONLY.'
$lines += '  - O6 is the key learned-reconstruction candidate because it doubles feature output vs RGB O3.'
$lines += '  - If one backbone has a sharp O6/O8 cliff while another does not, output-head alignment depends on C/L mapping.'
$lines += '  - Do not treat this as a quality ranking; these are random-weight structure probes.'
$lines += ''
$lines += "Rows   : $csv"
$lines += "Latency: $matrixCsv"
$lines += "Delta  : $deltaCsv"
$lines += "Full   : $full"
$lines | Set-Content -LiteralPath $txt -Encoding UTF8

foreach ($p in @($csv,$matrixCsv,$deltaCsv,$full,$txt)) { [void]$artifacts.Add($p) }
$compact = $baseName + '-COMPACT.zip'
$stage = Join-Path $resultsDir ("_p4oc_" + $stamp)
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
Write-Host '================ LATENCY MATRIX ================' -ForegroundColor Green
$latMatrix | Format-Table -AutoSize
Write-Host ''
Write-Host '================ DELTA VS SAME O3 ==============' -ForegroundColor Green
$deltaMatrix | Format-Table -AutoSize
Write-Host "[OK] Summary : $txt" -ForegroundColor Green
Write-Host "[OK] Compact : $compact" -ForegroundColor Green
Write-Host 'Upload the COMPACT.zip back to ChatGPT.' -ForegroundColor Cyan

$valid = @($rows | Where-Object { $_.status -eq 'NPU_ONLY' })
if ($valid.Count -eq 0) { exit 4 }
if ($valid.Count -lt $rows.Count) { exit 3 }
exit 0
