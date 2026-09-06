[CmdletBinding()]
param(
    [Parameter(Position=0)] [string]$PayloadPath = '',
    [int]$Warmup = 20,
    [int]$Runs = 600,
    [int]$CooldownSeconds = 1,
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

function FullPath([string]$PathValue) {
    return [System.IO.Path]::GetFullPath($PathValue)
}

function Is-UnderPath([string]$Child, [string]$Parent) {
    $childFull = (FullPath $Child).TrimEnd('\')
    $parentFull = (FullPath $Parent).TrimEnd('\')
    if ($childFull.Equals($parentFull, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    return $childFull.StartsWith($parentFull + '\', [System.StringComparison]::OrdinalIgnoreCase)
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
        Fail ("Required benchmark/runtime file is missing: {0}`nCopy the Runner files into the SAME root folder that successfully ran results-bm-5 (the folder containing runtime\\)." -f $required)
    }
}

$resultsDir = Join-Path $root 'results'
$payloadCache = Join-Path $root 'sweep_payload_cache'
New-Item -ItemType Directory -Force -Path $resultsDir, $payloadCache | Out-Null

function Resolve-ManifestFromFolder([string]$Folder) {
    if (-not (Test-Path -LiteralPath $Folder -PathType Container)) { return $null }
    $direct = Join-Path $Folder 'sweep_manifest.json'
    if (Test-Path -LiteralPath $direct -PathType Leaf) { return (Get-Item -LiteralPath $direct) }
    $preferred = Join-Path $Folder 'handheld_payload\sweep_manifest.json'
    if (Test-Path -LiteralPath $preferred -PathType Leaf) { return (Get-Item -LiteralPath $preferred) }
    $found = @(Get-ChildItem -LiteralPath $Folder -Filter 'sweep_manifest.json' -File -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
    if ($found.Count -gt 0) { return $found[0] }
    return $null
}

function Resolve-PayloadManifest([string]$InputPath) {
    if (-not [string]::IsNullOrWhiteSpace($InputPath)) {
        $candidate = FullPath ($InputPath.Trim().Trim('"'))
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $ext = [System.IO.Path]::GetExtension($candidate).ToLowerInvariant()
            if ($ext -eq '.json' -and [System.IO.Path]::GetFileName($candidate) -eq 'sweep_manifest.json') {
                return (Get-Item -LiteralPath $candidate)
            }
            if ($ext -eq '.zip') {
                $zipHash = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant().Substring(0,16)
                $zipBase = [System.IO.Path]::GetFileNameWithoutExtension($candidate) -replace '[^A-Za-z0-9._-]', '_'
                $dest = Join-Path $payloadCache ($zipBase + '-' + $zipHash)
                if (-not (Test-Path -LiteralPath $dest -PathType Container)) {
                    Write-Host "[PAYLOAD] Extracting Builder ZIP -> $dest" -ForegroundColor Cyan
                    New-Item -ItemType Directory -Force -Path $dest | Out-Null
                    Expand-Archive -LiteralPath $candidate -DestinationPath $dest -Force
                } else {
                    Write-Host "[PAYLOAD] Reusing extracted Builder ZIP: $dest" -ForegroundColor DarkCyan
                }
                $m = Resolve-ManifestFromFolder $dest
                if ($m) { return $m }
                Fail "ZIP did not contain handheld_payload/sweep_manifest.json: $candidate"
            }
            Fail "Unsupported payload file. Use Builder model ZIP or sweep_manifest.json: $candidate"
        }
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            $m = Resolve-ManifestFromFolder $candidate
            if ($m) { return $m }
            Fail "No sweep_manifest.json found under: $candidate"
        }
        Fail "Payload path not found: $candidate"
    }

    foreach ($folder in @(
        (Join-Path $root 'handheld_payload'),
        (Join-Path $root 'architecture_sweep\handheld_payload'),
        $root
    )) {
        $m = Resolve-ManifestFromFolder $folder
        if ($m) { return $m }
    }

    $zips = @(Get-ChildItem -LiteralPath $root -Filter 'GameSR-NPU-Architecture-Sweep-Models-*.zip' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
    if ($zips.Count -gt 0) {
        return Resolve-PayloadManifest $zips[0].FullName
    }
    Fail 'No Builder payload found. Put handheld_payload beside the Runner, copy the Builder model ZIP here, or drag the ZIP/folder onto RUN-GAMESR-ARCHITECTURE-SWEEP.cmd.'
}

$manifestFile = Resolve-PayloadManifest $PayloadPath
$payloadRoot = $manifestFile.DirectoryName
try {
    $manifest = Get-Content -Raw -LiteralPath $manifestFile.FullName | ConvertFrom-Json
} catch {
    Fail "Invalid sweep manifest: $($manifestFile.FullName). $($_.Exception.Message)"
}

if ($null -eq $manifest.models) { Fail 'Manifest has no models array.' }
$models = @($manifest.models | Where-Object { [string]$_.status -eq 'READY' })
if ($models.Count -eq 0) { Fail 'Manifest contains no READY models.' }

Write-Host ''
Write-Host "GameSR Architecture Sweep Runner v$runnerVersion" -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host 'Target      : Handheld Ryzen AI / VitisAI NPU only'
Write-Host 'Python      : NOT REQUIRED' -ForegroundColor Green
Write-Host "Payload     : $payloadRoot"
Write-Host "Builder     : $($manifest.builder) v$($manifest.builder_version)"
Write-Host "Architecture: $($manifest.architecture)"
Write-Host "Input shape : $(@($manifest.input_shape) -join 'x')"
Write-Host "Models      : $($models.Count) READY / $($manifest.requested_models) requested"
Write-Host "Warmup/Runs : $Warmup / $Runs (serial concurrency=1)"
Write-Host "Cooldown    : $CooldownSeconds sec between models"
Write-Host "Hash verify : $(-not $SkipHashValidation)"
Write-Host ''

$resolvedModels = @()
$validateIndex = 0
foreach ($m in $models) {
    $validateIndex++
    $rel = [string]$m.model_relpath
    if ([string]::IsNullOrWhiteSpace($rel)) {
        Fail "Manifest entry $($m.architecture) has no model_relpath."
    }
    $modelPath = FullPath (Join-Path $payloadRoot ($rel -replace '/', '\'))
    if (-not (Is-UnderPath $modelPath $payloadRoot)) { Fail "Unsafe model path in manifest: $rel" }
    if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) { Fail "Model missing: $modelPath" }
    if (-not $SkipHashValidation -and -not [string]::IsNullOrWhiteSpace([string]$m.sha256)) {
        Write-Host ("[VERIFY {0}/{1}] {2}" -f $validateIndex, $models.Count, $m.architecture) -ForegroundColor DarkGray
        $actual = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $expected = ([string]$m.sha256).ToLowerInvariant()
        if ($actual -ne $expected) { Fail "SHA256 mismatch for $($m.architecture). Expected $expected, got $actual" }
    }
    $resolvedModels += [pscustomobject]@{
        manifest_entry=$m
        model_path=$modelPath
        model_dir=(Split-Path -Parent $modelPath)
    }
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$sessionStart = Get-Date
$rows = @()
$sessionArtifacts = New-Object 'System.Collections.Generic.List[string]'
$total = $resolvedModels.Count
$order = 0

foreach ($resolved in $resolvedModels) {
    $order++
    $m = $resolved.manifest_entry
    $arch = [string]$m.architecture
    Write-Host ''
    Write-Host ("===== [{0}/{1}] {2} | C={3} L={4} | {5} GMAC =====" -f $order, $total, $arch, $m.channels, $m.conv_layers, $m.gmac_per_frame) -ForegroundColor Cyan

    $pattern = "GameSR-v24-$($arch -replace '[^A-Za-z0-9._-]', '_')-*-SUMMARY.json"
    $before = @{}
    foreach ($f in @(Get-ChildItem -LiteralPath $resultsDir -Filter $pattern -File -ErrorAction SilentlyContinue)) { $before[$f.FullName] = $true }

    $args = @('-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-File',$benchmark,$resolved.model_dir,'-Warmup',"$Warmup",'-Runs',"$Runs",'-OnlyInt8','-SkipCpu')
    if ($ForceRecompile) { $args += '-ForceRecompile' }

    # Capture the child benchmark output as an artifact. This makes infrastructure
    # failures diagnosable and prevents native stderr warnings from aborting this
    # outer runner under Windows PowerShell 5.1.
    $launchLog = Join-Path $resultsDir ("SweepRunner-{0}-{1}-launch.log" -f $stamp, ($arch -replace '[^A-Za-z0-9._-]', '_'))
    $savedOuterErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $benchOutput = @(& powershell.exe @args 2>&1)
        $benchExit = [int]$LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedOuterErrorAction
    }
    $benchOutput | Out-File -LiteralPath $launchLog -Encoding utf8
    foreach ($line in $benchOutput) { Write-Host ([string]$line) }
    [void]$sessionArtifacts.Add($launchLog)

    $newSummaries = @(
        Get-ChildItem -LiteralPath $resultsDir -Filter $pattern -File -ErrorAction SilentlyContinue |
        Where-Object { -not $before.ContainsKey($_.FullName) } |
        Sort-Object LastWriteTime -Descending
    )

    if ($newSummaries.Count -eq 0) {
        Write-Host "[WARN] No new SUMMARY.json for $arch (benchmark exit=$benchExit)" -ForegroundColor Yellow
        $rows += [pscustomobject]@{
            test_order=$order; architecture=$arch; channels=[int]$m.channels; conv_layers=[int]$m.conv_layers
            parameters=[int64]$m.parameters; gmac_per_frame=[double]$m.gmac_per_frame; sha256=[string]$m.sha256
            status='NO_SUMMARY'; benchmark_exit=$benchExit; average_ms=$null; p50_ms=$null; p95_ms=$null; pure_fps=$null
            effective_gmac_s=$null; npu_event_share_percent=$null; npu_duration_share_percent=$null; npu_events=$null; cpu_events=$null
            cache_present_before=$null; warmup_wall_seconds=$null; process_wall_seconds=$null; summary_json=''; log=$launchLog; profile=''
        }
        Write-Host '[STOP] Benchmark infrastructure failed before producing SUMMARY.json. Stopping the sweep instead of repeating the same failure for every model.' -ForegroundColor Red
        Write-Host ("[STOP] Diagnostic log: {0}" -f $launchLog) -ForegroundColor Yellow
        break
    } else {
        $summaryFile = $newSummaries[0]
        try { $sum = Get-Content -Raw -LiteralPath $summaryFile.FullName | ConvertFrom-Json } catch { $sum = $null }
        if ($sum) {
            $r = @($sum.results | Where-Object { $_.variant -eq 'int8' -and $_.provider -eq 'vitisai' }) | Select-Object -First 1
        } else { $r = $null }

        [void]$sessionArtifacts.Add($summaryFile.FullName)
        $summaryTxt = $summaryFile.FullName -replace '-SUMMARY\.json$', '-SUMMARY.txt'
        if (Test-Path -LiteralPath $summaryTxt -PathType Leaf) { [void]$sessionArtifacts.Add($summaryTxt) }

        if ($r) {
            foreach ($artifact in @([string]$r.log, [string]$r.warmup_log)) {
                if (-not [string]::IsNullOrWhiteSpace($artifact) -and (Test-Path -LiteralPath $artifact -PathType Leaf)) { [void]$sessionArtifacts.Add($artifact) }
            }
            $eff = $null
            if ($null -ne $r.average_ms -and [double]$r.average_ms -gt 0) { $eff = [math]::Round(([double]$m.gmac_per_frame * 1000.0 / [double]$r.average_ms), 3) }
            $rows += [pscustomobject]@{
                test_order=$order; architecture=$arch; channels=[int]$m.channels; conv_layers=[int]$m.conv_layers
                parameters=[int64]$m.parameters; gmac_per_frame=[double]$m.gmac_per_frame; sha256=[string]$m.sha256
                status=[string]$r.status; benchmark_exit=$benchExit; average_ms=$r.average_ms; p50_ms=$r.p50_ms; p95_ms=$r.p95_ms; pure_fps=$r.pure_fps
                effective_gmac_s=$eff; npu_event_share_percent=$r.npu_event_share_percent; npu_duration_share_percent=$r.npu_duration_share_percent
                npu_events=$r.npu_events; cpu_events=$r.cpu_events; cache_present_before=$r.cache_present_before
                warmup_wall_seconds=$r.warmup_wall_seconds; process_wall_seconds=$r.process_wall_seconds
                summary_json=$summaryFile.FullName; log=[string]$r.log; profile=[string]$r.profile
            }
            $c = if ([string]$r.status -eq 'NPU_ONLY') { 'Green' } elseif ([string]$r.status -eq 'NPU_PARTIAL') { 'Yellow' } else { 'Red' }
            Write-Host ("[RESULT] {0}: status={1}, avg={2} ms, P95={3} ms, FPS={4}, eff={5} GMAC/s" -f $arch, $r.status, $r.average_ms, $r.p95_ms, $r.pure_fps, $eff) -ForegroundColor $c

            # Stop immediately for a benchmark-infrastructure/parser failure.
            # Unsupported individual architectures may still fail for other reasons,
            # but malformed perf_test options would otherwise repeat for every model.
            if ([string]$r.status -eq 'FAIL') {
                $diagText = ''
                foreach ($diagPath in @([string]$r.log, [string]$r.warmup_log, $launchLog)) {
                    if (-not [string]::IsNullOrWhiteSpace($diagPath) -and (Test-Path -LiteralPath $diagPath -PathType Leaf)) {
                        try { $diagText += "`n" + [System.IO.File]::ReadAllText($diagPath) } catch {}
                    }
                }
                if ($diagText -match 'ParseSessionConfigs|run-time option|Use a .* to separate the key and value') {
                    Write-Host '[STOP] perf_test provider-option parser failure detected. Stopping immediately.' -ForegroundColor Red
                    Write-Host ("[STOP] Diagnostic log: {0}" -f $launchLog) -ForegroundColor Yellow
                    break
                }
            }
        } else {
            $rows += [pscustomobject]@{
                test_order=$order; architecture=$arch; channels=[int]$m.channels; conv_layers=[int]$m.conv_layers
                parameters=[int64]$m.parameters; gmac_per_frame=[double]$m.gmac_per_frame; sha256=[string]$m.sha256
                status='INVALID_SUMMARY'; benchmark_exit=$benchExit; average_ms=$null; p50_ms=$null; p95_ms=$null; pure_fps=$null
                effective_gmac_s=$null; npu_event_share_percent=$null; npu_duration_share_percent=$null; npu_events=$null; cpu_events=$null
                cache_present_before=$null; warmup_wall_seconds=$null; process_wall_seconds=$null; summary_json=$summaryFile.FullName; log=''; profile=''
            }
        }
    }

    if ($CooldownSeconds -gt 0 -and $order -lt $total) {
        Start-Sleep -Seconds $CooldownSeconds
    }
}

$valid = @($rows | Where-Object { $_.status -eq 'NPU_ONLY' -and $null -ne $_.average_ms -and [double]$_.average_ms -gt 0 } | Sort-Object {[double]$_.average_ms}, {[double]$_.p95_ms})
$rank = 0
foreach ($v in $valid) { $rank++; Add-Member -InputObject $v -NotePropertyName rank -NotePropertyValue $rank -Force }

$base = Join-Path $resultsDir "GameSR-NPU-Architecture-Sweep-$stamp"
$rankingCsv = $base + '-RANKING.csv'
$fullJson = $base + '-FULL.json'
$summaryTxt = $base + '-SUMMARY.txt'
$latencyMatrixCsv = $base + '-LATENCY-MATRIX.csv'
$efficiencyMatrixCsv = $base + '-EFFICIENCY-MATRIX.csv'
$bestChannelCsv = $base + '-BEST-BY-CHANNEL.csv'
$bestLayerCsv = $base + '-BEST-BY-LAYER.csv'

$valid | Select-Object rank,test_order,architecture,channels,conv_layers,parameters,gmac_per_frame,status,average_ms,p50_ms,p95_ms,pure_fps,effective_gmac_s,npu_event_share_percent,npu_duration_share_percent,npu_events,cpu_events,cache_present_before,warmup_wall_seconds,process_wall_seconds,sha256,summary_json,log,profile | Export-Csv -LiteralPath $rankingCsv -NoTypeInformation -Encoding UTF8

$payloadOut = [ordered]@{
    schema_version=1
    runner='GameSR Architecture Sweep Runner'
    runner_version=$runnerVersion
    created=(Get-Date).ToString('o')
    session_started=$sessionStart.ToString('o')
    source_manifest=$manifestFile.FullName
    source_manifest_schema=$manifest.schema_version
    builder=$manifest.builder
    builder_version=$manifest.builder_version
    architecture=$manifest.architecture
    input_shape=@($manifest.input_shape)
    requested_models=$manifest.requested_models
    ready_models=$models.Count
    warmup=$Warmup
    runs=$Runs
    measured_concurrency=1
    cooldown_seconds=$CooldownSeconds
    force_recompile=[bool]$ForceRecompile
    hash_validation=(-not [bool]$SkipHashValidation)
    result_count=$rows.Count
    valid_npu_only=$valid.Count
    results=$rows
}
$payloadOut | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $fullJson -Encoding UTF8

$channels = @($models | ForEach-Object { [int]$_.channels } | Sort-Object -Unique)
$layers = @($models | ForEach-Object { [int]$_.conv_layers } | Sort-Object -Unique)
function Write-Matrix([string]$PathValue, [string]$Metric) {
    $matrixRows = @()
    foreach ($c in $channels) {
        $h = [ordered]@{ channels=$c }
        foreach ($l in $layers) {
            $match = @($valid | Where-Object { [int]$_.channels -eq $c -and [int]$_.conv_layers -eq $l }) | Select-Object -First 1
            if ($match) {
                $value = $match.$Metric
                if ($null -ne $value) { $h["L$l"] = [math]::Round([double]$value, 4) } else { $h["L$l"] = '' }
            } else { $h["L$l"] = '' }
        }
        $matrixRows += [pscustomobject]$h
    }
    $matrixRows | Export-Csv -LiteralPath $PathValue -NoTypeInformation -Encoding UTF8
}
Write-Matrix $latencyMatrixCsv 'average_ms'
Write-Matrix $efficiencyMatrixCsv 'effective_gmac_s'

$bestByChannel = @()
foreach ($c in $channels) {
    $best = @($valid | Where-Object { [int]$_.channels -eq $c } | Sort-Object {[double]$_.average_ms}) | Select-Object -First 1
    if ($best) { $bestByChannel += $best }
}
$bestByChannel | Select-Object channels,architecture,conv_layers,average_ms,p95_ms,pure_fps,effective_gmac_s,gmac_per_frame,parameters | Export-Csv -LiteralPath $bestChannelCsv -NoTypeInformation -Encoding UTF8

$bestByLayer = @()
foreach ($l in $layers) {
    $best = @($valid | Where-Object { [int]$_.conv_layers -eq $l } | Sort-Object {[double]$_.average_ms}) | Select-Object -First 1
    if ($best) { $bestByLayer += $best }
}
$bestByLayer | Select-Object conv_layers,architecture,channels,average_ms,p95_ms,pure_fps,effective_gmac_s,gmac_per_frame,parameters | Export-Csv -LiteralPath $bestLayerCsv -NoTypeInformation -Encoding UTF8

$lines = @(
    "GameSR Architecture Sweep Runner v$runnerVersion",
    '============================================================',
    ('Finished: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
    ('Payload: ' + $payloadRoot),
    ('Builder: ' + $manifest.builder + ' v' + $manifest.builder_version),
    ('Input: ' + (@($manifest.input_shape) -join 'x')),
    ("Warmup/Runs: $Warmup / $Runs, concurrency=1, cooldown=${CooldownSeconds}s"),
    ("Models: $($rows.Count), NPU_ONLY: $($valid.Count)"),
    '',
    'TOP 30 - NPU_ONLY / lowest average latency:'
)
$i = 0
foreach ($v in @($valid | Select-Object -First 30)) {
    $i++
    $lines += ('{0,2}. {1,-10} C{2,-2} L{3,-2} avg {4,8:N4} ms  P95 {5,8:N4} ms  FPS {6,8:N2}  eff {7,9:N2} GMAC/s  {8,7:N3} GMAC' -f $i,$v.architecture,$v.channels,$v.conv_layers,$v.average_ms,$v.p95_ms,$v.pure_fps,$v.effective_gmac_s,$v.gmac_per_frame)
}
$nonNpu = @($rows | Where-Object { $_.status -ne 'NPU_ONLY' })
if ($nonNpu.Count -gt 0) {
    $lines += ''
    $lines += 'NON-NPU_ONLY / FAILED:'
    foreach ($r in $nonNpu) { $lines += ("  $($r.architecture): $($r.status), benchmark_exit=$($r.benchmark_exit)") }
}
$lines += ''
$lines += "Ranking          : $rankingCsv"
$lines += "Full JSON        : $fullJson"
$lines += "Latency matrix   : $latencyMatrixCsv"
$lines += "Efficiency matrix: $efficiencyMatrixCsv"
$lines += "Best/channel     : $bestChannelCsv"
$lines += "Best/layer       : $bestLayerCsv"
$lines | Set-Content -LiteralPath $summaryTxt -Encoding UTF8

foreach ($aggregate in @($rankingCsv,$fullJson,$summaryTxt,$latencyMatrixCsv,$efficiencyMatrixCsv,$bestChannelCsv,$bestLayerCsv)) { [void]$sessionArtifacts.Add($aggregate) }

$compactZip = $base + '-COMPACT.zip'
$stage = Join-Path $resultsDir ("_compact_" + $stamp)
New-Item -ItemType Directory -Force -Path $stage | Out-Null
try {
    $seen = @{}
    foreach ($artifact in $sessionArtifacts) {
        if ([string]::IsNullOrWhiteSpace($artifact) -or -not (Test-Path -LiteralPath $artifact -PathType Leaf)) { continue }
        if ($seen.ContainsKey($artifact)) { continue }
        $seen[$artifact] = $true
        $leaf = [System.IO.Path]::GetFileName($artifact)
        $dest = Join-Path $stage $leaf
        if (Test-Path -LiteralPath $dest) {
            $stem = [System.IO.Path]::GetFileNameWithoutExtension($leaf)
            $ext = [System.IO.Path]::GetExtension($leaf)
            $dest = Join-Path $stage ($stem + '-' + $seen.Count + $ext)
        }
        Copy-Item -LiteralPath $artifact -Destination $dest -Force
    }
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $compactZip -CompressionLevel Optimal -Force
} finally {
    Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ''
Write-Host '================ TOP 20 ================' -ForegroundColor Green
$valid | Select-Object -First 20 rank,architecture,channels,conv_layers,average_ms,p95_ms,pure_fps,effective_gmac_s | Format-Table -AutoSize
Write-Host ''
Write-Host "[OK] Ranking          : $rankingCsv" -ForegroundColor Green
Write-Host "[OK] Full JSON        : $fullJson" -ForegroundColor Green
Write-Host "[OK] Latency matrix   : $latencyMatrixCsv" -ForegroundColor Green
Write-Host "[OK] Efficiency matrix: $efficiencyMatrixCsv" -ForegroundColor Green
Write-Host "[OK] Compact upload   : $compactZip" -ForegroundColor Green
Write-Host ''
Write-Host 'Upload the COMPACT.zip to ChatGPT for architecture selection. Full profiler JSON files remain in results/ for deeper diagnosis.' -ForegroundColor Cyan

if ($valid.Count -eq 0) { exit 4 }
if ($valid.Count -lt $rows.Count) { exit 3 }
exit 0
