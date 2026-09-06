[CmdletBinding()]
param(
    [Parameter(Position=0)] [string]$DeploymentPath = '',
    [int]$Warmup = 5,
    [int]$Runs = 30,
    [switch]$OnlyInt8,
    [switch]$SkipCpu,
    [switch]$ForceRecompile,
    [switch]$NoProfile
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Fail([string]$Message, [int]$Code = 2) {
    Write-Host "[FAIL] $Message" -ForegroundColor Red
    exit $Code
}

if ($Warmup -lt 1) { Fail 'Warmup must be at least 1.' }
if ($Runs -lt 1) { Fail 'Runs must be at least 1.' }

$bin = Join-Path $root 'runtime\bin'
$perfExe = Join-Path $bin 'onnxruntime_perf_test.exe'
$config = Join-Path $root 'runtime\vaip_config_stride.json'
$xclbin = Join-Path $root 'runtime\AMD_AIE2P_4x4_Overlay.xclbin'
$modelsDir = Join-Path $root 'models'
$resultsDir = Join-Path $root 'results'
$cacheRoot = Join-Path $root 'cache'
New-Item -ItemType Directory -Force -Path $modelsDir, $resultsDir, $cacheRoot | Out-Null

foreach ($required in @($perfExe, $config, $xclbin)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        Fail "Required runtime file is missing: $required"
    }
}

function FullPath([string]$PathValue) {
    return [System.IO.Path]::GetFullPath($PathValue)
}

function Get-Variant([string]$Name) {
    if ($Name -match '-int8-qdq\.onnx$') { return 'int8' }
    if ($Name -match '-fp16\.onnx$') { return 'fp16' }
    if ($Name -match '\.onnx$') { return 'float' }
    return ''
}

function Get-Family([string]$Name) {
    $value = $Name -replace '-int8-qdq\.onnx$', ''
    $value = $value -replace '-fp16\.onnx$', ''
    $value = $value -replace '\.onnx$', ''
    return $value
}

function Choose-One([object[]]$Items, [string]$Title, [scriptblock]$Display) {
    if ($Items.Count -eq 1) { return $Items[0] }
    Write-Host ''
    Write-Host $Title -ForegroundColor Cyan
    for ($index = 0; $index -lt $Items.Count; $index++) {
        Write-Host ("  {0}. {1}" -f ($index + 1), (& $Display $Items[$index]))
    }
    $selection = Read-Host 'Select [1]'
    if ([string]::IsNullOrWhiteSpace($selection)) { $selection = '1' }
    $number = 0
    if (-not [int]::TryParse($selection, [ref]$number) -or $number -lt 1 -or $number -gt $Items.Count) {
        Fail 'Invalid selection.'
    }
    return $Items[$number - 1]
}

function Resolve-ModelSet([string]$SourcePath) {
    $source = $modelsDir
    if (-not [string]::IsNullOrWhiteSpace($SourcePath)) {
        $source = FullPath $SourcePath
    }
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        Fail "Deployment/model directory not found: $source"
    }

    $manifests = @(Get-ChildItem -LiteralPath $source -Filter 'deployment_manifest.json' -File -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName)
    if ($manifests.Count -gt 0) {
        $manifest = Choose-One $manifests 'Deployment manifests found:' { param($item) $item.FullName }
        try {
            $payload = Get-Content -Raw -LiteralPath $manifest.FullName | ConvertFrom-Json
        } catch {
            Fail "Invalid deployment manifest: $($manifest.FullName). $($_.Exception.Message)"
        }
        $entries = @()
        foreach ($relative in @($payload.models)) {
            $candidate = Join-Path $manifest.DirectoryName ([string]$relative)
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $variant = Get-Variant ([System.IO.Path]::GetFileName($candidate))
                if ($variant) {
                    $entries += [pscustomobject]@{ variant=$variant; path=(FullPath $candidate) }
                }
            }
        }
        if ($entries.Count -eq 0) { Fail "Manifest contains no usable ONNX models: $($manifest.FullName)" }
        return [pscustomobject]@{
            family = [string]$payload.model
            source = $manifest.DirectoryName
            manifest = $manifest.FullName
            entries = $entries
        }
    }

    $files = @(Get-ChildItem -LiteralPath $source -Filter '*.onnx' -File -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName)
    if ($files.Count -eq 0) { Fail "No ONNX model found under: $source" }
    $families = @($files | ForEach-Object { Get-Family $_.Name } | Sort-Object -Unique)
    $family = Choose-One $families 'Model families found:' { param($item) [string]$item }
    $entries = @()
    foreach ($file in $files) {
        if ((Get-Family $file.Name) -eq $family) {
            $entries += [pscustomobject]@{ variant=(Get-Variant $file.Name); path=$file.FullName }
        }
    }
    return [pscustomobject]@{ family=$family; source=$source; manifest=$null; entries=$entries }
}

function Get-ModelFingerprint([string]$ModelPath) {
    $onnxHash = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $externalPath = $ModelPath + '.data'
    $externalHash = ''
    if (Test-Path -LiteralPath $externalPath -PathType Leaf) {
        $externalHash = (Get-FileHash -LiteralPath $externalPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $material = $onnxHash + '|' + $externalHash
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($material)
        $digest = $sha.ComputeHash($bytes)
        $combined = ([System.BitConverter]::ToString($digest)).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
    return [pscustomobject]@{
        fingerprint = $combined
        short = $combined.Substring(0, 16)
        onnx_sha256 = $onnxHash
        external_sha256 = $externalHash
        external_present = (-not [string]::IsNullOrWhiteSpace($externalHash))
    }
}

function Get-RuntimeMetadata() {
    $items = @()
    foreach ($pathValue in @(
        $perfExe,
        (Join-Path $bin 'onnxruntime.dll'),
        (Join-Path $bin 'onnxruntime_providers_vitisai.dll'),
        (Join-Path $bin 'vitis-ai-runtime.dll'),
        $config,
        $xclbin
    )) {
        if (-not (Test-Path -LiteralPath $pathValue -PathType Leaf)) { continue }
        $file = Get-Item -LiteralPath $pathValue
        $version = ''
        try { $version = [string]$file.VersionInfo.FileVersion } catch {}
        $items += [pscustomobject]@{
            name=$file.Name
            size_bytes=$file.Length
            file_version=$version
            sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    return $items
}

function Read-Log([string]$PathValue) {
    if (-not (Test-Path -LiteralPath $PathValue -PathType Leaf)) { return '' }
    try { return [System.IO.File]::ReadAllText($PathValue) } catch { return '' }
}

function Match-Number([string]$Text, [string]$Pattern) {
    $match = [regex]::Match($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $match.Success) { return $null }
    try {
        return [double]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    } catch {
        return $null
    }
}

function Analyze-Profile([string]$ProfilePath) {
    $providers = @{}
    $ops = @{}
    $fallbackOps = @{}
    $uniqueNodes = New-Object 'System.Collections.Generic.HashSet[string]'
    $npuEvents = 0
    $cpuEvents = 0
    $otherEvents = 0
    $npuDuration = 0.0
    $cpuDuration = 0.0
    $otherDuration = 0.0
    $totalEvents = 0

    if (-not (Test-Path -LiteralPath $ProfilePath -PathType Leaf)) {
        return [pscustomobject]@{ found=$false; total_events=0; unique_nodes=0; npu_events=0; cpu_events=0; other_events=0; npu_event_share_percent=0.0; npu_duration_share_percent=0.0; provider_counts=@{}; op_counts=@{}; cpu_fallback_ops=@{} }
    }
    try { $events = Get-Content -Raw -LiteralPath $ProfilePath | ConvertFrom-Json } catch {
        return [pscustomobject]@{ found=$false; total_events=0; unique_nodes=0; npu_events=0; cpu_events=0; other_events=0; npu_event_share_percent=0.0; npu_duration_share_percent=0.0; provider_counts=@{}; op_counts=@{}; cpu_fallback_ops=@{} }
    }
    foreach ($event in $events) {
        if ($event.cat -ne 'Node') { continue }
        $provider = [string]$event.args.provider
        if ([string]::IsNullOrWhiteSpace($provider)) { continue }
        $op = [string]$event.args.op_name
        $name = [string]$event.name
        $duration = 0.0
        try { $duration = [double]$event.dur } catch {}
        $totalEvents++
        [void]$uniqueNodes.Add($name + '|' + $provider)
        if (-not $providers.ContainsKey($provider)) { $providers[$provider] = 0 }
        $providers[$provider]++
        if ($op) {
            if (-not $ops.ContainsKey($op)) { $ops[$op] = 0 }
            $ops[$op]++
        }
        if ($provider -match 'VitisAI|RyzenAI|NPU') {
            $npuEvents++; $npuDuration += $duration
        } elseif ($provider -eq 'CPUExecutionProvider') {
            $cpuEvents++; $cpuDuration += $duration
            if ($op) {
                if (-not $fallbackOps.ContainsKey($op)) { $fallbackOps[$op] = 0 }
                $fallbackOps[$op]++
            }
        } else {
            $otherEvents++; $otherDuration += $duration
        }
    }
    $eventShare = 0.0
    if ($totalEvents -gt 0) { $eventShare = [math]::Round(100.0 * $npuEvents / $totalEvents, 3) }
    $totalDuration = $npuDuration + $cpuDuration + $otherDuration
    $durationShare = 0.0
    if ($totalDuration -gt 0) { $durationShare = [math]::Round(100.0 * $npuDuration / $totalDuration, 3) }
    return [pscustomobject]@{
        found=$true; total_events=$totalEvents; unique_nodes=$uniqueNodes.Count
        npu_events=$npuEvents; cpu_events=$cpuEvents; other_events=$otherEvents
        npu_event_share_percent=$eventShare; npu_duration_share_percent=$durationShare
        provider_counts=$providers; op_counts=$ops; cpu_fallback_ops=$fallbackOps
    }
}

function Invoke-Perf([string]$ModelPath, [string]$Variant, [string]$Provider, [object]$HashInfo, [string]$Stamp) {
    $safeName = ([System.IO.Path]::GetFileNameWithoutExtension($ModelPath) -replace '[^A-Za-z0-9._-]', '_')
    $prefix = Join-Path $resultsDir ($safeName + '-' + $HashInfo.short + '-' + $Stamp + '-' + $Provider)
    $logPath = $prefix + '.log'
    $profilePrefix = $prefix + '-profile'
    $cacheDir = Join-Path $cacheRoot $HashInfo.short
    $cacheWasPresent = $null
    if ($Provider -ne 'cpu') {
        $cacheWasPresent = Test-Path -LiteralPath $cacheDir -PathType Container
        if ($ForceRecompile -and $cacheWasPresent) {
            Remove-Item -LiteralPath $cacheDir -Recurse -Force
            $cacheWasPresent = $false
        }
        New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
    }

    $providerArguments = @()
    if ($Provider -eq 'cpu') {
        $providerArguments += @('-e', 'cpu')
    } else {
        $providerOptions = "config_file|$config enable_cache_file_io_in_mem|0 target|X1-Benchmark cache_dir|$cacheDir cache_key|gamesr_$($HashInfo.short)"
        $providerArguments += @('-e', 'vitisai', '-i', $providerOptions)
    }
    # perf_test -c means parallel/concurrent runs, not warm-up count. GameSR is
    # a serial frame pipeline, so measured concurrency must remain exactly 1.
    $commonArguments = @('-I', '-m', 'times', '-c', '1', '-S', '42') + $providerArguments
    $warmupArguments = $commonArguments + @('-r', "$Warmup", $ModelPath)
    $arguments = $commonArguments + @('-r', "$Runs")
    if (-not $NoProfile) { $arguments += @('-p', $profilePrefix) }
    $arguments += @('-s', $ModelPath)

    Write-Host ''
    Write-Host ("[RUN] {0} | {1} | hash={2} | cache_before={3}" -f $Variant, $Provider, $HashInfo.short, $cacheWasPresent) -ForegroundColor Cyan
    Write-Host ("[WARMUP] separate process, serial runs={0}" -f $Warmup)
    $warmupTimer = [System.Diagnostics.Stopwatch]::StartNew()
    $savedErrorAction = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 converts any native stderr line (including
        # harmless ORT warnings) into a non-terminating NativeCommandError.
        # Temporarily continue so stderr is captured without aborting the run;
        # the real native exit code below remains authoritative.
        $ErrorActionPreference = 'Continue'
        $warmupOutput = @(& $perfExe @warmupArguments 2>&1)
        $warmupExitCode = [int]$LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedErrorAction
        $warmupTimer.Stop()
    }
    $warmupLog = $prefix + '-warmup.log'
    $warmupOutput | Out-File -LiteralPath $warmupLog -Encoding utf8
    if ($warmupExitCode -ne 0) {
        Write-Host ("[WARN] Warm-up process exited with code {0}; measured run will still be attempted." -f $warmupExitCode) -ForegroundColor Yellow
        foreach ($line in $warmupOutput) { Write-Host ([string]$line) }
    }
    Write-Host ("[MEASURE] serial concurrency=1, runs={0}" -f $Runs)
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $ErrorActionPreference = 'Continue'
        $nativeOutput = @(& $perfExe @arguments 2>&1)
        $exitCode = [int]$LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedErrorAction
        $timer.Stop()
    }
    $nativeOutput | Out-File -LiteralPath $logPath -Encoding utf8
    foreach ($line in $nativeOutput) { Write-Host ([string]$line) }

    $profilePath = $profilePrefix + '.json'
    if (-not $NoProfile) {
        $profileLeaf = [System.IO.Path]::GetFileName($profilePrefix)
        $profileFile = Get-ChildItem -LiteralPath $resultsDir -Filter ($profileLeaf + '*.json') -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($profileFile) { $profilePath = $profileFile.FullName }
    }
    $raw = Read-Log $logPath
    $averageMs = Match-Number $raw 'Average inference time cost:\s*([0-9.]+)\s*ms'
    if ($null -eq $averageMs) {
        $averageSec = Match-Number $raw 'Average inference time cost:\s*([0-9.]+)\s*s'
        if ($null -ne $averageSec) { $averageMs = $averageSec * 1000.0 }
    }
    $fps = Match-Number $raw 'Number of inferences per second:\s*([0-9.]+)'
    if ($null -eq $fps -and $null -ne $averageMs -and $averageMs -gt 0) { $fps = 1000.0 / $averageMs }
    $p50 = Match-Number $raw 'P50 Latency:\s*([0-9.]+)\s*s'
    $p95 = Match-Number $raw 'P95 Latency:\s*([0-9.]+)\s*s'
    if ($null -ne $p50) { $p50 *= 1000.0 }
    if ($null -ne $p95) { $p95 *= 1000.0 }
    $profile = Analyze-Profile $profilePath

    $status = 'FAIL'
    if ($exitCode -eq 0) {
        if ($Provider -eq 'cpu') {
            $status = 'CPU_BASELINE'
        } elseif (-not $profile.found -or $profile.total_events -eq 0) {
            $status = 'UNVERIFIED'
        } elseif ($profile.npu_events -gt 0 -and $profile.cpu_events -eq 0 -and $profile.other_events -eq 0) {
            $status = 'NPU_ONLY'
        } elseif ($profile.npu_events -gt 0) {
            $status = 'NPU_PARTIAL'
        } else {
            $status = 'CPU_FALLBACK'
        }
    }
    return [pscustomobject]@{
        variant=$Variant; provider=$Provider; status=$status; exit_code=$exitCode
        average_ms=$averageMs; p50_ms=$p50; p95_ms=$p95; pure_fps=$fps
        warmup_runs=$Warmup; measured_concurrency=1; random_seed=42
        warmup_exit_code=$warmupExitCode; warmup_wall_seconds=[math]::Round($warmupTimer.Elapsed.TotalSeconds, 3)
        process_wall_seconds=[math]::Round($timer.Elapsed.TotalSeconds, 3)
        model=$ModelPath; model_fingerprint=$HashInfo.fingerprint
        cache_key=$(if ($Provider -eq 'cpu') { $null } else { $HashInfo.short })
        cache_present_before=$cacheWasPresent
        cache_directory=$(if ($Provider -eq 'cpu') { $null } else { $cacheDir })
        profile=$profilePath; profile_found=$profile.found; total_profile_events=$profile.total_events
        unique_profile_nodes=$profile.unique_nodes; npu_events=$profile.npu_events
        cpu_events=$profile.cpu_events; other_provider_events=$profile.other_events
        npu_event_share_percent=$profile.npu_event_share_percent
        npu_duration_share_percent=$profile.npu_duration_share_percent
        provider_counts=$profile.provider_counts; op_counts=$profile.op_counts
        cpu_fallback_ops=$profile.cpu_fallback_ops; warmup_log=$warmupLog; log=$logPath
    }
}

$modelSet = Resolve-ModelSet $DeploymentPath
$orderedVariants = @('float', 'fp16', 'int8')
$selected = @()
foreach ($variant in $orderedVariants) {
    if ($OnlyInt8 -and $variant -ne 'int8') { continue }
    $matches = @($modelSet.entries | Where-Object { $_.variant -eq $variant })
    if ($matches.Count -gt 1) { Fail "More than one $variant model was resolved for family $($modelSet.family)." }
    if ($matches.Count -eq 1) { $selected += $matches[0] }
}
if ($OnlyInt8 -and @($selected | Where-Object { $_.variant -eq 'int8' }).Count -eq 0) {
    Fail 'No INT8/QDQ model was found. Build it on the training PC first.'
}
if ($selected.Count -eq 0) { Fail 'No model variants were selected.' }

Write-Host ''
Write-Host 'GameSR Ryzen AI NPU Benchmark v2.4.1' -ForegroundColor Cyan
Write-Host '==================================' -ForegroundColor Cyan
Write-Host "Family  : $($modelSet.family)"
Write-Host "Source  : $($modelSet.source)"
Write-Host "Manifest: $($modelSet.manifest)"
Write-Host "Warmup  : $Warmup"
Write-Host "Runs    : $Runs"
Write-Host "CPU     : $(-not $SkipCpu)"
Write-Host ''
Write-Host 'This benchmark measures synthetic-input model inference only; capture, preprocessing, display, and Magpie latency are not included.' -ForegroundColor Yellow
Write-Host 'NPU event/duration shares come from runtime profile events and are not claimed as static ONNX graph coverage.' -ForegroundColor Yellow

$oldPath = $env:PATH
$env:PATH = $bin + ';' + $env:PATH
$env:XLNX_ENABLE_CACHE = '1'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$allResults = @()
$modelMetadata = @()
try {
    foreach ($entry in $selected) {
        $hashInfo = Get-ModelFingerprint $entry.path
        $modelMetadata += [pscustomobject]@{
            variant=$entry.variant; path=$entry.path; fingerprint=$hashInfo.fingerprint
            onnx_sha256=$hashInfo.onnx_sha256; external_sha256=$hashInfo.external_sha256
            external_present=$hashInfo.external_present
        }
        Write-Host ''
        Write-Host ("Model {0}: {1}" -f $entry.variant, $entry.path) -ForegroundColor Green
        Write-Host ("SHA-256 fingerprint: {0}" -f $hashInfo.fingerprint)
        $allResults += Invoke-Perf $entry.path $entry.variant 'vitisai' $hashInfo $stamp
        if (-not $SkipCpu) {
            $allResults += Invoke-Perf $entry.path $entry.variant 'cpu' $hashInfo $stamp
        }
    }
} finally {
    $env:PATH = $oldPath
}

$primaryVariant = 'int8'
if (@($selected | Where-Object { $_.variant -eq 'int8' }).Count -eq 0) {
    if (@($selected | Where-Object { $_.variant -eq 'fp16' }).Count -gt 0) { $primaryVariant = 'fp16' } else { $primaryVariant = 'float' }
}
$primary = @($allResults | Where-Object { $_.variant -eq $primaryVariant -and $_.provider -eq 'vitisai' }) | Select-Object -First 1
$overall = if ($primary) { $primary.status } else { 'FAIL' }
$exitCode = 5
if ($overall -eq 'NPU_ONLY') { $exitCode = 0 }
elseif ($overall -eq 'NPU_PARTIAL') { $exitCode = 3 }
elseif ($overall -eq 'CPU_FALLBACK' -or $overall -eq 'UNVERIFIED') { $exitCode = 4 }

$summaryBase = Join-Path $resultsDir ("GameSR-v24-{0}-{1}" -f ($modelSet.family -replace '[^A-Za-z0-9._-]', '_'), $stamp)
$jsonPath = $summaryBase + '-SUMMARY.json'
$textPath = $summaryBase + '-SUMMARY.txt'
$payload = [ordered]@{
    schema_version=4
    benchmark='GameSR Ryzen AI NPU Benchmark v2.4.1'
    created=(Get-Date).ToString('o')
    family=$modelSet.family
    source=$modelSet.source
    deployment_manifest=$modelSet.manifest
    runtime_target='X1-Benchmark'
    input_mode='onnxruntime_perf_test synthetic input'
    warmup=$Warmup
    runs=$Runs
    measured_concurrency=1
    only_int8=[bool]$OnlyInt8
    cpu_baseline_enabled=(-not $SkipCpu)
    force_recompile=[bool]$ForceRecompile
    profile_enabled=(-not $NoProfile)
    primary_variant=$primaryVariant
    overall_status=$overall
    exit_code=$exitCode
    runtime_files=(Get-RuntimeMetadata)
    models=$modelMetadata
    results=$allResults
}
$payload | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

$lines = @(
    'GameSR Ryzen AI NPU Benchmark v2.4.1',
    '==================================',
    ('Finished: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
    ('Family: ' + $modelSet.family),
    ('Primary variant: ' + $primaryVariant),
    ('Overall status: ' + $overall),
    ''
)
foreach ($result in $allResults) {
    $avgText = if ($null -eq $result.average_ms) { 'N/A' } else { '{0:F4}' -f $result.average_ms }
    $fpsText = if ($null -eq $result.pure_fps) { 'N/A' } else { '{0:F3}' -f $result.pure_fps }
    $lines += ("[{0}] {1}/{2}" -f $result.status, $result.variant, $result.provider)
    $lines += ('  Average ms: ' + $avgText)
    $lines += ('  Pure FPS: ' + $fpsText)
    $lines += ('  NPU event share: {0:F3}%' -f $result.npu_event_share_percent)
    $lines += ('  NPU duration share: {0:F3}%' -f $result.npu_duration_share_percent)
    $lines += ('  NPU/CPU events: {0}/{1}' -f $result.npu_events, $result.cpu_events)
    $lines += ('  Warm-up/measured concurrency: separate process / 1')
    if ($result.cpu_fallback_ops.Count -gt 0) {
        $fallbackText = @($result.cpu_fallback_ops.GetEnumerator() | Sort-Object Value -Descending | ForEach-Object { $_.Key + '=' + $_.Value }) -join ', '
        $lines += ('  CPU fallback ops: ' + $fallbackText)
    }
    $lines += ('  Cache key: ' + $result.cache_key)
    $lines += ('  Log: ' + $result.log)
    $lines += ('  Profile: ' + $result.profile)
    $lines += ''
}
$lines += 'Interpretation:'
$lines += 'NPU_ONLY = profiled execution used VitisAI/RyzenAI with no CPU or other provider events.'
$lines += 'NPU_PARTIAL = NPU execution was observed, but some work fell back to CPU/another provider.'
$lines += 'CPU_FALLBACK = the VitisAI run completed but no NPU node execution was observed.'
$lines += 'UNVERIFIED = native run succeeded but no usable profile was produced.'
$lines += 'The reported shares are dynamic profile-event shares, not static graph-operator coverage.'
$lines | Set-Content -LiteralPath $textPath -Encoding UTF8

Write-Host ''
$color = if ($overall -eq 'NPU_ONLY') { 'Green' } elseif ($overall -eq 'NPU_PARTIAL') { 'Yellow' } else { 'Red' }
Write-Host ("[{0}] Benchmark complete; primary={1}" -f $overall, $primaryVariant) -ForegroundColor $color
foreach ($result in $allResults) {
    Write-Host ("  {0}/{1}: status={2}, avg={3} ms, FPS={4}, NPU-events={5}%, NPU-duration={6}%" -f $result.variant, $result.provider, $result.status, $result.average_ms, $result.pure_fps, $result.npu_event_share_percent, $result.npu_duration_share_percent)
}
Write-Host "Summary: $textPath"
Write-Host "JSON   : $jsonPath"
exit $exitCode
