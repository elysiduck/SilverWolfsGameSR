param(
  [Parameter(Mandatory=$true)][string]$Root
)
$ErrorActionPreference = 'Stop'
$cleanRoot = $Root.Trim().Trim([char]34)
$cleanRoot = $cleanRoot.TrimEnd([char]92, [char]47)
if ([string]::IsNullOrWhiteSpace($cleanRoot)) { throw 'Probe root path is empty.' }
$rootPath = [IO.Path]::GetFullPath($cleanRoot)
$deadline = (Get-Date).AddMinutes(10)
$probe = $null
Write-Host "[Probe] Waiting for a GameSR-Probe-* directory..."
while ((Get-Date) -lt $deadline) {
  $probe = Get-ChildItem -LiteralPath $rootPath -Directory -Filter 'GameSR-Probe-*' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($probe) { break }
  Start-Sleep -Milliseconds 500
}
if (-not $probe) { Write-Host '[ERROR] Probe did not start. Did you enable GameSR scaling?'; exit 2 }
Write-Host ("[Probe] Active: " + $probe.FullName)
while ((Get-Date) -lt $deadline) {
  if (Test-Path -LiteralPath (Join-Path $probe.FullName 'probe.done')) { break }
  $status = Join-Path $probe.FullName 'probe.status'
  if (Test-Path -LiteralPath $status) {
    $text = (Get-Content -LiteralPath $status -Raw -ErrorAction SilentlyContinue).Trim()
    if ($text) { Write-Host ("[Probe] " + ($text -replace "`r?`n", ' | ')) }
  }
  Start-Sleep -Seconds 2
}
if (-not (Test-Path -LiteralPath (Join-Path $probe.FullName 'probe.done'))) {
  Write-Host '[ERROR] Timed out before performance collection completed.'; exit 3
}

function Read-Metrics([string]$path) {
  $h = @{}
  if (Test-Path -LiteralPath $path) {
    Import-Csv -LiteralPath $path | ForEach-Object { $h[$_.metric] = [double]$_.value }
  }
  return $h
}
$q = Read-Metrics (Join-Path $probe.FullName 'quality_metrics.csv')
$p = Read-Metrics (Join-Path $probe.FullName 'performance_metrics.csv')
$log = Join-Path $rootPath 'logs\magpie.log'
$capture = 'unknown'
$fence = @()
if (Test-Path -LiteralPath $log) {
  $lines = Get-Content -LiteralPath $log -ErrorAction SilentlyContinue
  $cap = $lines | Where-Object { $_ -match '当前捕获模式:\s*(.+)$' } | Select-Object -Last 1
  if ($cap -and $cap -match '当前捕获模式:\s*(.+)$') { $capture = $Matches[1].Trim() }
  foreach ($line in $lines) {
    if ($line -match 'GameSR BACKEND v4:.*GPU fence等待\s*([0-9.]+)\s*ms') { $fence += [double]$Matches[1] }
  }
}
$fenceAvg = if ($fence.Count) { ($fence | Measure-Object -Average).Average } else { 0 }
$fenceMax = if ($fence.Count) { ($fence | Measure-Object -Maximum).Maximum } else { 0 }

$quality = 'UNKNOWN'
$qualityNotes = New-Object System.Collections.Generic.List[string]
if ($q.ContainsKey('fp32_available') -and $q['fp32_available'] -lt 0.5) {
  $quality = 'INCOMPLETE'; $qualityNotes.Add('FP32 sibling model was missing or CPU inference failed.')
} elseif ($q.ContainsKey('latent_cosine')) {
  $cos = $q['latent_cosine']; $rgb = $q['rgb_fp32_vs_int8_psnr_db']; $ratio = $q['residual_rms_ratio_int8_over_fp32']
  if ($cos -lt 0.98 -or $rgb -lt 35 -or $ratio -lt 0.60 -or $ratio -gt 1.60) {
    $quality = 'FAIL'; $qualityNotes.Add('QDQ/INT8 output differs strongly from the FP32 reference.')
  } elseif ($cos -lt 0.995 -or $rgb -lt 42 -or $ratio -lt 0.80 -or $ratio -gt 1.25) {
    $quality = 'WARN'; $qualityNotes.Add('Quantization difference is noticeable; inspect the generated FP32/INT8 images.')
  } else {
    $quality = 'PASS'; $qualityNotes.Add('FP32 and QDQ/INT8 are numerically close on the sampled real 720p frame.')
  }
}
$perf = 'UNKNOWN'
$perfNotes = New-Object System.Collections.Generic.List[string]
if ($p.ContainsKey('fresh_fps')) {
  $fps=$p['fresh_fps']; $reuse=$p['reuse_rate_per_fresh']; $drop=$p['drop_rate_per_submitted']; $npu=$p['npu_avg_ms']
  if ($fps -lt 50 -or $reuse -gt 0.15 -or $drop -gt 0.10) { $perf='FAIL' }
  elseif ($fps -lt 57 -or $reuse -gt 0.05 -or $drop -gt 0.03) { $perf='WARN' } else { $perf='PASS' }
  if ($npu -gt 16.67) { $perfNotes.Add('NPU average alone exceeds a 60-FPS frame budget.') }
  elseif ($npu -gt 12) { $perfNotes.Add('NPU consumes a large share of the 16.67 ms 60-FPS budget.') }
  if ($reuse -gt 0.05) { $perfNotes.Add(('Old-result reuse is high: {0:P1}.' -f $reuse)) }
  if ($drop -gt 0.03) { $perfNotes.Add(('Capture/worker drop rate is high: {0:P1}.' -f $drop)) }
  if ($capture -eq 'GDI') { $perfNotes.Add('Capture backend is GDI; this is a priority candidate for real-time stutter.') }
  if ($fenceAvg -gt 4) { $perfNotes.Add(('GPU fence waiting is material: avg {0:N2} ms, max {1:N2} ms.' -f $fenceAvg,$fenceMax)) }
}

$report = New-Object System.Collections.Generic.List[string]
$report.Add('GameSR One-Click Deployment Probe')
$report.Add(('Generated: ' + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')))
$report.Add(('Capture backend: ' + $capture))
$report.Add('')
$report.Add(('IMAGE QUALITY: ' + $quality))
foreach($n in $qualityNotes){$report.Add('  - '+$n)}
if($q.Count){
  $report.Add(('  FP32 CPU inference: {0:N1} ms' -f $q['fp32_cpu_inference_ms']))
  $report.Add(('  Latent cosine: {0:N6}' -f $q['latent_cosine']))
  $report.Add(('  FP32 vs INT8 RGB PSNR: {0:N2} dB' -f $q['rgb_fp32_vs_int8_psnr_db']))
  $report.Add(('  Residual RMS ratio INT8/FP32: {0:N3}' -f $q['residual_rms_ratio_int8_over_fp32']))
}
$report.Add('')
$report.Add(('PERFORMANCE: ' + $perf))
foreach($n in $perfNotes){$report.Add('  - '+$n)}
if($p.Count){
  $report.Add(('  Fresh GameSR result rate: {0:N2} FPS' -f $p['fresh_fps']))
  $report.Add(('  NPU avg/max: {0:N2}/{1:N2} ms' -f $p['npu_avg_ms'],$p['npu_max_ms']))
  $report.Add(('  Input pack: {0:N2} ms' -f $p['input_pack_avg_ms']))
  $report.Add(('  Output pack: {0:N2} ms' -f $p['output_pack_avg_ms']))
  $report.Add(('  Residual upload: {0:N2} ms' -f $p['residual_upload_avg_ms']))
  $report.Add(('  Reuse rate: {0:P1}' -f $p['reuse_rate_per_fresh']))
  $report.Add(('  Drop rate: {0:P1}' -f $p['drop_rate_per_submitted']))
  $report.Add(('  GPU fence avg/max from Magpie log: {0:N2}/{1:N2} ms' -f $fenceAvg,$fenceMax))
}
$report.Add('')
$report.Add('HOW TO READ THE PNGs')
$report.Add('  02_bilinear_1080p.png = deployment baseline')
$report.Add('  03_fp32_cpu_decoder_1080p.png = FP32 model + reference CPU phase decoder')
$report.Add('  04_int8_cpu_decoder_1080p.png = QDQ/INT8 model + same CPU phase decoder')
$report.Add('  05_zero_residual_1080p.png = must equal bilinear by construction')
$report.Add('  06/07 residual_view_x8 = residual enlarged 8x around mid-gray')
$report.Add('')
$report.Add('INTERPRETATION')
$report.Add('  FP32 good + INT8 bad => quantization/QDQ is the main image-quality suspect.')
$report.Add('  FP32 and INT8 probe images both good, but live Magpie bad => GPU decoder/capture/color path is suspect.')
$report.Add('  FP32 and INT8 probe images both poor on the real 720p frame => training/domain generalization is suspect.')
$report.Add('  Quality good but performance FAIL => optimize capture + transfer/pack/upload pipeline, not the training model first.')
$reportPath = Join-Path $probe.FullName 'REPORT.txt'
$report | Set-Content -LiteralPath $reportPath -Encoding UTF8

$zipPath = $probe.FullName + '.zip'
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Compress-Archive -Path (Join-Path $probe.FullName '*') -DestinationPath $zipPath -CompressionLevel Optimal
Write-Host ('[DONE] Report: ' + $reportPath)
Write-Host ('[DONE] Package: ' + $zipPath)
Start-Process explorer.exe -ArgumentList @('/select,',('"'+$zipPath+'"'))
exit 0
