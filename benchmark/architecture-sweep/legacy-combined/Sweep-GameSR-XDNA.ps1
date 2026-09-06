[CmdletBinding()]
param(
  [string]$Channels='4,6,8,10,12,14,16,20,24,28,32',
  [string]$Layers='2,3,4,5,6,8',
  [int]$Warmup=20,
  [int]$Runs=600,
  [switch]$ForceRecompile,
  [switch]$SkipModelBuild,
  [switch]$KeepFloatModels
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
function Fail($m){ Write-Host "[FAIL] $m" -ForegroundColor Red; exit 2 }
if(-not (Test-Path "$root\runtime\bin\onnxruntime_perf_test.exe")){ Fail 'runtime not found. Copy this patch into the root of your working GameSR XDNA Architecture Benchmark folder (the folder that contains runtime).' }
if(-not (Test-Path "$root\Benchmark-GameSR-v24.ps1")){ Fail 'Benchmark-GameSR-v24.ps1 missing.' }
$python='python'
if(-not $SkipModelBuild){
  & $python "$root\tools\generate_architecture_models.py" --root "$root" --channels $Channels --layers $Layers --skip-existing
  if($LASTEXITCODE -ne 0){ Fail 'Model generation/INT8 quantization failed. Use the same Python/Ryzen AI environment as GameSR training (torch + onnx + onnxruntime + amd-quark + Pillow).' }
}
$manifestPath="$root\sweep_manifest.json"
if(-not (Test-Path $manifestPath)){ Fail 'sweep_manifest.json missing.' }
$manifest=Get-Content -Raw $manifestPath | ConvertFrom-Json
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$rows=@()
$total=@($manifest.models).Count; $idx=0
foreach($m in @($manifest.models)){
  $idx++; $arch=[string]$m.architecture; $dir=Split-Path -Parent ([string]$m.model)
  Write-Host "`n[$idx/$total] $arch  C=$($m.channels) L=$($m.conv_layers)  GMAC=$($m.gmac_per_frame)" -ForegroundColor Cyan
  $before=@(Get-ChildItem "$root\results" -Filter "GameSR-v24-$arch-*-SUMMARY.json" -ErrorAction SilentlyContinue | Select-Object -Expand FullName)
  $args=@('-ExecutionPolicy','Bypass','-File',"$root\Benchmark-GameSR-v24.ps1",$dir,'-Warmup',$Warmup,'-Runs',$Runs,'-OnlyInt8','-SkipCpu')
  if($ForceRecompile){ $args += '-ForceRecompile' }
  & powershell.exe @args
  $after=@(Get-ChildItem "$root\results" -Filter "GameSR-v24-$arch-*-SUMMARY.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
  if($after.Count -eq 0){
    $rows += [pscustomobject]@{architecture=$arch;channels=$m.channels;conv_layers=$m.conv_layers;parameters=$m.parameters;gmac_per_frame=$m.gmac_per_frame;status='NO_SUMMARY';average_ms=$null;p50_ms=$null;p95_ms=$null;pure_fps=$null;effective_gmac_s=$null;npu_events=$null;cpu_events=$null;summary_json=''}
    continue
  }
  $sum=Get-Content -Raw $after[0].FullName | ConvertFrom-Json
  $r=@($sum.results | Where-Object {$_.variant -eq 'int8' -and $_.provider -eq 'vitisai'})[0]
  $eff=$null; if($r.average_ms -gt 0){$eff=[math]::Round(([double]$m.gmac_per_frame*1000.0/[double]$r.average_ms),3)}
  $rows += [pscustomobject]@{architecture=$arch;channels=[int]$m.channels;conv_layers=[int]$m.conv_layers;parameters=[int]$m.parameters;gmac_per_frame=[double]$m.gmac_per_frame;status=[string]$r.status;average_ms=[double]$r.average_ms;p50_ms=[double]$r.p50_ms;p95_ms=[double]$r.p95_ms;pure_fps=[double]$r.pure_fps;effective_gmac_s=$eff;npu_event_share_percent=[double]$r.npu_event_share_percent;npu_duration_share_percent=[double]$r.npu_duration_share_percent;npu_events=[int]$r.npu_events;cpu_events=[int]$r.cpu_events;cache_present_before=$r.cache_present_before;warmup_wall_seconds=[double]$r.warmup_wall_seconds;summary_json=$after[0].FullName}
}
$valid=@($rows | Where-Object {$_.status -eq 'NPU_ONLY' -and $_.average_ms -gt 0} | Sort-Object average_ms)
$rank=0; foreach($v in $valid){$rank++; Add-Member -InputObject $v -NotePropertyName rank -NotePropertyValue $rank -Force}
$csv="$root\results\GameSR-NPU-Sweep-$stamp-RANKING.csv"
$full="$root\results\GameSR-NPU-Sweep-$stamp-FULL.json"
$txt="$root\results\GameSR-NPU-Sweep-$stamp-SUMMARY.txt"
$valid | Select-Object rank,architecture,channels,conv_layers,parameters,gmac_per_frame,status,average_ms,p50_ms,p95_ms,pure_fps,effective_gmac_s,npu_event_share_percent,npu_duration_share_percent,npu_events,cpu_events,cache_present_before,warmup_wall_seconds,summary_json | Export-Csv -NoTypeInformation -Encoding UTF8 $csv
[pscustomobject]@{schema_version=1;benchmark='GameSR NPU Architecture Sweep v1.0';created=(Get-Date).ToString('o');channels=$Channels;layers=$Layers;warmup=$Warmup;runs=$Runs;force_recompile=[bool]$ForceRecompile;result_count=$rows.Count;valid_npu_only=$valid.Count;results=$rows} | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $full
$lines=@('GameSR NPU Architecture Sweep v1.0','',"Created: $(Get-Date)","Grid: channels=[$Channels], layers=[$Layers]","Warmup=$Warmup Runs=$Runs",'', 'TOP 20 (NPU_ONLY):')
$i=0; foreach($v in $valid | Select-Object -First 20){$i++; $lines += ('{0,2}. {1,-10}  avg {2,8:N4} ms  P95 {3,8:N4} ms  FPS {4,8:N2}  {5,8:N2} GMAC/s' -f $i,$v.architecture,$v.average_ms,$v.p95_ms,$v.pure_fps,$v.effective_gmac_s)}
$lines += ''; $lines += "CSV : $csv"; $lines += "FULL: $full"; $lines | Set-Content -Encoding UTF8 $txt
Write-Host "`n===== TOP 20 =====" -ForegroundColor Green
$valid | Select-Object -First 20 architecture,channels,conv_layers,average_ms,p95_ms,pure_fps,effective_gmac_s | Format-Table -AutoSize
Write-Host "`n[OK] Ranking: $csv" -ForegroundColor Green
Write-Host "[OK] Full JSON: $full" -ForegroundColor Green
Write-Host "[OK] Summary: $txt" -ForegroundColor Green
if(-not $KeepFloatModels -and (Test-Path "$root\float_models")){ Remove-Item "$root\float_models" -Recurse -Force }
