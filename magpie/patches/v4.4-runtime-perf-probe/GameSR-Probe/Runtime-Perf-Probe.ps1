param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath($Root)
$log=Join-Path $root 'logs\magpie.log'
$deadline=(Get-Date).AddMinutes(8)
$startCount=0
if(Test-Path -LiteralPath $log){$startCount=(Get-Content -LiteralPath $log -ErrorAction SilentlyContinue).Count}
Write-Host '[Probe] Waiting for GameSR PIPELINE v4 ASYNC...'
$samples=@()
$backend=@()
$capture='unknown'
while((Get-Date)-lt $deadline -and $samples.Count -lt 10){
  if(Test-Path -LiteralPath $log){
    $lines=Get-Content -LiteralPath $log -ErrorAction SilentlyContinue
    if($lines.Count -gt $startCount){
      $new=$lines[$startCount..($lines.Count-1)]; $startCount=$lines.Count
      foreach($line in $new){
        if($line -match '当前捕获模式:\s*(.+)$'){$capture=$Matches[1].Trim()}
        if($line -match '^(?<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\|.*GameSR PIPELINE v4 ASYNC: render线程 (?<render>[0-9.]+) ms, GPU->CPU行拷贝 (?<copy>[0-9.]+) ms, worker input-pack (?<pack>[0-9.]+) ms, NPU (?<npu>[0-9.]+) ms, worker output-pack (?<out>[0-9.]+) ms, residual上传 (?<upload>[0-9.]+) ms, GPU合成提交 (?<compose>[0-9.]+) ms, 处理帧 (?<fresh>\d+), 提交 (?<submitted>\d+), worker完成 (?<completed>\d+), 捕获/结果丢弃 (?<drop>\d+), worker队列丢弃 (?<wdrop>\d+), 复用 (?<reuse>\d+)'){
          $samples += [pscustomobject]@{Time=[datetime]::ParseExact($Matches.ts,'yyyy-MM-dd HH:mm:ss.fff',$null);Render=[double]$Matches.render;Copy=[double]$Matches.copy;Pack=[double]$Matches.pack;NPU=[double]$Matches.npu;Out=[double]$Matches.out;Upload=[double]$Matches.upload;Compose=[double]$Matches.compose;Fresh=[int64]$Matches.fresh;Submitted=[int64]$Matches.submitted;Completed=[int64]$Matches.completed;Drop=[int64]$Matches.drop;WDrop=[int64]$Matches.wdrop;Reuse=[int64]$Matches.reuse}
          Write-Host ("[Probe] pipeline sample {0}/10: NPU {1:N2} ms" -f $samples.Count,[double]$Matches.npu)
        }
        if($line -match 'GameSR BACKEND v4: 总 (?<total>[0-9.]+) ms, SESR提交/收结果 (?<sesr>[0-9.]+) ms, effects CPU提交 (?<cpu>[0-9.]+) ms, GPU fence等待 (?<fence>[0-9.]+) ms, shared Acquire\+Copy\+Flush (?<shared>[0-9.]+) ms'){
          $backend += [pscustomobject]@{Total=[double]$Matches.total;SESR=[double]$Matches.sesr;CPU=[double]$Matches.cpu;Fence=[double]$Matches.fence;Shared=[double]$Matches.shared}
        }
      }
    }
  }
  Start-Sleep -Milliseconds 500
}
if($samples.Count -lt 2){Write-Host '[ERROR] Not enough GameSR samples. Make sure scaling is active with GameSR NPU O4.';exit 2}
$first=$samples[0];$last=$samples[-1]
$seconds=($last.Time-$first.Time).TotalSeconds
$freshDelta=$last.Fresh-$first.Fresh
$submittedDelta=$last.Submitted-$first.Submitted
$dropDelta=$last.Drop-$first.Drop
$wDropDelta=$last.WDrop-$first.WDrop
$reuseDelta=$last.Reuse-$first.Reuse
$freshFps=if($seconds -gt 0){$freshDelta/$seconds}else{0}
$reuseRate=if($freshDelta -gt 0){$reuseDelta/$freshDelta}else{0}
$dropRate=if($submittedDelta -gt 0){($dropDelta+$wDropDelta)/$submittedDelta}else{0}
function Avg($arr,$prop){return ($arr|Measure-Object -Property $prop -Average).Average}
function MaxV($arr,$prop){return ($arr|Measure-Object -Property $prop -Maximum).Maximum}
$npuAvg=Avg $samples 'NPU';$npuMax=MaxV $samples 'NPU'
$fenceAvg=if($backend.Count){Avg $backend 'Fence'}else{0};$fenceMax=if($backend.Count){MaxV $backend 'Fence'}else{0}
$status='PASS';$notes=New-Object System.Collections.Generic.List[string]
if($freshFps -lt 50 -or $reuseRate -gt .15 -or $dropRate -gt .10){$status='FAIL'}elseif($freshFps -lt 57 -or $reuseRate -gt .05 -or $dropRate -gt .03){$status='WARN'}
if($capture -eq 'GDI'){$notes.Add('Capture backend is GDI; test WGC/Desktop Duplication if compatible.')}
if($npuAvg -gt 16.67){$notes.Add('NPU average alone exceeds the 16.67 ms 60-FPS budget.')}elseif($npuAvg -gt 12){$notes.Add('NPU consumes a large share of the 60-FPS budget.')}
if($reuseRate -gt .05){$notes.Add(('Old-output reuse is high: {0:P1}.' -f $reuseRate))}
if($dropRate -gt .03){$notes.Add(('Capture/worker drop rate is high: {0:P1}.' -f $dropRate))}
if($fenceAvg -gt 4){$notes.Add(('GPU fence wait is material: avg {0:N2} ms.' -f $fenceAvg))}
$stamp=(Get-Date).ToString('yyyyMMdd-HHmmss');$outDir=Join-Path $root ('GameSR-Runtime-Perf-Probe-'+$stamp);New-Item -ItemType Directory -Path $outDir|Out-Null
$samples|Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $outDir 'pipeline_samples.csv')
if($backend.Count){$backend|Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath (Join-Path $outDir 'backend_samples.csv')}
$report=@()
$report+='GameSR Runtime Performance Probe'
$report+=('Status: '+$status)
$report+=('Capture backend: '+$capture)
$report+=('Fresh GameSR result rate: {0:N2} FPS' -f $freshFps)
$report+=('NPU avg/max: {0:N2}/{1:N2} ms' -f $npuAvg,$npuMax)
$report+=('GPU->CPU sample avg: {0:N2} ms' -f (Avg $samples 'Copy'))
$report+=('Input pack avg: {0:N2} ms' -f (Avg $samples 'Pack'))
$report+=('Output pack avg: {0:N2} ms' -f (Avg $samples 'Out'))
$report+=('Residual upload avg: {0:N2} ms' -f (Avg $samples 'Upload'))
$report+=('Reuse rate: {0:P1}' -f $reuseRate)
$report+=('Drop rate: {0:P1}' -f $dropRate)
$report+=('GPU fence avg/max: {0:N2}/{1:N2} ms' -f $fenceAvg,$fenceMax)
$report+='';$report+='Diagnosis:';foreach($n in $notes){$report+=(' - '+$n)}
$report+='';$report+='Note: This runtime-only patch cannot compare FP32 vs QDQ/INT8 image quality because the current Magpie.exe has no one-shot quality-probe hook. Use the source Probe build for the complete A/B test.'
$report|Set-Content -LiteralPath (Join-Path $outDir 'REPORT.txt') -Encoding UTF8
$zip=$outDir+'.zip';Compress-Archive -Path (Join-Path $outDir '*') -DestinationPath $zip -CompressionLevel Optimal
Write-Host ('[DONE] '+$zip);Start-Process explorer.exe -ArgumentList @('/select,',('"'+$zip+'"'));exit 0
