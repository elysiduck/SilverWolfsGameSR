# Reproducible build entry points

All scripts run from any working directory. They do not commit generated files.

```powershell
# Verify installed tools. Add -VerifyArtifacts after downloading release assets.
.\scripts\bootstrap.ps1 -Component All

# Create the isolated training environment and run its smoke test.
.\scripts\build-training.ps1 -TorchBuild cu128

# Create the Collector/Cleaner environment and run its self-test.
.\scripts\build-dataset-tools.ps1
```

Magpie and the XDNA benchmark require AMD/ORT runtime files that are intentionally not in Git. Download and extract release assets beneath one artifact root, verify them, then stage into ignored build locations:

```text
<artifact-root>/
  Magpie-SESR-GameSR-v4.1.1-AsyncNPU-CurrentInput-full-source.zip
  GameSR-XDNA-Architecture-Benchmark-v1.3.zip
  magpie-v4.1.1-runtime/third_party/sesr/{model,onnx}/...
  benchmark-v1.3-runtime/{runtime,models,control}/...
```

```powershell
.\scripts\bootstrap.ps1 -Component Magpie -ArtifactRoot D:\GameSR-artifacts -VerifyArtifacts
.\scripts\build-magpie.ps1 -ArtifactRoot D:\GameSR-artifacts -StageRuntime
.\scripts\build-benchmark.ps1 -ArtifactRoot D:\GameSR-artifacts -StageRuntime -Warmup 20 -Runs 600
```

`artifacts.lock.json` is the source of truth for artifact hashes. `ARTIFACTS.md` describes the intended Release/LFS/external-storage destination of each class of output.
