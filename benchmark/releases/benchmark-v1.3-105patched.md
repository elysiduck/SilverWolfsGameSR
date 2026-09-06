# Benchmark v1.3 — 105patched release

Release asset: `GameSR-XDNA-Architecture-Benchmark-v1.3-105patched.zip`

| Field | Value |
| --- | --- |
| SHA256 | `52ECDBD9145DE4768789CDAB0A26AF934B19812D0B93A273480812DD7BE34A1B` |
| Size | 308,876,364 bytes |
| Bundle content | Benchmark v1.3, Sweep Runner v1.0.2, 105 compiled XDNA caches, vendor runtime, model payloads, raw benchmark logs/results |
| Source status | All tracked scripts and manifests are byte-identical to the repository's canonical Benchmark v1.3 and Runner v1.0.2 sources. |

This is a binary release asset, not Git source. It is intentionally excluded by `.gitignore` because it contains vendor runtime binaries, ONNX models, compiled `.xmodel` caches and raw logs.

Before use, verify it against `artifacts.lock.json`:

```powershell
.\scripts\bootstrap.ps1 -Component Benchmark -ArtifactRoot D:\GameSR-artifacts -VerifyArtifacts
```
