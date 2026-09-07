# Artifact policy and local inventory

Git tracks source code, configuration, documentation and compact benchmark summaries. Large generated or redistributable files are excluded by `.gitignore`.

## Storage destinations

- GitHub Releases: versioned source bundles, deployable ONNX packages and compact benchmark bundles.
- Git LFS: only when a model must be versioned together with code; track it explicitly.
- Independent artifact storage: checkpoints, full profiler traces, datasets, virtual environments and vendor runtimes.

## Local release assets held outside Git

| Component | Local artifact | Intended destination |
| --- | --- | --- |
| Training | `GameSR-training-code-v1.8.2-Late-Polish-Information-Aware-Patch.zip` | Release `training-v1.8.2` |
| Dataset tools | Collector v3.2 ZIP and Cleaner v3.2.1 patch ZIP | Release `dataset-tools-v3.2.1` |
| Benchmark | `GameSR-XDNA-Architecture-Benchmark-v1.3.zip` | Release `benchmark-v1.3` after redistribution review |
| Benchmark 105patched | `GameSR-XDNA-Architecture-Benchmark-v1.3-105patched.zip` | Release `benchmark-v1.3-105patched`; 105 compiled caches, runtime and raw results |
| Output-channel probe | v1.0.2 patch, O3/O4/O6/O8 model payload and Compact result | Release `output-channel-probe-v1.0.2` |
| Potential-4 output probe | v1.1.0 PC generator, handheld Runner, 16-model payload and Compact result | Release `potential4-output-probe-v1.1.0` |
| Sweep | 105-model bundle and compact result ZIP | Release or independent artifact storage |
| Magpie | v4.1.1 full-source ZIP | Release `magpie-v4.1.1` |
| Magpie | v4.3 runtime ZIP (296 MB) + v4.3 source-patch ZIP | Release `magpie-v4.3` (runtime binary must be uploaded as Release asset; source patch extracted to `magpie/patches/v4.3-phase3x3-modelfolders`) |
| Magpie | v4.4 runtime ZIP (296 MB) + v4.4 full-source ZIP | Release `magpie-v4.4` (runtime binary must be uploaded as Release asset; 9-file GameSR modification set extracted to `magpie/patches/v4.4-phase3x3-vectorized`) |
| Magpie | v4.4 FullSource RuntimeDeps Fix ZIP (132 MB) | Release `magpie-v4.4`; restores 4 RyzenAI/VitisAI DLLs under `third_party/sesr/onnx/bin`. **DLLs are gitignored (`*.dll`) — Release asset only.** Scripts (`README_FIX_CN.txt`, `VERIFY-RUNTIME-DEPS.cmd`) extracted to `magpie/patches/v4.4-fullsource-runtimedeps-fix` |
| Magpie | v4.4 Runtime Perf Probe patch ZIP (4 KB, scripts) | In git at `magpie/patches/v4.4-runtime-perf-probe` (no binaries; overlay onto the v4.4 runtime root, no rebuild needed) |
| Magpie | v4.5 One-Click Probe source patch ZIP (+ FIXED) | In git at `magpie/patches/v4.5-one-click-probe` (FIXED = canonical; original v1 retained by SHA256 in `artifacts.lock.json`) |
| Models | C32-L3 ONNX/QDQ and checkpoints | Model release or independent artifact storage |

Vendor runtimes and full profiler output must not enter Git or LFS without a licensing and size review.
