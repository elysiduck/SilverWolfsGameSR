does not contain the user's NVIDIA CUDA environment or the Windows AMD Quark
environment. `00_smoke_test.cmd` performs the CUDA forward/backward/checkpoint
test on the training PC. The normal pipeline performs ONNX and Quark graph gates
before it publishes an NPU deployment bundle.


---

## `docs/20-training/current-v1.7.1/dependencies/deployment_requirements.txt`

```text
# FP16/general deployment helpers used by the CUDA training environment.
# Ryzen AI INT8 uses the separate quark_requirements.txt environment.
numpy
Pillow
onnx
onnxruntime
onnxconverter-common

```

---

## `docs/20-training/current-v1.7.1/dependencies/quark_requirements.txt`

```text
amd-quark==0.12.post1
onnxruntime==1.23.2
onnx>=1.21.0,<=1.22.0
numpy>=2.0
Pillow>=10.0

```

---

## `docs/20-training/current-v1.7.1/guides/README.md`

# GameSR Training Code v1.7.1

PC-side end-to-end training and deployment preparation for 720p -> 1080p GameSR.

The training machine handles:

`clean -> benchmark -> full-image training -> NPU core ONNX -> AMD Quark XINT8/QDQ -> deployment bundle`

The handheld does not use this training environment. It only receives the model bundle and uses the separate standalone NPU benchmark/runtime.

Start with `先看这里-新手步骤.txt`. Technical details are in
`README_V1_7_1.md`. Legacy v1.7 design details remain in `README_V1_7.md`.


---

## `docs/20-training/current-v1.7.1/guides/README_V1_4_2.md`

# GameSR Training Code v1.4.2

## One-click workflow
Double-click `GameSR_Train.cmd`.

The pipeline performs:
1. Environment check
2. Dataset check/prepare
3. Benchmark read or benchmark run
4. Batch selection (Enter = fastest valid batch)
5. Training
6. Full test evaluation
7. ONNX export

## Re-export an existing checkpoint
After training, double-click `GameSR_Export.cmd`.
- no argument: Tiny-C8
- `GameSR_Export.cmd balanced-c12`
- `GameSR_Export.cmd quality-c16`

## Validate ONNX
`ONNX_Validate.cmd` or `ONNX_Validate.cmd tiny-c8`.

Keep `.onnx` and `.onnx.data` together whenever external data is present.

## Benchmark cache
Saved under `outputs/benchmarks/<GPU>/<config>.json` and reused when the benchmark signature is still valid.
Use `GameSR_Train.cmd --force-benchmark` to refresh.


---

## `docs/20-training/current-v1.7.1/guides/README_V1_5.md`

# GameSR Training Code v1.5.3

v1.5 keeps the v1.4.2 training pipeline and adds a PC-side deployment build stage for INT8/QDQ and a clean deployment bundle. The handheld benchmark remains Python/venv-free.

## Architecture

PC training machine:

`clean -> train -> test -> FP ONNX -> INT8/QDQ -> deployment bundle`

Handheld:

`deployment bundle -> standalone AMD VitisAI benchmark`

The handheld does **not** need the clean dataset, Python, `.venv`, PyTorch, or the PTQ tooling.

## One-click training

Double-click `GameSR_Train.cmd` and select Tiny-C8 / Balanced-C12 / Quality-C16 / ALL. After training and full test, the pipeline exports FP ONNX, builds INT8/QDQ from calibration data, and creates `deployment/<model>/`.

If the INT8 build is not possible because the PC environment lacks ONNX/ORT dependencies, the float model and test result are still kept; the pipeline reports the deployment build failure and does not delete the checkpoint.

## Manual PC-side INT8 build

With a float model already exported:

```cmd
Build-INT8.cmd tiny-c8 "D:\path\to\clean"
```

Or use the existing `dataset/train.csv` as calibration source:

```cmd
Build-INT8.cmd tiny-c8
```

The builder uses 128 representative images, deterministic seed 1234, QDQ format, QUInt8 activations, QInt8 weights, and per-channel weights.

## Deployment bundle

```cmd
Build-Deployment.cmd tiny-c8
```

Output:

```text
deployment\tiny-c8\
  tiny-c8-720p-to-1080p.onnx
  tiny-c8-720p-to-1080p.onnx.data   (when used)
  tiny-c8-720p-to-1080p-int8-qdq.onnx
  tiny-c8-720p-to-1080p-int8-qdq.onnx.data (when used)
  deployment_manifest.json
```

Copy only this model bundle into the standalone NPU benchmark package on the handheld. No clean data is copied.

## INT8 is deployment-time PTQ, not training-time precision

The training checkpoint remains a normal floating-point checkpoint. INT8/QDQ is produced after training from representative images. This does not require retraining unless PTQ causes unacceptable quality loss or the NPU requires a different architecture.

## Important

The INT8/QDQ output is a **candidate**. It is not assumed to be NPU-compatible. Use the standalone NPU benchmark to verify actual VitisAI coverage and latency. Only if INT8 remains at 0% NPU coverage should the architecture be redesigned around the NPU operator/layout constraints.

## What is required on the handheld

Only the standalone NPU benchmark package/runtime and the model files. The handheld does **not** need:

- `.venv`
- Python/Conda
- PyTorch
- NumPy/Pillow
- `onnxruntime` Python package
- calibration images
- `dataset\*.csv`

This separation is intentional: conversion/calibration belongs on the training PC; native NPU inference/benchmark belongs on the handheld.


---

## `docs/20-training/current-v1.7.1/guides/README_V1_6.md`

# GameSR Training Code v1.6.2

v1.6 replaces the generic ONNX Runtime PTQ path with AMD Quark's Ryzen AI
`XINT8` flow. Training remains FP32/AMP; only deployment conversion changes.

## Why v1.5.3 was rejected

The handheld profile showed all 14 `QLinearConv` nodes on CPU. The VitisAI
run used the NPU for only a tiny output subgraph and was slower than CPU.
Ryzen AI's XINT8 operator table does not support `QLinearConv`.

v1.6 therefore uses `QConfig.get_default_config("XINT8")` through AMD Quark
and refuses to publish a graph containing `QLinearConv`.

## Separate environments

- `.venv`: existing CUDA/PyTorch training environment.
- `.venv-quark`: AMD Quark ONNX quantization environment.

Run `01_Install_RyzenAI_INT8.cmd` once. It creates `.venv-quark` with PyTorch
2.11 CPU, the matching AMD Quark 0.12.post1 CPU wheel, ONNX Runtime 1.23.2,
and compatible ONNX packages.
Keeping it separate prevents Quark dependency changes from altering CUDA
training packages.

## Rebuild INT8 without retraining

If the FP32 model and calibration manifest already exist:

```cmd
Build-INT8.cmd tiny-c8
```

Or pass an image folder:

```cmd
Build-INT8.cmd tiny-c8 "D:\path\to\clean"
```

The default calibration source is `dataset\train.csv`; 128 deterministic
full-size samples are used.

## Mandatory graph gate

Before replacing the public output, the builder checks:

- valid ONNX structure;
- at least one `Conv` node;
- Q/DQ nodes exist;
- `QLinearConv` count equals zero;
- Q/DQ zero points use a valid XINT8 storage encoding: signed zero, or the
  unsigned activation centre value 128 (with zero allowed after Relu/Clip);
- output is a self-contained ONNX file;
- SHA-256 and graph report are recorded in `.meta.json`.

These checks prevent recurrence of the known v1.5.3 graph. They cannot prove
actual NPU coverage; only the VitisAI runtime profile on the handheld can do
that.

## Deployment bundle gate

`build_deployment.py` packages INT8 only when its v1.6 metadata says AMD
Quark/XINT8, the graph gate passed, and the current ONNX SHA-256 matches the
metadata. A stale or manually replaced INT8 file is excluded.

## Current architecture

The trained `lr_residual` architecture and existing checkpoints are retained.
AMD Ryzen AI 1.8 broadly lists Conv, Add, Mul, Resize, Transpose, Clip, and Q/DQ
operators for XINT8. No retraining is required merely to rebuild the old FP32
ONNX with the new quantizer.

If the new handheld profile still shows poor Conv coverage, the next step is a
compiler-specific architecture/export investigation based on that new profile,
not another blind PTQ parameter change.


---

## `docs/20-training/current-v1.7.1/guides/README_V1_7.md`

# GameSR Training Code v1.7 — Ryzen AI core-only deployment

The structural probes isolated the compiler partition failure to the complete
output topology: a learned residual branch plus two bilinear Resize paths,
base-image bypass, Add and Clip. The same 720p C8 convolution cores run as a
single NPU subgraph when those post-processing operations are absent.

## New profiles

| Profile | Deployment graph | Parameters | Purpose |
|---|---:|---:|---|
| `tiny-npu-fast` | 8 sequential Conv layers, C8 | 3,947 | first 60-FPS candidate |
| `tiny-npu-quality` | 14 sequential Conv layers, C8 | 7,451 | quality/speed comparison |

Both models are trained end-to-end against the full 1.5x image. Their PyTorch
`forward()` path still computes bilinear base plus upsampled learned residual,
so training and validation metrics remain full-image metrics.

Only ONNX deployment changes. `forward_core()` exports the low-resolution RGB
residual with shape `[1,3,720,1280]`. Magpie owns this post-process:

```text
display = clamp(bilinear(lr, 1.5) + bilinear(residual_lr, 1.5), 0, 1)
```

Sampling must match PyTorch `align_corners=False` (half-pixel coordinates).

## Graph gates

For NPU profiles, export and Quark validation reject graphs containing
`Resize`, `Add` or `Clip`. Existing XINT8 checks remain: no `QLinearConv`, valid
Q/DQ structure and zero-point encoding, self-contained ONNX, and SHA-256 bound
metadata. The deployment manifest explicitly declares `core_only`, output
meaning and the Magpie post-process contract.

## Measured evidence and limits

The second structure probe measured the 14-Conv sequential core at 20.8493 ms
(47.934 FPS) on the target handheld and runtime. This is core-only synthetic
inference, not full Magpie latency and not a hardware limit. The 8-Conv profile
is a design candidate whose actual post-quantization latency must be measured.

## Compatibility

Legacy `tiny-c8`, `balanced-c12`, `quality-c16` configs and checkpoints remain
available. They keep their complete-output export unless `--core-only` is
explicitly selected. New NPU profiles use separate output folders and cannot
silently resume a legacy checkpoint.



---
