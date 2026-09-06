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
