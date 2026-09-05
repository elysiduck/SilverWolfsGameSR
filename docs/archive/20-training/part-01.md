- Preserves the v1.5 PC-side INT8/QDQ workflow and handheld-independent deployment design.

```

---

## `docs/20-training/current-v1.7.1/history/RELEASE_NOTES_V1_6.txt`

```text
GameSR Training Code v1.6
========================

- Replaced generic ONNX Runtime PTQ with AMD Quark XINT8.
- Added isolated .venv-quark installation; CUDA training .venv is untouched.
- Added Ryzen AI graph gate that rejects QLinearConv and asymmetric zero points.
- Added deterministic calibration progress and robust CSV-relative paths.
- Added atomic model publication and self-contained ONNX output.
- Added SHA-256/graph metadata for every accepted INT8 model.
- Deployment now excludes stale or unvalidated INT8 files.
- Existing FP32 ONNX/checkpoints remain usable; retraining is not required.

```

---

## `docs/20-training/current-v1.7.1/history/RELEASE_NOTES_V1_6_1.txt`

```text
GameSR Training Code v1.6.1
==========================

- Fixed `No module named 'torch'` during AMD Quark verification.
- Installs the exact PyTorch 2.11 CPU runtime required by the selected
  `amd-quark-0.12.post1+cpu.torch2.11` wheel.
- Verification now reports the PyTorch version and rejects mismatched builds.
- Existing `.venv-quark` environments can be repaired in place.

```

---

## `docs/20-training/current-v1.7.1/history/RELEASE_NOTES_V1_6_2.txt`

```text
GameSR Training Code v1.6.2
==========================

- Fixed a false rejection after successful AMD Quark XINT8 calibration.
- Quark can store symmetric signed activations as UINT8 with centre value 128.
- Validation now checks zero-point values according to their storage dtype.
- Signed INT8/INT16/INT32 zero points must remain 0.
- UINT8 activation zero points may be 128, or 0 after Relu/Clip constraints.

```

---

## `docs/20-training/current-v1.7.1/history/RELEASE_NOTES_V1_7.txt`

```text
GameSR Training Code v1.7
========================

- Added tiny-npu-fast: C8, 8 sequential Conv layers.
- Added tiny-npu-quality: C8, 14 sequential Conv layers.
- Training/validation still optimize the complete 720p-to-1080p image.
- Ryzen AI ONNX now exports only the 720p RGB residual core.
- Resize, base bypass, Add and Clip move to the future Magpie shader.
- Export and INT8 validation reject those partition-breaking operators.
- Deployment manifest schema v3 records the core-only shader contract.
- Added beginner one-click training and re-export entry points.
- Kept all v1.6.2 AMD Quark XINT8 safety checks and legacy configs.

```

---

## `docs/20-training/current-v1.7.1/history/RELEASE_NOTES_V1_7_1.txt`

```text
GameSR training code v1.7.1

Reliability fix for the first real NPU Fast/Quality training run.
No model architecture or deployment contract was changed.

See README_V1_7_1.md for the complete fix list.

```

---

## `docs/20-training/current-v1.7.1/history/UPGRADE_FROM_V1_1.txt`

```text
GameSR training code v1.2 upgrade notes

1. Keep using GameSR-local-training-env-v1.1. No new Python package is required.
2. Stop any old v1.1 training process before replacing the code files.
3. Copy all files from this v1.2 package into the same folder that contains .venv.
4. Existing dataset/train.csv, dataset/val.csv and dataset/test.csv can be reused.
5. Existing v1.1 checkpoints are NOT recommended for automatic resume because v1.2 changes the optimizer/scheduler/loss and adds EMA. Start v1.2 training from epoch 1 for clean comparisons.
6. Existing v1.1 images in clean can be reused without conversion; v1.2 creates LR images online during training.
7. The default v1.2 configs use larger batch sizes tuned for a 12 GB-class GPU, but GameSR_Train.cmd runs an automatic batch benchmark first and passes the fastest usable batch size to train.py.
8. To rebuild manifests after adding more clean screenshots, run:
   GameSR_Train.cmd --model 1 --source "D:\path\to\clean"
   This explicitly rebuilds dataset manifests.
9. To use existing manifests without rebuilding them, just run:
   GameSR_Train.cmd --model 1
10. For all models:
   GameSR_Train.cmd --model all
11. To skip batch benchmarking:
   GameSR_Train.cmd --model 1 --no-auto-batch

```

---

## `docs/20-training/current-v1.7.1/history/UPGRADE_FROM_V1_2.txt`

```text
GameSR Training Code v1.4 upgrade notes

1. Training degradation moved from CPU/PIL into the GPU training loop.
2. Benchmark now exercises the v1.4 GPU-side degradation path and saves persistent JSON results.
3. GameSR_Train.cmd reads saved benchmark files, shows every measured batch, asks for a batch, and uses the fastest batch on Enter.
4. Benchmark files are separated by GPU name and model config.
5. Training metrics include loader wait, H2D+degradation, compute stage, images/s, peak VRAM and selected batch.
6. JPEG CPU roundtrip is disabled by default; quantization provides a lightweight compression-like perturbation.
7. Horizontal flipping remains disabled for full game screenshots.
8. SSIM is computed less often by default to reduce overhead on tiny models.
9. Existing v1.2 checkpoints are not recommended for formal v1.4 runs; start a fresh v1.4 run.

```

---

## `docs/20-training/current-v1.7.1/history/UPGRADE_FROM_V1_3.txt`

```text
GameSR Training Code v1.4

Major changes:
1. Benchmark_Training.cmd can now be launched directly without arguments.
2. If dataset/train.csv, val.csv, or test.csv is missing, the benchmark pipeline asks for the clean screenshot folder and prepares the manifests automatically.
3. Benchmark cache is validated with a signature that includes model/config, data split fingerprint, degradation, loss settings, benchmark steps/warmup, and runtime-related settings.
4. Failed benchmark runs are never treated as valid cache and never allow training to start.
5. GameSR_Train.cmd automatically reads valid cached benchmark results, displays every tested batch, and lets the user choose a batch. Enter accepts the fastest valid batch.
6. Each model has a separate cache: tiny-c8.json, balanced-c12.json, quality-c16.json, under the detected GPU name.
7. --force-benchmark forces a fresh benchmark.
8. Benchmark now follows the real training loss path: L1 + edge + periodic SSIM, with GPU-side game degradation.
9. Training configs keep game-oriented degradation, EMA, AMP, TF32, channels-last, pinned memory, persistent workers, and prefetching.
10. Training and benchmark failures are fail-fast; an invalid benchmark cannot silently produce a recommended batch.
11. Full evaluation remains PSNR + SSIM on the full test set and ONNX export remains fixed 720p -> 1080p.

Backward compatibility:
- Existing dataset CSVs can be reused.
- Existing v1.2/v1.4 checkpoints can still be loaded by train.py where architecture/config is compatible, but v1.4 changes in degradation/loss mean fresh v1.4 runs are recommended for formal comparison.

Additional v1.4 architecture change:
- New default `lr_residual` architecture performs the learned feature stack at LR resolution and upsamples only the 3-channel residual before adding it to the bilinear base. This is substantially cheaper than the former HR-resolution residual stack and is intended to improve training/inference efficiency while preserving the NPU-friendly residual design.

```

---

## `docs/20-training/current-v1.7.1/history/UPGRADE_FROM_V1_5_3.txt`

```text
Upgrade from v1.5.3
===================

1. Copy all v1.6 files over the existing GameSR project and choose Replace.
2. Keep the existing .venv, dataset, models, outputs, and checkpoints.
3. Run 01_Install_RyzenAI_INT8.cmd once.
4. Run Build-INT8.cmd.
5. Do not reuse the v1.5.3 INT8 model; use the newly written same-name ONNX.

No retraining is required when models\<name>-720p-to-1080p.onnx already exists.

```

---

## `docs/20-training/current-v1.7.1/deployment/DEPLOYMENT_WORKFLOW.md`

# GameSR deployment workflow (legacy complete-output profiles)

## PC (training machine)

1. Train and evaluate.
2. Export FP32 ONNX.
3. Run `01_Install_RyzenAI_INT8.cmd` once to create the isolated AMD Quark environment.
4. Optionally build FP16 with `Build-FP16.cmd`.
5. Build Ryzen AI XINT8/QDQ with `Build-INT8.cmd`, or let `GameSR_Train.cmd` do it automatically.
6. `Build-Deployment.cmd` packages only an AMD Quark/XINT8 model that passes graph and hash validation.

## Handheld (NPU machine)

Do not install Python, PyTorch, the GameSR training `.venv`, NumPy, Pillow, ONNX Runtime Python, or calibration data just to benchmark the model.

Copy the model files from `deployment/<model>/` into the standalone NPU benchmark's `models\` folder. Keep `.onnx.data` beside its corresponding `.onnx` whenever present.

The benchmark should invoke its own packaged native ONNX Runtime/VitisAI runtime. It should not call `python` or `.venv`.

## Model variants

- FP32: reference quality/debug model.
- FP16: optional GPU/deployment candidate.
- INT8 QDQ: AMD Quark XINT8 NPU candidate. It is accepted locally only when `QLinearConv=0`, but remains a candidate until VitisAI coverage and latency are measured.

Calibration is a PC-only build-time concern. The handheld never needs the calibration images.
# v1.7 notice

For `tiny-npu-fast` and `tiny-npu-quality`, the exported ONNX is a core-only
720p RGB residual. Follow `README_V1_7.md`; any complete-output instructions
below apply only to legacy profiles.

---

## `docs/20-training/current-v1.7.1/verification/VERIFICATION_V1_7_1.md`

# v1.7.1 verification record

Completed in the migration workspace:

- All five YAML configurations passed schema and value validation.
- Independent parameter-count checks passed for NPU Fast (3,947) and NPU Quality (7,451).
- A synthetic Collector V3.2 `game/session/image` dataset was prepared successfully.
- The test confirmed that no capture session appeared in more than one of train/val/test.
- Every Python source file passed bytecode compilation.
- The original direct `model.blocks` lookup and unsafe SSIM interval pattern are absent.

Hardware-dependent checks cannot run in the Linux migration workspace because it
