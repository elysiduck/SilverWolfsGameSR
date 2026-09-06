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
