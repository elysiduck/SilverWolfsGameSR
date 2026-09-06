# GameSR Training Code v1.7.1 — reliability fix

This release keeps the v1.7 model architecture and Ryzen AI core-only design.
It fixes and hardens the workflow before the first real `tiny-npu-fast` run.

## Fixed

- NPU YAML profiles build without an unused `model.blocks` key.
- The smoke test loads both real NPU YAML files and checks parameter counts.
- Auto-resume no longer loads an incompatible checkpoint after skipping it.
- Checkpoints are written atomically to protect `latest.pth` during interruption.
- Invalid batch sizes and `ssim_every_n_steps: 0` are handled safely.
- Dataset manifests verify referenced image files before a long run.
- Collector V3.2 `game/session/image` folders split by capture session, reducing adjacent-frame leakage between train, validation and test.
- ONNX export refuses a checkpoint from another model/version configuration.
- INT8 export verifies static `[1,3,720,1280]` input and removes stale sidecars.
- NPU packaging fails closed unless a hash-bound Quark XINT8 core is valid and contains no Resize/Add/Clip.

## Unchanged

- `tiny-npu-fast`: 8 sequential Conv layers, C8, 3,947 parameters.
- `tiny-npu-quality`: 14 sequential Conv layers, C8, 7,451 parameters.
- Deployment output is a low-resolution RGB residual.
- Magpie must use half-pixel sampling for bilinear base plus bilinear residual.

Run `00_smoke_test.cmd` first. Train `tiny-npu-fast` before the quality profile.
