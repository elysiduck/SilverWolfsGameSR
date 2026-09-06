# v1.7.1 verification record

Completed in the migration workspace:

- All five YAML configurations passed schema and value validation.
- Independent parameter-count checks passed for NPU Fast (3,947) and NPU Quality (7,451).
- A synthetic Collector V3.2 `game/session/image` dataset was prepared successfully.
- The test confirmed that no capture session appeared in more than one of train/val/test.
- Every Python source file passed bytecode compilation.
- The original direct `model.blocks` lookup and unsafe SSIM interval pattern are absent.

Hardware-dependent checks cannot run in the Linux migration workspace because it
does not contain the user's NVIDIA CUDA environment or the Windows AMD Quark
environment. `00_smoke_test.cmd` performs the CUDA forward/backward/checkpoint
test on the training PC. The normal pipeline performs ONNX and Quark graph gates
before it publishes an NPU deployment bundle.
