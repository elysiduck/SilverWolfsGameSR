# Output-channel probe v1.0.2

This experiment holds the C16-L4 convolution body fixed and varies only the final output width: O3, O4, O6 and O8. It measures core ONNX/VitisAI runtime cost; it does not measure Magpie transfer, output staging, GPU decoding or image quality.

| Output | Average NPU latency | Delta vs O3 | Output FP32 size |
| --- | ---: | ---: | ---: |
| O3 | 7.0105 ms | baseline | 10.547 MiB |
| O4 | 6.7868 ms | -3.192% | 14.063 MiB |
| O6 | 7.0749 ms | +0.919% | 21.094 MiB |
| O8 | 7.6792 ms | +9.537% | 28.125 MiB |

All four observations were `NPU_ONLY`. The complete compact result is recorded under `benchmark/results/output-channel-probe-20260906/` and published with the release assets.

Use `handheld/` only after copying its files into a complete Benchmark v1.3 root with the runtime in place. Use the training-side generator in `training/current/` to create new payloads.
