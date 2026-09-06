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
