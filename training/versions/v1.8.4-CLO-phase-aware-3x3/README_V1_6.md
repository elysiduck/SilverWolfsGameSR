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
