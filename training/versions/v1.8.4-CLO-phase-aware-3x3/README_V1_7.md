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

