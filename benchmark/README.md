# Benchmark

- `xdna-v1.3/`: current XDNA benchmark scripts and manifests; vendor runtime and probe models are external artifacts.
- `architecture-sweep/`: Builder and handheld Runner sources for the 105-architecture sweep.
- `results/architecture-sweep-20260905/`: aggregate results only.

The recorded sweep contains 105 `NPU_ONLY` results.

The ready-to-run 105-cache bundle is a Release artifact: see [`releases/benchmark-v1.3-105patched.md`](releases/benchmark-v1.3-105patched.md).

`output-channel-probe/handheld/` is the v1.0.2 handheld Runner for the C16-L4 output-width probe. Its O3/O4/O6/O8 models and compact result are Release artifacts; see [`releases/output-channel-probe-v1.0.2.md`](releases/output-channel-probe-v1.0.2.md).
