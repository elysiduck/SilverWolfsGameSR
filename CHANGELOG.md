# Changelog

## 2026-09-06

### Magpie v2.1 / Pipeline v4 Async NPU

- 将 GameSR full-frame residual 的 CPU NCHW pack 与 ORT/VitisAI NPU 推理移出 Magpie render thread。
- 增加三槽异步状态机：Free → GpuPending → CpuQueued → WorkerRunning → ResultReady。
- 增加 latest-frame-wins 调度，避免 NPU 排队造成输入延迟累积。
- 使用 persistent CPU / residual buffers，减少逐帧大块内存分配。
- 将 NCHW residual → RGBA interleave 移至 worker。
- 启动阶段使用 zero residual + bilinear base，避免首帧黑屏。
- 新增 `GameSR PIPELINE v4 ASYNC` 与 `GameSR BACKEND v4` 性能日志。
- 保留 GameSR contract v3 / NativePresetModels v2.2 兼容性。

目标：在 C16L4 720p→1080p 场景中，将旧 Pipeline v3 的约 50–53 FPS 推向 60+ FPS，并继续定位 render/present 路径剩余阻塞。

