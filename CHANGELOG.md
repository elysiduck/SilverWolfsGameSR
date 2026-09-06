# Changelog

## 2026-09-06

### Magpie v4.1.1 AsyncNPU CurrentInput

- 在 Pipeline v4.1 异步 NPU 基础上加入 Current Input 路径。
- NPU 输入改为当前游戏捕获分辨率；720p 捕获时直接进行 720p → 1080p，不再强制预缩放为 960×540。
- 保留三槽异步状态机、latest-frame-wins、persistent buffers、worker 侧 residual interleave 和启动 zero-residual fallback。
- 保留 Pipeline v3 fixed3 的 legacy tiled SESR 路径与 GameSR contract v3 / NativePresetModels v2.2 兼容性。
- 新增 `SESR-M7 NPU Current Input` 效果、模式选择及相关 UI/配置字段。
- 补齐 `GameSR-Models/README_CN.txt`，修复 `PrepareGameSrModelLibrary` 的 MSB3030 构建失败。
- 增强 `Magpie.vcxproj`：发布时复制 `GameSR-Models/*.onnx` 与 `*.onnx.data`。
- Runtime build ID：`magpie-v2.1-pipeline-v4.1.1-async-npu-current-input | contractv3 | canonical-current`。
- 完整补丁源码位于 `magpie/patches/pipeline-v4.1.1-async-npu-current-input/`。

### Magpie v2.1 / Pipeline v4 Async NPU

- 将 GameSR full-frame residual 的 CPU NCHW pack 与 ORT/VitisAI NPU 推理移出 Magpie render thread。
- 增加三槽异步状态机：Free → GpuPending → CpuQueued → WorkerRunning → ResultReady。
- 增加 latest-frame-wins 调度，避免 NPU 排队造成输入延迟累积。
- 使用 persistent CPU / residual buffers，减少逐帧大块内存分配。
- 将 NCHW residual → RGBA interleave 移至 worker。
- 启动阶段使用 zero residual + bilinear base，避免首帧黑屏。
- 新增 `GameSR PIPELINE v4 ASYNC` 与 `GameSR BACKEND v4` 性能日志。
