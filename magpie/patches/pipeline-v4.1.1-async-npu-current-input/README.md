# Magpie Pipeline v4.1.1 AsyncNPU CurrentInput

本目录保存本次 GameSR 对 Magpie 的完整补丁文件，不包含未修改的上游 Magpie 源码与第三方依赖。

## 核心变化

- NPU 输入使用当前捕获分辨率。
- 1280×720 捕获直接送入模型，不再固定 GPU 预缩放为 960×540。
- 输出仍由模型契约与目标缩放链决定，典型用途为 720p → 1080p。
- 保留 Pipeline v4.1 的异步 worker、三槽状态机、latest-frame-wins 和持久缓冲。
- 保留 Pipeline v3 fixed3 legacy tiled SESR 路径。
- 补齐 `GameSR-Models/README_CN.txt` 并让工程复制 `*.onnx` / `*.onnx.data`。

## 应用方式

将本目录内文件按相对路径覆盖到对应的 Magpie v2.1 源码树，再执行 `Build-SESR.ps1`。模型文件不提交到 Git；请按 `GameSR-Models/README_CN.txt` 放置。

## 标识

- Build version: `GameSR-Magpie-v4.1.1-AsyncNPU-CurrentInput`
- Runtime build ID: `magpie-v2.1-pipeline-v4.1.1-async-npu-current-input | contractv3 | canonical-current`
- Full source ZIP SHA256: `e45eec28c582629c3df8bb9a1564a43ca620f73921c194b356d7e1195a4b0bb2`
