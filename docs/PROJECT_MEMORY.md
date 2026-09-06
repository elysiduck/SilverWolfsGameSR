# PROJECT MEMORY

更新时间：2026-09-06

## 当前 Magpie 主线

- 版本：v4.1.1 AsyncNPU CurrentInput
- 源码补丁：`magpie/patches/pipeline-v4.1.1-async-npu-current-input/`
- Runtime build ID：`magpie-v2.1-pipeline-v4.1.1-async-npu-current-input | contractv3 | canonical-current`
- Build version：`GameSR-Magpie-v4.1.1-AsyncNPU-CurrentInput`
- 完整源码 ZIP SHA256：`e45eec28c582629c3df8bb9a1564a43ca620f73921c194b356d7e1195a4b0bb2`

## 部署链结论

Current Input 模式让 NPU 接收游戏当前捕获分辨率。若游戏捕获为 1280×720，则模型输入为 1280×720，不再强制经过 960×540 中间输入。训练/部署倍率应以运行日志和实际模型契约继续校验。

## 延迟链保留优化

- NPU worker 与 render thread 解耦
- 三槽异步状态机
- latest-frame-wins，避免陈旧帧排队
- persistent CPU / residual buffers
- worker 侧 NCHW residual → RGBA interleave
- zero-residual + bilinear base 首帧回退
- Pipeline v3 fixed3 legacy 路径兼容

## 构建修复

`GameSR-Models/README_CN.txt` 已补齐；`Magpie.vcxproj` 会复制 README、`*.onnx` 和 `*.onnx.data`，解决 MSB3030 缺文件错误。

## GitHub 同步规则

每一版 GameSR 修改完成后同步到 `elysiduck/SilverWolfsGameSR`，并在交付时明确报告仓库、分支、commit SHA 和 tag（未打 tag 也要说明）。大型训练集、完整上游依赖树、模型与 checkpoint 不直接写入 Git 历史。
