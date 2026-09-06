# SESR-M7 NPU v2.2 当前分辨率输入模式

## 这版解决什么

新增缩放模式：`SESR-M7 NPU 当前分辨率输入`。

它不会执行 `Exact 1080p` 的 GPU 预缩放。游戏捕获为 1280×720 时，链路是：

```text
游戏捕获 1280×720
  -> NPU 直接输入 1280×720
  -> SESR-M7 固定 ×2 输出 2560×1440
  -> Magpie 适配到 1920×1080 屏幕
```

原来的 `SESR-M7 NPU Exact 1080p` 保留不变，可用于 A/B 对照：

```text
游戏捕获 1280×720
  -> GPU 预缩放 960×540
  -> NPU ×2 输出 1920×1080
```

## 怎么使用

1. 双击 `构建-SESR.cmd` 编译。
2. 启动 `publish\x64\Magpie.exe`。
3. 在该预设的 `GameSR 模型` 下拉菜单中选择你的 C16L4 ONNX；它应位于
   `publish\\x64\\models\\GameSR\\`。
4. 选择 `SESR-M7 NPU Exact 1080p`（推荐）或 `SESR-M7 NPU 当前分辨率输入`（显式对照）。
5. 将游戏设为 1280×720 窗口或无边框窗口并启动缩放。
6. 打开日志，确认出现：

```text
GameSR BUILD ID: magpie-v2.1-pipeline-v4.1.1-async-npu-current-input | contractv3 | canonical-current
GameSR 预设: 模型 ... 按模型库合约请求 Exact 输入 1280x720
GameSR ONNX 阶段 4/4: ... PIPELINE v4 异步 worker 已启动
```

若已经选中 GameSR 模型却仍出现 `GPU 预缩放 1280x720 -> 960x540`，说明没有实际
加载到 GameSR 预设；先检查 ONNX 是否放在 `models\\GameSR`，并重新选择该模型。

## 必须知道的限制

若没有选择 GameSR 预设模型，内置 SESR-M7 仍是固定 ×2：720p 直接输入会先生成
1440p，再由 Magpie 缩到 1080p。

若选择 `models\\GameSR` 中的 C16L4 等 GameSR 预设，v4.1 会使用原生的
`720p NPU residual -> GPU 1.5x Resize/Add/Clip -> 1080p` 链路；不再将 GameSR
输入缩为 540p。此时使用 `SESR-M7 NPU Exact 1080p` 也会自动保留模型所需的 1280×720
输入，当前分辨率输入模式仅用于显式诊断和对照。

## 合并的性能补丁

本包合并 `Pipeline v3 fixed3` 和当前标准 `Pipeline v4.1 Async NPU`。对 GameSR
full-frame residual 模型，v4.1 使用三槽 latest-frame-wins 异步 worker，使 NPU Run
不再直接阻塞 Magpie 渲染线程；legacy tiled SESR 路径保留 v3 行为。
