# Changelog

## 2026-09-07

### Training v1.8.4 / v1.8.5 / v1.8.6 / v1.8.7 — Phase-Aware 3×3 家族归档

- 新增训练代码四版本，独立归档于 `training/versions/`（构建产物按 `.gitignore` 不入仓），并打 tag `training-v1.8.4 … training-v1.8.7`；`training/current/` 现有 v1.8.2 + CLO v1.0.1 基线保持不变。
- **v1.8.4** `gamesr-local-training-v1.8.4-CLO-phase-aware-3x3-shared-env.zip`：phase-aware 3×3 decoder（九相位 27 路卷积）+ 五类新损失（Charbonnier / MS-SSIM / 梯度 / 频率 / 振铃）+ 直接 720p→1080p 重建 + 20% bilinear skip；schema v2 `direct_rgb`。SHA256 `7D377B71…CDC9E`。
- **v1.8.5** `gamesr-local-training-v1.8.5-CLO-phase3x3-residual-shared-env.zip`：九相位 decoder 改学 RGB residual、权重零初始化 ⇒ 第 0 轮逐像素=双线性；MS-SSIM 每步、SSIM/高频/振铃渐入、梯度 25%→100%、困难样本 3%~8%、退化 30%→65%；schema v3 `rgb_residual` + decoder 契约（防误续训）。SHA256 `9DD052D3…B73C26`。
- **v1.8.6** `GameSR-v1.8.6-Stable-PhaseResidual-Training-Patch.zip`：稳定训练增量补丁（未动解码器），新增 `training_contract` 契约 + 质量刹车；导出 `-phase3x3r-v3`。SHA256 `074015E7…61D7C`。
- **v1.8.7** `GameSR-v1.8.7-Dataset-Manager-Stable-Training-Full-Patch.zip`：增量补丁，新增 `prepare_dataset.py` Dataset Manager（会话原子组防泄漏切分、默认 80/10/10）+ `dataset_contract`；导出 `-phase3x3r-v4`。SHA256 `2D0FE4AB…F21ED`。
- 梳理与检查（源码审查 + 真实 torch 执行 + 纯 Python 几何复现）见 `training/versions/REVIEW-v1.8.4-v1.8.7.md`：四个版本核心改造均属实，无阻断性问题；补丁相对基线语义 / 契约 / 调度自洽。

## 2026-09-06

### Training CLO v1.0 — C/L/O 三维训练选择器

- 在 **v1.7.2 C/L 选择器**基础上增加第三维 **O = NPU 输出通道数**，形成 C / L / O 三维自定义训练选择器，取代原 `training/current/` 的 v1.8.2 覆盖式训练包。
- 入口：`training/一键训练-选择C-L-O.cmd`（中文）/ `training/Train-Choose-C-L.cmd`（英文），均调用 `pipeline.py --model custom`。
- **O 三种模式**（`training/gamesr/model.py`）：
  - **O3**：`output_channels=3`，`decoder=None` —— 保留原 RGB residual + GPU Bilinear 路径（契约 `magpie_bilinear_residual_v1`）。
  - **O4 / O6**：`output_channels=4/6`，尾部追加 `nn.Conv2d(O, 3, 1)` learned 1×1 GPU decoder（契约 `magpie_learned_1x1_decoder_v1`）；导出额外生成 `<name>.onnx.decoder.json`（含 decoder weight/bias 与合约公式），供 Magpie / HLSL。
- 自动命名隔离：`gamesr-c{C}-l{L}-o{O}`；一键 8 段流水线（环境检查 → 数据集清单 → batch 基准 → 训练 → 全量测试 → ONNX 导出 → FP16 → AMD Quark XINT8/QDQ → deployment 打包），导出用 `--use-ema`，O>3 时 `--output-channels` 透传量化。
- 已验证 O3/O4/O6 前向计算正确；完整训练与 INT8 编译待真实数据集 + 较长时间执行。说明见 `training/CLO改造说明.txt`、`training/README_C_L_SELECTOR_CN.md`、`training/训练与帧率说明.md`。

### Training CLO v1.0.1 — 批处理中文编码修复

- 修复**批处理中文编码问题**（见 `training/FIX-v1.0.1.txt`）：原中文名 `training/一键训练-选择C-L-O.cmd` 在部分系统下因 GBK/UTF-8 编码与 `%~dp0` 尾部反斜杠导致 `--root` 路径解析失败。
- 移除中文启动器，改为纯英文 `training/Train-Choose-C-L-O.cmd`，并以 `%CD%` 传参避免尾部反斜杠问题；仓库现仅保留英文启动器。
- Builder 自带 v1.3 `training/tools/int8_ptq.py`，自包含，不再依赖外部 int8_ptq 路径。
- O3/O4/O6 模式、8 段流水线、`onnx.decoder.json`、C/L/O 命名隔离与部署契约均未变。
- 新增 3 份中文安装说明：`training/安装说明-V1.8.2.txt`、`training/覆盖安装说明-V1.7.7.txt`、`training/覆盖安装说明-V1.8.1.txt`。

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
