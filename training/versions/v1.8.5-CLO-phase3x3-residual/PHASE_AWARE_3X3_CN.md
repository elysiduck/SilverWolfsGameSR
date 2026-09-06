# GameSR Phase-Aware 3×3 训练路径

## 如何使用

双击 `Train-Choose-C-L-O.cmd`，选择 `O4`、`O6` 或 `O8` 即可启用新路径。

- `O3`：保留旧的双线性 RGB residual，兼容既有模型。
- `O4/O6/O8`：使用 `phase_aware_3x3_v1`，模型名自动增加 `-phase3x3r-v2`，不会续训旧的 direct-RGB 或 1×1 decoder 检查点。
- 第一轮推荐：`C16 / L4 / O4`。

## 新 decoder

NPU 仍只导出 720p latent。GPU decoder 针对 1.5 倍缩放的 3×3 个亚像素相位，分别学习一套 3×3 空间卷积核。最终输出为完整双线性基线加 learned RGB residual。卷积核和 bias 均从零开始，因此第 0 轮严格等于双线性，不再浪费小模型容量重建低频颜色。边界规则为 replicate，base skip 固定为 1.0。

导出的 `.onnx.decoder.json` 包含 decoder 类型、相位数、kernel、3/2 缩放、anchor 规则、权重、bias、base skip 和边界规则。

Magpie 必须实现此契约后才能显示新模型；旧版 Magpie 只能使用旧 O3/1×1 模型。

## 新损失函数

- 1.00 × Charbonnier 重建
- 最终 0.10 × 多尺度 SSIM（每一步计算，渐入）
- 最终 0.05 × 梯度（从 25% 权重渐入）
- 最终 0.02 × 细节感知高频（渐入）
- 最终 0.01 × 振铃/光晕惩罚（渐入）

困难样本只占少量混合权重；退化强度从 30% 开始，到训练进度 65% 才达到完整强度。验收首先看 `delta PSNR`，未达到 0 dB 前不应进入 INT8 或 Magpie 集成。

权重位于生成配置的 `train.phase_aware_loss`，可在训练前调整。

## 验证顺序

先短训 5～10 轮检查预览、尺寸和收敛，再进行完整训练。最终分别比较 Bilinear、O3、phase-aware FP32 和 phase-aware INT8；如果 FP32 正常但 INT8 明显退化，应进行量化感知微调，而不是继续增加 O。
