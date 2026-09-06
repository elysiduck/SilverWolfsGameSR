# GameSR Phase-Aware 3×3 训练路径

## 如何使用

双击 `Train-Choose-C-L-O.cmd`，选择 `O4`、`O6` 或 `O8` 即可启用新路径。

- `O3`：保留旧的双线性 RGB residual，兼容既有模型。
- `O4/O6/O8`：使用 `phase_aware_3x3_v1`，模型名自动增加 `-phase3x3`，不会续训旧的 1×1 decoder 检查点。
- 第一轮推荐：`C16 / L4 / O4`。

## 新 decoder

NPU 仍只导出 720p latent。GPU decoder 针对 1.5 倍缩放的 3×3 个亚像素相位，分别学习一套 3×3 空间卷积核，直接重建 1080p RGB。边界规则为 replicate，默认保留 0.20 的双线性 base skip 以稳定训练。

导出的 `.onnx.decoder.json` 包含 decoder 类型、相位数、kernel、3/2 缩放、anchor 规则、权重、bias、base skip 和边界规则。

Magpie 必须实现此契约后才能显示新模型；旧版 Magpie 只能使用旧 O3/1×1 模型。

## 新损失函数

- 1.00 × Charbonnier 重建
- 0.10 × 多尺度 SSIM
- 0.05 × 梯度
- 0.02 × 细节感知高频
- 0.01 × 振铃/光晕惩罚

权重位于生成配置的 `train.phase_aware_loss`，可在训练前调整。

## 验证顺序

先短训 5～10 轮检查预览、尺寸和收敛，再进行完整训练。最终分别比较 Bilinear、O3、phase-aware FP32 和 phase-aware INT8；如果 FP32 正常但 INT8 明显退化，应进行量化感知微调，而不是继续增加 O。
