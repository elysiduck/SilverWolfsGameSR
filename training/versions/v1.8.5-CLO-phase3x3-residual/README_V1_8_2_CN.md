# GameSR v1.8.2：Late Polish / Information-Aware Training

这是覆盖式训练补丁，直接覆盖到 `gamesr-local-training` 根目录即可。

## 这版解决什么

你当前 c32-l3 在 epoch 45~54 已进入明显平台：5 轮 PSNR 只增加约 0.03~0.04 dB，
而 `edge_ratio` 已到约 1.05。v1.8.2 不推翻 v1.8.1，而是在训练后半程自动从“抢细节”切换到“收敛/抛光”。

### 1. Late-Polish

大约 50% 训练进度开始，75% 左右完全进入 polish：

- HF：从当前高位逐渐降到 `0.012`
- Hard top：逐渐降到 `15%`
- Hard mix：逐渐降到 `9%`
- Edge：只轻微回落到约 `0.0675`，因为 Edge Loss 本身是 pred/GT 梯度差，不是单纯锐化
- SSIM：保持约 `0.030`
- L1：保持主导

如果你现在在 epoch 54/100，覆盖后从 epoch 55 Auto Resume，会直接进入 `phase=polish`，但变化是平滑的，不会突然扭转权重。

### 2. Information-Aware L1

只在 polish 阶段逐渐启用。所有 patch 仍然训练，但：

- 低信息 patch：略低权重（目标约 0.80）
- 高信息 UI / 文字 / 发丝 / 几何：略高权重（目标约 1.15）
- 权重每个 batch 自动归一化，避免总体梯度突然变大
- 最终权重有 0.75~1.25 的保守限制

### 3. Black / Flat Patch Suppression

接近全黑、低方差、低细节的 patch（例如纯电影黑边）会再略降权。
黑底白字/字幕不会被简单当成黑边，因为文字会提高方差和 detail score。

### 4. 新 Validation 指标

日志新增：

- `detail=xx.xx dB`：高细节区域 PSNR
- `dDelta=+x.xx dB`：高细节区域相对 Bilinear 的提升
- `smooth=xx.xx dB`：平滑区域 PSNR，观察是否产生假纹理
- `over=x.xxx`：预测边缘超过 GT 的正向超额比例
- `undersharp_deficit_ratio`：写入 CSV / JSON / TensorBoard
- `detail_edge_error`：写入 CSV / JSON / TensorBoard

训练日志的 L1 会区分：

- `RawL1`：原始均匀 L1，方便和 v1.8.1 对比
- `InfoL1`：实际用于 polish 的信息加权基础 L1
- `HardL1`：Hard Mining 选中的难样本原始 L1

## 安装

1. 停止当前训练。
2. 解压本补丁。
3. 将补丁目录中的内容覆盖到 `gamesr-local-training` 根目录。
4. 再运行原来的 `Train-Choose-C-L.cmd`。
5. 选择与原模型完全相同的 C/L（例如 `C32 / L3`）。
6. Auto Resume 继续 `latest.pth`。

不需要从 epoch 1 重训；模型结构、ONNX 拓扑、参数量和 NPU 推理路径均不变。

## 预期日志

从 epoch 55 附近会看到：

```text
[V1.8.2] phase=polish ... polish=...% info=...%
[V1.8.2 LOSS] RawL1=... InfoL1=... HardL1=... ... flat_black=...%
[QUALITY] ... detail=...dB dDelta=+...dB smooth=...dB over=...
```

## 兼容说明

- checkpoint signature 仍只取 `version/model/scale`，因此同 C/L 的 v1.8.1 checkpoint 可直接续训。
- 本包包含 `gamesr/streaming.py`，不会再次出现 `PersistentValidationLoader` 漏文件的问题。
- 本包也包含 v1.7.5.2 的 `gamesr/utils.py` CSV schema 迁移代码，新指标可以安全追加到旧 `training_log.csv`。
