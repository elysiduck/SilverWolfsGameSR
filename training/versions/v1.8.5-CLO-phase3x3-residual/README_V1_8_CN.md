# GameSR v1.8 Training Algorithm Upgrade

这是基于 GameSR v1.7.6 高速训练底座，并合并 v1.7.7 Quality Diagnostics 的训练算法升级补丁。

## 目标

**不增加最终 ONNX / NPU 部署模型的层数和参数量，真正改变训练行为，让同一个小模型把容量更多用在困难区域、边缘和高频细节上。**

本版不是“只加日志”。训练目标与样本权重会实际改变，最终得到的权重也会不同。

---

## v1.8 的四项核心升级

### 1. GPU Online Hard Patch Mining（在线困难样本挖掘）

旧版每个随机 patch 在 L1 中基本等权。

v1.8 每个 batch 会先计算每张 patch 当前的重建误差，然后：

- 训练初期：重点关注误差最大的约 **15%** patch
- 后期逐步提高到约 **40%**
- 困难 patch 的额外混合权重从约 **10%** 提升到 **35%**

也就是说模型会逐渐变成：

> 哪里目前学得最差，就把更多梯度给哪里。

这里特意采用 **GPU 在线排序**，而不是在 CPU 解码阶段对一张源图反复裁 4~8 个候选区域。原因是你当前日志已经显示 DataLoader/Streaming wait 很高，CPU 端 Hard Crop 会进一步拖慢训练。

### 2. Loss Curriculum（分阶段损失）

v1.8 默认按总 epoch 百分比自动切换重点：

- **Foundation：0%~6%**
  - L1 仍为主
  - Edge = 0.05
  - SSIM = 0.02
  - High Frequency = 0
- **Detail：6%~30%**
  - Edge / SSIM / High Frequency 快速爬升
  - Hard Mining 权重同步提高
- **Refine：30%~100%**
  - Edge 最高约 0.09
  - SSIM 最高约 0.035
  - High Frequency 最终约 0.06
  - Hard Mining 最终 top 40%，mix 35%

这套阶段比旧版更早进入细节训练，是因为 GameSR 小模型在现有数据上 10~15 epoch 就已经能很快得到高 PSNR。

### 3. Lightweight High-Frequency Loss

新增轻量高频损失：

- 先转亮度通道
- 用 3x3 平滑得到低频
- 原图减低频得到高频残差
- 比较预测与 GT 的高频残差

重点约束：

- UI 小字和细线
- 草地 / 树叶
- 发丝
- 栅栏
- 装备纹理
- 细小边缘

为了不破坏当前训练吞吐量，它**不用 FFT、不引入 VGG/LPIPS、无新增 Python 依赖**，并默认每 2 step 计算一次。

这个 loss 只存在于训练阶段，**最终 ONNX/NPU 模型完全不增加算子。**

### 4. Degradation Curriculum

不推翻已有 degradation，只控制强度：

- 开始约 90%
- 到训练进度约 25% 时达到完整 100%

这样既保留原来已经验证过的 blur / sharpen / resize / brightness / contrast / quantization / noise 体系，又避免模型最初阶段同时承担所有最难扰动。

对于从 v1.7.x 中途续训的 checkpoint，这个 90% 的高起点也能避免突然出现很大的训练分布变化。

---

## 如果你现在是 epoch 13/100

按默认 v1.8 曲线，续训后约为：

```text
phase      = detail
Edge       ≈ 0.0565
SSIM       ≈ 0.0224
HF         ≈ 0.0083
hard_top   ≈ 18%
hard_mix   ≈ 13%
degradation≈ 95%
```

随后到 20~30 epoch，细节约束会明显增强，而不是像旧版那样一直使用同一套固定 loss。

---

## v1.7.7 Quality Diagnostics 已合并

v1.8 同时保留/包含：

- Bilinear baseline
- Delta PSNR
- MSE reduction
- PSNR P10 / min / std
- Edge error / edge strength ratio
- Quality trend
- `quality_summary.json`

另外新增训练日志字段：

```text
train_l1
train_hard_l1
train_edge
train_ssim_loss
train_high_frequency_loss
v18_phase
v18_edge_weight
v18_ssim_weight
v18_high_frequency_weight
v18_hard_fraction
v18_hard_mix
v18_degradation_strength
```

控制台会看到：

```text
[V1.8] phase=detail ...
[V1.8 LOSS] L1=... HardL1=... Edge=... SSIMLoss=... HFLoss=...
[QUALITY] ...
[TREND] ...
```

---

## 兼容性 / 断点续训

v1.8 **没有改变模型 topology**。

`pipeline.py` 会把：

```yaml
trainer_version: 1.8.0
```

写入新配置，但为了兼容现有同结构 checkpoint，保留：

```yaml
version: 1.7.4
```

GameSR 当前 checkpoint compatibility signature 只检查：

- `version`
- `model`
- `scale`

因此只要你继续选择**相同的 C / L 和 scale**，原来的 `latest.pth` 可以 Auto Resume。

> 不要用不同 C/L 的 checkpoint 强行续训。

---

## 安装

1. 先停止正在运行的训练。
2. 解压本补丁。
3. 将补丁内文件复制到现有 `gamesr-local-training` 根目录并允许覆盖。
4. 继续运行：

```text
Train-Choose-C-L.cmd
```

5. 选择与原训练相同的 C / L / profile，pipeline 会继续使用原 output 目录并自动续训。

本补丁会覆盖/新增：

```text
train.py
pipeline.py
Train-Choose-C-L.cmd
gamesr/data.py
gamesr/metrics.py
```

不会覆盖模型结构文件 `gamesr/model.py`。

---

## 回滚 / 关闭 v1.8 算法

如果只想临时关闭算法，不必删 checkpoint。在生成的 YAML 中设置：

```yaml
train:
  algorithm_v18:
    enabled: false
```

此时会回到旧版固定 L1 + Edge + SSIM 训练行为（High Frequency 与 Hard Mining 关闭，degradation 使用完整强度）。

---

## 关于训练速度

v1.8 会增加少量 GPU 训练开销：

- 每 batch 一次很小的 per-image L1 + `topk`
- 每 2 step 一次亮度高频 loss

但它**没有在 CPU 解码阶段增加候选 patch 搜索**，这是为了保护你当前 RAM Streaming / Prefetch 的吞吐量。

最终实际 img/s 仍应以你的机器日志为准。

---

## 判断 v1.8 是否真的更好

不要只看单个 PSNR 数字。建议至少比较同一 test set 的：

1. PSNR / SSIM
2. Delta PSNR vs Bilinear
3. PSNR P10 / min（困难样本）
4. Edge error / edge strength ratio
5. 游戏画面实际截图：文字、草地、发丝、栅栏、UI

v1.8 的设计目标不是保证 PSNR 一定暴涨，而是**减少“平均 PSNR 已很高，但困难纹理仍然糊”的情况。**
