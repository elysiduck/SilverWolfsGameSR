# GameSR v1.8.1 — Game-Aware / Detail-Aware Training

这一版是在 v1.8 Training Algorithm Upgrade 上，根据实际 GameSR 游戏截图进一步收敛训练策略。

目标不是增加部署模型大小，而是避免 tiny SR 模型把有限容量浪费在天然模糊、Bloom、DOF、电影滤镜或少数极端 VFX 帧上，同时仍然重点学习 UI、文字、人物轮廓、发丝、建筑线条、草叶等真正影响游戏超分观感的内容。

## 核心变化

### 1. Detail-aware Hard Mining

v1.8：Hard sample 只按当前 L1 reconstruction error 排名。

v1.8.1：改为近似：

`hard_score = reconstruction_error × target_detail_factor`

其中 detail factor 来自 HR target 自身的亮度梯度密度。

因此：
- UI / 字体 / 细线 / 发丝 / 建筑几何：detail 高，真正难时更容易被重点训练。
- DOF / Bloom / 梦境柔焦 / 大面积平坦区域：detail 低，不会仅因为误差偏高就长期霸占 Hard Mining。
- 所有样本仍保留基础 L1 监督，没有任何样本被丢弃。

默认：
- hard fraction：10% -> 25%
- requested hard mix：6% -> 20%
- selected-sample gradient cap：1.50x
- detail ranking floor：0.45

### 2. Hard gradient cap

旧的 top-k mixed loss 在 hard fraction 很小时，单个被选样本可能获得过大的相对梯度。

v1.8.1 根据 batch size / selected K 自动限制 effective hard mix，使被选 hard sample 的理论梯度倍率不超过 1.50x。

日志新增：
- `mix_req`：curriculum 请求的 hard mix
- `mix_eff`：经过 cap 后实际使用的 hard mix
- `detail`：当前训练 batch 的平均 target detail score（0~1）

### 3. Target-detail-aware High-Frequency Loss

v1.8 的 HF Loss 对整张 patch 等权。

v1.8.1 根据 HR target 的局部细节建立 spatial mask：
- 真实细节/边缘区域：高权重
- 天然平滑/景深/Bloom 区域：低权重
- 平滑区域仍保留 0.15 floor，因此模型如果产生 ringing / 假高频仍会被惩罚

HF 最大权重由 0.06 降为 0.03。

这不是 GAN，不会要求模型“凭空生成纹理”；它仍然只要求预测高频接近真实 HR target 中存在的高频。

### 4. 更保守的游戏型 Edge / SSIM curriculum

保持 L1 = 1.0 为主目标。

默认最终权重：
- Edge：0.05 -> 最大约 0.0725
- SSIM：0.02 -> 最大约 0.030
- HF：0 -> 最大 0.030

原因：数据集中既有清晰 UI/Gameplay，也有游戏故意加入的柔焦、景深、Bloom 和电影效果。

### 5. Game-aware Degradation

真实输入是数字游戏画面，不是相机照片，因此降低不必要的 sensor-noise / compression 类随机退化：

- blur: 0.10
- pre sharpen: 0.08
- post sharpen: 0.08
- brightness: 0.03
- contrast: 0.05
- quantization: 0.025
- noise: 0.04

Resize kernel 改为加权采样：
- Bicubic 50%
- Bilinear 40%
- Area 10%

Degradation curriculum：
- start strength = 0.85
- full strength at = 35% training progress

## 对现有 checkpoint 的兼容性

部署模型结构没有变化：
- 不增加 C / L
- 不增加 ONNX 算子
- 不增加 NPU 推理计算
- 不改变 scale

因此同 C/L/scale 的 v1.7.x / v1.8 checkpoint 可以继续 Auto Resume。

`train.py` 还加入了 stock v1.8 配置的内存迁移逻辑：如果现有 YAML 仍是 v1.8 默认参数，会自动升级为 v1.8.1 默认值；如果某项已经被手动修改，则尽量保留用户自定义值。

## 你当前 epoch 13/100 的情况

如果从 epoch 13 后继续，epoch 14 大约会进入：

- phase: detail
- Edge ≈ 0.0548
- SSIM ≈ 0.0221
- HF ≈ 0.0054
- hard_top ≈ 12.6%
- requested hard mix ≈ 8.4%
- degradation strength ≈ 89.8%

之后逐步增强，到后期最大值才到上述 Game-aware 上限，不会突然把现有模型训练目标剧烈翻转。

注意：切换训练目标后，PSNR / train loss 短期出现小幅波动是正常的。更应该同时观察 `delta PSNR`、P10、edge ratio、HardL1 和后续实际游戏预览。

## 新日志示例

```text
[V1.8.1] phase=detail ... hard_top=13% mix=8% cap=1.50x detail_floor=0.45 ...
[V1.8.1 LOSS] L1=... HardL1=... Edge=... SSIMLoss=... HFLoss=... hard_top=... mix_req=... mix_eff=... detail=...
```

## 安装

1. 停止当前训练。
2. 解压本补丁。
3. 将补丁目录中的文件覆盖到 `gamesr-local-training` 根目录。
4. 再运行 `Train-Choose-C-L.cmd`。
5. 选择与当前 checkpoint 完全相同的 C / L / scale/profile，使用 Auto Resume。

不需要从 epoch 1 重练。
