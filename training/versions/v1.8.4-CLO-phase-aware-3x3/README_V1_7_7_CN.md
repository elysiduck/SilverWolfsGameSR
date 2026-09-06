# GameSR Training v1.7.7 — 训练质量诊断 / 边际收益补丁

这是在 **v1.7.6 Validation RAM Cache** 上继续叠加的一版训练流程补丁。

这次的原则非常明确：

> **不改模型算法，只把“训练到什么程度了、继续训练还有没有收益、模型到底比普通缩放强多少”测清楚。**

## 这版没有改什么

以下全部保持 v1.7.6 原样：

- 模型架构、C/L、参数量
- GPU degradation
- AdamW
- warmup + cosine scheduler
- `L1 + Edge + sparse SSIM` 训练 loss
- EMA
- checkpoint 结构与续训逻辑
- Persistent RAM Cache
- Next-Epoch Prewarm
- Validation RAM Cache

因此 **已有 checkpoint 可以直接续训**，不需要从 Epoch 1 重来。

也没有新增 pip 依赖。

---

# 为什么需要这一版

你目前已经出现了典型情况：

```text
epoch 8 : PSNR=37.85 SSIM=0.9821
epoch 13: PSNR=38.13 SSIM=0.9828
```

单看 PSNR 会产生两个问题：

1. 38 dB 看起来已经很高，但不知道 **GameSR 比“什么都不学、只做 Bilinear”到底强多少**；
2. 平均 PSNR 很高，不知道是不是大量容易图片把少数困难图片遮住了；
3. 不知道最近 5 个 epoch 的收益到底是 `+0.30 dB` 还是只剩 `+0.02 dB`；
4. `train_loss` 是组合 loss，而旧版 `val_loss` 实际只有 L1，两个数字不能直接比较。

v1.7.7 就是解决这些“训练仪表盘”问题。

---

# 新增指标

每轮验证后新增：

```text
[QUALITY] bilinear=xx.xx dB
          delta=+x.xx dB
          MSE_reduction=+xx.x%
          P10=xx.xx dB
          min=xx.xx dB
          edge_err=0.xxxxx
          edge_ratio=0.xxx
          val_obj=0.xxxxx

[TREND] last 5 epoch(s):
        PSNR +0.xxx dB (+0.xxx dB/epoch)
        SSIM +0.xxxxx
        best=xx.xx dB@epochN
```

## 1. `bilinear`

同一批 Validation LR，直接 Bilinear 放大到 HR 后的 PSNR。

它回答：

> 如果完全不用神经网络，只普通拉伸，能有多少分？

## 2. `delta`

```text
Delta PSNR = GameSR PSNR - Bilinear PSNR
```

例如：

```text
Bilinear = 34.90 dB
GameSR   = 38.13 dB
Delta    = +3.23 dB
```

这比只看“38.13”更有意义。

## 3. `MSE_reduction`

GameSR 相比 Bilinear 把平均平方误差减少了多少百分比。

例如：

```text
MSE_reduction = +52.0%
```

表示网络把普通 Bilinear 的像素平方误差又削掉了约一半。

## 4. `P10`

PSNR 最差 10% 图片所在位置的指标。

平均 PSNR 可能是：

```text
38.1 dB
```

但如果：

```text
P10 = 31.5 dB
```

说明仍有一批困难场景表现明显较差。

因此后面看泛化时，`P10` 往往比平均值更敏感。

## 5. `min`

这一轮 Validation 里最差一张图片的 PSNR。

它容易受极端样本影响，所以不要单独拿它选模型，但很适合发现异常。

## 6. `edge_err`

和训练 Edge Loss 使用同一类一阶横向/纵向梯度误差。

越低表示：

- UI 线条
- 轮廓
- 发丝
- 栅栏
- 小文字边缘

更接近 GT。

## 7. `edge_ratio`

```text
预测图平均边缘强度 / GT 平均边缘强度
```

大致解释：

```text
< 1.0  → 整体偏软
≈ 1.0  → 边缘能量接近 GT
> 1.0  → 可能偏锐 / 过锐
```

**它不是越大越好。**

这是诊断值，不是排行榜分数。

## 8. `val_obj`

旧版：

```text
train_loss = L1 + Edge + 间歇性 SSIM
val_loss   = 只有 L1
```

所以以前：

```text
train=0.00899
val=0.00745
```

不能直接比较。

v1.7.7 保留旧 `val_loss` 字段，避免破坏历史日志；同时新增 `val_obj`。

`val_obj` 使用当前训练 loss 的平均权重口径，包括：

```text
SSIM 每 N step 才参与一次
```

这一点也会折算进去。

注意：训练集有随机 degradation，而 Validation 是固定 deterministic pair，因此 `train_loss` 与 `val_obj` 仍然不会完全相同；但至少它们现在使用的是相近的 loss 组成口径。

## 9. `[TREND]`

默认看最近 5 epoch：

```text
PSNR +0.280 dB (+0.056 dB/epoch)
```

这样以后不用自己翻日志计算“边际收益还有多少”。

可选 YAML：

```yaml
train:
  quality_trend_window: 5
```

不写就默认 5。

---

# 新输出文件

每个 epoch 会更新：

```text
outputs\你的模型\quality_summary.json
```

里面保存：

- 当前 PSNR / SSIM
- Bilinear PSNR
- Delta PSNR
- MSE 改善率
- P10 / min / std
- Edge Error / Edge Ratio
- 最近窗口边际收益
- 当前历史最佳 Epoch

以后你直接把这个 JSON 或训练日志发给我，就能更快判断：

> 继续训练值不值、是不是平台期、是平均指标好看还是真正困难样本也在改善。

`training_log.csv` 和 TensorBoard 同时增加这些字段。

---

# 为什么这版没有把 LPIPS / DISTS 塞进每个 Epoch

这是刻意的。

LPIPS / DISTS 都需要额外的感知网络和依赖，会让 Validation 更慢、占更多显存/内存；而你刚刚已经花了 v1.7.6 专门优化 Validation 时间。

所以 v1.7.7 先加入：

- 零新增依赖
- 不需要额外网络
- 可以每轮实时跑
- 对游戏 SR 很直接的指标

LPIPS / DISTS 更适合后面做成 **离线最终模型 Benchmark**，而不是拖慢每个训练 Epoch。

---

# 安装

前提：你当前训练目录已经至少覆盖到 v1.7.6。

训练进程停止后，把本补丁目录中的内容覆盖到：

```text
GameSR-local-training-env-v1.1\gamesr-local-training\
```

实际只新增/修改：

```text
train.py
gamesr\metrics.py
```

文档文件可留可删。

不要删除：

```text
.venv
.venv-quark
dataset
outputs
latest.pth
best.pth
```

然后仍然使用原来的：

```text
Train-Choose-C-L.cmd
```

已有训练直接 Auto Resume 即可。

---

# 兼容性

- v1.7.6 checkpoint：兼容
- v1.7.5.x checkpoint：按原兼容规则兼容
- 原 YAML：兼容
- 原 C/L 模型：兼容
- ONNX 导出：无变化
- NPU 部署：无变化
- 模型参数量：无变化
- 推理速度：无变化

第一行训练日志也继续保留原格式：

```text
train=... val=... PSNR=... dB SSIM=... time=...
```

因此之前依赖这一行的观察方式不会失效。
