## `docs/20-training/current-v1.7.1/guides/README_V1_7_1.md`

# GameSR Training Code v1.7.1 — reliability fix

This release keeps the v1.7 model architecture and Ryzen AI core-only design.
It fixes and hardens the workflow before the first real `tiny-npu-fast` run.

## Fixed

- NPU YAML profiles build without an unused `model.blocks` key.
- The smoke test loads both real NPU YAML files and checks parameter counts.
- Auto-resume no longer loads an incompatible checkpoint after skipping it.
- Checkpoints are written atomically to protect `latest.pth` during interruption.
- Invalid batch sizes and `ssim_every_n_steps: 0` are handled safely.
- Dataset manifests verify referenced image files before a long run.
- Collector V3.2 `game/session/image` folders split by capture session, reducing adjacent-frame leakage between train, validation and test.
- ONNX export refuses a checkpoint from another model/version configuration.
- INT8 export verifies static `[1,3,720,1280]` input and removes stale sidecars.
- NPU packaging fails closed unless a hash-bound Quark XINT8 core is valid and contains no Resize/Add/Clip.

## Unchanged

- `tiny-npu-fast`: 8 sequential Conv layers, C8, 3,947 parameters.
- `tiny-npu-quality`: 14 sequential Conv layers, C8, 7,451 parameters.
- Deployment output is a low-resolution RGB residual.
- Magpie must use half-pixel sampling for bilinear base plus bilinear residual.

Run `00_smoke_test.cmd` first. Train `tiny-npu-fast` before the quality profile.


---

## `docs/20-training/current-v1.7.1/guides/修复版使用说明.txt`

```text
GameSR v1.7.1 修复版使用说明
=============================

这个版本修复了 v1.7 第一次训练前发现的问题，模型设计没有改变。

你现在继续使用 Collector V3.2 收集和清洗图片即可，不必立刻训练。

准备训练时：

1. 先备份原来的 gamesr-local-training 文件夹。
2. 把本压缩包内 GameSR-training-code-v1.7.1 的所有内容复制到原训练文件夹。
3. 如果 Windows 询问是否替换，选择替换。
4. 不要删除 dataset、.venv、.venv-quark 和 outputs。
5. 双击 00_smoke_test.cmd。
6. 看到两行 [PASS] 后，双击 Train-NPU-Fast.cmd。
7. 第一次选择 batch size 时直接按 Enter。

如果出现 [FAIL]：

- 不要反复重装，也不要删除数据。
- 截图整个窗口，并把 outputs\pipeline_logs 中最新的日志发回来。

训练期间可以正常中断。程序采用安全保存，并会从 latest.pth 继续。

先只训练 tiny-npu-fast。等它完成 NPU 实测后，再决定是否训练
tiny-npu-quality，不需要两个一起训练。

```

---

## `docs/20-training/current-v1.7.1/guides/先看这里-新手步骤.txt`

```text
GameSR 训练代码 v1.7.1 —— 新手只看这里
========================================

这版有两个新模型：

1. tiny-npu-fast：8 层卷积，优先追求接近 60 FPS。建议先训练它。
2. tiny-npu-quality：14 层卷积，画质潜力更高；结构探针实测核心约 47.9 FPS。

注意：47.9 FPS 是掌机上 14 层卷积“核心 ONNX”的实测结果，60 FPS 是
8 层快版的设计目标；二者都不是 NPU 硬件理论上限。Magpie 最终帧率还会受
截图、内存传输、着色器合成和显示影响。


第一次使用
----------

1. 把本包所有文件复制到原 gamesr-local-training 文件夹，选择替换。
2. 保留原来的 dataset、.venv 和 .venv-quark，不要删除。
3. 双击 00_smoke_test.cmd。必须同时看到配置检查和训练步骤的 [PASS]。
4. 如果从未成功安装 Quark，才需要双击 01_Install_RyzenAI_INT8.cmd。


推荐训练：NPU Fast
------------------

1. 双击 Train-NPU-Fast.cmd。
2. 第一次会测试合适的训练 batch。出现选择时直接按 Enter 使用最快值。
3. 程序会自动训练、测试、导出 FP32、生成 AMD Quark XINT8 并打包。
4. 训练中途关闭也没关系；再次双击会从 latest.pth 继续。
5. 完成后使用：

   deployment\tiny-npu-fast\tiny-npu-fast-720p-to-1080p-int8-qdq.onnx


画质版
------

需要比较画质时，再双击 Train-NPU-Quality.cmd。它更慢、训练时间也更长。
完成后使用：

   deployment\tiny-npu-quality\tiny-npu-quality-720p-to-1080p-int8-qdq.onnx


重要部署变化
------------

v1.7.1 的 ONNX 输出不是可直接显示的 1080p 图片，而是：

   [1, 3, 720, 1280] 低分辨率 RGB 残差

Magpie 必须在着色器中执行：

   output = clamp(bilinear(input, 1.5) + bilinear(residual, 1.5), 0, 1)

并使用 half-pixel / align_corners=false 坐标规则。

这是为了把 Resize、基础图旁路、Add 和 Clip 移出 ONNX，让全部卷积保持为一个
连续 NPU 子图。当前 Benchmark 可以测核心速度，但在 Magpie 完成这一着色器集成
以前，不要把 core-only ONNX 当成完整超分模型直接显示。


只重新导出/量化
----------------

已有 best.pth 时，不用重训：

   快版：双击 Export-INT8-NPU-Fast.cmd
   画质版：双击 Export-INT8-NPU-Quality.cmd

如果生成过程成功，必须看到：

   [PASS] QLinearConv: 0
   [PASS] Core-only graph: Resize/Add/Clip = 0/0/0
   [OK] Ryzen AI INT8 model

```

---

## `docs/20-training/current-v1.7.1/guides/训练与帧率说明.md`

# GameSR v1.7.1 训练与帧率说明

## 推荐顺序

先训练 `tiny-npu-fast`。确认它在掌机上能够形成单一 NPU 子图且速度合格后，
再训练 `tiny-npu-quality` 做画质对照。不要同时开始两个训练任务。

训练时模型仍输出完整 (1080p) 图像并与真值计算损失：

\[
\hat y=\operatorname{clamp}\left(B_{1.5}(x)+B_{1.5}(R_\theta(x)),0,1\right).
\]

其中 (R_\theta(x)) 是网络学习到的 (720p) RGB 残差，(B_{1.5}) 是
倍率为 (1.5) 的双线性放大。因此 core-only 只改变部署边界，不会把训练目标
改成低分辨率图片。

## 实际操作

1. 确认 `dataset/train.csv`、`dataset/val.csv`、`dataset/test.csv` 已存在。
2. 双击 `00_smoke_test.cmd`，确认训练环境正常。
3. 双击 `Train-NPU-Fast.cmd`。
4. 第一次 batch 测试后直接按 Enter；脚本会自动使用最快的安全 batch。
5. 等待全部 epoch 完成。再次启动会自动读取 `outputs/tiny-npu-fast/latest.pth`。
6. 最佳权重位于 `outputs/tiny-npu-fast/best.pth`。
7. 最终 XINT8 位于 `deployment/tiny-npu-fast/`。

如果训练结束但量化因意外中断，双击 `Export-INT8-NPU-Fast.cmd` 即可重新导出
和量化，不需要重新训练。

## (47.9\,\mathrm{FPS}) 与 (60\,\mathrm{FPS}) 的含义

结构探针在目标掌机上测得 14 层顺序卷积核心：

\[
t_{14}=20.8493\,\mathrm{ms},\qquad
f_{14}=\frac{1000}{20.8493}\approx47.934\,\mathrm{FPS}.
\]

这是 `tiny-npu-quality` 的结构级参考值，不是 NPU 理论极限，也不包含 Magpie
截图、CPU/NPU 或 GPU/NPU 传输、着色器和显示等待。

已知 2 层探针为 (9.184\,\mathrm{ms})。若只做粗略线性插值，8 层候选为：

\[
t_8\approx 9.184+6\times\frac{20.8493-9.184}{12}
\approx15.02\,\mathrm{ms},
\]

\[
f_8\approx\frac{1000}{15.02}\approx66.6\,\mathrm{FPS}.
\]

真实编译和内存开销并不完全线性，所以 (66.6\,\mathrm{FPS}) 只是估算。
`tiny-npu-fast` 的合理验证目标是核心约 (55\text{--}65\,\mathrm{FPS})，最终值必须
由 Quark 量化后的掌机 Benchmark 给出。

## 仍有优化空间

- 把 8 层进一步降到 6 层，可继续换取速度，但画质可能下降。
- 训练时使用可重参数化模块、部署时折叠成普通顺序卷积，有机会增加画质而不增加
  部署算子种类；这需要下一轮专项验证。
- Magpie 侧使用异步流水线、复用缓冲区并减少 NPU 前后的内存复制，往往比继续删
  一两层卷积更影响整机帧率。
- 把双线性放大、残差相加和裁剪合并成一个 GPU shader，避免重新进入 CPU。
- 新版 Ryzen AI runtime/驱动可能改善编译或调度，但必须做同模型 A/B 测试。

因此，当前不能说硬件最大值就是 (60\) 或 (47\)。能确定的是：旧完整图因分区
只能跑约 (6\,\mathrm{FPS})，14 层连续核心已经实测约 (47.9\,\mathrm{FPS})，而
8 层快版有希望越过 (60\,\mathrm{FPS}) 的核心吞吐门槛。
