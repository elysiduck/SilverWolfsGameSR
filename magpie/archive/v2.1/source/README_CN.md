# GameSR × Magpie v2.1 — 原生预设模型选择 v2

这是基于以下版本继续整合的源码补丁：

- Magpie-SESR-M7-NPU-Exact1080p-source-v2.1
- GameSR-Magpie-SESR-v2.1-ProfilerPatch-v2
- GameSR-Magpie-SESR-v2.1-Pipeline-v3-fixed3
- 上一版 GameSR v1.7.1 full-frame residual 接入思路

## 这一版取消了什么

彻底取消：

- `选择GameSR模型.cmd`
- `打开GameSR模型文件夹.cmd`
- `active_model.txt`
- 全局“当前模型”概念

模型选择现在属于 **Magpie 缩放模式/预设本身**。

## 编译后怎么用

编译完成后，Magpie 输出目录会自动创建：

```text
models\GameSR\
```

把训练好的 `.onnx` 直接拖进去，例如：

```text
models\GameSR\
  tiny-fast-e50.onnx
  tiny-fast-e100.onnx
  tiny-quality.onnx
```

然后进入 Magpie 的“缩放模式”：

1. 复制或新建一个缩放模式；
2. 这个缩放模式里使用 `SESR-M7 NPU Exact 1080p`；
3. 展开缩放模式，会看到新增的 `GameSR NPU 模型`；
4. 在下拉框中直接选择 `.onnx`；
5. Magpie 自动保存，不需要 CMD；
6. 如果是在 Magpie 已经打开后才复制模型，点一次“重新扫描模型”。

例如可以建立：

```text
GameSR Fast     -> tiny-fast.onnx
GameSR Quality  -> tiny-quality.onnx
GameSR Epoch100 -> tiny-fast-e100.onnx
```

以后只切换 Magpie 缩放预设即可比较模型。

## 兼容行为

下拉框第一项：

```text
内置 SESR-M7（原版）
```

选择它时继续使用原来的：

```text
sesr\sesr_nhwc_int8_256x256.onnx
```

旧配置 JSON 没有 `gameSrModel` 字段时也会自动视为“内置 SESR-M7”，因此旧预设不会失效。

## GameSR v1.7.1

外部模型仍保留上一版的自动接口识别：

- legacy SESR: INT8 NHWC `[1,256,256,3] -> [1,512,512,3]`
- GameSR v1.7.1 residual core: FP32/QDQ NCHW `[1,3,H,W] -> [1,3,H,W]`

对于 1280×720 GameSR residual 模型，Exact 1080p 路径会使用 1280×720 NPU 输入，然后 GPU 完成 half-pixel 1.5× base/residual 合成。

Pipeline v3 的三槽异步 readback、latest-ready-frame、旧输出复用和帧匹配逻辑继续保留。

## NPU Cache

外部模型缓存目录：

```text
cache\GameSR\
```

cache key 按 ONNX 文件内容哈希生成，而不是只看文件名。
因此即使覆盖同名 `tiny-fast.onnx`，只要模型内容变化就会获得新的 NPU cache key。

## 安装

把这个补丁 ZIP 解压到 `Magpie-SESR` 源码根目录并覆盖同名文件，然后按原来的方式重新编译 x64。

这份补丁已经包含与本功能相关的 Profiler/Pipeline v3 核心文件，不要再在它后面覆盖旧的 ModelHub v1。

## 当前验证范围

已做源码级一致性检查和 XAML/XML 结构检查。
当前执行环境不是 Windows + Ryzen AI，无法在这里完成 Visual Studio/WinUI 编译和 VitisAI NPU 实机运行验证；第一次掌机编译/运行后的日志仍应保留用于继续修正。
