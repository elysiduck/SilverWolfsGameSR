# GameSR v1.7.2 C/L 训练选择器补丁

本补丁直接覆盖到现有 `GameSR-training-code-v1.7.1` 根目录，不新建 Python 环境。

## 最简单用法

双击：

`Train-Choose-C-L.cmd`

随后输入：

- `C` = channels（通道宽度）
- `L` = npu_sequential 的总卷积层数
- `Epochs` = 训练轮数，默认 100

例如输入 C=24、L=3，会自动生成：

`configs/gamesr-c24-l3.yaml`

训练输出自动隔离到：

`outputs/gamesr-c24-l3/`

最终 ONNX / XINT8 / deployment 名称也使用 `gamesr-c24-l3`，不会与其他 C/L 覆盖。

## 推荐的第一轮画质对照

- C8-L5
- C16-L4
- C24-L3
- C32-L3

这些结构在当前掌机 NPU Sweep 中有接近的纯 NPU 延迟预算，适合比较“宽度 vs 深度”的真实画质收益。

## 命令行用法

无需交互：

`GameSR_Train.cmd --channels 24 --layers 3 --epochs 100`

自定义名称：

`GameSR_Train.cmd --channels 24 --layers 3 --name my-c24-l3`

## 安全性

- 继续复用 `.venv` 训练环境。
- 继续复用 `.venv-quark` 做 AMD Quark XINT8。
- 每种 C/L 有独立 config、output、benchmark cache、checkpoint 和 deployment。
- 自动续训只会续同一个 C/L 的 checkpoint；checkpoint signature 仍会阻止错误架构混用。
- 旧的 `Train-NPU-Fast.cmd` / `Train-NPU-Quality.cmd` 仍可继续使用。
