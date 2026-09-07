GameSR 独立 NPU 模型目录
========================

这版 GameSR 已做成独立超分模板，不再依赖 SESR 的 540p 切片路径。
四个模式可自由选择：GameSR NPU O3 / O4 / O6 / O8。

最简单的使用方法
----------------
1. 把训练包生成的 deployment 文件夹整体复制到 models\GameSR 下；
   可以每个模型使用一个独立子文件夹，不再需要全部堆在根目录。
2. 启动 Magpie，在缩放模式里选择对应的 GameSR NPU O3/O4/O6/O8。
3. 在“GameSR NPU 模型”下拉框中选择匹配的 *-int8-qdq.onnx。
4. 开始缩放。

模型和模式必须对应
------------------
- O3 模式只能选择输出通道为 3 的模型。
- O4 模式只能选择输出通道为 4 的模型。
- O6 模式只能选择输出通道为 6 的模型。
- O8 模式只能选择输出通道为 8 的模型。

O4/O6/O8 除 ONNX 外，还必须带训练导出的：
  同模型名.onnx.decoder.json

若选择的是 *-int8-qdq.onnx，程序会自动寻找其 FP32 基名对应的
.onnx.decoder.json。deployment 文件夹保持原样复制即可，不要手工改名。

当前运行路径
------------
- 输入：当前捕获分辨率直接进入 NPU，不预缩放到 540p。
- NPU：输出同分辨率 O3/O4/O6/O8 latent。
- GPU：旧模型继续使用 half-pixel Bilinear latent + 1x1 decoder；schema v3
  新模型使用九相位 3x3 decoder 直接从 720p latent 生成 RGB residual；
  两者最终都和 1.5x bilinear 基础画面合成。
- 调度：沿用三槽异步 NPU pipeline，优先显示最新完成帧。

注意
----
- 只能放导出的 ONNX 部署模型，不要放 .pth checkpoint。
- ONNX 的固定输入尺寸必须等于当前捕获尺寸。例如 1280x720 模型应配合
  1280x720 游戏输入，最终输出为 1920x1080。
- O3 保留旧 A/B 路径，不要求 decoder.json；它使用恒等 RGB residual 解码。
- 若 ONNX 使用外部数据，还要复制同名 .onnx.data 文件。
