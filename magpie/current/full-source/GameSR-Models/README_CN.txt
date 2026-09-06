GameSR 模型目录
===============

用途
----
把 GameSR 的 INT8/QDQ ONNX 模型放在本目录后再运行“构建-SESR.cmd”。
构建系统会自动将模型和本说明复制到：

  publish\x64\models\GameSR\

例如 C16L4：

  gamesr-c16-l4-720p-to-1080p-int8-qdq.onnx

也可以在编译完成后，直接把 ONNX 放入上述 publish 目录。

当前模型合约
------------
- 输入：NCHW FP32/QDQ，[1,3,720,1280]
- 输出：同尺寸 720p RGB residual
- 最终输出：Magpie GPU 执行 half-pixel Bilinear 1.5x + Add + Clip，得到 1920x1080

使用
----
1. 启动 Magpie。
2. 编辑 SESR/GameSR 缩放预设。
3. 在“GameSR 模型”下拉菜单选择相应 ONNX。
4. 推荐使用“SESR-M7 NPU Exact 1080p”。当选中 GameSR 模型时，它会保留
   1280x720 模型输入，不会执行 720p -> 540p 预缩放。

注意
----
- 不要把训练 checkpoint（.pth）放进来；部署只使用导出的 INT8/QDQ ONNX。
- 若 ONNX 使用外部数据，还要把同名 .onnx.data 文件一并放入本目录。
- 模型文件名必须与预设中选择的文件完全一致。
