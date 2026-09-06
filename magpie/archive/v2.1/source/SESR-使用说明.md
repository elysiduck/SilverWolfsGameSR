# Magpie SESR-M7 NPU ×2 测试版

这个版本把 AMD SESR-M7 INT8 模型接入 Magpie，并通过 ONNX Runtime 的 `VitisAIExecutionProvider` 在 Ryzen AI 9 HX 365 的 XDNA2 NPU 上执行推理。

## 推荐分辨率

你的屏幕是 1920×1080，因此游戏输入建议设置为 960×540：

\[
960\times540 \xrightarrow{\text{SESR-M7 }2\times} 1920\times1080
\]

不要先用 1280×720 测试。固定的 2× 模型会生成 2560×1440，随后又要缩回 1920×1080，会浪费大量性能。

## 编译（不用自己研究）

1. 双击 `0-先双击我-检查环境.cmd`。
2. 它会在同一文件夹生成 `环境检查结果.txt`。
3. 把这个文本发给我，不用看懂，也不用自己安装东西。
4. 环境齐全后才双击 `构建-SESR.cmd`。编译完成后，程序位于 `publish\x64\Magpie.exe`。

## 第一次测试

1. 将游戏改为 960×540、窗口模式或无边框窗口模式。
2. 启动这个版本的 Magpie。
3. 缩放模式选择 `SESR-M7 NPU ×2`。
4. 暂时不要叠加其他锐化、降噪或插帧效果。
5. 启动缩放并运行游戏约两分钟。
6. 退出 Magpie，打开 `logs\magpie.log`。
7. 搜索 `SESR ONNX 帧耗时`，把相关行发给我。

日志会分别显示读回切片、NPU 推理和拼接上传耗时。理论处理帧率可按下式估算：

\[
FPS_{\max}\approx\frac{1000}{t_{\text{读回}}+t_{\text{NPU}}+t_{\text{上传}}}
\]

## 当前限制

- 这是需要在实际 XDNA2 设备上验证的第一版，尚未验证最终帧率。
- 使用 16 像素重叠切片，逐切片提交给单个 ONNX Runtime NPU 会话，避免旧 GraphEngine 直接创建上下文时的兼容性崩溃。
- Direct3D 纹理与 NPU 之间仍有一次读回和一次上传；日志可以判断瓶颈究竟在 NPU 还是内存搬运。
- 目前仅支持 x64 Ryzen AI 设备。

## 第三方组件

- Magpie：GPL-3.0
- ONNX Runtime / AMD VitisAI EP：MIT 及对应组件许可证
- AMD SESR-M7 模型：Apache-2.0
