# GameSR v4.5 一键部署 Probe（源码补丁）

基于：Magpie-GameSR-v4.4-Phase3x3-Vectorized-full-source

## 安装
将本 ZIP 内容直接覆盖到 v4.4 full-source 根目录，然后正常编译 Release x64。
编译出的 Magpie.exe 与本补丁中的 `RUN-GAMESR-PROBE.cmd`、`GameSR-Probe/` 一起放入运行版根目录。

## 使用
1. 游戏设置 1280x720，Render Scale 100%，关闭游戏内 FSR/DLSS/XeSS/动态分辨率。
2. 双击 `RUN-GAMESR-PROBE.cmd`。
3. 脚本会启动/复用 Magpie。按平时的缩放热键，使用 `GameSR NPU O4`。
4. 首个 Probe 帧会用 CPU 跑一次 FP32 参考模型并生成参考图片，因此短暂停顿属于正常诊断行为。
5. 随后继续运行，自动收集 180 个 fresh NPU 结果；完成后脚本生成 REPORT.txt 并自动 ZIP。

## 自动输出
- 01_input_720p.png：真实捕获 720p 同一帧
- 02_bilinear_1080p.png：双线性基线
- 03_fp32_cpu_decoder_1080p.png：FP32 ONNX + CPU 参考 Phase3x3 decoder
- 04_int8_cpu_decoder_1080p.png：当前 QDQ/INT8 NPU 输出 + 同一 CPU decoder
- 05_zero_residual_1080p.png：零 residual 基线（应与 Bilinear 相同）
- 06/07_residual_view_x8.png：残差放大图
- quality_metrics.csv：latent/RGB FP32-vs-INT8 数值差异
- performance_metrics.csv：fresh FPS、NPU、pack、upload、drop、reuse
- REPORT.txt：自动诊断摘要
- 最终 `GameSR-Probe-时间.zip`：直接发给 ChatGPT 分析即可

## 重要
Probe 仅在根目录存在 `GameSR-Probe.request` 时启动；正常使用 Magpie 时没有任何额外 CPU FP32 推理和图片生成开销。
