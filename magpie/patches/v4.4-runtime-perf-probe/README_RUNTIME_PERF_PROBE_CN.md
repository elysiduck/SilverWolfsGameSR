# GameSR v4.4 运行版即时性能 Probe

这个补丁可以**直接覆盖当前 v4.4 runtime**，不需要重新编译。

双击 `RUN-GAMESR-RUNTIME-PERF-PROBE.cmd`，然后正常启动 1280x720 游戏的 `GameSR NPU O4` 缩放。脚本自动从 `logs\magpie.log` 收集 10 组 Pipeline v4 数据，输出 fresh GameSR FPS、NPU、pack/unpack、upload、drop/reuse、GPU fence 和捕获后端，并自动打包 ZIP。

注意：当前 Magpie.exe 本身没有 FP32-vs-INT8 同帧 A/B hook，所以这个“运行版即时补丁”只做性能/部署日志诊断。完整画质 A/B 在另一个 **v4.5 OneClick Probe 源码补丁**中实现，需要用 Windows/MSVC 编译新的 Magpie.exe。
