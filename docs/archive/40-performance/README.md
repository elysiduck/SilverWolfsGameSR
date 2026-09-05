# GameSR 性能 / Pipeline / Profiler 文档归档

## `docs/40-performance/pipeline-v3/README_CN-v3-fixed2.md`

# GameSR Magpie SESR Pipeline v3 修正版 2

上一个 v3 修正版仍缺少 `SesrNpuProcessor::Impl` 的结构体结束括号，导致 MSVC C3254/C2838/C2382/C1075。此包已修正：`Process()` 函数关闭后再关闭 `Impl` 结构体。

请覆盖原源码中的：

- `src/Magpie.Core/SesrNpuProcessor.cpp`
- `src/Magpie.Core/GraphicsCaptureFrameSource.cpp`
- `src/Magpie.Core/GraphicsCaptureFrameSource.h`

然后按原来的 Windows x64 Release/Rebuild 流程编译。

本补丁仍保持 v3 的 pipeline 设计：3-slot 异步 readback、非阻塞检查、优先处理最近完成帧，以及更快的 CPU tile merge。

---

## `docs/40-performance/profiler-patch-v2/README_CN.md`

# GameSR / Magpie SESR v2.1 - Profiler Round 2

这不是完整的 Magpie 发布包，而是第二轮源码补丁。

## 本轮改动

1. 将 `SesrNpuProcessor::Process()` 的粗粒度计时拆成：
   - `copy提交`
   - `input Map等待`
   - `CPU切片`
   - `NPU`
   - `output Map等待`
   - `CPU拼接`
   - `Unmap提交`
   - `合计`
2. 预计算 X/Y 反射坐标，移除每个像素重复的整数取模/边界反射计算。
3. `GraphicsCaptureFrameSource::_Update()` 增加捕获阶段日志，并记录主动丢弃的旧帧数量。

## 使用

1. 将本包 `src/Magpie.Core/` 中的 3 个文件覆盖到原源码对应目录。
2. 不需要给 Magpie 增加任何启动参数，也不需要在掌机上输入 PowerShell 命令。
3. 按你原来的方式重新编译 `Release x64`。原项目根目录已有 `Build-SESR.ps1`，它会生成 `publish\\x64\\Magpie.exe`。
4. 把新的 `publish\\x64` 整个目录复制到掌机，像原来一样双击 Magpie，选择 SESR NPU 并正常缩放游戏。
5. 测试约 30-60 秒，分别做一次 15 W 和 10 W。
6. 把新的 `magpie.log` 发回来。

## 重点看

日志中应该出现：

`Graphics Capture PROFILE: ...`

`SESR PROFILE: ...`

我们主要比较：

`input Map等待` 是否占主要时间；
`CPU切片` 是否仍然明显；
`output Map等待` 是否出现尖峰；
`CPU拼接` 是否随功耗变化；
`本次跳过旧帧` 是否大量出现。

注意：当前 Linux 容器无法执行你的 Windows/MSVC + WinUI 构建，因此本补丁未在此环境完成最终 Windows 编译验证。

## 原始日志

原始 `magpie-sesr-v2.1-2026-09-04.log` 已纳入本次本地文档清单，但不作为说明文档复制到 GitHub 文档中心；性能日志应继续放在 `benchmark/` 或 GitHub Release/Artifacts，避免把大段运行日志混入长期文档历史。
