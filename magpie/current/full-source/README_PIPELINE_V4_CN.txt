GameSR / Magpie SESR-M7 Pipeline v4 Async NPU Source Patch
==========================================================

目标
----
针对 C16L4 720p->1080p GameSR 模型实机表现：原始约 74 FPS，旧 Pipeline v3 只有约 50~53 FPS；
日志显示 NPU 本体约 9~10 ms，但旧版 SesrNpuProcessor::Process() 在渲染线程同步执行 CPU NCHW pack + NPU Run + residual upload，
完整 Process 约 13~15 ms，导致 NPU 时间直接阻塞 Magpie backend/render thread。

本补丁基于：
- Magpie-SESR-M7-NPU-Exact1080p-source-v2.1
- Pipeline-v3 / ProfilerPatch-v2
- NativePresetModels v2.2 StableModelContractFix（contractv3）

核心修改
--------
1. GameSR full-frame residual 路径改为真正异步 NPU worker：
   - Render thread：GPU readback ready 检查 -> 快速 BGRA 行 memcpy -> 提交 CPU slot；
   - Worker thread：BGRA/RGBA -> FP32 NCHW pack -> ORT/VitisAI NPU Run -> NCHW residual 转 RGBA；
   - Render thread：只在 result ready 时 memcpy residual 到动态纹理并提交 GPU 1.5x compose。
   - Render thread 不再调用 full-frame GameSR 的 Ort::Session::Run。

2. 三槽状态机：
   Free -> GpuPending -> CpuQueued -> WorkerRunning -> ResultReady -> Free
   每个 slot 在结果被消费之前不会复写其 gpuTexture，保证 base frame 和 residual 是同一帧。

3. latest-frame-wins：
   - Worker 只处理最新的 CpuQueued 帧；更旧的排队帧直接释放。
   - 不允许 NPU 工作无限排队，优先低延迟而不是“每帧必算”。

4. Persistent buffers：
   - 三个 slot 各自保留 compact BGRA capture buffer 和 RGBA residual buffer；
   - NCHW input/output 只保留一套全局 persistent FP32 buffer，因为 NPU worker 只有一个；
   - 不做每帧大块内存分配。

5. residual interleave 从 render thread 移到 worker：
   - Worker 在 NPU 完成后做 NCHW -> RGBA；
   - Render thread 上传时每行一次 memcpy，减少 render thread CPU 开销。

6. 启动阶段避免黑帧：
   - residual texture 初始化为 0；
   - 第一个 NPU result 出来之前用当前 base frame + zero residual 做纯 bilinear 1.5x fallback。

7. 新性能日志：
   GameSR PIPELINE v4 ASYNC:
     render线程
     GPU->CPU行拷贝
     worker input-pack
     NPU
     worker output-pack
     residual上传
     GPU合成提交
     slot 状态 / 丢弃 / 复用

   GameSR BACKEND v4:
     BackendRender 总耗时
     SESR提交/收结果
     effects CPU提交
     GPU fence等待
     shared Acquire+Copy+Flush

安装
----
1. 解压本补丁。
2. 将 src/ 等目录覆盖到你当前的 Magpie-SESR-M7-NPU-Exact1080p-source-v2.1 源码根目录。
3. 允许覆盖同名文件。
4. 用你当前已经能正常编译 v2.1 的 Visual Studio / x64 Release 流程重新 Rebuild。
5. 模型文件、models/GameSR、sesr overlay/config 均无需移动或重新训练。

验证
----
启动 SESR-M7 NPU Exact 1080p 后，日志应出现：
  PIPELINE v4 异步 worker 已启动

运行一段时间后应出现：
  GameSR PIPELINE v4 ASYNC: ...
  GameSR BACKEND v4: ...

重点看：
- NPU 预计仍在约 9~10 ms（C16L4）；
- 旧版 Process墙钟 13~15 ms 不再代表 render thread 被阻塞；
- 新版“render线程”目标应明显低于旧 Process 墙钟；
- 如果 GPU fence等待 / shared Acquire+Copy+Flush 仍很高，下一轮就针对 Renderer/Presenter 继续砍；
- 如果 worker队列丢弃持续快速增长，说明 worker (input pack + NPU + output pack) 已追不上输入，需要继续优化 pack/NPU。

重要说明
--------
- 本补丁没有假装实现“VitisAI <-> D3D11 真正零拷贝”。目前仍使用 CPU tensor + D3D11 staging/dynamic texture，
  因为当前环境没有可靠证据表明 VitisAI EP 可直接安全绑定 D3D11 texture memory。
- 本版优先解决最大的同步问题：不再让 9~10 ms NPU Run 直接堵住 render thread。
- 异步流水线会增加约 1~2 帧处理管线延迟，但 latest-frame-wins 保证不会无限积压。
- Legacy tiled SESR INT8 路径保持原 Pipeline v3 逻辑，不改行为。

修改文件
--------
主要代码修改：
- src/Magpie.Core/SesrNpuProcessor.cpp
- src/Magpie.Core/Renderer.cpp

其余文件沿用 NativePresetModels v2.2 StableModelContractFix source patch，方便直接覆盖而不丢失现有 ModelHub/contractv3 修复。
