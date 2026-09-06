# GameSR NPU Architecture Sweep v1.0

这是给现有 **GameSR XDNA Architecture Benchmark** 的小型覆盖补丁，不重复打包 486MB Ryzen AI runtime。

## 安装

1. 找到你刚才能正常跑 `results-bm-5` 的 **GameSR-XDNA-Architecture-Benchmark-v1.3** 目录。
2. 把本压缩包里的全部文件复制到那个目录根部。
3. 必须能看到：`runtime\bin\onnxruntime_perf_test.exe`。
4. 用你安装了 GameSR/Ryzen AI INT8 环境的 Python 运行（需要 torch、onnx、onnxruntime、amd-quark、Pillow）。
5. 双击 `RUN-GAMESR-NPU-SWEEP.cmd`。

## 默认扫描范围

- Channels: `4,6,8,10,12,14,16,20,24,28,32`
- Conv layers: `2,3,4,5,6,8`
- 共 **66 个结构**。
- 固定输入：`NCHW FP32 [1,3,720,1280]`
- 每个结构导出相同的 `Conv/ReLU ... Conv` residual core，并量化成 Ryzen AI XINT8 QDQ。
- 默认 warmup 20 次、正式 600 次。
- 每个模型必须由原 Benchmark v2.4.1 profile 验证是否 `NPU_ONLY`。

## 输出

结果目录会出现：

- `GameSR-NPU-Sweep-时间-RANKING.csv`：按平均 NPU 延迟自动排序。
- `GameSR-NPU-Sweep-时间-FULL.json`：完整原始汇总。
- `GameSR-NPU-Sweep-时间-SUMMARY.txt`：Top 20 快速查看。

字段包括 channels、layers、parameters、GMAC/frame、avg/P50/P95、FPS、effective GMAC/s、NPU/CPU events、cache 状态等。

## 自定义扫描

PowerShell 示例：

```powershell
.\Sweep-GameSR-XDNA.ps1 -Channels "8,12,16,20,24,32,40,48" -Layers "2,3,4,5,6,8,10" -Warmup 20 -Runs 600
```

重新编译所有 VitisAI cache：

```powershell
.\Sweep-GameSR-XDNA.ps1 -ForceRecompile
```

如果模型已经生成，只想重新测：

```powershell
.\Sweep-GameSR-XDNA.ps1 -SkipModelBuild
```

## 为什么保留 C6/C10/C12/C14 这些“不规则”通道

这正是 Sweep 的意义。`results-bm-5` 已经发现 C12-L6 相对异常，因此不能提前假定只有 8/16/24/32 才值得测。完整扫描可以直接看出 XDNA 的 shape 性能台阶和异常区域。

## 注意

第一次生成 66 个 XINT8 模型会比纯 benchmark 慢，后续模型会保存在 `sweep_models`，再跑时自动复用。真正重复测试时可用 `-SkipModelBuild`。
