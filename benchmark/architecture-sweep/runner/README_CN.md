# GameSR Architecture Sweep Runner v1.0.2（掌机）

这个 Runner **只负责在掌机 NPU 上测试 Builder 已经生成好的 INT8/QDQ 模型**。

它不需要、也不会尝试调用：Python、PyTorch、ONNX Python、AMD Quark、训练环境。

## 安装位置

把本补丁里的文件复制到你之前能够成功运行 `results-bm-5` 的 GameSR/XDNA Benchmark 根目录。

复制后同一目录必须能看到：

```text
runtime\
  bin\onnxruntime_perf_test.exe
  vaip_config_stride.json
  AMD_AIE2P_4x4_Overlay.xclbin

Benchmark-GameSR-v24-Sweep.ps1
Runner-GameSR-XDNA.ps1
RUN-GAMESR-ARCHITECTURE-SWEEP.cmd
```

本补丁**不重复包含 Ryzen AI runtime**，直接复用已经在掌机上验证成功的 runtime。

## 放入 Builder 模型

Builder 在笔记本会生成：

```text
GameSR-NPU-Architecture-Sweep-Models-xxx-xxxxxxxx.zip
```

三种使用方式任选一种：

1. 最方便：把这个 ZIP 复制到 Runner 所在目录，然后双击 `RUN-GAMESR-ARCHITECTURE-SWEEP.cmd`。
2. 把 ZIP 直接拖到 `RUN-GAMESR-ARCHITECTURE-SWEEP.cmd` 上。
3. 解压后把 `handheld_payload\` 放到 Runner 旁边，再双击运行。

Runner 会自动读取 `sweep_manifest.json`，验证模型 SHA256，然后逐个跑 VitisAI/NPU benchmark。

## 两个启动器

- `RUN-GAMESR-ARCHITECTURE-SWEEP.cmd`：正式精确测试，Warmup=20、Runs=600。
- `RUN-GAMESR-ARCHITECTURE-SWEEP-QUICK.cmd`：快速预筛，Warmup=5、Runs=120。

最终选模型时建议以正式精确测试为准。

## 输出

结果都在 `results\`：

```text
GameSR-NPU-Architecture-Sweep-...-RANKING.csv
GameSR-NPU-Architecture-Sweep-...-FULL.json
GameSR-NPU-Architecture-Sweep-...-SUMMARY.txt
GameSR-NPU-Architecture-Sweep-...-LATENCY-MATRIX.csv
GameSR-NPU-Architecture-Sweep-...-EFFICIENCY-MATRIX.csv
GameSR-NPU-Architecture-Sweep-...-BEST-BY-CHANNEL.csv
GameSR-NPU-Architecture-Sweep-...-BEST-BY-LAYER.csv
GameSR-NPU-Architecture-Sweep-...-COMPACT.zip
```

`COMPACT.zip` 是最适合直接发回 ChatGPT 分析的结果包：包含聚合结果、每个模型的 SUMMARY 和日志，但自动排除体积很大的 profiler JSON。

原始 profiler JSON 仍保留在 `results\` 中，遇到异常架构时可以再单独分析。

## 判断原则

Runner 只把 `NPU_ONLY` 结果放进正式排名。`NPU_PARTIAL`、`CPU_FALLBACK`、`UNVERIFIED`、失败项会保留在 FULL/SUMMARY 中，但不会被错误地当成最快模型。

## v1.0.1 修复

- 修复 Windows PowerShell 5.1 把 ONNX Runtime/VitisAI 写到 stderr 的普通警告包装为 `NativeCommandError`，并在 `$ErrorActionPreference='Stop'` 下提前终止 benchmark 的问题。
- Sweep benchmark 现在直接采用已在 `results-bm-5` 路线验证成功的 Benchmark v2.4.1 native-stderr hotfix。
- Runner 会保存每个子 benchmark 的 launch log。
- 如果出现 `NO_SUMMARY` 这种基础设施级错误，Runner 会在第一个模型后立即停止，不再把同一个错误重复 105 次。

升级时直接把 v1.0.1 文件覆盖到原 Runner/benchmark 根目录即可；Builder 生成的模型 ZIP 不需要重新生成。


## v1.0.2 修复（路径含空格）

这版修复一个已经从实际掌机日志确认的问题：`onnxruntime_perf_test -i` 的 VitisAI provider options 会按空格拆成多个 `key|value` token。

当 benchmark 根目录是例如：

```text
D:\Game SR\GameSR-XDNA-Architecture-Benchmark-v1.3\...
```

旧版把 `config_file` 和 `cache_dir` 以**绝对路径**放进 `-i`，路径中的 `Game SR` 会被 perf_test 拆成两个 token，从而报：

```text
ParseSessionConfigs Use a '|' to separate the key and value ...
```

v1.0.2 改为在 benchmark 根目录下传递无空格的相对路径：

```text
config_file|runtime\vaip_config_stride.json
cache_dir|cache\<hash>
```

模型 ONNX 的绝对路径仍作为单独命令行参数传入，所以模型所在路径包含空格没有问题。

另外，如果再次检测到 `ParseSessionConfigs` 这类基础设施错误，Sweep 会立即停止，不会再连续失败几十个模型。

升级时直接覆盖 Runner 的 4 个脚本即可；Builder 产出的 105 个模型无需重新生成。
