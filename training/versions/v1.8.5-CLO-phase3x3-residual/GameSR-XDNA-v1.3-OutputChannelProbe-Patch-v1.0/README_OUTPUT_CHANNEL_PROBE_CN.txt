GameSR XDNA Architecture Benchmark v1.3
Output-Channel Probe 补丁 v1.0
=======================================

目的
----
这个补丁补上我们刚刚确定的新测试：

  C16-L4-O3   当前 RGB residual 基线
  C16-L4-O4   最重要的新方案：4-channel feature 输出
  C16-L4-O6   压力点
  C16-L4-O8   压力点 / 对齐观察点

四个模型只有“最后输出通道数”不同：

  O3: C16 -> 3
  O4: C16 -> 4
  O6: C16 -> 6
  O8: C16 -> 8

C16-L4 主体、720p 输入、卷积层数均保持不变。
因此这个测试可以隔离回答：把当前 3-channel residual 改为 4/6/8-channel feature，
会不会让 XDNA/VitisAI 的纯 NPU 延迟突然变差。

重要：它现在测什么 / 不测什么
-------------------------------
会测：
  - VitisAI 是否仍然 NPU_ONLY
  - pure NPU average / P50 / P95
  - O4/O6/O8 相对 O3 的延迟增量
  - 输出 tensor 理论大小
  - 是否出现 channel shape 的 XDNA 映射 cliff

暂时不会测：
  - Magpie 的 output Map / CPU stitch / GPU upload
  - learned 1.5x GPU decoder 的开销
  - 画质

所以这是第一关：“NPU 本身怕不怕 O3 -> O4？”
如果 O4 几乎不涨，再做下一步 Magpie C4 learned reconstruction 实机链路。

安装补丁
--------
把本补丁 ZIP 的内容直接复制/覆盖到：

  GameSR-XDNA-Architecture-Benchmark-v1.3 根目录

同一目录应该已经有：

  runtime\
  Benchmark-GameSR-v24-Sweep.ps1
  tools\int8_ptq.py

本补丁不会修改现有 105-model sweep，也不会删除原 results/cache。

第一步：在有 AMD Quark 的训练/Builder 环境生成 4 个模型
-----------------------------------------------------
在“之前成功生成 105 个 XINT8/QDQ sweep 模型”的那台机器/Python 环境中，双击：

  BUILD-GAMESR-OUTPUT-CHANNEL-PROBES.cmd

它会生成：

  GameSR-NPU-Output-Channel-Probe-Models-4-YYYYMMDD-HHMMSS.zip

如果脚本没找到正确 Python，可以先在 CMD 设置：

  set GAMESR_QUARK_PYTHON=D:\你的环境\.venv-quark\Scripts\python.exe

然后再运行 BUILD-GAMESR-OUTPUT-CHANNEL-PROBES.cmd。

第二步：掌机测试
----------------
把上一步生成的模型 ZIP 放到掌机现有 benchmark 根目录。

先快速测试：

  RUN-GAMESR-OUTPUT-CHANNEL-PROBE-QUICK.cmd

确认四个都是 NPU_ONLY 后，再正式测试：

  RUN-GAMESR-OUTPUT-CHANNEL-PROBE.cmd

也可以直接把模型 ZIP 拖到这两个 CMD 上。

正式测试默认：
  Warmup = 20
  Runs   = 600
  Cooldown = 2 sec

输出
----
results\ 中会新增：

  GameSR-NPU-Output-Channel-Probe-*-OUTPUT-CHANNELS.csv
  GameSR-NPU-Output-Channel-Probe-*-FULL.json
  GameSR-NPU-Output-Channel-Probe-*-SUMMARY.txt
  GameSR-NPU-Output-Channel-Probe-*-COMPACT.zip

把 COMPACT.zip 发回即可。

判断标准
--------
最关键看 O4 相对 O3：

  O4 ~ O3（例如只多 0.x ms）
    -> 很好。C4 feature 输出对 NPU 很便宜，可以继续做 learned 1.5x reconstruction。

  O4 明显跳高
    -> 最后一层输出 shape 本身触发了不理想的 XDNA/VitisAI 映射，需要重新设计输出编码。

O6/O8 的用途不是现在就拿去训练，而是帮助判断：
  - 输出成本是否近似线性增长；
  - 是否又存在类似 C10/C12、C18/C20 那种“对齐坏点”；
  - O8 是否反而是一个硬件甜点位。

公平性
------
Builder 对每个 O 变体都会重置同一个随机 seed，因此共同的 C16-L4 主干权重一致；
只有最后 Conv 的输出形状不同。测试模型是结构探针，不代表训练后画质。
