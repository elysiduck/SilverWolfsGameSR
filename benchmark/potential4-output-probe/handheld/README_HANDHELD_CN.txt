GameSR Potential-4 Output Probe v1.1.0 - 掌机端
============================================================

这个包只包含掌机 Runner，不包含 PC Builder、PyTorch、Quark 或量化脚本。
请覆盖到现有：GameSR-XDNA-Architecture-Benchmark-v1.3-105patched 的 benchmark 根目录。

需要的四个 backbone
------------------
- C8-L5
- C16-L4
- C24-L3
- C32-L3

每个测试 O3/O4/O6/O8，共16个模型。

使用
----
1. PC 端先运行单独的 PC 包，生成：
   GameSR-NPU-Potential4-Output-Probe-Models-16-*.zip
2. 把那个 Models-16 ZIP 放到掌机 benchmark 根目录。
3. 先运行：
   RUN-GAMESR-POTENTIAL4-OUTPUT-PROBE-QUICK.cmd
4. Quick 全部 NPU_ONLY 后，再运行：
   RUN-GAMESR-POTENTIAL4-OUTPUT-PROBE.cmd
5. 最终上传 results 下生成的：
   GameSR-NPU-Potential4-Output-Probe-*-COMPACT.zip

输出内容
--------
- ROWS.csv：16个模型逐项结果
- LATENCY-MATRIX.csv：4个 backbone x O3/O4/O6/O8 延迟矩阵
- DELTA-MATRIX.csv：每个 backbone 内相对自身 O3 的百分比变化
- FULL.json
- SUMMARY.txt
- COMPACT.zip

重要
----
- O3 基线按“同一个 backbone”计算；不会再出现拿 C8-O4 对 C16-O3 这种错误比较。
- 这是纯 NPU 结构探针。仍然不包含 Magpie 的 output Map / CPU stitch / GPU upload / learned 1.5x decoder 成本。
- 掌机包里没有 PC 生成器，这是刻意拆开的。
