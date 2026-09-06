GameSR Potential-4 Output Probe v1.1.0 - PC端
============================================================

目的
----
基于 105 模型 C/L sweep，先剔除明显低效/被同档位甜点宽度压过的结构，只保留四个有代表性的潜力 backbone：

1) C8-L5   : tiny
2) C16-L4  : balanced / 当前重点
3) C24-L3  : wide+shallow / 105 sweep 中 C24 明显优于同档 C22
4) C32-L3  : high-capacity / 高有效吞吐

每个 backbone 只测试最后输出宽度：O3 / O4 / O6 / O8。
总计 4 x 4 = 16 个 INT8/QDQ 模型。

为什么不把 C22/C14/C30 等一起测
-------------------------------
这次不是再做完整 C/L 搜索，而是做“幸存 backbone 的 output head 搜索”。
例如原 105 sweep：
- C24-L3 10.4399 ms vs C22-L3 10.4446 ms，同时 C24 容量更高；因此不再给 C22 分配 output probe 预算。
- C16-L4 10.5284 ms vs C14-L4 10.5514 ms；同理保留 C16。
- C32-L3 11.1148 ms vs C30-L3 11.1313 ms；同理保留 C32。

PC端怎么用
-----------
1. 只在有 torch + onnx + amd-quark 的 PC 环境运行。
2. 推荐把本 PC 包解压到你之前成功生成 105 模型的 Quark 环境目录；也可以通过环境变量 GAMESR_QUARK_PYTHON 指向正确 python.exe。
3. 双击：
   BUILD-GAMESR-POTENTIAL4-OUTPUT-PROBES.cmd
4. 输出位于：
   output_potential4\GameSR-NPU-Potential4-Output-Probe-Models-16-*.zip
5. 只把这个生成的 Models-16 ZIP 拷到掌机 benchmark 根目录。

重要
----
- PC 包里没有掌机 Runner；这是刻意分开的。
- O3 也重新生成，确保每个 backbone 的 O3/O4/O6/O8 在共同层上使用同一随机 seed，避免拿旧 sweep 的不同随机权重作不干净比较。
- 这里只测试 NPU 图/runtime 成本，不代表实际画质模型，也不包含 learned 1.5x decoder。
