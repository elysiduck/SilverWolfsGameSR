GameSR XDNA Architecture Benchmark v1.3
Output-Channel Probe 补丁 v1.0.1 FIX
===========================================

本修复解决 v1.0 Builder 的 Windows 路径解析错误。

你遇到的错误形如：
  ...gamesr-local-training" --outputs 3,4,6,8 --skip-existing\tools\int8_ptq.py

原因：
  BUILD 脚本把 %~dp0 直接作为带引号参数传给 Python。%~dp0 永远以反斜杠结尾，
  在 Windows argv 解析中，末尾反斜杠会干扰闭合引号，导致后面的 --outputs 等参数
  被吞进 --root 字符串。

v1.0.1 FIX 做了两项修改：
  1. 先 cd 到脚本目录，再使用无尾反斜杠的 %CD% 作为 --root。
  2. 补丁内直接附带原 v1.3 benchmark 的 tools\int8_ptq.py。
     因此 Builder 可以放在训练环境目录单独运行，不再强制要求旁边已有 benchmark quantizer。

生成模型：
  双击 BUILD-GAMESR-OUTPUT-CHANNEL-PROBES.cmd

仍然测试四个结构：
  C16-L4-O3
  C16-L4-O4
  C16-L4-O6
  C16-L4-O8

如果自动找到的 Python 不对，在 CMD 先执行：
  set GAMESR_QUARK_PYTHON=D:\你的环境\.venv-quark\Scripts\python.exe

成功后会生成：
  GameSR-NPU-Output-Channel-Probe-Models-4-YYYYMMDD-HHMMSS.zip

把这个模型 ZIP 复制到掌机上“原 v1.3 benchmark 根目录”。
掌机端 Runner 仍需要原 benchmark 的 runtime、Benchmark-GameSR-v24-Sweep.ps1 等文件。
然后运行：
  RUN-GAMESR-OUTPUT-CHANNEL-PROBE-QUICK.cmd
确认 NPU_ONLY 后再运行：
  RUN-GAMESR-OUTPUT-CHANNEL-PROBE.cmd

注意：
Builder 可以在训练环境目录运行；但 RUN-*.cmd / Runner-GameSR-OutputChannels.ps1
必须覆盖到完整的 GameSR-XDNA-Architecture-Benchmark-v1.3 根目录后再在掌机执行。


[v1.0.2 掌机修复]
如果 v1.0.1 在 Expand-Archive 阶段报 Remove-Item 找不到 C16-L4-O3-int8-qdq.onnx，
这是 Windows PowerShell 5.1 在较长解压路径/Force 覆盖路径上的 Archive 模块问题。
v1.0.2 改用根目录下的短缓存 _ocp\<hash>，并在临时目录完整解压后再移动，
不再使用 Expand-Archive -Force。无需重新生成 O3/O4/O6/O8 模型 ZIP。
