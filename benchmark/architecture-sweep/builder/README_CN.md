# GameSR NPU Architecture Sweep Builder v1.0

这是 **笔记本端 Builder**。它直接接入现有 `GameSR-training-code-v1.7.1` 环境，不创建新的 Python/Quark 环境。

## 接入方式

把本补丁中的文件复制到你当前已经能训练/构建 INT8 的 `GameSR-training-code-v1.7.1` 根目录。复制后应看到：

```text
GameSR-training-code-v1.7.1\
  .venv\
  .venv-quark\
  gamesr\
  int8_ptq.py
  check_quark_env.py
  Build-NPU-Architecture-Sweep.cmd   <- 新增
  tools\architecture_sweep_builder.py <- 新增
```

然后双击 `Build-NPU-Architecture-Sweep.cmd`。

## 它复用了什么

- `.venv-quark\Scripts\python.exe`
- `check_quark_env.py`
- `gamesr.model.GameSR`
- 现有 `npu_sequential` 架构
- `int8_ptq.py`
- AMD Quark XINT8/QDQ

因此 Sweep probe 与 GameSR 正式 NPU 模型使用同一套模型定义和量化链路。

## 默认扫描

FULL（推荐）：

- Channels: `4,6,8,10,12,14,16,18,20,22,24,26,28,30,32`
- Layers: `2,3,4,5,6,7,8`
- 共 105 个组合

STANDARD：保留之前 66 组方案，方便与旧 Sweep 直接对照。

也可以在启动器中选择 CUSTOM。

## Calibration

默认只使用 8 张**固定、可重复的合成 720p 图像**做 Quark calibration。这些模型用于测 NPU 架构/编译器延迟，不用于比较画质，所以不需要消耗正式训练数据做 128 张重校准。

## 输出

```text
architecture_sweep\
  int8_models\           # Builder 工作区
  handheld_payload\      # 给掌机 Runner 使用的展开目录
    sweep_manifest.json
    sweep_manifest.csv
    models\...
  exports\
    GameSR-NPU-Architecture-Sweep-Models-*.zip
```

最终 `exports` 里的 ZIP 就是之后复制到掌机的**预构建模型包**。掌机不需要 Python / PyTorch / AMD Quark。

## 断点续跑

启动器默认带 `--skip-existing`。中途某个模型失败或关掉窗口后，再运行一次会验证并跳过已经成功生成的 INT8 模型，只继续缺失的组合。
