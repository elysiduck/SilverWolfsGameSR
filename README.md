# SilverWolfsGameSR

SilverWolfsGameSR 是一个面向游戏实时超分的实验项目，目标是在 AMD Ryzen AI / XDNA NPU 上运行轻量级神经网络超分模型，并与 Magpie 集成。

## 项目结构

- `training/`：训练、验证、导出、量化相关代码
- `magpie/`：Magpie NPU 集成与源码补丁
- `benchmark/`：NPU / GPU / 端到端性能测试
- `dataset-tools/`：采集、清洗、数据集处理工具
- `models/`：模型说明与发布索引
- `docs/`：架构、性能、实验记录和项目记忆

## 当前重点

- GameSR full-frame residual SR
- ONNX Runtime + VitisAIExecutionProvider
- QDQ INT8 NPU 推理
- Magpie Pipeline v4.1：异步 NPU worker / latest-frame-wins / 三槽流水线
- Current Input：NPU 直接接收当前捕获分辨率，不再强制把 1280×720 预缩放为 960×540
- 当前补丁：`magpie/patches/pipeline-v4.1.1-async-npu-current-input/`
- **训练主线 CLO v1.0.1**：`training/current/` 的 C/L/O 三维自定义训练选择器（O3 = RGB 残差 + GPU Bilinear；O4/O6 = learned 1×1 GPU decoder）。入口 `training/Train-Choose-C-L-O.cmd`（v1.0.1 修复批处理中文编码，移除中文启动器）。

## 当前源码基线

- Training：`training/current/`，**CLO v1.0.1**（在 v1.8.2 训练包基础上扩展 C/L/O 三维选择器；原 v1.8.2 覆盖式训练包内容已被 CLO 取代；v1.0.1 修复批处理中文编码并移除中文启动器）
- Dataset tools：`dataset-tools/collector-v3.2/`，Collector v3.2 + Cleaner v3.2.1
- Benchmark：`benchmark/xdna-v1.3/`，另含 105 结构 Sweep Builder/Runner 与小型结果摘要
- Magpie：`magpie/current/full-source/`，v4.1.1 Async NPU Current Input
- 历史 Magpie v2.1：`magpie/archive/v2.1/`

模型、checkpoint、厂商运行时和大型 benchmark 输出不进入 Git，见 [`ARTIFACTS.md`](ARTIFACTS.md)。

## 构建

统一构建入口和外部制品目录约定见 [`scripts/README.md`](scripts/README.md)。所有外部包均由 [`artifacts.lock.json`](artifacts.lock.json) 的 SHA256 锁定。

## 文档中心

GameSR 的历史迁移资料、数据采集/Cleaner、训练、Magpie/NPU 部署、Pipeline/Profiler 和上游参考资料已分类整理到 [`docs/README.md`](docs/README.md)。

各训练代码版本、架构扫描/探针工具家族的作用与相互关系见 [`docs/VERSIONS.md`](docs/VERSIONS.md)（版本目录）。

当前项目状态以本 README、`CHANGELOG.md`、`docs/PROJECT_MEMORY.md` 和当前补丁源码为准；`docs/archive/` 中的 v1.7、C8、Pipeline v3 等资料用于历史追溯，不代表当前主线。

## 版本规则

- Training: `training-vX.Y.Z`
- Magpie: `magpie-v2.1-pipeline-vX`
- Benchmark: `benchmark-vX.Y`
- Model: `model-c16l4-rX` / `model-c32l3-rX`

大型训练数据集、完整 Magpie 上游依赖树和全量 checkpoint 不直接提交到 Git 历史；正式 ONNX / 关键 checkpoint 优先通过 Release 或专门模型存储发布。
