# SilverWolfsGameSR

SilverWolfsGameSR 是一个面向游戏实时超分的实验项目，目标是在 AMD Ryzen AI / XDNA NPU 上运行轻量级神经网络超分模型，并与 Magpie 集成，实现 720p → 1080p 的实时 AI 超分。

## 项目结构

- `training/`：训练、验证、导出、量化相关代码
- `magpie/`：Magpie NPU 集成与源码补丁
- `benchmark/`：NPU / GPU / 端到端性能测试
- `dataset-tools/`：采集、清洗、数据集处理工具
- `models/`：模型说明与发布索引
- `docs/`：架构、性能、实验记录

## 当前重点

- C16L4 GameSR 720p → 1080p residual SR
- ONNX Runtime + VitisAIExecutionProvider
- QDQ INT8 NPU 推理
- Magpie Pipeline v4：异步 NPU worker / latest-frame-wins / 多槽流水线

## 版本规则

- Training: `training-vX.Y.Z`
- Magpie: `magpie-v2.1-pipeline-vX`
- Benchmark: `benchmark-vX.Y`
- Model: `model-c16l4-rX` / `model-c32l3-rX`

大型训练数据集和全量 checkpoint 不直接提交到 Git 历史；正式 ONNX / 关键 checkpoint 优先通过 Release 或专门模型存储发布。
