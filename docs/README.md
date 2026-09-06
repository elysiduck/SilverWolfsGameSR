# GameSR 文档中心

本目录集中保存 SilverWolfsGameSR 的项目文档、历史迁移资料、数据工具说明、训练/部署说明和性能诊断记录。

> **当前项目状态以仓库根目录 `README.md` 与 `CHANGELOG.md` 为准。**  
> `docs/archive/` 中很多内容来自 2026-09-04～09-06 的迁移包和阶段性版本，包含旧的 C8 / v1.7 / Pipeline v3 等历史设计，不应覆盖当前 C16L4 / Pipeline v4 主线。

## 分类

### 1. 项目历史与迁移

- [`archive/00-project.md`](archive/00-project.md)
  - PROJECT_MEMORY
  - 项目迁移包 README
  - SHA256SUMS

适合追溯 GameSR 从早期 tiny-C8、Quark XINT8、core-only residual 到当前路线的演进过程。

### 2. 数据采集与 Cleaner

- [`archive/10-dataset.md`](archive/10-dataset.md)
  - Dataset Collector V3.2
  - Cleaner V3.2.1 StorageStats Patch
  - Collector / Cleaner requirements

### 3. 训练系统

- [`archive/20-training/part-00.md`](archive/20-training/part-00.md)
- [`archive/20-training/part-01.md`](archive/20-training/part-01.md)
- [`archive/20-training/part-02.md`](archive/20-training/part-02.md)
- [`archive/20-training/part-03.md`](archive/20-training/part-03.md)

覆盖：

- 本地隔离训练环境 v1.1
- Training v1.4 → v1.7.1 发布历史
- AMD Quark / Ryzen AI XINT8
- core-only ONNX 部署契约
- NPU Fast / Quality 旧模型说明
- smoke test、验证、导出、deployment workflow

### 4. Magpie / NPU 部署

- [`archive/30-deployment/part-00.md`](archive/30-deployment/part-00.md)
- [`archive/30-deployment/part-01.md`](archive/30-deployment/part-01.md)

包括：

- 训练模型接入 Magpie
- Ryzen AI runtime 版本绑定
- SESR v1.2 / v1.7 / v1.8 / v1.9 / v2.0 诊断和性能模式
- Exact 1080p 路径
- NPU 利用率、宽浅网络与流水线优化思路

### 5. Pipeline / Profiler / 性能

- [`archive/40-performance/README.md`](archive/40-performance/README.md)

包括：

- Pipeline v3 fixed2
- Profiler Round 2
- Map 等待、CPU 切片、NPU、拼接、捕获旧帧等性能指标说明

原始大体积运行日志建议进入 `benchmark/`、Release 或 CI Artifact，而不是长期混在说明文档里。

### 6. 历史版本

- [`archive/70-history/training-v1.7.md`](archive/70-history/training-v1.7.md)

只保留和 v1.7.1 当前归档内容不同的 v1.7 文档，避免重复。

### 7. 版本目录（训练代码 / 工具家族）

- [`VERSIONS.md`](VERSIONS.md)

逐版本作用与说明的总索引：训练代码主线 v1.4 → v1.8.2 与 **CLO v1.0**（C/L/O 选择器），以及架构扫描构建器、NPU 架构扫描 / 输出通道探针、数据集工具等独立工具家族的角色与相互关系；并附 `training/` 文件索引。

### 8. 上游参考

- [`reference/UPSTREAM_MAGPIE.md`](reference/UPSTREAM_MAGPIE.md)
- [`reference/LICENSES.md`](reference/LICENSES.md)

Magpie 通用说明与第三方许可证属于上游资料，不和 GameSR 自有设计文档混在一起。

## 文档使用规则

1. **当前事实**：优先看根目录 `README.md`、`CHANGELOG.md` 和当前源码。
2. **历史原因/设计演变**：看 `docs/archive/`。
3. **第三方 Magpie 行为**：看 `docs/reference/` 中的上游索引和对应 upstream 文档。
4. **Benchmark 原始数据**：放 `benchmark/`，不要把几万行日志当操作手册。
5. **模型与 checkpoint**：不直接塞进 docs；正式 ONNX / checkpoint 优先使用 Release 或专门模型存储。

## 来源快照

本轮归档主要来自：

- `GameSR-FSR-project-export-2026-09-04.zip`
- `GameSR_Dataset_Collector_V3_2.zip`
- `GameSR-Cleaner-V3.2.1-StorageStats-Patch.zip`
- `GameSR-training-code-v1.7.1-fixed.zip`
- `GameSR-training-code-v1.7.zip`
- `GameSR-local-training-env-v1.1.zip`
- `gamesr-local-training-CLO-v1.0.zip`（当前训练主线：C/L/O 三维选择器）
- `Magpie-SESR-M7-NPU-Exact1080p-source-v2.1.zip`
- `GameSR-Magpie-SESR-v2.1-Pipeline-v3-fixed3.zip`
- `GameSR-Magpie-SESR-v2.1-ProfilerPatch-v2.zip`
- `训练模型接入Magpie.txt`

归档时优先保留最新版本；完全相同的旧版文档不重复复制。
