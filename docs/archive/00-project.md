# GameSR 文档归档：00-project

> 本文件由项目现有文档自动归档生成；每个原始文件均保留路径标题与完整文本内容。

---

## `docs/00-project/PROJECT_MEMORY.md`

# GameSR 项目重要记忆

## 1. 项目目标与硬件

- 目标：针对特定游戏训练轻量超分模型，把游戏内部约 $1280\times720$ 放大到 $1920\times1080$，优先部署到 Ryzen AI NPU，以降低 GPU 占用和掌机功耗。
- 目标设备：AMD Ryzen AI 9 HX 365 掌机，核显为 Radeon 880M；日常游戏功耗通常在约 $15\,\mathrm{W}$ 范围内。
- 实时链路：Magpie 捕获游戏画面，调用 ONNX Runtime / Vitis AI NPU，再把结果合成并显示。
- 核心取舍：先达到可用实时速度，再在不破坏 NPU 单子图覆盖的条件下提高画质。

## 2. 数据策略

- 只采集原生 $1080p$ 高质量截图即可。训练时从 HR 动态退化生成 LR，不需要同时截一份 $720p$。
- 游戏内截图应尽量关闭动态分辨率、FSR/DLSS/XeSS、插帧、动态锐化和会改变分辨率的选项；保持最高可承受的原生画质、固定分辨率和稳定 UI 比例。
- 数据要覆盖战斗、移动、站桩对话、剧情、室内外、昼夜、粒子、细线条、文字/UI 等。连续重复帧价值有限，场景多样性比单纯数量更重要。
- 已出现的一次数据状态：raw 约 $6500$ 张、clean $2081$ 张、review $2592$ 张。严格使用 clean 会提高平均质量，但意味着需要更长采集周期；review 用于人工复查边缘样本。
- raw、clean、review 若完整复制会占三份空间。更理想的实现是保留 raw 原图，并让 clean/review 使用索引、硬链接或移动策略；迁移前需确认 Cleaner 实际模式。
- 对话已推进 Cleaner `V3.1`，期望包含：进度条、既有数据迁移、清晰的 soft-frame/分类指标与更合理的存储方式。但本次导出没有找到 `V3.1` 成品，不能假定已经保存。

## 3. 当前训练主线：v1.7

当前应使用 `GameSR-training-code-v1.7.zip`。主要模型：

| 配置 | 部署核心 | 参数量 | 当前定位 |
|---|---:|---:|---|
| `tiny-npu-fast` | $8$ 层顺序 Conv，C8 | $3947$ | 第一优先，冲击约 $60\,\mathrm{FPS}$ 核心吞吐 |
| `tiny-npu-quality` | $14$ 层顺序 Conv，C8 | $7451$ | 画质对照，已知核心约 $47.9\,\mathrm{FPS}$ |

训练仍以完整 $1080p$ 图像为监督目标：

$$
\hat y=\operatorname{clamp}\!\left(B_{1.5}(x)+B_{1.5}(R_\theta(x)),0,1\right),
$$

其中 $R_\theta(x)$ 是网络在 $720p$ 空间预测的 RGB 残差，$B_{1.5}$ 是 $1.5$ 倍双线性放大。

部署时只导出 core-only ONNX，输出形状仍为 $[1,3,720,1280]$ 的低分辨率残差。Magpie 需要在 GPU shader 中完成：

$$
y=\operatorname{clamp}\!\left(B_{1.5}(x)+B_{1.5}(r),0,1\right),
$$

并匹配 PyTorch 的 `align_corners=False`，即 half-pixel 采样规则。

这样做的原因是把 `Resize`、基础图旁路、`Add` 和 `Clip` 移出 ONNX，避免 Vitis AI 编译器把图切碎或让主要算子回退 CPU。

## 4. 已确认的 NPU 结论

- 普通 ORT PTQ 产生的 `QLinearConv` 不适合 Ryzen AI XINT8 路径。旧模型中 $14$ 个 `QLinearConv` 全部落在 CPU，NPU 只执行很小的输出子图，结果甚至可能慢于 CPU。
- v1.6 起改用 AMD Quark `XINT8`，强制要求 `QLinearConv=0`，并检查 Q/DQ、zero-point 编码、自包含 ONNX 和哈希绑定 metadata。
- 已有一次通过的图检查：ONNX checker 通过，`QLinearConv: 0`，`Conv/Q/DQ: 14/24/52`，zero-point 编码包含 `INT8:0` 与 `UINT8:128`。
- 即使量化格式正确，完整输出拓扑仍会导致编译分区问题。三轮结构探针最终把问题定位到 learned residual 后的双 `Resize`、base-image bypass、`Add`、`Clip` 组合。
- 顺序卷积 core-only 图可以形成单一 NPU 子图。这是 v1.7 架构变更的直接依据。

## 5. 性能数字如何理解

结构探针在目标掌机和当时运行时上测得 $14$ 层连续卷积核心：

$$
t_{14}=20.8493\,\mathrm{ms},\qquad
f_{14}=\frac{1000}{20.8493}\approx47.934\,\mathrm{FPS}.
$$

$2$ 层探针约为 $9.184\,\mathrm{ms}$。用粗略线性插值估算 $8$ 层：

$$
t_8\approx 9.184+6\times\frac{20.8493-9.184}{12}
\approx15.02\,\mathrm{ms},
$$

$$
f_8\approx\frac{1000}{15.02}\approx66.6\,\mathrm{FPS}.
$$

结论：$47.9\,\mathrm{FPS}$ 是 $14$ 层 core-only 的实测值；$60\,\mathrm{FPS}$ 是 $8$ 层快版的设计目标，不是硬件理论上限。合理验证区间为核心约 $55$--$65\,\mathrm{FPS}$，但必须以 Quark 量化后在掌机的实测为准。

这些数字都不包含完整 Magpie 开销。最终帧时间近似为：

$$
t_{\mathrm{frame}}=t_{\mathrm{capture}}+t_{\mathrm{transfer,in}}+t_{\mathrm{NPU}}+t_{\mathrm{transfer,out}}+t_{\mathrm{shader}}+t_{\mathrm{sync}}.
$$

旧完整图/旧管线曾出现约 $6\,\mathrm{FPS}$、约 $30\,\mathrm{FPS}$ 等结果，不能代表连续卷积核心的上限。日志也出现过约 $33.3\,\mathrm{ms}$ 的完整帧处理和约 $24.8\,\mathrm{ms}$ NPU 段，说明捕获、切片、上传和分区方式会显著改变最终表现。

## 6. 当前操作步骤

### 在训练 PC 上

1. 解压 `GameSR-local-training-env-v1.1.zip`，安装隔离环境。
2. 把 `GameSR-training-code-v1.7.zip` 的内容覆盖到原 `gamesr-local-training` 项目目录。
3. 保留已有 `dataset`、`.venv` 和 `.venv-quark`。
4. 确认 `dataset/train.csv`、`dataset/val.csv`、`dataset/test.csv` 存在。
5. 双击 `00_smoke_test.cmd`，必须看到通过结果。
6. 若 Quark 从未成功安装，运行 `01_Install_RyzenAI_INT8.cmd`。
7. 先运行 `Train-NPU-Fast.cmd`。首次 batch 测试时直接按 Enter 使用最快安全值。
8. 训练中断后可再次启动，从 `outputs/tiny-npu-fast/latest.pth` 续训；最佳权重在 `best.pth`。
9. 完成后目标文件为 `deployment/tiny-npu-fast/tiny-npu-fast-720p-to-1080p-int8-qdq.onnx`。
10. 之后再按需运行 `Train-NPU-Quality.cmd` 做画质/速度对照。

已有 checkpoint 时，可运行 `Export-INT8-NPU-Fast.cmd` 或 `Export-INT8-NPU-Quality.cmd` 重新导出，不必重训。成功输出必须包含：

- `[PASS] QLinearConv: 0`
- `[PASS] Core-only graph: Resize/Add/Clip = 0/0/0`
- `[OK] Ryzen AI INT8 model`

### 在掌机上

1. 使用独立 NPU benchmark 包和匹配的 Ryzen AI / Vitis AI runtime。
2. 先测 core-only XINT8 ONNX，检查是否为单一 NPU 子图，并记录首帧、稳态中位数、P95 和 NPU 利用率。
3. 不要把 core-only ONNX 直接当完整图片显示；它输出的是残差。
4. 完成 Magpie shader 后处理后，再测端到端帧率和功耗。

## 7. 接下来的优先级

1. 继续采集并严格清洗覆盖面更广的 $1080p$ 数据；目前已经接近“主要去采集并训练”的阶段。
2. 找回或重建 Cleaner `V3.1`，确认迁移、进度条和存储策略。
3. 训练 `tiny-npu-fast`，完成 Quark XINT8 导出，先验证核心速度和 NPU 覆盖。
4. 修改 Magpie：输入原图与 residual 共用高效路径，将双线性放大、残差相加、裁剪融合为单个 GPU shader；复用缓冲区并减少 CPU/NPU/GPU 间复制和同步。
5. 测端到端延迟。若核心已达标但总帧率低，优先优化 capture/transfer/shader pipeline，而不是继续删卷积层。
6. 再训练 `tiny-npu-quality` 比较画质。若仍需提升，在部署算子不变的前提下研究可重参数化训练、蒸馏或更好的损失，而不是贸然加入 NPU 不友好的算子。
7. 多帧超分理论上可提高时序稳定性与细节，但会增加光流/对齐、缓存、延迟和 NPU 编译风险。当前单帧链路尚未完成端到端优化，因此不是近期优先项。

## 8. 文件对应关系

| 文件 | 用途 | 状态 |
|---|---|---|
| `GameSR-training-code-v1.7.zip` | 当前训练、导出、Quark XINT8、core-only 部署 | 主线 |
| `GameSR-local-training-env-v1.1.zip` | Windows 隔离训练环境 | 主线配套 |
| `GameSR_Dataset_Collector_V2_6.zip` | 最近可恢复的采集/清理工具 | 临时保留，非 V3.1 |
| `GameSR-Magpie-SESR-v2.1-Pipeline-v3-fixed3.zip` | 早期 Magpie SESR 管线修正版 | 参考，尚非 v1.7 core-only 完整集成 |
| `GameSR-RyzenAI-NPU-Benchmark-v2.4.zip` | 掌机独立 NPU benchmark/runtime 包 | 测试主工具 |
| `GameSR-Benchmark-v2.4.1-Hotfix.zip` | benchmark hotfix | 与 v2.4 配套 |
| `GameSR-NPU-Compatibility-Probe-v1.zip` | 算子/量化兼容性探针 | 诊断工具 |
| `GameSR-NPU-Structure-Probe-v1.zip` | 输出拓扑/连续卷积结构探针 | 诊断工具 |
| `probe.zip`, `probe2.zip`, `probe3.zip` | 三轮用户返回的探针结果 | 关键原始证据 |
| `results-int8*.zip`, `packed1.zip` | INT8 和打包模型测试结果 | 关键原始证据 |
| `GameSR-1.5x-Models-ShapeFix-v3.zip` | 早期 $1.5\times$ 模型形状修复 | 历史参考 |
| `GameSR_Quickstart_Colab.ipynb` | 最早的 $27$ 图训练链路验证 | 仅历史参考 |

## 9. 迁移到新对话时的建议开场

上传 `PROJECT_MEMORY.md`、训练代码 `v1.7` 和当前正在处理的结果文件，然后说明：

> 继续 GameSR 项目。以 PROJECT_MEMORY.md 为事实基线，当前主线是 v1.7 core-only XINT8。请先核对我最新一次掌机 benchmark 结果，再决定是优化模型结构还是 Magpie 端到端管线。不要退回 QLinearConv PTQ，也不要把 core-only 残差直接作为最终画面。

---

## `docs/00-project/README_先看这里.md`

# GameSR / FSR 项目迁移包

导出日期：2026-09-04（UTC）

这个包保留了项目目前最重要、可继续工作的版本，以及用于证明 NPU 性能瓶颈的原始测试结果。迁移到新账号后，先上传本文件和 `PROJECT_MEMORY.md`，再按需要上传对应压缩包。

## 推荐阅读顺序

1. `PROJECT_MEMORY.md`：项目目标、已确认结论、当前状态、下一步和关键命令。
2. `01_core/GameSR-training-code-v1.7.zip`：当前训练与部署主线。
3. `01_core/GameSR-local-training-env-v1.1.zip`：Windows 本地隔离环境安装器。
4. `01_core/GameSR_Dataset_Collector_V2_6.zip`：目前能够恢复到文件的采集/清理工具版本。
5. `03_npu_tools/` 与 `04_results/`：NPU 基准工具、结构探针和原始结果。
6. `02_magpie/`：旧版 Magpie 管线补丁参考；core-only 模型仍需新的 Magpie shader 后处理集成。

## 重要缺口

- 对话里已经讨论并要求制作 `GameSR_Dataset_Collector_V3.1`，但导出时可访问文件中没有找到其成品 ZIP。包内保留 `V2.6` 作为最近可恢复版本，不能视为 `V3.1`。
- 用户本机上的完整 Magpie 源码目录、训练数据集、`.venv`、`.venv-quark`、模型 checkpoint（如 `best.pth`）不在项目文件库中，因此未包含。
- 部分用户上传附件在文件库中使用 UUID 名称，未能可靠判定用途的文件没有盲目加入主线包。

## 目录说明

- `01_core/`：训练环境、训练代码、数据工具。
- `02_magpie/`：Magpie SESR 管线补丁。
- `03_npu_tools/`：独立 benchmark、hotfix、兼容性/结构探针和早期模型。
- `04_results/`：用户在目标掌机上返回的 benchmark、INT8 与结构探针结果。
- `05_legacy_reference/`：早期 Colab 验证 Notebook，仅供追溯，不建议作为当前训练入口。

## 不必继续保留的旧版

本迁移包没有重复收录训练代码 `v1.0` 到 `v1.6.2`、Collector `V1` 到 `V2.5` 以及多次 Magpie fixed 中间版。它们已经由当前主线或最终修正版取代；若要审计历史，原项目文件库仍有这些文件。

---

## `docs/00-project/SHA256SUMS.txt`

```text
efbed9ce7aace7da91100c19c1fdd164f4b0736486810e4feefd8516461cc1c8  GameSR-project-export/01_core/GameSR-local-training-env-v1.1.zip
21a6c1e94b8a5fa1ce449bbd48f0ece89c652fa34d7b993ceb3ee88e4f115ef3  GameSR-project-export/01_core/GameSR-training-code-v1.7.zip
229d686003d1782fe8e965d0581f0a44b079322e793b183fb407436f4e32e8b0  GameSR-project-export/01_core/GameSR_Dataset_Collector_V2_6.zip
1a91e1062e685303a47382d1ff44aeb44e38c819d1566b32e7bd110c612b554a  GameSR-project-export/02_magpie/GameSR-Magpie-SESR-v2.1-Pipeline-v3-fixed3.zip
2c2ff3240c5209afd523de0ec278f95e98a9240af5d8719f07a753e804a0fe8b  GameSR-project-export/03_npu_tools/GameSR-1.5x-Models-ShapeFix-v3.zip
7e10f7fc3fb7fbc9571e0a930e2f4ae63763c4267cdf4665d72460d04eaa9a96  GameSR-project-export/03_npu_tools/GameSR-Benchmark-v2.4.1-Hotfix.zip
134bf6cecde01af37968ef1d631a557b2b9fc2b2d85088472e377d482dbd3cc9  GameSR-project-export/03_npu_tools/GameSR-NPU-Compatibility-Probe-v1.zip
83e1b1b0717db9088b6135d1ed29686a4a6f5377b05ee508e261fdde810c20cb  GameSR-project-export/03_npu_tools/GameSR-NPU-Structure-Probe-v1.zip
4c7840f87f6b3a6b8fa2ec27f12b07fe3934787d1a6c9c19cd7a72f37595f1a6  GameSR-project-export/03_npu_tools/GameSR-RyzenAI-NPU-Benchmark-v2.4.zip
7629cf3db5424741cda545b1899a4485e6cb1976bfb16cf6a3cf712e52b4ca50  GameSR-project-export/04_results/packed1.zip
d675d4c7fe413e218aa19118465d1d5915875f54e375c7590fe451a525043c41  GameSR-project-export/04_results/probe.zip
33807db76924dda075ef1062614947ca0a35175765a783ee6e99f27299fa962a  GameSR-project-export/04_results/probe2.zip
69849dc77a87c00ad5f658430a7067a1ff419490c1905aa0e680075952db6214  GameSR-project-export/04_results/probe3.zip
d0e6ea62ef477aa382687a9b3efd30a1f629e96d7fa98829b4d49d0fe7d782bc  GameSR-project-export/04_results/results-int8(1).zip
4e46a647b1110853625ff07f395562d7f177282e482f83cb6a9a32bb98dfab57  GameSR-project-export/04_results/results-int8-2.zip
d0e6ea62ef477aa382687a9b3efd30a1f629e96d7fa98829b4d49d0fe7d782bc  GameSR-project-export/04_results/results-int8.zip
899558d1ec31c747c70079b739fbab71c5ca8c6892dff6ab0a2a220ef769938d  GameSR-project-export/04_results/results.zip
cf686c82ba580ac33792a4204ec991f4370f5c26c2f8d0ac3e34968ab89bd8d2  GameSR-project-export/05_legacy_reference/GameSR_Quickstart_Colab.ipynb
6a78f5a889c5f28af05a76dc04f738852740265b2b61b75744999f2938c6927b  GameSR-project-export/PROJECT_MEMORY.md
8dcbc15bded30de1a55ed6e301c2cf7472431787557b22487ed36858de7eb3be  GameSR-project-export/README_先看这里.md
df3495f54a23031e03d9ed96a174020657cc8778d02e111378e511c829203cb8  GameSR-project-export/SHA256SUMS.txt
```
