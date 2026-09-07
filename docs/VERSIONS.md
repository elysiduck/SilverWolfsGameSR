# GameSR 版本目录 / Version Catalog

> 集中说明 SilverWolfsGameSR 各**训练代码版本**、**架构扫描 / 探针工具**的作用与相互关系。
> 代码在 `training/`；Magpie / NPU 部署在 `magpie/`；历史迁移资料在 `docs/archive/`。
> **当前训练主线：`training/` 下的 CLO v1.0（C / L / O 三维自定义训练选择器）。**

版本号规则（见根 `README.md`）：

- 训练代码：`training-vX.Y.Z`
- Magpie：`magpie-v2.1-pipeline-vX`
- Benchmark：`benchmark-vX.Y`
- 模型：`model-c16l4-rX` / `model-c32l3-rX`

---

## 1. 训练代码主线（training-vX.Y.Z）

演进时间线：

```
v1.4  LR-residual 基线
  └ v1.4.2  可靠性热修 / 一键重导出 / ONNX 校验
v1.5  PC 端 INT8/QDQ + FP16 一站式流水线
  └ v1.5.1 / v1.5.3  INT8 PTQ 热修
v1.6  AMD Quark XINT8 替换通用 ORT PTQ（独立 .venv-quark）
  └ v1.6.1 / v1.6.2  补强
v1.7  Ryzen AI core-only 部署（NPU 只跑卷积核心，后处理交 Magpie）
  └ v1.7.1  首次真实 NPU 训练的可靠性修复
v1.7.2  C/L Selector（交互式自定义 C + L）
  └ v1.7.3  吞吐 / 启动优化
v1.7.4  RAM Streaming Prefetch（SMB/掌机数据集场景）
v1.7.5  Persistent RAM Cache + 自动剩余内存检测
  └ v1.7.5.1  EMA 续训热修
  └ v1.7.5.2  training_log.csv 旧格式兼容热修
v1.7.6  Validation RAM Cache（底座）
v1.7.7  Quality Diagnostics / 边际收益诊断（不改架构）
v1.8    Training Algorithm Upgrade（在线困难样本挖掘 + 损失课程 + 高频损失）
  └ v1.8.1  Game-Aware / Detail-Aware Training
  └ v1.8.2  Late Polish / Information-Aware Training
CLO v1.0  ⭐ C/L/O 三维选择器（在 v1.7.2 上加 O 维）
CLO v1.0.1  批处理中文编码修复（去中文启动器，自带 v1.3 tools/int8_ptq.py）
Phase-Aware 3×3 家族（独立于 training/current/ 的新分支，归档于 training/versions/）
v1.8.4  Phase-Aware 3×3 decoder（direct_rgb + 20% bilinear skip，schema v2）
  └ v1.8.5  Phase-Aware 3×3 RGB-residual（权重零初始化 ⇒ 第 0 轮=双线性，schema v3）
     └ v1.8.6  Stable Phase-Residual Training（增量补丁，training_contract v186，质量刹车）
        └ v1.8.7  Dataset Manager + Stable Training（增量补丁，training+dataset_contract v187）
```

### v1.4 / v1.4.1 / v1.4.2 — LR-residual 基线

- **v1.4**：默认架构 **LR-residual**——LR 分辨率做特征提取，只上采样 3 通道残差，再叠加双线性全局残差（比在 HR 做残差栈省约 scale² 卷积）。引入 benchmark 缓存 + 签名校验、缺失数据集清单自动准备、自动续训拒绝跨版本/跨配置 checkpoint。
- **v1.4.1**：与 v1.4 同内容补发说明。
- **v1.4.2**：可靠性热修。强制 UTF-8 安全 stdout/stderr；新增 `GameSR_Export.cmd` 一键从 `best.pth` 重导出；新增 `ONNX_Validate.cmd` / `validate_onnx.py`；`export_onnx.py` 支持 `--use-ema` / `--no-ema` 并校验 external data。

### v1.5 / v1.5.1 / v1.5.3 — PC 端一站式部署

- **v1.5**：把 PC 端 INT8/QDQ PTQ 与 FP16 转换并入训练流水线。一键流程 `benchmark → train → full test → FP32 ONNX → FP16 → INT8/QDQ → deployment bundle`；标定只在 PC，掌机端无需数据集 / Python / `.venv`。
- **v1.5.1**：修复外部数据 ONNX 上的静态 INT8/QDQ 失败（先在 Python 端做 shape inference 再量化自包含图）。
- **v1.5.3**：Windows / ONNX Runtime 上 INT8 PTQ 热修。

### v1.6 / v1.6.1 / v1.6.2 — AMD Quark XINT8

- **v1.6**：用 **AMD Quark `XINT8`** 替换通用 ORT PTQ 路径。根因：v1.5.3 的 handheld profile 把 14 个 `QLinearConv` 全落在 CPU，NPU 只跑极小子图且比 CPU 慢；Ryzen AI 的 XINT8 算子表不支持 `QLinearConv`。v1.6 拒绝发布含 `QLinearConv` 的图。引入独立 `.venv-quark`（PyTorch 2.11 CPU + Quark 0.12.post1 + ORT 1.23.2），与 CUDA 训练环境隔离，避免 Quark 依赖改动污染训练包。
- **v1.6.1 / v1.6.2**：小幅修复 / 补强。

### v1.7 / v1.7.1 — Ryzen AI core-only 部署

- **v1.7**：结构性突破。结构探针定位到完整输出拓扑（learned residual + 两条双线性 Resize + base bypass + Add + Clip）导致编译器分区失败；把这些后处理移出 NPU 后，720p 卷积核心可成为单一 NPU 子图。新增 NPU profile：`tiny-npu-fast`（8 层 C8，3947 参数，首档 ~60 FPS 候选）、`tiny-npu-quality`（14 层 C8，7451 参数）。`forward_core()` 只导出 `[1,3,720,1280]` 低分辨率 RGB residual，Magpie 负责后处理 `display = clamp(bilinear(lr,1.5) + bilinear(residual_lr,1.5), 0, 1)`，必须 half-pixel（`align_corners=False`）采样。导出 / Quark 校验拒绝含 `Resize` / `Add` / `Clip` 的图。
- **v1.7.1**：首次真实 NPU Fast/Quality 训练前的可靠性修复（原子写 `latest.pth`、自动续训拒绝不兼容 checkpoint、数据集清单校验引用文件、Collector V3.2 按 `game/session/image` 切分防相邻帧泄漏、ONNX/INT8 校验失败关闭等）。不改架构 / 契约。

### v1.7.2 — C/L Selector（CLO 前身）

- 增加**交互式自定义 NPU 架构训练**：输入 Channels C + Conv Layers L，生成可复现的 `configs/gamesr-c{C}-l{L}.yaml`；各 C/L 独立 `outputs/` / checkpoints / ONNX / INT8 / deployment 命名，互不覆盖；新增 `Train-Choose-C-L.cmd` 与 CLI `--channels/--layers/--epochs/--name`。这是 CLO v1.0 的直接前身。

### v1.7.3 / v1.7.4 / v1.7.5 / v1.7.5.1 / v1.7.5.2 — 吞吐 / 缓存线

- **v1.7.3**：吞吐 / 启动优化（batch benchmark 复用 pool 等）。
- **v1.7.4**：RAM Streaming Prefetch。针对“笔记本 RTX 4060 训练 + 数据集在掌机 / SMB 共享盘”场景，后台并发预取（默认窗口 256 张、24 读线程、6 解码线程、4 batch 队列）+ CUDA 异步 copy-stream 预取，减少 GPU 等数据。
- **v1.7.5**：Persistent RAM Cache + 自动剩余内存检测（无 psutil 依赖）。压缩源字节跨 epoch 保留在 RAM（保护 + 滚动-LRU 两级缓存、紧急驱逐），避免每轮重新从网络取同一批图；不创建 NVMe 副本。
- **v1.7.5.1**：EMA 续训热修——修复从 `latest.pth` 续训时 EMA tensor 留在 CPU 的 device 不匹配。
- **v1.7.5.2**：`training_log.csv` 旧格式兼容热修——原子重写、无法映射的多余列存入 `legacy_extra_N` 不静默丢弃。

### v1.7.6 / v1.7.7 — 验证缓存 + 质量诊断

- **v1.7.6**：Validation RAM Cache（验证集缓存底座）。
- **v1.7.7**：Quality Diagnostics / 边际收益补丁。**刻意不改**架构 / 损失 / EMA / 续训 ABI / 缓存，只把“训练到什么程度、继续有没有收益、比普通缩放强多少”测清楚：双线性基线 PSNR、Delta-PSNR、MSE 降幅%、验证 Edge Error、edge-strength ratio（1.0 为诊断中心非优化目标）、PSNR 分布（标准差 / P10 / 最小）、SSIM P10、边际收益跟踪（默认 5 epoch 窗口）、`quality_summary.json`。

### v1.8 / v1.8.1 / v1.8.2 — 训练算法升级线

- **v1.8**：Training Algorithm Upgrade（基于 v1.7.6，合入 v1.7.7 诊断）。**不增加部署层数 / 参数**，真实改变训练行为：① GPU 在线困难样本挖掘（hard 比例 15%→40%，hard mix 10%→35%）② 损失课程 Foundation/Detail/Refine（Edge 0.05→0.09、SSIM 0.02→0.035、轻量高频 0→0.06）③ 轻量高频重建损失（亮度高通残差，无 FFT/VGG/LPIPS，默认每 2 步）④ 退化课程（90%→100% by 25%）。config 兼容版本保持 1.7.4，使同 C/L/scale 的 v1.7.x checkpoint 可自动续训。
- **v1.8.1**：Game-Aware / Detail-Aware Training。`hard_score = 重建误差 × HR target detail factor`，让 UI / 字体 / 发丝 / 几何被重点学，而 DOF / Bloom / 柔焦不因误差高被误当 hard；hard gradient cap 1.50×；HF 用 HR detail mask（max 0.06→0.03）；游戏向退化（更少合成噪声 / 量化 / 色彩扰动）；加权 resize 采样（Bicubic 50% / Bilinear 40% / Area 10%）。不改部署图，同 C/L/scale checkpoint 可续训。
- **v1.8.2**：Late Polish / Information-Aware Training（覆盖式补丁，直接覆盖 `gamesr-local-training` 根）。约 50% 进度进入 late-polish、75% 完全抛光：HF→0.012、hard 15% / 9%；Information-Aware L1（低信息 patch 略降权、高信息 UI/文字/发丝略升权，0.75~1.25 限制）；黑 / 平 patch 抑制（黑底白字受方差 / detail 保护不被当黑边）；新增 Detail PSNR / oversharp / undersharp 等指标。Resume-safe 覆盖 v1.8.1 checkpoint。

### CLO v1.0 ⭐（当前训练主线）

在 **v1.7.2 C/L 选择器**基础上增加第三维 **O = NPU 输出通道数**，形成 C / L / O 三维自定义训练选择器。

- **入口**：`一键训练-选择C-L-O.cmd`（中文）/ `Train-Choose-C-L.cmd`（英文），均调用 `pipeline.py --model custom`。
- **交互输入**：C（中间通道数）、L（卷积层数）、O（输出通道数，默认 3）、Epochs（默认 100）。也可非交互：`pipeline.py --model custom --channels 16 --layers 4 --output-channels 4 --epochs 100`。
- **O 的三种模式**（`gamesr/model.py`）：
  - **O3**：`output_channels=3`，`decoder=None` —— 保留原 RGB residual + GPU Bilinear 路径（部署契约 `magpie_bilinear_residual_v1`）。完全兼容旧方案。
  - **O4 / O6**：`output_channels=4/6`，模型尾部追加一个 `nn.Conv2d(O, 3, 1)` 的 **learned 1×1 GPU decoder**（部署契约 `magpie_learned_1x1_decoder_v1`）。导出 ONNX 时额外生成 `<name>.onnx.decoder.json`，内含 decoder `weight`(3×O)、`bias`(3) 与合约公式 `display = clamp(bilinear(lr,1.5) + decoder_1x1(bilinear(latent_lr,1.5)), 0, 1)`，供后续 Magpie / HLSL 使用。
- **自动命名隔离**：profile 名 `gamesr-c{C}-l{L}-o{O}`，不同 C/L/O 的 config / checkpoint / 导出 / deployment 互不混用、不跨架构续训。
- **一键 8 段流水线**（`pipeline.py`，导出用 `--use-ema`）：`[0/8]` 环境检查 → `[1/8]` 数据集清单 → `[2/8]` batch 基准 → `[3/8]` 训练 → `[4/8]` 全量测试 → `[5/8]` ONNX 导出 → `[6/8]` FP16 → `[7/8]` AMD Quark XINT8/QDQ（O>3 时把 `--output-channels` 透传给量化）→ `[8/8]` deployment 打包。
- **首次建议**：C=16、L=4、O=4（或直接 O3 走原方案），Epochs=100（回车默认）。
- **已验证**：O3 / O4 / O6 前向计算正确（`gamesr/model.py`、`pipeline.py`、`export_onnx.py` 的 `onnx.decoder.json`、`int8_ptq.py` / `build_deployment.py` 的 O 透传）。完整训练与 INT8 编译待真实数据集 + 较长时间执行。
- 使用说明见 `training/CLO改造说明.txt` 与 `training/README_C_L_SELECTOR_CN.md`；帧率 / 结构探针背景见 `training/训练与帧率说明.md`。

### CLO v1.0.1 — 批处理中文编码修复

在 CLO v1.0 基础上修复**批处理中文编码问题**（见 `training/FIX-v1.0.1.txt`）：

- **去中文启动器**：原中文名 `一键训练-选择C-L-O.cmd` 在部分系统下因 GBK/UTF-8 编码与 `%~dp0` 尾部反斜杠导致 `--root` 路径解析失败；v1.0.1 改为纯英文 `Train-Choose-C-L-O.cmd`，并以 `%CD%` 传参避免尾部反斜杠问题。仓库现仅保留英文启动器。
- **自带 v1.3 `tools/int8_ptq.py`**：Builder 自包含，不再依赖外部 int8_ptq 路径。
- 其余（O3/O4/O6 模式、一键 8 段流水线、`onnx.decoder.json`、C/L/O 命名隔离、部署契约 `magpie_bilinear_residual_v1` / `magpie_learned_1x1_decoder_v1`）与 CLO v1.0 完全一致，未改架构 / 契约。
- 同步新增 3 份中文安装说明（`安装说明-V1.8.2.txt`、`覆盖安装说明-V1.7.7.txt`、`覆盖安装说明-V1.8.1.txt`）。

### v1.8.4 — Phase-Aware 3×3 decoder + 新损失

- **包**：`gamesr-local-training-v1.8.4-CLO-phase-aware-3x3-shared-env.zip`（`training/versions/v1.8.4-CLO-phase-aware-3x3/`）。
- **SHA256**：`7D377B71713CA2F21BFE6221C7602F16B1CD89C02A79D50F4AB9190CE20CDC9E`。
- 引入 **phase-aware 3×3 decoder**（九相位卷积 `(phase_y, phase_x, out_ch, in_ch, 3, 3)` = 27 路）：O4/O6/O8 自动走该 decoder，O3 保留原 `bilinear + 1×1` 兼容路径。
- 新增五类损失：`charbonnier_loss` / `ssim_loss`（`multi_scale_ssim_loss`）/ `edge_loss`（梯度）/ `high_frequency_loss`（频率）/ `ringing_loss`（振铃抑制），写入 `gamesr/data.py`。
- 直接学 **720p → 1080p** 重建（不再以双线性缩放结果作为唯一目标），保留 **20% bilinear base skip**（`base_skip=0.20`）避免初期崩坏。
- 模型名自动加 `-phase3x3`，不与旧 1×1 decoder 模型续训；支持向上查找共用 `.venv` / 数据集 / 环境。
- 导出 **schema v2** decoder JSON，含九组相位卷积核（`direct_rgb`，`base_skip=0.20`）。
- 梳理与检查见 `training/versions/REVIEW-v1.8.4-v1.8.7.md`。

### v1.8.5 — Phase-Aware 3×3 RGB-residual（零初始化）

- **包**：`gamesr-local-training-v1.8.5-CLO-phase3x3-residual-shared-env.zip`（`training/versions/v1.8.5-CLO-phase3x3-residual/`）。
- **SHA256**：`9DD052D34C343385A47C0FD1B4914F515AE7D6EB266CC7EF7B50589265B73C26`（与发布值一致）。
- 九相位 3×3 decoder 改为学习 **RGB residual**；`base_skip=1.0`；decoder 权重 / bias **全零初始化** ⇒ **第 0 轮输出与双线性逐像素完全一致**（已用真实 torch `torch.equal` 验证 `maxerr=0`）。
- MS-SSIM 改为**每步**计算；SSIM / 高频 / 振铃损失**渐入调度**；梯度损失从 **25% 权重逐步提高**到 100%。
- 恢复少量困难样本训练，混合权重控制在 **3%~8%**（`pipeline.py` O>3 分支覆盖 `min_mix=0.03/max_mix=0.08`）；退化强度从 **30%** 开始、训练进度 **65%** 才到完整强度（`start_strength=0.30/full_strength_at=0.65`）。
- 新模型目录自动用 `-phase3x3r-v2`；**checkpoint 校验加入 decoder 契约**（`checkpoint_signature` 含 `deployment`：`direct_rgb` ≠ `rgb_residual` 判异），即使手工同目录也阻止误加载旧模型。
- decoder JSON 升级 **schema v3**，`output:"rgb_residual"`。
- ⚠️ v1.8.4 的 `checkpoint_signature` 不含 `deployment` 字段（契约是 v1.8.5 才引入）；v1.8.4 仅靠前/后缀与"不指定旧目录"防误用。

### v1.8.6 — Stable Phase-Residual Training（增量补丁，叠加 v1.8.5）

- **包**：`GameSR-v1.8.6-Stable-PhaseResidual-Training-Patch.zip`（`training/versions/v1.8.6-CLO-stable-phaseresidual-patch/`）。
- **SHA256**：`074015E7AA3C5353D002BA8964AD04FFC61E455D70981A4B827585B255F61D7C`。
- 改动 5 个核心源文件（`gamesr/data.py`、`gamesr/utils.py`、`pipeline.py`、`train.py`、config），**未动 `model.py` 解码器** → 基线=双线性语义保持。
- `checkpoint_signature` 纳入 **`training_contract=phase_residual_stable_v186`**；与 v1.8.5（无该字段）判异 ⇒ 手工 `--resume` 旧 checkpoint 被 `SystemExit` 拒绝。
- 加入**质量刹车**（`update_quality_controller`）：edge 比例超硬阈值时梯度 / 高频损失归零、振铃损失不关闭；phase polish 调度。
- 自带 `verify_v186_patch.py` 静态校验全 PASS；真实 torch 复核：解码器基线=双线性（maxerr=0）、8×10→12×15、v1.8.6 签名 ≠ v1.8.5。
- 导出目录后缀 `-phase3x3r-v3`。

### v1.8.7 — Dataset Manager + Stable Training（增量补丁，叠加 v1.8.6）

- **包**：`GameSR-v1.8.7-Dataset-Manager-Stable-Training-Full-Patch.zip`（`training/versions/v1.8.7-CLO-dataset-manager-stable-patch/`）。
- **SHA256**：`2D0FE4ABDC0437A1743644C6AF547B076A3ACA2158EBBE27AB4AF0B9AFDF21ED`。
- 新增 **`prepare_dataset.py`（Dataset Manager）**：按 `game/session` 二级目录为**原子组**切分，保证会话绝不跨 Train/Val/Test；默认 **80/10/10**（`--val 0.1/--test 0.1`，`--min-width 1280 --min-height 720`）；支持 `--mode full`（重建）与 `incremental`。
- `gamesr/data.py` 的 SHA256 与 v1.8.6 **完全相同** → 稳定训练损失 / 课程逻辑未动，仅打包全套。
- `checkpoint_signature` 新增 **`dataset_contract`** 字段（与 `training_contract` 双保险）。
- 自带 `verify_v187_patch.py`：首次 `dataset_manager` smoke 失败仅因测试环境缺 `tqdm`，安装后全 PASSED；真实执行 `prepare_dataset.py` 40 会话 full rebuild ⇒ ≈80/10/10、组不跨集、`health` 无 leaked_groups。
- 导出目录后缀 `-phase3x3r-v4`。

### Magpie GameSR v4.3 — Phase3x3 + 模型子文件夹（NPU 推理端，叠加 v4.2.6 SIMD）

- **包**：
  - 源码补丁 `Magpie-GameSR-v4.3-Phase3x3-ModelFolders-source-patch.zip`（`magpie/patches/v4.3-phase3x3-modelfolders/`）。SHA256 `34706F7197DDE8655AC448019378A2CD2BA28E8A5877BB23F5EC9D9CD3246AA3`。
  - 运行包 `Magpie-GameSR-v4.3-Phase3x3-ModelFolders-runtime.zip`（296 MB，GitHub Release `magpie-v4.3` 资产，不入 git）。SHA256 `0A461A282EF13A48FBE86B98E37218EE59296D84523353BD3776BCD9049738E2`。
- 基于 **v4.2.6 全模型输入/输出 SIMD** 版改造，**完整保留** SIMD、当前 720p 直进 NPU、异步三槽流水线（BUILD ID：`magpie-v4.3-phase3x3-folders | SIMD-v4.2.6 | decoder-schema-v3 | recursive-models`）。
- 支持新 **`phase_aware_3x3_v1 + rgb_residual + schema v3`** decoder：O4 = **972 权重 + 27 bias**、O6 = 1458+27、O8 = 1944+27（权重布局 `phase_y,phase_x,rgb,input_channel,kernel_y,kernel_x`，3·3·3·O·3·3）。已实测运行包 `onnx.decoder.json` 该契约成立。
- **保留旧 O3 RGB residual**（不需 decoder.json）与旧 O4/O6/O8 `bilinear_then_1x1`（旧 12 权重 + 3 bias 仍可识别），向后兼容。
- **phase-aware HLSL decoder**（`cs_5_0`，嵌入 `SesrNpuProcessor.cpp` 并提供 `tools/GameSRPhaseDecoderTest.hlsl` 独立测试）：按 3/2 输出坐标选九相位、replicate 边界读 720p latent 3×3 邻域，最后 + 1.0× 双线性。与训练端 v1.8.5 解码器张量布局逐维对齐。
- `models/GameSR` 支持**任意层级子文件夹**；下拉框显示并保存**相对路径**；**阻止绝对路径与 `..` 越界**（`SesrNpuProcessor.cpp:233-250`）。
- 修复**首帧等待 NPU 时 decoder bias 污染双线性画面**：首帧用独立 baseline 着色器占位，decoder bias 不参与（`:1155`、`:1637`）。
- 运行包已含新模型：`models/GameSR/gamesr-c16-l4-o4-phase3x3r-v3-720p-to-1080p/`（int8-qdq.onnx + meta.json + onnx.decoder.json）与 `gamesr-c24-l3-o4-720p-to-1080p/`。
- 梳理与检查见 `magpie/REVIEW-v4.3-phase3x3-modelfolders.md`：契约与训练端 v1.8.5 schema v3 对齐，源码逐项核对属实；Release x64 编译 / HLSL `cs_5_0` 编译 / 设备端 NPU 缩放待用户解压后最终验证。

---

### Magpie GameSR v4.4 — Phase3x3 GPU 向量化（叠加 v4.3）

- **包**：
  - 源码 `Magpie-GameSR-v4.4-Phase3x3-Vectorized-full-source.zip`（`magpie/patches/v4.4-phase3x3-vectorized/`，9 文件，与 v4.3 同构的 GameSR 修改集，其中 3 个文件在 v4.4 更新）。SHA256 `29D4D09A0547FF75E43271A35AE583711D2541C5944E2CA7C25F2F9BCC37936B`。
  - 运行包 `Magpie-GameSR-v4.4-Phase3x3-Vectorized-runtime.zip`（296 MB，GitHub Release `magpie-v4.4` 资产，不入 git）。SHA256 `4BB0E16CA0CE228CAA672A16B82F9D08AFB8D9B2C472D356486969A6117582AC`。
- 基于 **v4.3**，**完整保留** SIMD-v4.2.6、异步三槽、模型子文件夹、**不修改 ONNX / 不重新训练 / 不重新量化**（运行包 `int8-qdq.onnx.meta.json` 明确 `source_model=...v1.8.5-CLO-phase3x3-residual...`，由 AMD Quark INT8 从训练 v1.8.5 导出）。
- **GPU 向量化（核心改动）**：启动时把 schema v3 权重**重排为 GPU `float4` 通道组**，`StructuredBuffer<float4>` 替代 `StructuredBuffer<float>`；一个 latent `float4` 邻域读取同时用于 R/G/B 三路输出，不再按 RGB / 单通道重复 Load。
  - O4 每个 1080p 输出像素：latent 纹理读取由 **108 次标量 Load → 9 次 `float4` Load**；权重读取由 **108 次标量 → 27 次 `float4`**（利用 GPU 参数缓存）。
  - O6 / O8 自动用**两个 `float4` latent 平面**（`gameSrLatentTextureCount=(OC+3)/4`：O4→1、O6/O8→2）；缺失通道补零。
  - bias 在权重块后追加为 9 个 `float4`（每相位 1 个，`.xyz` 取 3 bias，`.w=0`）。
  - 权重重排索引 `((phase*3+rgb)*OC+channel)*9+ky*3+kx` 与 HLSL `tapBase=((27*phase+3*ky+kx)*TC+t)+rgb*9*TC` 严格对应，单测着色器 `tools/GameSRPhaseDecoderTest.hlsl` 同步更新。
- **向后兼容完全保留**：`SesrNpuProcessor.cpp` 默认 `Legacy1x1`；O3 旧 `bilinear + identity residual` 路径、旧 O4/O6/O8 `bilinear_then_1x1`（12 权重 + 3 bias，schema v1）均可识别。运行包实测含 `gamesr-c24-l3-o4`（schema v1 `bilinear_then_1x1`）+ `gamesr-c16-l4-o4-phase3x3r-v3`（schema v3 `phase_aware_3x3_v1`，与 v4.3 / 训练 v1.8.5 逐维对齐）。
- **与 v4.3 的文件级差异**（v4.4 仅改 3 文件 + 更新说明，其余 5 文件与 v4.3 完全相同）：
  - `src/Magpie.Core/SesrNpuProcessor.cpp`：新增 float4 重排 + 嵌入 HLSL 改 `StructuredBuffer<float4>`、单趟 `PhaseResidual` 输出 RGB 三路（`dot` 累加）。
  - `src/Magpie.Core/Renderer.cpp`：仅 `_InitBackend` 日志 Build ID 串改为 `magpie-v4.4-phase3x3-vectorized | float4-decoder`。
  - `tools/GameSRPhaseDecoderTest.hlsl`：镜像上述着色器改动。
  - `BUILD_ID.txt` / `RELEASE_NOTES_v4.4_*` 更新；`ScalingModeItem.cpp` / `ScalingModesPage.xaml` / `GameSR-Models/README_CN.txt` 不变。
- 梳理与检查见 `magpie/REVIEW-v4.4-phase3x3-vectorized.md`：源码逐项核对属实、索引自洽、契约与训练 v1.8.5 schema v3 对齐（meta.json 实证）；Release x64 编译 / HLSL `cs_5_0` 编译 / 设备端 NPU 720p→1080p 实际加速与逐像素一致性待用户解压后最终验证。

---

### Magpie GameSR v4.4 — Runtime Perf Probe（运行版性能探针，无需编译）

- **包**：`Magpie-GameSR-v4.4-Runtime-Perf-Probe-patch.zip`（4 KB，纯脚本，入仓于 `magpie/patches/v4.4-runtime-perf-probe/`）。SHA256 `16603246338DD4958ED33E39CD7B7C9A6CE4F48F5DFDC1966B3C272EF1AA093D`。
- **定位**：直接覆盖 v4.4 **运行版**根目录，**不需要 Visual Studio / 重新编译**；双击 `RUN-GAMESR-RUNTIME-PERF-PROBE.cmd` 即用。
- 测量的是**现有 v4.4 链路**（`GameSR PIPELINE v4 ASYNC` 日志）：GPU→CPU、Input Pack、NPU、Output Pack、GPU Upload、GPU Fence、丢帧、旧结果复用、实际 fresh GameSR FPS，共 10 组样本。
- ⚠️ **不做** FP32 vs INT8 同帧 A/B（那需要源码版 Probe 重新编译，见下条 v4.5）。
- 已知并已修的坑：`%~dp0` 带尾部 `\` 传给 PowerShell 会触发 `GetFullPath … Illegal characters in path`；修正版改用 `%CD%` 并在 PS1 端清理引号 / 斜杠。

### Magpie GameSR v4.4 — FullSource RuntimeDeps 修复（补齐编译依赖 DLL）

- **包**：`Magpie-GameSR-v4.4-FullSource-RuntimeDeps-Fix.zip`（132 MB，GitHub Release `magpie-v4.4` 资产，**DLL 不入 git**）。SHA256 `D2A3010A1E1855E5770D539149FFAB36BF86954976AB895CDB99B141BD5F276A`。
- 根因：v4.4 full-source 打包时**漏掉** `third_party\sesr\onnx\bin\`，导致环境检查报 `[MISSING] … VitisAI runtime files`。
- 补 4 个 DLL：`onnxruntime.dll`、`onnxruntime_vitisai_ep.dll`、`dyn_dispatch_core.dll`、`aiecompiler_client.dll`。
- `onnxruntime.dll` SHA256 `919AC18B9AD61588C1D65EDBA775B012DA626FACD01623FEAB3A470042EAF4E3`，与 `Check-Environment.ps1` 要求的 Ryzen AI 1.7.1 一致。
- 本仓仅收录脚本部分（`README_FIX_CN.txt` + `VERIFY-RUNTIME-DEPS.cmd`）于 `magpie/patches/v4.4-fullsource-runtimedeps-fix/`，并附 `README.md` 说明 DLL 需从 Release 资产获取。

### Magpie GameSR v4.5 — One-Click Probe（源码补丁，画质 + 性能一键探针）

- **包**：
  - `Magpie-GameSR-v4.5-OneClick-Probe-source-patch.zip`（45 KB，初版）。SHA256 `4386CE5A80E691C1F723F83E50DD6D2A41032EF04ADD4A30AE42653A0745B7D6`。
  - `Magpie-GameSR-v4.5-OneClick-Probe-source-patch-FIXED.zip`（45.8 KB，**修正版 = 本仓归档的正式版**）。SHA256 `6F09308BFA2D2C1A7B41882E0D39A782F9375BBBE0F294C92B2A55AD851056BA`。
  - 归档于 `magpie/patches/v4.5-one-click-probe/`（覆盖 v4.4 full-source 的 `src/` 后重新编译）。
- **两版差异仅脚本路径**：FIXED 改 `RUN-GAMESR-PROBE.cmd` 传 `%CD%`（原 `%~dp0` 尾部反斜杠）+ `Finalize-Probe.ps1` 清理引号/斜杠 + 新增 `README_PATH_HOTFIX_CN.txt`；**`Renderer.cpp` 与 `SesrNpuProcessor.cpp` 两版完全相同**（C++ Probe 实现未动）。
- **相对 v4.4 是纯增量**：`SesrNpuProcessor.cpp` 2058 → 2504 行（**+447 行、仅删 1 行**）；v4.4 的 float4 向量化全部保留（`StructuredBuffer<float4>`、`LatentVectorAt`、`packedPhase` 均在），未回退。
- 新增 Probe 能力：`GameSrProbeTimestamp()`、`CompareFloatVectors()`（FP32 vs INT8 误差）、`CompareRgba8()`（画质比对）；由根目录 `GameSR-Probe.request` 文件触发，分画质 / 性能两个阶段。
- 自动输出 `GameSR-Probe-<日期时间>/`：`REPORT.txt`、`metrics.csv`、`perf*.csv`、`probe.log`；`REPORT.txt` 直接给 `[FAIL]/[WARN]` 结论（如 INT8 量化是否损伤残差、fresh 结果率能否撑 60 FPS、捕获后端是否为 GDI）而非原始日志。
- 检查项：**画质**（Bilinear / FP32 GameSR / INT8 GameSR / Zero Residual / 残差强度 / FP32-vs-INT8 误差 / Phase3x3 decoder 状态）+ **性能**（GPU→CPU、Input pack、NPU、Output pack、GPU upload、GPU fence、丢帧、旧结果复用、实际 fresh FPS、捕获后端）。

---

## 2. 相关工具家族（独立归档，未全部入仓）

- **架构扫描构建器** `GameSR-Architecture-Sweep-Builder` / `GameSR-Architecture-Sweep-Runner`（v1.0 / v1.0.1 / v1.0.2）：在目标掌机批量跑 NPU 架构组合（C / L / 层数）的延迟探针，产出候选延迟预算表，供“宽度 vs 深度”取舍。
- **NPU 架构扫描 / 基准** `GameSR-NPU-Architecture-Sweep`（v1.0）+ `GameSR-XDNA-Architecture-Benchmark`（v1.0）：XDNA NPU 架构基准与扫描。
- **NPU 输出通道探针** `GameSR-NPU-Output-Channel-Probe`（Models-4）/ `GameSR-NPU-Potential4-Output-Probe`（Models-16）：探测不同输出通道数的 NPU 行为与精度，产出 `handheld_payload` 与 `sweep_manifest`。**其补丁已入仓**：`training/GameSR-XDNA-v1.3-OutputChannelProbe-Patch-v1.0/`，说明见 `training/README_OUTPUT_CHANNEL_PROBE_CN.txt`。
- **数据集工具** `GameSR-Dataset-Collector` V3.2 / `GameSR-Cleaner` V3.2.1 StorageStats Patch：采集与清洗游戏截图数据集（Collector V3.2 按 `game/session/image` 切分防相邻帧泄漏，是 v1.7.x 训练数据干净度的前提）。
- **Magpie SESR 集成**：位于 `magpie/`（当前主线 Pipeline v4.1 AsyncNPU CurrentInput）。

---

## 3. `training/` 文件索引

- **启动器**：`Train-Choose-C-L-O.cmd`、`Train-Choose-C-L.cmd`、`GameSR_Train.cmd`、`Train-NPU-Fast.cmd`、`Train-NPU-Quality.cmd`、`Build-INT8.cmd`、`Build-Deployment.cmd`、`Build-FP16.cmd`、`GameSR_Export.cmd`、`ONNX_Validate.cmd`、`00_smoke_test.cmd`、`01_Install_RyzenAI_INT8.cmd`
- **核心代码**：`pipeline.py`（8 段流水线 + C/L/O 选择器）、`train.py`、`evaluate.py`、`export_onnx.py`、`int8_ptq.py`、`build_deployment.py`、`fp16_convert.py`、`prepare_dataset.py`、`validate_onnx.py`、`tools/int8_ptq.py`、`gamesr/{model,data,streaming,metrics,utils}.py`
- **配置**：`configs/*.yaml`（c8 / c12 / c16 / c32-l3 / npu-fast / npu-quality 预设，C/L/O 自定义生成 `gamesr-c{C}-l{L}-o{O}.yaml`）、`config/potential_backbones.json`
- **版本 / 使用说明**：`README_V1_*.md`、`RELEASE_NOTES_V1_*.txt`、`README_C_L_SELECTOR_CN.md`、`CLO改造说明.txt`、`训练与帧率说明.md`、`DEPLOYMENT_WORKFLOW.md`、`README_OUTPUT_CHANNEL_PROBE_CN.txt`、`README_PC_CN.txt`
- **输出通道探针补丁**：`GameSR-XDNA-v1.3-OutputChannelProbe-Patch-v1.0/`

> 不入仓：`.venv` / `.venv-quark` / `__pycache__` / `dataset` / `models` / `deployment` / `output_*` / `outputs` / `*.onnx` / `*.onnx.data` / `*.zip` / `*.rar`（见 `training/.gitignore` 与根 `README.md` 版本规则）。
