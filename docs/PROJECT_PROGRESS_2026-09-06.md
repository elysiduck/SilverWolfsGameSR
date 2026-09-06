# GameSR 项目进度总览

> 整理日期：2026-09-06  
> 文档定位：项目事实基线、版本盘点、问题清单与下一阶段执行路线  
> 信息来源：项目历史对话、当前项目空间中的代码/补丁/日志，以及 2026-09-04 项目迁移包

---

## 1. 一句话结论

GameSR 已经完成了从 **1080p 游戏数据采集 → 清洗 → 训练 → ONNX/Quark XINT8 导出 → Vitis AI NPU 执行 → Magpie 720p→1080p 实时合成** 的完整技术闭环；当前主要问题已经不是“能否运行”，而是：

1. **画质尚未稳定超过 FSR1 / Lossless Scaling LS1**；
2. **端到端性能开销仍明显高于 GPU 空间超分**；
3. **训练目标与实际游戏 720p 输入分布仍不够一致**；
4. **v1.7.4～v1.8.1、Magpie v4.1.1 等后续迭代尚未形成统一、可审计的项目归档**。

因此，项目当前处于：

> **技术可行性已验证，产品竞争力尚未验证。**

下一阶段不应该盲目继续扩大同一种训练，而应同时推进两条主线：

- 用部署一致的数据和大模型容量探针确认“画质上限”；
- 继续压缩 Magpie/NPU 的搬运、打包和同步开销，确认“性能下限”。

---

## 2. 项目目标与目标硬件

### 2.1 目标

针对游戏画面训练轻量 AI 超分模型，将约 `1280×720` 的游戏输入放大为 `1920×1080`，优先把神经网络计算放到 Ryzen AI NPU，以达到：

- 比普通双线性/双三次插值更好的细节恢复；
- 尽量接近或超过 FSR1、LS1 的观感；
- 降低 GPU 超分负担，给游戏渲染留出更多 GPU 资源；
- 在掌机约 10～15 W 的功耗环境中保持可用帧率和稳定帧时间。

### 2.2 主要设备与软件栈

| 项目 | 当前信息 |
| --- | --- |
| 目标设备 | AMD Ryzen AI 9 HX 365 掌机 |
| GPU | Radeon 880M |
| NPU | AMD Ryzen AI / XDNA NPU |
| 捕获与显示 | Magpie |
| 推理接口 | ONNX Runtime 1.23.3（现有日志） |
| NPU EP | VitisAIExecutionProvider |
| INT8 工具链 | AMD Quark XINT8 / QDQ |
| 训练设备 | NVIDIA GPU 训练服务器/电脑 |
| 训练数据访问 | 通过局域网/SMB 从数据服务器读取，使用 RAM 缓存与预取 |

---

## 3. 当前实际部署架构

### 3.1 GameSR 不是让 NPU 直接生成完整 1080p 图像

当前 GameSR NPU core 的输入和输出均为：

```text
输入：1×3×720×1280 RGB
输出：1×3×720×1280 RGB residual（残差）
```

完整 1080p 结果由 Magpie/GPU 合成：

```text
720p 原图 ──half-pixel bilinear 1.5×──┐
                                       ├── Add + Clamp ──> 1920×1080
720p NPU residual ─bilinear 1.5×───────┘
```

数学形式为：

\[
\hat y=\operatorname{clamp}\left(B_{1.5}(x)+B_{1.5}(R_\theta(x)),0,1\right)
\]

其中：

- `x` 是 720p 输入；
- `Rθ(x)` 是 NPU 在 720p 空间预测的 RGB 残差；
- `B1.5` 是符合 PyTorch `align_corners=False` 的 half-pixel 双线性采样；
- 训练损失仍然在最终 1080p 合成结果上计算。

### 3.2 为什么采用 core-only residual

早期完整输出 ONNX 把 `Resize`、原图旁路、`Add` 和 `Clip` 一并放进模型，导致 Vitis AI 分图或算子回退，NPU 无法高效执行。

结构探针确认：

- 顺序卷积 core-only 图可以形成单一 NPU 子图；
- 普通 ORT PTQ 生成的 `QLinearConv` 不适合目标 Ryzen AI XINT8 路径；
- 当前导出必须满足 `QLinearConv = 0`，并检查 Q/DQ、静态输入形状和模型哈希；
- 后处理移到 GPU 后，NPU 只负责连续卷积残差预测。

这条部署边界已基本固定，后续模型研究应尽量不破坏它。

---

## 4. 各子系统进度

| 子系统 | 当前状态 | 完成度判断 |
| --- | --- | --- |
| 数据采集 | Collector V3.2 已形成一键采集流程 | 已完成基础能力 |
| 数据清洗 | Cleaner V3.2 + V3.2.1 空间统计补丁 | 已完成并可继续使用 |
| 数据审计 | JSON/CSV/HTML/contact sheet 报告 | 已完成基础能力 |
| 本地训练环境 | v1.1 隔离 `.venv`、CUDA PyTorch 安装与检查 | 已完成 |
| 训练/续训 | 自动 batch、EMA、checkpoint、Auto Resume | 已实现；后续版本继续修复 |
| ONNX 导出 | FP32/FP16/Quark XINT8 QDQ | 已实现 |
| NPU 图验证 | 静态 720p、QLinearConv=0、core-only 检查 | 已实现 |
| NPU 独立 benchmark | 已有 v2.4/hotfix/结构与兼容性探针 | 已使用，但历史结果较混乱 |
| Magpie 集成 | 已从旧 SESR 管线推进到 GameSR v4.1.1 异步流水线 | 已跑通 |
| 720p→1080p 合成 | GPU half-pixel base + residual 合成 | 已跑通 |
| 实际游戏画质 | C16-L4、C32-L3 已进行视觉测试 | 未达到目标 |
| 实际游戏性能 | 可运行约 50+ FPS，但开销高于 FSR/LS1 | 未达到目标 |
| 统一版本库 | GitHub 本地 clone 曾显示与 `origin/main` 一致 | 后续版本是否全部入库未确认 |

---

## 5. 数据采集与清洗进度

### 5.1 当前数据策略

采集端只需要保存高质量原生 `1920×1080` 截图。训练时动态生成 720p LR，不需要日常同时采集两套分辨率。

建议的游戏设置已经明确：

- 1920×1080、Render Scale 100%；
- 关闭 DLSS/FSR/XeSS 超分；
- 关闭动态分辨率、插帧和会改变渲染比例的选项；
- 光追、Bloom、景深等真实游戏效果可以保留，并需要覆盖；
- 数据应覆盖战斗、移动、UI/文字、室内外、昼夜、粒子、细线条、暗场和高亮场景。

历史上曾记录的数据规模为：

| 分类 | 历史记录数量 | 说明 |
| --- | ---: | --- |
| raw | 约 6500 | 原始采集图 |
| clean | 2081 | Cleaner 判定的训练候选 |
| review | 2592 | 需要人工复查的边缘样本 |

这些是历史快照，不应当作当前实时数量。

### 5.2 Collector V3.2

已实现：

- 采集与离线清洗分离；
- 实时清洗进度、速度、ETA 与分类计数；
- 增量处理；
- 跨游戏、跨 Session 的历史全局去重；
- dHash + pHash 双指纹；
- 曝光、对比度、熵、细节、黑边、偏色等质量指标；
- KEEP / REVIEW / REJECT 三层分类；
- 数据集审计报告及人工抽查 contact sheet；
- 按 `game/session/image` 组织数据，降低相邻帧同时进入训练集和验证集的泄漏。

### 5.3 Cleaner V3.2.1

V3.2.1 补丁没有改变 KEEP/REVIEW/REJECT 判断，主要补充存储统计：

- 优先为 clean/review 创建硬链接；
- 硬链接失败时才退回复制；
- 统计 `hardlinks_created`、`copies_created`、节省空间和复制额外占用；
- Audit 会检查现有文件的硬链接、普通复制和缺失状态。

这符合当前项目取舍：**不建立永久 SSD 图片缓存，也尽量避免 raw、clean、review 三份完整副本占用磁盘。**

### 5.4 仍存在的数据问题

目前最大的画质风险并不是图片数量，而是训练 LR 与实际部署输入的差异：

- 训练 LR 主要由 1080p HR 用 bicubic/bilinear/area、轻量噪声、模糊、锐化等人工退化生成；
- 实际游戏的 720p 画面包含引擎 TAA、锐化、HUD、动态效果、材质 LOD、后处理和采样历史；
- “1080p 缩小到 720p”不完全等于“游戏引擎原生渲染 720p”。

因此目前模型可能很擅长恢复人工降采样图，却没有学到部署时真正遇到的 720p 图像分布。

---

## 6. 训练代码演进与当前版本混乱

### 6.1 当前项目空间中可直接核验的版本

当前附件可以完整核验到 `GameSR-training-code-v1.7.1-fixed.zip`。

v1.7.1 的核心配置：

| 配置 | 通道/层数 | 参数量 | 设计定位 |
| --- | --- | ---: | --- |
| tiny-npu-fast | C8 / 8 Conv | 3947 | 快速核心验证 |
| tiny-npu-quality | C8 / 14 Conv | 7451 | 画质对照 |

v1.7.1 已验证：

- 5 份 YAML 配置通过校验；
- Fast/Quality 参数量独立核对通过；
- Session 不跨 train/val/test；
- Python 文件可编译；
- checkpoint 原子写入；
- 不兼容 checkpoint 不会错误 Auto Resume；
- 导出拒绝错误模型/错误配置；
- NPU 打包对静态形状、Quark XINT8、core-only 图执行 fail-closed 检查。

### 6.2 后续已讨论/使用，但本次附件中没有完整归档的版本

| 版本 | 已知定位 | 当前可核验程度 |
| --- | --- | --- |
| v1.7.4 | RAM Streaming Prefetch；网络预取、解码线程、batch 队列、pinned RAM、CUDA copy stream | 有历史记录，当前无 ZIP |
| v1.7.5 | 跨 epoch 的压缩源字节 RAM Cache、LRU/内存压力淘汰、预热 | 有历史记录，当前无 ZIP |
| v1.7.5.1 | EMA Resume Hotfix | 仅能由版本名/历史对话确认 |
| v1.7.5.2 | Training Log Hotfix | 仅能由版本名/历史对话确认 |
| v1.7.7 | Quality Diagnostics Patch | 仅能由版本名/历史对话确认 |
| v1.8 | Training Algorithm Upgrade Patch | 仅能由版本名/历史对话确认 |
| v1.8.1 | Game-Aware Training Patch；C32-L3 当前训练曾使用此版本 | 有训练记录，当前无 ZIP |

这意味着：**实际训练主线已经超过 v1.7.1，但项目空间里最晚可审计的完整训练包仍是 v1.7.1。**

### 6.3 训练吞吐优化结果

远程 SMB 读取曾是训练的主要瓶颈。记录中一轮约：

- epoch 训练墙钟约 58～59 秒；
- 数据等待约 50 秒；
- GPU stage 约 8～9 秒；
- 推算 GPU stream busy 约 14%～15%；
- 网络读取约 41～42 MiB/s；
- RAM Cache 约 3.05 GiB，命中率约 53%。

这说明当时训练并非算力不足，而是主要受网络读取、解码和 batch 构建限制。v1.7.4/v1.7.5 的预取与 RAM Cache 正是为此设计。

当前明确不采用“把所有远程图片永久缓存到本机 SSD”的方案，因为占用空间过大。可接受方向是：

- 只使用内存级压缩源缓存；
- 自动按剩余内存控制上限；
- 跨 epoch 复用并按压力淘汰；
- 不在 SSD 中永久保留数据副本。

---

## 7. 模型训练与画质结果

### 7.1 C16-L4

已知状态：

- 已训练完成；
- 独立测试图数量：208；
- 最终记录的 GameSR PSNR：`35.417 dB`；
- 已导出并接入 Magpie 的模型名记录为 `gamesr-c16-l4-720p-to-1080p-int8-qdq.onnx`；
- 实际游戏观感没有明显超过 FSR1 或 LS1。

训练过程还出现过 `PSNR 37.85 dB / SSIM 0.9821` 一类验证指标。该值与最终 208 图测试结果不应直接横向比较，因为验证集、退化方式、EMA 权重或评估入口可能不同。

### 7.2 C32-L3

最近可确认的训练记录：

- 使用 v1.8.1；
- epoch 20：PSNR `33.49 dB`，SSIM `0.9717`，相对基线 ΔPSNR `+0.80 dB`；
- epoch 54：PSNR `34.07 dB`，SSIM `0.9730`，相对基线 ΔPSNR `+1.38 dB`，MSE 改善约 `33.4%`；
- 模型也已被用于实际 Magpie 画质/性能比较；
- 实际观感仍不理想，与 C16-L4 各有优劣，没有形成稳定领先。

当前资料不足以严谨确认 C32-L3 的最终 epoch、最终独立测试表和最佳 checkpoint 文件，因此不能把 epoch 54 记录写成最终成绩。

### 7.3 当前画质判断

目前用户实际观察是：

- C16-L4 与 C32-L3 都没有达到原生 1080p；
- 与 FSR1、LS1 相比只是有来有回，难以说稳定胜出；
- 模型容量变化没有解决核心问题；
- 单看 PSNR/SSIM 较高并不等于游戏实际观感更好，尤其是 UI 文字、细线条、动态边缘、锐化感和时序稳定性。

这表明下一步不能只继续追逐验证集 PSNR，而需要加入部署匹配测试、细节/边缘指标和固定场景主观 A/B。

---

## 8. NPU 与 Magpie 部署进展

### 8.1 必须区分的两条历史链路

#### 旧 SESR-M7 v2.1 Exact 1080p

当前项目附件中的 `Magpie-SESR-M7-NPU-Exact1080p-source-v2.1.zip` 是旧 SESR 路线：

- 捕获 1280×720；
- GPU 先缩到 960×540；
- NPU 以 x2 方式输出 1920×1080；
- 12 个切片；
- 日志显示所有节点由 Vitis AI EP 执行。

它不是当前 GameSR C16-L4/C32-L3 的最终部署架构，只能作为历史实现与性能参考。

#### 当前 GameSR v4.1.1

后续历史记录显示当前完整源码主线已推进到：

`Magpie-SESR-GameSR-v4.1.1-AsyncNPU-CurrentInput-full-source`

其关键变化是：

- 1280×720 输入不再预缩放到 960×540；
- NPU 直接接收 1280×720 NCHW；
- 输出同尺寸 full-frame residual；
- GPU 用 half-pixel 1.5× 放大 base/residual 并合成为 1920×1080；
- 使用 PIPELINE v4 异步 worker；
- Current Input 与 Exact 1080p 输入契约已经校准；
- C16-L4 日志显示 ONNX 3 个节点全部进入 VitisAIExecutionProvider。

### 8.2 Pipeline 优化历程

已实现或测试过：

- Pipeline v3：3-slot 异步 readback、非阻塞完成检查、优先处理最近完成帧、加速 CPU tile merge；
- Profiler Round 2：拆分 copy、Map 等待、CPU 切片、NPU、CPU 拼接、Unmap、捕获丢帧；
- Pipeline v4：独立异步 NPU worker，允许渲染与 NPU 工作部分重叠；
- full-frame residual 路径，减少旧切片拓扑对 GameSR 的不匹配；
- GPU 上执行 1.5× 合成后处理。

### 8.3 日志证据

2026-09-04 旧 SESR v2.1 日志显示：

- 输入 960×540，输出 1920×1080；
- 12 切片；
- 所有节点放置到 VitisAI EP；
- 较稳定阶段 NPU 约 15.3～16.6 ms；
- Process 墙钟约 19.6～21.8 ms；
- 较早一段 NPU 可达 17.5～21.7 ms，Process 墙钟约 22～27 ms；
- CPU 切片约 1～2 ms，CPU 拼接约 2.5～4.8 ms；
- 捕获阶段本身通常很小，仅偶发跳过旧帧。

后续 GameSR 日志记录过：

- Pipeline v3 墙钟约 16.14～16.19 ms、NPU 约 9.12～10.14 ms（特定测试配置）；
- Pipeline v4 样例：render 3.33 ms、GPU→CPU 0.55 ms、input-pack 4.05 ms、NPU 13.96 ms、output-pack 2.41 ms、residual upload 2.74 ms；
- 这些阶段通过异步重叠后不能简单相加，但 input-pack、NPU、output-pack/上传依然是主要成本。

### 8.4 当前端到端体验

用户的实际对比记录：

| 模式 | 实际帧率记录 | 观感结论 |
| --- | ---: | --- |
| 游戏原始输出 | 约 74 FPS | 基准 |
| FSR / LS1 | 约 67 FPS | 开销较小 |
| GameSR NPU | 约 50+ FPS | 开销明显更大，画质未形成优势 |

另外，曾有一次锁定 45 FPS 的测试，GameSR 能够跑满 45 FPS；这只能证明它达到了锁帧目标，不能说明实际余量充足。

NPU 运行时频率约 800 MHz，但占用率没有达到 100%。这不必然意味着 NPU 没有工作；日志已证明节点确实进入 Vitis AI。更合理的解释是模型/内存搬运/调度没有让阵列持续饱和，或整条流水线由 NPU 外阶段限制。

---

## 9. 当前已经确认的成功与失败

### 9.1 已确认成功

- 原生 1080p 游戏数据采集和离线清洗流程可用；
- 数据集 Session 级拆分降低相邻帧泄漏；
- Windows 隔离 CUDA 训练环境可部署；
- 训练、EMA、Auto Resume、最佳 checkpoint 与任意 checkpoint 再导出路径已具备；
- Quark XINT8/QDQ 导出可生成 NPU 候选模型；
- core-only 模型可避免旧完整图的分区问题；
- C16-L4/C32-L3 能在 Ryzen AI NPU 上通过 Magpie 实际运行；
- 720p 输入契约和 GPU half-pixel 1.5× 合成已校准；
- Magpie 异步 NPU worker 已工作；
- 项目证明了“游戏 AI 超分放到 NPU”在工程上可行。

### 9.2 已确认未达标

- C16-L4/C32-L3 画质没有稳定超过 FSR1/LS1；
- GameSR 端到端开销明显高于 FSR/LS1；
- PSNR/SSIM 与真实游戏观感存在偏差；
- 训练数据退化与实际部署 720p 输入仍不完全一致；
- NPU core 与整机流水线之间仍存在明显搬运、打包、上传与同步成本；
- 当前版本散落在多个 ZIP、补丁、本机目录和 GitHub 之间，缺乏单一事实源。

### 9.3 已知失败案例

- 旧 `QLinearConv` PTQ 模型出现卷积落 CPU、NPU 只执行小子图的问题；
- 旧完整输出拓扑中的双 Resize、base bypass、Add、Clip 导致 Vitis AI 分图；
- Benchmark 中存在 int8/vitisai FAIL、统计项为空和 PowerShell RemoteException 的历史结果；
- 早期 Magpie Pipeline 补丁出现过括号/结构体结束位置错误，后由 fixed3 修复；
- 个别 C/L 结构存在硬件映射异常，不能只按参数量推算 NPU 性能。

---

## 10. GitHub 与版本管理状态

用户本机曾执行：

```text
git status
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean
```

这只能证明**当时本地仓库与远端 main 一致**，不能证明所有后续 ZIP 都已经提交。

当前存在明显归档缺口：

- 当前附件最新训练代码是 v1.7.1，但实际使用已到 v1.8.1；
- 当前附件 Magpie 完整源码是旧 SESR v2.1，但实际主线已到 GameSR v4.1.1；
- v1.7.4～v1.8.1 的补丁、C16-L4/C32-L3 模型与训练记录没有在本次空间中形成统一归档；
- 仅凭 `git status` 无法确认这些内容是否存在于其他分支、Release 或尚未提交的本机目录。

因此，GitHub 当前应视为：

> **仓库本身可正常同步，但是否覆盖最新项目状态尚未完成审计。**

建议把“整理仓库”设为 P0，而不是继续产生更多独立 ZIP。

---

## 11. 当前项目空间文件盘点

### 11.1 唯一内容

| 文件 | 用途 | 当前地位 |
| --- | --- | --- |
| `GameSR-Cleaner-V3.2.1-StorageStats-Patch.zip` | Cleaner 硬链接/复制空间统计 | 当前数据工具补丁 |
| `GameSR_Dataset_Collector_V3_2.zip` | 数据采集、清洗、审计 | 当前可核验数据工具 |
| `GameSR-training-code-v1.7.1-fixed.zip` | 训练/导出可靠性修复 | 当前可核验训练包 |
| `GameSR-training-code-v1.7.zip` | core-only XINT8 主线基础版 | 历史基础版 |
| `GameSR-local-training-env-v1.1.zip` | Windows 隔离训练环境 | 当前配套环境 |
| `GameSR-Magpie-SESR-v2.1-Pipeline-v3-fixed3.zip` | 旧 Magpie Pipeline v3 修复 | 历史参考 |
| `GameSR-Magpie-SESR-v2.1-ProfilerPatch-v2.zip` | 旧管线细分性能日志 | 历史诊断工具 |
| `Magpie-SESR-M7-NPU-Exact1080p-source-v2.1.zip` | 旧 SESR M7 完整源码/运行时 | 历史参考，不是当前 GameSR v4.1.1 |
| `GameSR-FSR-project-export-2026-09-04.zip` | 迁移包、旧 benchmark 与探针结果 | 历史证据集合 |
| `64a2a6fb-346f-4b32-a230-d6ff71df32ad.log` | 旧 SESR v2.1 Pipeline v3 性能日志 | 已分析 |

### 11.2 完全重复附件

当前空间中以下文件各有重复副本，内容哈希相同：

- `Magpie-SESR-M7-NPU-Exact1080p-source-v2.1.zip`；
- `GameSR-training-code-v1.7.zip`；
- `GameSR-Magpie-SESR-v2.1-Pipeline-v3-fixed3.zip`；
- `GameSR-Magpie-SESR-v2.1-ProfilerPatch-v2.zip`；
- `GameSR-local-training-env-v1.1.zip`；
- 两份 2026-09-04 项目导出包。

---

## 12. 最关键的技术判断

### 12.1 继续只扩大同类小模型，收益可能有限

C16-L4 与 C32-L3 已证明：单纯在通道数和层数之间换算，并没有带来稳定画质优势。原因可能包括：

- 训练退化与真实部署输入不匹配；
- 小模型容量不足；
- 损失函数更偏平均误差，不够关注游戏细节；
- 模型学到的是人工降采样逆过程，而非游戏引擎 720p→原生 1080p 的差异；
- INT8 量化进一步压缩了细节表达；
- 单帧模型无法解决 TAA 闪烁或历史信息缺失。

### 12.2 必须先做容量探针

应先训练一个不考虑 NPU 实时性的较大教师/容量探针，例如 C64-L8（NVIDIA GPU 训练和推理即可）。目的不是部署，而是回答：

> 在现有数据与退化策略下，更大模型能否明显超过 FSR/LS1？

- 如果大模型仍然不行，问题主要在数据/目标，不应继续缩小部署模型；
- 如果大模型明显更好，说明任务可学，再通过蒸馏、可重参数化或信息感知训练压回 C16-L4 一类结构。

### 12.3 必须做部署匹配数据实验

建立 30～50 个固定静态场景：

1. 同一机位、同一游戏状态分别采集原生 1080p 与原生 720p；
2. 尽量固定 UI、相机、动画、时间和随机效果；
3. 对无法严格配准的样本人工剔除；
4. 将其作为小型 deployment-matched 验证/微调集；
5. 与当前 synthetic degradation 训练做 A/B。

这不要求立刻替换全部数据集，但可以快速判断“部署不一致”到底占多大比例。

### 12.4 性能优化应针对流水线，不只针对卷积层数

Pipeline v4 的记录中，`input-pack 4.05 ms`、`output-pack 2.41 ms`、`residual upload 2.74 ms` 已经与 NPU 的 `13.96 ms` 同属重要成本。

下一步性能工作应优先验证：

- 是否仍存在整帧 GPU→CPU→NPU 与 NPU→CPU→GPU 往返；
- input/output pack 是否能减少复制、转换或合并；
- 缓冲区是否真正长期复用；
- 当前 async worker 是否存在等待最新帧或过期 residual；
- 是否能用更低成本的纹理上传与后处理融合；
- NPU 推理是否可以与游戏渲染和上一帧合成更充分重叠；
- 是否能减少日志模式、同步点和不必要的 CPU 格式转换。

---

## 13. 下一阶段执行路线

### P0：先统一版本与证据

1. 找回并归档训练代码 v1.7.4～v1.8.1；
2. 归档 Magpie GameSR v4.1.1 完整源码；
3. 保存 C16-L4/C32-L3 的 YAML、best/latest checkpoint、FP32 ONNX、INT8 ONNX、量化 sidecar、训练日志和测试报告；
4. 核对 GitHub `main`、其他分支和 Releases；
5. 为每个版本写 changelog，不再只以 ZIP 文件名管理；
6. 建立 `CURRENT.md` 指向唯一当前版本。

### P1：冻结可重复基线

对 C16-L4、C32-L3、FSR1、LS1 使用完全相同的固定场景：

- 相同 720p 输入；
- 相同 1080p 输出截图；
- 相同游戏设置和锐化参数；
- 记录 PSNR、SSIM、LPIPS/感知指标、Detail/Edge 指标；
- 保存局部裁剪图和盲测结果；
- 记录平均 FPS、1% low、P95/P99 帧时间、NPU 时延与功耗。

### P2：部署匹配 A/B

1. 建立 30～50 场景的 720p/1080p 配对小集；
2. 先只作为验证集判断 domain gap；
3. 再做小规模 fine-tune；
4. 与原 synthetic degradation 结果比较；
5. 根据结果调整 resize、锐化、TAA/噪声和 UI 权重。

### P3：训练大模型容量探针

1. 训练 C64-L8 或类似 GPU 教师模型；
2. 先不考虑 NPU 部署；
3. 检查它能否在同一固定测试集上明显超过 FSR1/LS1；
4. 若能，研究蒸馏到 C16-L4/C32-L3；
5. 若不能，优先重做数据和目标，而非继续堆训练 epoch。

### P4：继续砍 Magpie/NPU 流水线开销

1. 基于 Pipeline v4 日志确定 pack/upload/sync 的真实重叠关系；
2. 分别记录首帧、稳态 median、P95、P99；
3. 关闭详细日志后复测；
4. 以“原始 74 FPS、FSR/LS1 67 FPS”为现实基准；
5. 目标不是 NPU 利用率 100%，而是在相同画质下把端到端 FPS 拉近 60～67 FPS。

### P5：最后再考虑多帧和单游戏专模

多帧模型理论上可以利用前两帧提高稳定性与细节，但会增加：

- 帧缓存；
- 运动对齐/光流；
- 遮挡处理；
- 延迟；
- NPU 算子兼容风险；
- 训练数据配对难度。

当前单帧链路仍未达到画质/性能目标，因此多帧不是近期优先级。按游戏训练专模可能有价值，但应先证明统一模型在部署匹配数据上确实可学，否则只会放大维护成本。

---

## 14. 建议的阶段验收标准

### 画质验收

- 固定测试集上明显优于 bilinear/bicubic；
- 对 FSR1/LS1 至少在 UI、细线、角色轮廓或纹理中的两个类别形成稳定优势；
- 不以单一 PSNR 作为结论；
- INT8 相对 FP32 的退化有单独报告；
- 连续游戏 10～20 分钟无明显闪烁、重影或锐化异常。

### 性能验收

- 1280×720→1920×1080 真实游戏链路稳定；
- 所有目标卷积进入 Vitis AI EP；
- 无持续队列堆积或大量过期帧；
- 报告平均/P95/P99，而非只报单帧；
- 端到端帧率至少接近 60 FPS；
- 相对原始游戏的损耗显著缩小，并接近 FSR/LS1 的损耗水平。

### 工程验收

- GitHub 中存在唯一主线源码；
- 训练、导出、部署配置可追溯；
- 模型文件与配置/commit/hash 绑定；
- 新电脑可按 README 复现训练与导出；
- 新掌机环境可按 README 复现 Magpie 部署与 benchmark。

---

## 15. 当前推荐决策

现在最合理的项目顺序是：

1. **先整理并同步版本库**，防止继续丢失 v1.8.1/v4.1.1 等最新成果；
2. **冻结 C16-L4/C32-L3 基线**，停止凭肉眼和零散日志反复比较；
3. **做 deployment-matched 小实验**，验证训练目标是否错位；
4. **训练大模型容量探针**，判断数据和任务的画质上限；
5. **并行继续优化 Pipeline v4 的 pack/upload/sync**；
6. 只有当大模型证明画质可学后，再做蒸馏或新一代 NPU 小模型。

换句话说：

> 当前不应直接开始第三个类似 C16-L4/C32-L3 的小模型。先回答“数据能不能学出明显优势”和“流水线最低能压到多少开销”这两个问题，项目才会真正向前推进。

---

## 16. 文档可信度说明

本文件将信息分为三类：

- **附件可核验**：来自当前 ZIP、README、源码结构或日志；
- **项目历史记录**：来自此前训练、部署和测试对话；
- **待确认**：最新文件没有进入当前项目空间，无法从文件本身复核。

尤其需要注意：

- v1.7.1 是当前可核验训练包，不等于实际最新训练代码；
- SESR-M7 v2.1 是当前可核验 Magpie 完整源码，不等于实际最新 GameSR 部署；
- 实际最新已知主线是训练 v1.8.1、Magpie GameSR v4.1.1；
- C32-L3 的 epoch 54 指标是最近可确认记录，不应冒充最终测试成绩；
- GitHub 是否已包含全部后续版本仍需以仓库文件和 commit 审计为准。
