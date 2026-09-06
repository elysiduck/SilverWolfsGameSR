# GameSR 训练代码 v1.8.4 → v1.8.7 梳理与检查报告

> 审查范围：用户交付的 4 个训练代码包。审查方式：**源码审查 + 真实 torch 执行 + 纯 Python 几何复现**三重验证。
> 审查结论：**四个版本核心改造均属实，未发现阻断性问题**；补丁相对基线的语义、契约、调度均自洽。
> 仓库落地：每个版本原样放入 `training/versions/` 独立子目录（构建产物按 `.gitignore` 不入库），并打 `training-v1.8.4 … v1.8.7` tag。

---

## 0. 版本关系速览

```
v1.8.2 (CLO v1.0.1, training/current/ 现有基线)
  └─（phase-aware 3×3 新家族，独立于 current/）
v1.8.4  Phase-Aware 3×3 decoder（直接学 RGB + 20% bilinear skip，schema v2，direct_rgb）   ← 完整包
  └ v1.8.5  Phase-Aware 3×3 RGB-residual（权重零初始化 ⇒ 第0轮=双线性，schema v3，rgb_residual）← 完整包
     └ v1.8.6  Stable Phase-Residual Training（补丁，叠加 v1.8.5；training_contract v186）  ← 增量补丁
        └ v1.8.7  Dataset Manager + Stable Training（补丁，叠加 v1.8.6；training_contract v187 + dataset_contract） ← 增量补丁
```

注：v1.8.6 / v1.8.7 是**增量补丁**，必须叠加在对应基线（v1.8.5 / v1.8.6 结果）上才是完整可运行版本；本仓库按"原样归档补丁 + 文档说明基线/应用方式"处理，未做任何代码合并。

---

## 1. v1.8.4 — Phase-Aware 3×3 decoder + 新损失（完整包）

**包**：`gamesr-local-training-v1.8.4-CLO-phase-aware-3x3-shared-env.zip`
**SHA256**：`7D377B71713CA2F21BFE6221C7602F16B1CD89C02A79D50F4AB9190CE20CDC9E`

| 用户所述改造 | 代码层核对 | 结论 |
|---|---|---|
| O4/O6/O8 自动走 phase-aware 3×3 decoder | `gamesr/model.py` 9 相位卷积 `(phase_y, phase_x, out_ch, in_ch, 3, 3)`=27 路；`output_channels>3` 分支 | ✅ |
| O3 保留原兼容路径 | O3 → `decoder=None` + `magpie_bilinear_residual_v1` 路径 | ✅ |
| 新增 Charbonnier / MS-SSIM / 梯度 / 频率 / 振铃损失 | `gamesr/data.py`：`charbonnier_loss` / `ssim_loss` / `multi_scale_ssim_loss` / `edge_loss` / `high_frequency_loss` / `ringing_loss` | ✅ |
| 直接学 720p→1080p，不再以双线性为唯一目标 | `GameSR.forward` 直接重建 HR；`decoder(latent_lr, lr)` 内部含 `base_skip=0.20` | ✅ |
| 保留 20% bilinear base skip | `PhaseAware3x3Decoder(base_skip=0.20)` | ✅ |
| 模型名自动 `-phase3x3`，不误续训旧 1×1 | 输出目录后缀 `-phase3x3` | ✅ |
| 向上查找共用 `.venv`/数据集/环境 | `pipeline.py` `quark_python_cmd` 父目录向上查找 | ✅ |
| 导出 schema v2 decoder JSON（九相位核） | `export_onnx.py`：`schema_version:2` / `direct_rgb` / 九相位 `weight_layout` | ✅ |

**执行级验证（真实 torch 2.14 CPU）**：8×10 → 12×15 形状正确；phase-aware decoder 前向/反向有限；复合损失前向/反向通过；O3 旧架构兼容路径保留。

---

## 2. v1.8.5 — Phase3x3 RGB-residual + 零初始化（完整包）

**包**：`gamesr-local-training-v1.8.5-CLO-phase3x3-residual-shared-env.zip`
**SHA256**：`9DD052D34C343385A47C0FD1B4914F515AE7D6EB266CC7EF7B50589265B73C26`（与用户发布值一致）

| 用户所述改造 | 代码层核对 | 结论 |
|---|---|---|
| 完整保留 1.0× Bilinear 基线 | `base_skip=1.0`（外部不再叠加 base） | ✅ |
| 九相位 3×3 decoder 学 RGB residual | `PhaseAware3x3Decoder(base_skip=1.0)`，输出叠加 `bilinear(latent_lr)` | ✅ |
| 权重/bias 零初始化 | `nn.init.zeros_(self.phase_rgb)` + `bias` 零初始化 | ✅ |
| 第 0 轮输出与双线性逐像素一致 | **真实 torch：`torch.equal(output, bilinear(lr))` → `True, maxerr=0`** | ✅ 已坐实 |
| MS-SSIM 每步计算 | `multi_scale_ssim_loss` 每步 | ✅ |
| SSIM/高频/振铃渐入调度 | `phase_detail_ramp`（8%→40%） | ✅ |
| 梯度损失 25%→100% 升权 | `gradient_weight = 0.25 + 0.75*ramp` | ✅ |
| 困难样本 3%~8% | **`pipeline.py` O>3 分支覆盖：`min_mix=0.03 / max_mix=0.08`**（非默认 0.06/0.20） | ✅ |
| 退化强度 30%→65% | **`pipeline.py` O>3 分支：`start_strength=0.30 / full_strength_at=0.65`** | ✅ |
| 新目录 `-phase3x3r-v2` | 输出目录后缀 | ✅ |
| checkpoint 契约阻止误加载旧模型 | `checkpoint_signature` 含 `deployment`（schema/输出类型）；`direct_rgb` ≠ `rgb_residual` → 判异 | ✅ |
| decoder JSON schema v3 `rgb_residual` | `export_onnx.py`：`schema_version:3` / `output:"rgb_residual"` | ✅ |

**执行级验证（真实 torch）**：`[PASS] Phase-aware v2 starts as the exact bilinear baseline`（逐像素 `torch.equal`）；复合损失 forward/backward + grad 有限；`direct_rgb` 与 `rgb_residual` 的 `checkpoint_signature` 不同（契约成立）。

> ⚠️ 诚实边界：v1.8.4 的 `checkpoint_signature` **不含 `deployment` 字段**（仅 `{version, model, scale}`）——decoder 契约是 v1.8.5 才引入。因此 v1.8.4 无法靠签名拒绝误续训；v1.8.5 起用"新目录后缀 + 签名含 deployment"双重保险。

---

## 3. v1.8.6 — Stable Phase-Residual Training Patch（增量补丁，叠加 v1.8.5）

**包**：`GameSR-v1.8.6-Stable-PhaseResidual-Training-Patch.zip`
**SHA256**：`074015E7AA3C5353D002BA8964AD04FFC61E455D70981A4B827585B255F61D7C`
**改动文件**：`gamesr/data.py`、`gamesr/utils.py`、`pipeline.py`、`train.py`、config、`verify_v186_patch.py`（仅 5 个核心源文件，未动 `model.py` 解码器）

| 用户所述 / 包说明 | 代码层核对 | 结论 |
|---|---|---|
| decoder 输出语义不变（`rgb_residual`, `base_skip=1.0`） | `model.py` 未改动，沿用 v1.8.5 | ✅ |
| 训练控制器 / 课程调度 / 损失权重 / 质量刹车 / checkpoint 分离 / epoch 输入修复 | `train.py`、`pipeline.py` 改动对应实现 | ✅ |
| 加入 `training_contract` 契约 | `checkpoint_signature`（utils.py）纳入 `training_contract=phase_residual_stable_v186` | ✅ |
| 质量刹车（edge 比例超硬阈值时梯度/HF 归零、振铃不关） | `update_quality_controller`：`edge_ratio>hard ⇒ gradient/HF=0%` | ✅ |
| 误续训阻断 | `train.py:1007`：`current_sig != saved_sig` 且非 `--allow-incompatible-resume` ⇒ `SystemExit` 拒绝；`--auto-resume` 不兼容则跳过、从零开始 | ✅ |
| 导出目录 `-phase3x3r-v3` | config `phase3x3r-v3` | ✅ |

**执行级验证**：自带 `verify_v186_patch.py` 全 PASS；真实 torch 复核：解码器基线 = 双线性（`torch.equal` maxerr=0，未回归）；v1.8.6 签名含 `training_contract` 与 v1.8.5（无该字段）判异 → 手工 `--resume` 旧 checkpoint 会被拒绝；8×10→12×15 正确。

---

## 4. v1.8.7 — Dataset Manager + Stable Training Full Patch（增量补丁，叠加 v1.8.6）

**包**：`GameSR-v1.8.7-Dataset-Manager-Stable-Training-Full-Patch.zip`
**SHA256**：`2D0FE4ABDC0437A1743644C6AF547B076A3ACA2158EBBE27AB4AF0B9AFDF21ED`
**新增**：`prepare_dataset.py`（Dataset Manager）；`gamesr/data.py` 的 SHA256 与 v1.8.6 **完全相同**（稳定训练逻辑未动，仅打包全套）

| 用户所述 / 包说明 | 代码层核对 | 结论 |
|---|---|---|
| 稳定训练损失/课程逻辑不变 | `gamesr/data.py` SHA256 与 v1.8.6 一致 | ✅ |
| Dataset Manager（会话级防泄漏切分） | `prepare_dataset.py`：`group_key` 取 `game/session` 二级目录为**原子组**，整组进同一 split | ✅ |
| 默认 80/10/10 切分 | `--val 0.1 / --test 0.1`（默认）；`--min-width 1280 --min-height 720` | ✅ |
| full rebuild + incremental 两种模式 | `--mode full / incremental` | ✅ |
| `dataset_contract` 隔离 | `checkpoint_signature` 新增 `dataset_contract` 字段（与 `training_contract` 双保险） | ✅ |
| 导出目录 `-phase3x3r-v4` | config `phase3x3r-v4` | ✅ |

**执行级验证**：
- 自带 `verify_v187_patch.py`：首跑 `dataset_manager` smoke 失败，根因是**测试环境缺 `tqdm`**（非补丁 bug）；`pip install tqdm` 后全部 PASSED。
- 真实执行 `prepare_dataset.py`：造 40 个不等大模型（≥720p）跑 `--mode full`，结果 **train/val/test ≈ 80/10/10、各组不跨集（no cross-split leak）、health 无 leaked_groups**；少会话场景（如 3 个等大模型）因"会话为原子组不可拆"会得到 15/15/15 —— 这是约束下的正确结果，非 bug。
- 解码器基线 = 双线性（未回归）；v1.8.7 / v1.8.6 / v1.8.5 三者 `checkpoint_signature` **两两不同**（防跨版本误续训）。

> 说明：dataset CSV 只有 `path` 一列、无 session 列，这是正常的——session 归属是"分组决策"不进 manifest 列；"会话绝不跨 Train/Val/Test"由原子组约束保证（已实测验证）。

---

## 5. 诚实边界（未在本机执行 / 待办）

1. **未在本机执行**：完整 NPU 顺序训练（需 CUDA）、INT8/Quark 量化编译、Magpie 端 C++/HLSL 九相位 3×3 解码路径。
2. **下一步仍需给 Magpie 补**：读取 `phase_aware_3x3_v1` / schema v2 / schema v3 的参数上传，以及 HLSL 九相位 3×3 解码实现；原 O3 路径继续保留。
3. **v1.8.4 无 decoder 契约**：仅靠"新目录后缀 + 不与旧 1×1/direct-RGB 续训"避免误用；v1.8.5+ 才有签名级契约。
4. **契约防误续训已实测**：v1.8.5→v1.8.7 任意跨版本 `--resume` 都会被签名判异拒绝；操作上仍需"解压到独立子目录 + 不手动指定旧输出目录"。

---

## 6. 一句话操作结论

- 停旧训练 → 解压新版到独立子目录 → 跑 `Train-Choose-C-L-O.cmd` 选 `C16/L4/O4` → **不要手动指定旧输出目录**。
- v1.8.5+ 前几轮 `delta PSNR` 应 ≈ 0 dB（第 0 轮逐像素=双线性）；若见 `-4 dB` 即说明误用了旧 1×1 / direct-RGB checkpoint，契约会拒绝但建议手动核对目录。
- v1.8.6/v1.8.7 为增量补丁，须叠加对应基线使用（见各版本目录内 `README_*.md` 与 `verify_v1*.py`）。
