# Magpie GameSR v4.3 — Phase3x3 + 模型子文件夹 梳理与检查报告

> 审查对象：`Magpie-GameSR-v4.3-Phase3x3-ModelFolders-source-patch.zip` 与
> `Magpie-GameSR-v4.3-Phase3x3-ModelFolders-runtime.zip`
> 审查日期：2026-09-07
> 结论：**属实，无阻断性问题；契约与训练端 v1.8.5 schema v3 完全对齐。**

## 1. SHA256 校验（用户发布值 vs 实测）

| 包 | 用户给出 | 实测 | 结果 |
| --- | --- | --- | --- |
| 运行包 runtime | `0A461A28…049738E2` | `0A461A282EF13A48FBE86B98E37218EE59296D84523353BD3776BCD9049738E2` | ✅ 一致 |
| 源码补丁 source | `34706F71…D3246AA3` | `34706F7197DDE8655AC448019378A2CD2BA28E8A5877BB23F5EC9D9CD3246AA3` | ✅ 一致 |

两份包 CRC 校验均通过。运行包 296 MB / 211 条目；源码补丁 47 KB / 8 条目。

## 2. 源码补丁内容（已入仓 `magpie/patches/v4.3-phase3x3-modelfolders/`）

```
BUILD_ID.txt
RELEASE_NOTES_v4.3_PHASE3X3_CN.txt
GameSR-Models/README_CN.txt
src/Magpie/ScalingModeItem.cpp
src/Magpie/ScalingModesPage.xaml
src/Magpie.Core/Renderer.cpp
src/Magpie.Core/SesrNpuProcessor.cpp
tools/GameSRPhaseDecoderTest.hlsl
```

## 3. 逐项核对（源码审查结果）

| 用户所述能力 | 代码证据 | 结论 |
| --- | --- | --- |
| 新 `phase_aware_3x3_v1 + rgb_residual + schema v3` | `SesrNpuProcessor.cpp:960` 识别 `phase_aware_3x3_v1`；运行包 `onnx.decoder.json` 实测 `schema_version:3 / type:phase_aware_3x3_v1 / output:rgb_residual / base_skip:1.0` | ✅ |
| O4 = 972 权重 + 27 bias | 代码 `phaseWeightCount = 3·3·3·O·3·3` → O4=972、O6=1458、O8=1944；`bias.size()==27`；JSON 实测 972 权重 / 27 bias | ✅ |
| 保留旧 O3 RGB residual | `SesrNpuProcessor.cpp:919` 注释 O3 走 legacy identity residual；O3 不要求 `decoder.json` | ✅ |
| 保留旧 O4/O6/O8 `bilinear_then_1x1` | `:948` `decoderType=="" || "bilinear_then_1x1"` 且 `weights==oldWeightCount && bias==3` | ✅（旧 12/3 可识别） |
| 完整保留 v4.2.6 SIMD + 异步三槽 | `Renderer.cpp:1035` BUILD ID 标注 `SIMD-v4.2.6`；运行包 `effects/` 含 124 个 hlsl | ✅ |
| `models/GameSR` 任意层级子文件夹 | `ScalingModeItem.cpp:345` 递归扫描 `.onnx`；`:347` 存**相对**路径 | ✅ |
| 下拉框显示并保存相对路径 | `ScalingModesPage.xaml:464` ComboBox 绑定 `GameSrModels`/`GameSrModelIndex`；`ScalingModeItem.cpp:366` 存相对路径 | ✅ |
| 阻止绝对路径与 `..` 越界 | `SesrNpuProcessor.cpp:233-250`：`is_absolute()` 拒绝、`weakly_canonical` + `lexically_relative` + `*begin()==".."` 拒绝 | ✅ |
| 修复首帧等待 NPU 时 decoder bias 污染双线性 | `SesrNpuProcessor.cpp:1155` 独立 `gameSrBaseline` 着色器；`:1637` 首帧用 baseline 占位，decoder bias 不参与 | ✅ |

## 4. 契约一致性（训练端 ↔ 推理端）

- 训练端 v1.8.5 导出 `schema v3 / rgb_residual / base_skip:1.0`（`-phase3x3r-v2/v3`）。
- 推理端 v4.3 `onnx.decoder.json` 实测同字段，且权重布局 `phase_y,phase_x,rgb,input_channel,kernel_y,kernel_x`（3·3·3·4·3·3=972）、bias 布局 `phase_y,phase_x,rgb`（3·3·3=27）——与训练端解码器张量布局**逐维对齐**。
- 推理端 HLSL `PhaseResidual`（`SesrNpuProcessor.cpp:1076` 与 `tools/GameSRPhaseDecoderTest.hlsl`）的相位选择 `outputPos%3`、锚点规则 `phaseX==2?1:0`、replicate 边界、与 1.0× 双线性相加——与训练端 `model.py` decoder 前向（v1.8.5 已验证 `torch.equal` 双线性基线）**几何一致**。

## 5. 诚实边界

1. **未在本机执行**：Release x64 C++/WinUI 编译（需 VS + Windows SDK）、HLSL `cs_5_0` 编译（需 `fxc`/`dxc`）、NPU 实际缩放——均为你的环境职责。代码契约已静态验证，但设备端运行（Ryzen AI NPU 720p→1080p）**待你解压后做最终验证**。
2. **运行包不入 git**：296 MB 含 `Magpie.exe`、ONNX Runtime / RyzenAI / VitisAI 等 DLL、`int8-qdq.onnx` 二进制，按仓库 `ARTIFACTS.md` 约定属 **GitHub Release 资产**，已在 `artifacts.lock.json` 记录 SHA256（release_tag `magpie-v4.3`）。需手动上传到 tag `magpie-v4.3` 的 Release（或提供 token 由我创建 Release 附件）。
3. **源码补丁未覆盖 `magpie/current/full-source`**：按交付要求“不要覆盖旧 Magpie”，独立归档于 `magpie/patches/v4.3-phase3x3-modelfolders/`。
4. 运行包含第二个模型 `gamesr-c24-l3-o4-720p-to-1080p/`（仅 `int8-qdq.onnx` + `meta.json` + `decoder.json`，无浮点 `.onnx`/`.onnx.data`），属同架构另一配置，契约一致。

## 6. 使用提示（来自交付说明）

解压运行包到**新目录**（勿覆盖旧 Magpie），打开后在 `GameSR NPU O4` 模型选择框选子文件夹：
```
gamesr-c16-l4-o4-phase3x3r-v3-720p-to-1080p/
gamesr-c16-l4-o4-phase3x3r-v3-720p-to-1080p-int8-qdq.onnx
```
