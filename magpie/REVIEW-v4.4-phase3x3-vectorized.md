# 复核：Magpie GameSR v4.4 — Phase3x3 GPU 向量化

> 复核对象：`Magpie-GameSR-v4.4-Phase3x3-Vectorized-full-source.zip` + `-runtime.zip`
> 归档目标：`magpie/patches/v4.4-phase3x3-vectorized/`（源码，9 文件）+ GitHub Release `magpie-v4.4`（运行包）
> 复核方法：源码逐项 diff（vs v4.3）+ 运行包 `onnx.decoder.json` / `*.meta.json` 实测 + 索引数学自洽性推导
> 结论：**源码改造属实、向后兼容完整保留、解码契约与训练 v1.8.5 schema v3 逐维对齐；可归档。**

---

## 1. 包与 SHA256

| 包 | SHA256 | 处置 |
| --- | --- | --- |
| `Magpie-GameSR-v4.4-Phase3x3-Vectorized-full-source.zip` | `29D4D09A0547FF75E43271A35AE583711D2541C5944E2CA7C25F2F9BCC37936B` | 源码 → `magpie/patches/v4.4-phase3x3-vectorized/` |
| `Magpie-GameSR-v4.4-Phase3x3-Vectorized-runtime.zip` | `4BB0E16CA0CE228CAA672A16B82F9D08AFB8D9B2C472D356486969A6117582AC` | 运行包 → GitHub Release `magpie-v4.4`（不入 git） |

---

## 2. 与 v4.3 的文件级差异（仅改 3 文件 + 更新说明）

用 `diff` 逐文件比对 v4.4 staging 与 `magpie/patches/v4.3-phase3x3-modelfolders/`：

- `src/Magpie/ScalingModeItem.cpp` — **字节完全相同**（cmp 通过）
- `src/Magpie/ScalingModesPage.xaml` — **字节完全相同**（cmp 通过）
- `GameSR-Models/README_CN.txt` — 同 v4.3
- **`src/Magpie.Core/SesrNpuProcessor.cpp`** — 新增 +32 行（2026 → 2058 行）：float4 重排 + 嵌入 HLSL 改造
- **`src/Magpie.Core/Renderer.cpp`** — 仅 `_InitBackend` 日志 Build ID 串变化
- **`tools/GameSRPhaseDecoderTest.hlsl`** — 镜像着色器改造
- `BUILD_ID.txt` / `RELEASE_NOTES_v4.4_*` 更新

即 v4.4 相对 v4.3 **只有向量化这一处功能性改动**，与 RELEASE_NOTES 描述一致。

---

## 3. 向量化实现核对（SesrNpuProcessor.cpp）

### 3.1 权重重排（C++ 端）
- 遍历 `(phase, rgb, ky, kx, texture)`，每个 `float4` 由 `channel = texture*4 + component` 四通道聚成；
  `channel < gameSrOutputChannels` 时取权重，否则补零（缺失通道补零 ✓）。
- `gameSrLatentTextureCount = (gameSrOutputChannels+3)/4`（第 827 行赋值，第 428 行默认 1）：
  O4→1、O6→2、O8→2（两个 float4 latent 平面 ✓）。
- `packedWeightCount = 9 * 3 * 3 * 3 * textureCount`（O4=243 float4）；bias 在权重块后追加 9 个 `float4`
  `{bias[3p], bias[3p+1], bias[3p+2], 0}`（`.xyz` 取 3 bias ✓）。
- `StructuredBuffer<float4>`，`StructureByteStride = sizeof(float4)`（原为 `float`）。

### 3.2 着色器（嵌入 HLSL + tools 单测）
- `LatentVectorAt(position, textureIndex)` 一次返回整组 `float4` latent（原 `LatentAt(channel)` 逐标量提取）。
- `PhaseResidual(outputPosition)` 单趟算出 RGB 三路：`result.x += dot(latent, PhaseParams[tapBase])`，
  `result.y += dot(latent, PhaseParams[tapBase + rgbStride])`，`result.z += dot(latent, PhaseParams[tapBase + rgbStride*2])`；
  `rgbStride = 3*3*textureCount`（原 3 次 `PhaseResidual(id, rgb)` 调用 → 1 次 ✓）。
- 索引：`tapBase = ((phase*3)*3 + ky)*3 + kx)*TC + textureIndex = (27*phase + 3*ky + kx)*TC + t`；
  叠加 `rgb*9*TC` 恰好等于 C++ 重排索引 `(27*phase + 9*rgb + 3*ky + kx)*TC + t` → **读取与写入严格对应，无错位**。

### 3.3 性能声明核对
RELEASE_NOTES 称 O4 每像素 latent 读取 `108 标量 → 9 float4`，权重读取 `108 标量 → 27 float4`。
由重排结构可直接推出：9 个 3×3 tap × 1 float4 邻域 = 9 次 latent Load；9 tap × 3 rgb = 27 次权重 float4 Load，
与声明一致（属于 **读取次数** 层面的削减，非改变数值结果）。

---

## 4. 向后兼容核对

- `SesrNpuProcessor.cpp:445` 默认 `GameSrDecoderMode::Legacy1x1`；
  `:919` 注释 "O3 is the legacy identity residual decoder. O4/O6/O8 accept both"；
  `:948` 旧 `bilinear_then_1x1`（12 权重 + 3 bias）路径保留；`:963` `PhaseAware3x3` 分支独立。
- 运行包实测模型：
  - `gamesr-c24-l3-o4-720p-to-1080p/` → `decoder.json`：`schema_version:1`、`bilinear_then_1x1`（legacy 路径 ✓）
  - `gamesr-c16-l4-o4-phase3x3r-v3-720p-to-1080p/` → `decoder.json`：`schema_version:3`、`phase_aware_3x3_v1`、`rgb_residual`、`base_skip:1.0`、权重布局 `phase_y,phase_x,rgb,input_channel,kernel_y,kernel_x`（与 v4.3 / 训练 v1.8.5 逐维对齐 ✓）

---

## 5. 契约对齐（与训练 v1.8.5）

- 运行包 `gamesr-c16-l4-o4-phase3x3r-v3/int8-qdq.onnx.meta.json` 明确：
  `"source_model": "...gamesr-local-training-v1.8.5-CLO-phase3x3-residual-shared-env..."`、
  `"builder": "AMD Quark for ONNX"`、`"builder_version": "1.7.1"`。
  ⇒ v4.4 运行包模型**实证来自训练 v1.8.5**，INT8/QDQ 由 Quark 导出；向量化不改模型、不改契约。
- decoder JSON 权重布局 `phase_y,phase_x,rgb,input_channel,kernel_y,kernel_x` → O4 = 3·3·3·4·3·3 = 972 权重 + 3·3·3 = 27 bias，
  与训练端 v1.8.5 schema v3 完全一致。

---

## 6. 诚实边界（未执行项）

- ❌ **未编译 Release x64 C++/WinUI**：本机无 MSVC / Windows SDK / vcpkg 构建环境。`BUILD_ID.txt` 称
  "Release x64 and HLSL cs_5_0 compile verified" 为用户侧声明，本次仅做静态源码审查 + 索引自洽推导。
- ❌ **未在 Ryzen AI NPU 设备上实跑 720p→1080p**：向量化带来的实际加速比、与 v4.3 的逐像素输出一致性、
  以及 RELEASE_NOTES 要求的 `magpie-43.log` GPU fence / 复用旧输出 / worker NPU / output-pack / residual 上传对比，
  均待用户在设备端解压运行后最终确认。
- ❌ **未重训 / 未重量化**：meta.json 已实证模型源自 v1.8.5，无需重做。
