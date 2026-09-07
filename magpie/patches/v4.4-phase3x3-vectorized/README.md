# Magpie GameSR v4.4 — Phase3x3 GPU 向量化（源码补丁）

基于 **v4.3** 的源码补丁，与 `v4.3-phase3x3-modelfolders/` 同构：都是 GameSR 修改集的 8 个源文件 + 说明，
其中 **3 个文件在 v4.4 被更新**（向量化），其余 5 个文件与 v4.3 完全相同。

## 内容（9 文件）

| 文件 | v4.4 状态 | 作用 |
| --- | --- | --- |
| `src/Magpie.Core/SesrNpuProcessor.cpp` | **更新** | float4 权重重排 + 嵌入 HLSL 改 `StructuredBuffer<float4>` |
| `src/Magpie.Core/Renderer.cpp` | **更新** | 仅 `_InitBackend` 日志 Build ID 串改为 `magpie-v4.4-phase3x3-vectorized \| float4-decoder` |
| `tools/GameSRPhaseDecoderTest.hlsl` | **更新** | 镜像着色器：单趟输出 RGB 三路（`dot` 累加） |
| `src/Magpie/ScalingModeItem.cpp` | 同 v4.3 | 模型子文件夹 / 相对路径绑定 |
| `src/Magpie/ScalingModesPage.xaml` | 同 v4.3 | UI 配置 |
| `GameSR-Models/README_CN.txt` | 同 v4.3 | 模型库说明 |
| `BUILD_ID.txt` | 更新 | `magpie-v4.4-phase3x3-vectorized` |
| `RELEASE_NOTES_v4.3_PHASE3X3_CN.txt` | 同 v4.3 | v4.3 说明 |
| `RELEASE_NOTES_v4.4_VECTOR_DECODER_CN.txt` | 新增 | v4.4 向量化解码器说明 |

## 向量化核心改动（SesrNpuProcessor.cpp）

- 启动时把 schema v3 权重**重排为 GPU `float4` 通道组**，`StructuredBuffer<float4>` 替代 `StructuredBuffer<float>`。
- 一个 latent `float4` 邻域读取同时用于 R/G/B 三路输出，不再按 RGB / 单通道重复 Load。
- O4 每 1080p 输出像素：latent 纹理读取 `108 标量 Load → 9 次 float4 Load`；权重读取 `108 标量 → 27 次 float4`。
- O6 / O8 自动用**两个 float4 latent 平面**（`gameSrLatentTextureCount = (OC+3)/4`：O4→1、O6/O8→2）；缺失通道补零。
- bias 在权重块后追加为 9 个 `float4`（每相位 1 个，`.xyz` 取 3 bias，`.w=0`）。
- 权重重排索引 `((phase*3+rgb)*OC+channel)*9+ky*3+kx` 与 HLSL `tapBase=((27*phase+3*ky+kx)*TC+t)+rgb*9*TC` 严格对应（详见 REVIEW 文档的索引自洽性证明）。

## 重建方式

叠加在 `magpie/current/full-source/`（v4.1.1-AsyncNPU-CurrentInput 基线）之上即可获得 v4.4 完整源码：

```
magpie/current/full-source/   (基线)
   + magpie/patches/v4.3-phase3x3-modelfolders/   (v4.3 GameSR 修改集)
   + magpie/patches/v4.4-phase3x3-vectorized/     (v4.4 向量化增量，覆盖同名文件)
   = v4.4 完整源码
```

运行包（`Magpie-GameSR-v4.4-Phase3x3-Vectorized-runtime.zip`，GitHub Release `magpie-v4.4` 资产）含编译好的
`Magpie.exe`、RyzenAI / VitisAI DLL、`models/GameSR/`（由 AMD Quark INT8 从训练 v1.8.5 导出的 `gamesr-c16-l4-o4-phase3x3r-v3` 与 legacy `gamesr-c24-l3-o4`）。

## 契约对齐

- 运行包 `onnx.decoder.json`：`schema_version:3` / `phase_aware_3x3_v1` / `rgb_residual` / `base_skip:1.0`，
  权重布局 `phase_y,phase_x,rgb,input_channel,kernel_y,kernel_x` → O4 = 972 权重 + 27 bias。
- `int8-qdq.onnx.meta.json` 明确 `source_model=...v1.8.5-CLO-phase3x3-residual...`，与训练端 v1.8.5 逐维对齐。
- **不修改 ONNX / 不重新训练 / 不重新量化**；向量化纯属 GPU 运行时优化。
