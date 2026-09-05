# GameSR 文档归档：20-training

> 本文件由项目现有文档自动归档生成；每个原始文件均保留路径标题与完整文本内容。

---

## `docs/20-training/local-env-v1.1/dependencies/requirements.txt`

```text
numpy>=2.0,<3
Pillow>=10,<13
opencv-python>=4.10,<5
tqdm>=4.66,<5
PyYAML>=6,<7
tensorboard>=2.17,<3
onnx>=1.17,<2
onnxruntime>=1.20,<2
onnxscript>=0.2,<1
einops>=0.8,<1
scikit-image>=0.24,<1
lpips>=0.1.4,<1
safetensors>=0.4,<1


```

---

## `docs/20-training/local-env-v1.1/guides/README.md`

# GameSR 本地训练环境（Windows 11）

这个目录使用项目内的 `.venv` 和 `.tools`，不会调用现有 Conda、系统 Python、CUDA Toolkit 或其他虚拟环境。PyTorch 的 CUDA wheel 自带 CUDA 运行库和 cuDNN；系统只需要可用的 NVIDIA 驱动。

## 一键安装

1. 解压到空间充足的英文路径，例如 `D:\GameSR-Train`。
2. 双击 `install.cmd`。默认安装 Python 3.11 与 PyTorch CUDA 12.8 版本。
3. 看到 `[PASS]` 即表示 GPU 训练环境正常。

安装器会最后一次从 PyTorch 专用索引重装 CUDA 版 PyTorch，防止 `lpips` 等通用依赖将其替换为 CPU 版。

不需要管理员权限。首次安装需要联网，并可能下载数 GB 文件。

如果显卡驱动不支持默认版本，在 PowerShell 中使用：

```powershell
.\setup.ps1 -TorchBuild cu130
```

如需彻底重建本项目环境：

```powershell
.\setup.ps1 -TorchBuild cu128 -Rebuild
```

`-Rebuild` 只删除当前目录内名为 `.venv` 的环境，不接触任何其他环境。

## 检查环境

```powershell
.\run.ps1
```

它会输出 Python 路径、PyTorch 版本、PyTorch 自带的 CUDA/cuDNN 版本、驱动信息、GPU、显存，并执行一次 CUDA 矩阵运算。

## 运行训练代码

把训练项目放在本目录内，例如 `train.py`，然后执行：

```powershell
.\run.ps1 train.py --config configs\tiny-c8.yaml
```

无需手动激活虚拟环境。也可以在 VS Code 中选择：

```text
.venv\Scripts\python.exe
```

## 只读盘点旧环境

```powershell
.\audit_existing_envs.ps1
```

脚本会生成 `environment-audit.csv`，记录找到的环境、Python、PyTorch 与 CUDA 状态。它不会删除或修改旧环境。递归扫描用户目录可能需要几分钟。

## 关于电脑里已有的 CUDA 和 cuDNN

训练本项目时，以 `check_env.py` 输出为准：

- `python_executable` 必须指向本目录 `.venv`；
- `cuda_available` 应为 `true`；
- `torch_cuda_runtime` 和 `cudnn` 是当前 PyTorch 实际使用的版本；
- `nvcc --version` 显示的是系统 CUDA Toolkit，训练普通 PyTorch 模型时通常无关；
- 不要把其他环境的 `site-packages`、DLL 或 `PYTHONPATH` 复制进来。

暂时不要清理旧环境。等 GameSR 的真实训练代码和数据管线在新环境中通过一次训练后，再根据审计表逐项删除。

## 修复早期版本的 CPU PyTorch 覆盖问题

如果检查结果显示 `torch` 末尾为 `+cpu`，在本目录打开 PowerShell，执行：

```powershell
.\.tools\uv.exe pip install --python .\.venv\Scripts\python.exe --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe .\check_env.py
```

看到 `[PASS]` 后环境即可继续使用，不需要重建 `.venv`。


---

## `docs/20-training/current-v1.7.1/history/FIX_V1_4_1.txt`

```text
GameSR Training Code v1.4.1

Hotfix: pipeline benchmark cache validation now accepts schema_version=5, matching benchmark_speed.py.
This fixes the false "Benchmark produced no valid result" failure after a successful benchmark.
No training model/data format changes.
Existing valid v1.4 benchmark JSON caches with schema_version=5 can be reused.

```

---

## `docs/20-training/current-v1.7.1/history/HOTFIX_V1_6_1.txt`

```text
GameSR v1.6.1 installer hotfix
================================

Problem fixed:
AMD Quark 0.12.post1+cpu.torch2.11 does not automatically install PyTorch.
The v1.6 installer therefore reached verification and failed with:

No module named 'torch'

v1.6.1 explicitly installs the matching PyTorch 2.11 CPU wheel before Quark
verification. Existing .venv-quark packages are preserved; rerunning the
installer only downloads and adds the missing PyTorch package.

```

---

## `docs/20-training/current-v1.7.1/history/RELEASE_NOTES_V1_4.txt`

```text
GameSR Training Code v1.4 release notes

Primary fixes and upgrades:
- Benchmark_Training.cmd works with no arguments and enters the same data-preparation + benchmark flow as the main pipeline.
- Missing dataset manifests trigger automatic clean-folder prompt and prepare_dataset.py.
- A benchmark that contains only errors/OOM/no measurements is rejected and never cached as valid.
- Cached benchmarks use schema/signature validation over dataset manifests, model architecture, degradation, loss settings, benchmark settings, and runtime-relevant config.
- GameSR_Train.cmd uses a valid cache when possible, prints all tested batch sizes, and Enter selects the fastest valid batch.
- Per-model benchmark files remain separate for Tiny-C8, Balanced-C12, and Quality-C16.
- v1.4 default model architecture is LR-residual: feature extraction at LR resolution, 3-channel residual upsample, then bilinear global residual add.
- Benchmark uses the real training loss path: L1 + edge + periodic SSIM plus gradient clipping.
- GPU-side game degradation is retained; CPU JPEG encode/decode is not used by default.
- Training logs expose batch size, throughput, loader wait, H2D/degradation time, compute time, peak VRAM, and benchmark provenance.
- Automatic resume refuses a checkpoint created by a different training code version/config unless explicitly overridden.

Recommended formal experiment policy:
- Start fresh at epoch 1 for v1.4.
- Run the benchmark once per model/GPU; use Enter for fastest throughput unless there is a deliberate reason to select another tested batch.

```

---

## `docs/20-training/current-v1.7.1/history/RELEASE_NOTES_V1_4_1.txt`

```text
GameSR Training Code v1.4 release notes

Primary fixes and upgrades:
- Benchmark_Training.cmd works with no arguments and enters the same data-preparation + benchmark flow as the main pipeline.
- Missing dataset manifests trigger automatic clean-folder prompt and prepare_dataset.py.
- A benchmark that contains only errors/OOM/no measurements is rejected and never cached as valid.
- Cached benchmarks use schema/signature validation over dataset manifests, model architecture, degradation, loss settings, benchmark settings, and runtime-relevant config.
- GameSR_Train.cmd uses a valid cache when possible, prints all tested batch sizes, and Enter selects the fastest valid batch.
- Per-model benchmark files remain separate for Tiny-C8, Balanced-C12, and Quality-C16.
- v1.4 default model architecture is LR-residual: feature extraction at LR resolution, 3-channel residual upsample, then bilinear global residual add.
- Benchmark uses the real training loss path: L1 + edge + periodic SSIM plus gradient clipping.
- GPU-side game degradation is retained; CPU JPEG encode/decode is not used by default.
- Training logs expose batch size, throughput, loader wait, H2D/degradation time, compute time, peak VRAM, and benchmark provenance.
- Automatic resume refuses a checkpoint created by a different training code version/config unless explicitly overridden.

Recommended formal experiment policy:
- Start fresh at epoch 1 for v1.4.
- Run the benchmark once per model/GPU; use Enter for fastest throughput unless there is a deliberate reason to select another tested batch.

```

---

## `docs/20-training/current-v1.7.1/history/RELEASE_NOTES_V1_4_2.txt`

```text
GameSR Training Code v1.4.2

Hotfix + reliability release.

- Forces UTF-8/replace-safe Python stdout/stderr for Windows ONNX export.
- GameSR_Train.cmd and Benchmark_Training.cmd now set UTF-8 environment variables.
- Added GameSR_Export.cmd for one-command re-export from an existing best.pth.
- Added ONNX_Validate.cmd / validate_onnx.py for graph validation and external-data checks.
- export_onnx.py explicitly supports --use-ema and --no-ema and validates external data.
- Pipeline subprocesses inherit UTF-8-safe environment and decode output as UTF-8 with replacement.
- Existing checkpoints and dataset manifests are not modified by the hotfix.

```

---

## `docs/20-training/current-v1.7.1/history/RELEASE_NOTES_V1_5.txt`

```text
GameSR Training Code v1.5

- PC-side INT8/QDQ PTQ integrated into the training pipeline.
- Optional FP16 ONNX conversion added.
- Deployment bundle now copies FP32, FP16, and INT8 candidates when present.
- Calibration happens only on the PC; handheld never needs clean data or Python/.venv.
- One-click pipeline: benchmark -> train -> full test -> FP32 ONNX -> FP16 candidate -> INT8/QDQ -> deployment bundle.
- Float model/checkpoint are preserved if a deployment conversion fails.

```

---

## `docs/20-training/current-v1.7.1/history/RELEASE_NOTES_V1_5_1.txt`

```text
GameSR Training Code v1.5.1

Hotfix:
- Fix static INT8/QDQ PTQ failures on ONNX models stored with external data.
- Explicitly run ONNX shape inference in Python, flatten a temporary inferred model,
  then quantize that self-contained graph to avoid ORT *-inferred.onnx lookup failures.
- Validate the generated INT8 ONNX with onnx.checker before reporting success.
- Delete the temporary inferred model after successful validation.
- Existing FP32/FP16/checkpoints/dataset formats remain unchanged.

```

---

## `docs/20-training/current-v1.7.1/history/RELEASE_NOTES_V1_5_3.txt`

```text
GameSR Training Code v1.5.3

Hotfix for INT8 PTQ on Windows/ONNX Runtime.
- Keeps the temporary self-contained PTQ input model shape-inferred in Python.
- Sets ORT extra option DisableShapeInference=True to prevent the inferred-inferred filename bug.
- Cleans up temporary PTQ artifacts after successful or failed runs where possible.
