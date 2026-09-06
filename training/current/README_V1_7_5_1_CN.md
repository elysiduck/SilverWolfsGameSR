# GameSR training v1.7.5.1 — EMA 续训热修复

这是 v1.7.5 的小型累计热修复，专门修复 **从 `latest.pth` 续训时 EMA tensor 留在 CPU** 的问题。

## 症状

续训开始后第一步在 `ema.update(model)` 报错：

```text
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

## 根因

checkpoint 使用 `torch.load(..., map_location="cpu")` 安全读取。旧版 `ModelEMA.load_state_dict()` 直接用 checkpoint 中的 CPU tensors 替换了已经在 CUDA 上创建的 `ema.shadow`，导致第一次 EMA update 时 CPU/CUDA 混用。

## 修复

- EMA load 不再替换 shadow tensor。
- 保存值会复制进现有 shadow，因此自动保持当前 `device` 与 `dtype`。
- EMA checkpoint 以后统一以 CPU tensor 保存，使 checkpoint 更可移植。
- 加入 EMA key 完整性检查。

## 安装

把补丁里的 `train.py` 覆盖到现有 v1.7.5 训练目录即可。

不需要删除：

- `.venv`
- `.venv-quark`
- `outputs`
- `latest.pth`
- 已有 C/L 配置

## 续训

直接重新运行原来的 C32-L3 训练。现有 epoch 5 `latest.pth` 可以继续使用，应该从 epoch 6 重新开始。

之前崩溃发生在 checkpoint 保存之前，因此不会破坏 epoch 5 的 `latest.pth`。
