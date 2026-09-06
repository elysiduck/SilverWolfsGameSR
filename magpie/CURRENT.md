# Magpie Current

**唯一当前版本：`magpie-v2.1-pipeline-v4.1-async-npu`**

从这个版本开始，本地只保留一个可编译 / 可运行的 Magpie 工作目录：

```text
SilverWolfsGameSR/magpie/current
```

旧版本不再复制成多个本地目录，只通过 GitHub commit / tag / release 回退。

## 如何确认没有跑错版本

启动缩放后，`magpie.log` 必须出现：

```text
GameSR BUILD ID: magpie-v2.1-pipeline-v4.1-async-npu | contractv3 | canonical-current
```

没有这一行，就不是当前标准版本。

## 当前模型

```text
models/GameSR/gamesr-c16-l4-720p-to-1080p-int8-qdq.onnx
```

## 当前流水线

- Pipeline v4 Async NPU
- NPU worker 脱离 render thread
- three-slot async pipeline
- latest-frame-wins
- GameSR contract v3
- C16L4 720p → 1080p residual SR

## 本地管理规则

1. 只编译 `magpie/current`。
2. 只运行从 `magpie/current` 构建出的 Magpie。
3. 新升级直接覆盖 `current` 并提交 Git。
4. 历史版本不留第二份“可运行副本”，需要时从 Git tag/commit 恢复。
