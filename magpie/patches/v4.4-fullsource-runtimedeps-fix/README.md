# v4.4 FullSource RuntimeDeps 修复（补齐 4 个 Ryzen AI / VitisAI 运行时 DLL）

## 背景

`Magpie-GameSR-v4.4-Phase3x3-Vectorized-full-source.zip` 打包时漏掉了
`third_party\sesr\onnx\bin\`，导致源码工程运行 `0-first-check-environment.cmd` 时报：

```text
[MISSING] SESR-M7 ONNX model, X1 config, NPU overlay, or VitisAI runtime files
```

本补丁补齐该目录，使环境检查变为：

```text
[PASS] SESR-M7 model, precompiled X1 cache, and matching Ryzen AI 1.7.1 runtime
RESULT: READY TO BUILD
```

## ⚠️ 本目录只有脚本，DLL 不在 git 里

仓库 `.gitignore` 排除 `*.dll`，因此 **4 个 DLL 不能进 git**，只能作为 GitHub Release 资产分发。

本目录包含：

- `README_FIX_CN.txt` — 原始修复说明
- `VERIFY-RUNTIME-DEPS.cmd` — 双击检查 4 个 DLL 是否已放到正确位置

完整补丁（含 DLL）：

- `Magpie-GameSR-v4.4-FullSource-RuntimeDeps-Fix.zip`
- SHA256 `D2A3010A1E1855E5770D539149FFAB36BF86954976AB895CDB99B141BD5F276A`
- GitHub Release `magpie-v4.4` 资产

## 用法

1. 下载上述 ZIP，解压到 full-source 根目录并**合并/覆盖**目录。
2. 最终结构：

```text
Magpie-GameSR-v4.4-Phase3x3-Vectorized-full-source
└─ third_party
   └─ sesr
      └─ onnx
         └─ bin
            ├─ onnxruntime.dll               (21925456 B)
            ├─ onnxruntime_vitisai_ep.dll    (149825616 B)
            ├─ dyn_dispatch_core.dll         (151886416 B)
            └─ aiecompiler_client.dll        (10092624 B)
```

3. 双击 `VERIFY-RUNTIME-DEPS.cmd` 确认 4 个 DLL 就位。
4. 重新运行 `0-first-check-environment.cmd`，应显示 `RESULT: READY TO BUILD`。

> 这些 DLL 与 v4.4 runtime 包中的版本一致；`onnxruntime.dll` SHA256
> `919AC18B9AD61588C1D65EDBA775B012DA626FACD01623FEAB3A470042EAF4E3`，
> 与 `Check-Environment.ps1` 要求的 Ryzen AI 1.7.1 版本匹配。
