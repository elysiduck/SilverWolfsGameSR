# SESR-M7 NPU v1.8 测试说明

这一版同时修复两项关键问题：

1. 使用 AMD Ryzen AI Deployment 1.7.1 的整套运行库。其 ONNX Runtime 提交号
   `58b2075a54578015d2c573c209b2ca462228672d` 与模型缓存记录完全一致。
2. 将原先遗漏的 X1 预编译缓存文件夹复制到 `publish\x64\sesr`。

## 编译

双击 `构建-SESR.cmd`。看到 `BUILD COMPLETED` 后，打开 `publish\x64`。

## 检查最终文件

运行前确认下面三个文件存在：

- `publish\x64\onnxruntime.dll`
- `publish\x64\sesr\modelcachekey_sesr_nhwc_int8_256x256\context.json`
- `publish\x64\sesr\modelcachekey_sesr_nhwc_int8_256x256\compiled.0xa000205001fdb46.xmodel`

## 测试

把整个 `publish\x64` 文件夹复制到掌机，不要只复制 `Magpie.exe`。退出所有旧版
Magpie，启动这一版，选择 `SESR-M7 NPU x2` 后缩放一个窗口。

日志第一阶段应显示 ORT 版本以及缓存名
`modelcachekey_sesr_nhwc_int8_256x256`。测试时同时打开任务管理器的 NPU 页面。
若失败或仍然卡顿，请上传新的 `logs\magpie.log`；不要重复安装编译环境。
