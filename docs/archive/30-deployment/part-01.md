4. 先选择 `Lanczos` 并启动缩放。
5. 如果 Lanczos 正常，退出缩放，再选择 `SESR-M7 NPU x2`。
6. 启动 SESR 后等待最多 20 秒。首次创建 NPU 上下文可能不是瞬间完成。

## 发生崩溃后发回什么

进入新版 Magpie 文件夹中的 `logs`，发回：

- `magpie.log`
- 最新的 `magpie-crash-*.dmp`（如果存在）

日志中的最后一个“SESR 诊断阶段”可以直接定位崩溃位置。

## 输入尺寸

第一次只用于查崩溃，不要先追求画质。建议用约 `960×540` 的窗口。之前日志中的捕获尺寸是 `1128×634`，经过二倍模型会先生成 `2256×1268`，然后再缩到掌机的 `1920×1080`，会多做不少无用计算。



---

## `docs/30-deployment/magpie-sesr-v2.1/SESR-v1.7-X1混合节点诊断说明.md`

# SESR-M7 X1 混合节点诊断 v1.7

v1.6 证明旧运行库的 X1 编译器仍会把少量节点分配到 CPU。AMD 官方成功示例本身也可能显示少量 `VITIS_EP_CPU` 节点，因此禁止任何 CPU 节点会错误拦截可用的 NPU 子图。

v1.7 继续使用本机编译的 X1 缓存，但允许必要的 CPU 边界节点，并把 ONNX Runtime 的 NPU、VitisAI、节点分配和警告信息写入 Magpie 日志。程序不再仅凭“会话初始化完成”判断成功：首帧推理超过 250 ms 时会明确警告可能发生主要 CPU 回退。

## 测试

1. 解压到纯英文路径并编译。
2. 把整个 `publish\x64` 放到掌机新目录，直接运行其中的 `Magpie.exe`。
3. 确认日志版本为 `SESR-M7-ONNX-NPU-v1.7`。
4. 使用约 `960×540` 的普通窗口测试。
5. 第一次创建会话可能生成 `sesr\modelcachekey_sesr_nhwc_int8_256x256_ort1230_x1_v17`。
6. 测试十几秒，同时观察任务管理器 NPU，然后把完整日志发回。

判断标准：日志中应出现 VitisAI/NPU 子图信息，NPU 曲线应有活动，首帧推理应远低于 v1.4 的约 1400 ms。


---

## `docs/30-deployment/magpie-sesr-v2.1/SESR-v1.8-测试说明.md`

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


---

## `docs/30-deployment/magpie-sesr-v2.1/SESR-v1.9-性能模式说明.md`

# SESR-M7 NPU v1.9 性能模式

## 两个模式

- `SESR-M7 NPU x2`：画质模式，每块保留 16 像素重叠，减少接缝。
- `SESR-M7 NPU x2 Fast`：快速模式，不做重复边缘计算。

对于日志中的 1024×751 输入：

- 画质模式：5×4，共 20 块；
- 快速模式：4×3，共 12 块。

快速模式预计可减少约 40% 的 NPU 提交次数，但块边缘可能出现轻微接缝。
本版还会在初始化阶段创建并复用 ONNX 输入输出张量，避免每帧重复创建 40 个
张量对象。

## 测试方法

先选择 `SESR-M7 NPU x2 Fast`，使用与 v1.8 完全相同的游戏窗口和功耗设置运行
一分钟。日志第一阶段应显示“快速（无重叠）”和“切片数 12”。测试后上传
`logs\magpie.log`。


---

## `docs/30-deployment/magpie-sesr-v2.1/SESR-v2.0-精确1080p模式说明.md`

# SESR-M7 NPU v2.0 精确 1080p 模式

> v2.1 修复了 `Renderer.cpp` 中 GPU 预缩放效果编译函数缺少前置声明、导致
> MSVC 报 `C3861: CompileEffect 找不到标识符` 的问题。模式算法保持不变。

新增效果：`SESR-M7 NPU Exact 1080p`。

该模式自动读取 Magpie 的渲染矩形和源画面长宽比。以测试日志中的
1024×751 源画面、1920×1080 屏幕为例，有效显示区域为 1473×1080：

1. GPU Bicubic 将捕获纹理从 1024×751 缩到 737×540；
2. SESR-M7 NPU 以无重叠模式处理 3×3，共 9 个切片；
3. NPU 输出 1474×1080，与有效显示区域只差 1 个水平像素。

游戏本身不需要支持修改分辨率。预缩放在 GPU 上完成，CPU 只负责读取已经缩小的
纹理、切片和拼接。

尺寸不是固定值。Magpie 每次建立或重建捕获纹理时，都会根据当前捕获尺寸、输出
区域和原始长宽比重新计算模型输入尺寸。上面的 737×540 只对应 1024×751 的示例。

测试时选择 `SESR-M7 NPU Exact 1080p`。日志应包含：

`GPU 预缩放 1024x751 -> 737x540，NPU 输出 1474x1080`

以及：

`模式 快速（无重叠），输入 737x540，切片数 9`

若 Magpie 最外层没有自动出现新缩放模式，请编辑现有 SESR 缩放模式，在效果链中
把第一个效果替换为 `SESR-M7 NPU Exact 1080p`。


---

## `docs/30-deployment/magpie-sesr-v2.1/SESR-使用说明.md`

# Magpie SESR-M7 NPU ×2 测试版

这个版本把 AMD SESR-M7 INT8 模型接入 Magpie，并通过 ONNX Runtime 的 `VitisAIExecutionProvider` 在 Ryzen AI 9 HX 365 的 XDNA2 NPU 上执行推理。

## 推荐分辨率

你的屏幕是 1920×1080，因此游戏输入建议设置为 960×540：

\[
960\times540 \xrightarrow{\text{SESR-M7 }2\times} 1920\times1080
\]

不要先用 1280×720 测试。固定的 2× 模型会生成 2560×1440，随后又要缩回 1920×1080，会浪费大量性能。

## 编译（不用自己研究）

1. 双击 `0-先双击我-检查环境.cmd`。
2. 它会在同一文件夹生成 `环境检查结果.txt`。
3. 把这个文本发给我，不用看懂，也不用自己安装东西。
4. 环境齐全后才双击 `构建-SESR.cmd`。编译完成后，程序位于 `publish\x64\Magpie.exe`。

## 第一次测试

1. 将游戏改为 960×540、窗口模式或无边框窗口模式。
2. 启动这个版本的 Magpie。
3. 缩放模式选择 `SESR-M7 NPU ×2`。
4. 暂时不要叠加其他锐化、降噪或插帧效果。
5. 启动缩放并运行游戏约两分钟。
6. 退出 Magpie，打开 `logs\magpie.log`。
7. 搜索 `SESR ONNX 帧耗时`，把相关行发给我。

日志会分别显示读回切片、NPU 推理和拼接上传耗时。理论处理帧率可按下式估算：

\[
FPS_{\max}\approx\frac{1000}{t_{\text{读回}}+t_{\text{NPU}}+t_{\text{上传}}}
\]

## 当前限制

- 这是需要在实际 XDNA2 设备上验证的第一版，尚未验证最终帧率。
- 使用 16 像素重叠切片，逐切片提交给单个 ONNX Runtime NPU 会话，避免旧 GraphEngine 直接创建上下文时的兼容性崩溃。
- Direct3D 纹理与 NPU 之间仍有一次读回和一次上传；日志可以判断瓶颈究竟在 NPU 还是内存搬运。
- 目前仅支持 x64 Ryzen AI 设备。

## 第三方组件

- Magpie：GPL-3.0
- ONNX Runtime / AMD VitisAI EP：MIT 及对应组件许可证
- AMD SESR-M7 模型：Apache-2.0
