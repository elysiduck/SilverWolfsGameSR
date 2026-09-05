# GameSR 文档归档：30-deployment

> 本文件由项目现有文档自动归档生成；每个原始文件均保留路径标题与完整文本内容。

---

## `docs/30-deployment/训练模型接入Magpie.txt`

```text
对，**你这个判断方向基本是对的**，但要稍微改一句：

> **不是“占用率没到 100% 所以 NPU 没用好”这么简单，而是我们当前模型很可能没有把 XDNA NPU 的计算阵列喂得足够饱。**

而你这次日志其实给了很强的证据。

Fast 模型稳定时，NPU 大约 **16.6–17.5 ms**；Quality 则大约 **24.7–25.7 ms**。而且两者日志都明确显示 `All nodes placed on [VitisAIExecutionProvider]`，也就是说不是因为大量算子偷偷掉到 CPU 上。fileciteturn4file0

### 先说一个很反直觉的点

我们的 Fast 模型虽然只有：

```text id="lynght"
约 3947 参数
```

但它并不代表计算量很小。

因为它是在：

```text id="l4ksu9"
1280 × 720
```

这么大的整张图上，**连续做很多层 3×3 卷积**。

可以把它想成：

> 工人只有几千件工具，但这几千件工具要在 92 万个像素上反复干活很多遍。

所以“参数少”≠“推理快”。

这也是为什么一个只有几千参数的小 CNN，居然还能让 NPU 跑 17ms。

---

## 而你看到“不满 100%”，最值得怀疑的是 C8

我们现在 Fast 大概是这种东西：

```text id="74qytt"
RGB
 ↓
C8
 ↓
C8
 ↓
C8
 ↓
C8
 ...
 ↓
RGB residual
```

这里的 **C8 = 每层只有 8 个通道**。

对于 CPU 来说：

> 通道少 → 运算少 → 快。

但对于一个高度并行的 NPU，不一定。

NPU更像有一大群工人：

```text id="poqtau"
████████████████████████████
很多并行计算单元
```

结果我们每次只给它：

```text id="zc68mb"
████████
8 个通道的活
```

可能出现：

```text id="4lmw0s"
有些计算单元工作
有些计算单元等着
有些周期搬数据
有些周期计算
```

于是你看到的就是：

> 800 MHz 没问题，但 utilization 上不去。

AMD自己的性能分析工具也明确把 **Utilization、计算性能、内存带宽**分开衡量，因为模型可能受到计算利用率或内存访问等不同因素限制。([docs.amd.com](https://docs.amd.com/r/1.4-English/ug1414-vitis-ai/Fine-Grained-Profiling?utm_source=chatgpt.com))

---

# ASR 100% 但帧数不高，并不矛盾

这个特别重要。

比如微软 ASR：

```text id="wwq1oi"
NPU利用率：100%
处理一次：30ms
```

而我们的：

```text id="ilr894"
NPU利用率：60%
处理一次：17ms
```

完全可能同时成立。

100%代表的是：

> **在那段时间里，NPU非常忙。**

不是：

> **它一定很快。**

举个极端例子。

一个工厂：

```text id="a4kaym"
100 个工人全部满负荷
做一个东西要 30 秒
```

另一个：

```text id="vrtxlh"
只有 60 个工人在干活
但 17 秒做完
```

第二个虽然利用率低，仍然更快。

所以微软 ASR 能到100%，只能说明它的 workload **更容易持续填满 NPU**，不能直接说明它的模型延迟一定优秀。

---

# 但对 GameSR 来说，低利用率依然是一个优化机会

这才是最关键的。

你的 Fast：

```text id="bcjcri"
NPU ≈ 16.7 ms
```

如果此时利用率只有比如 50～70%，那说明理论上存在这种可能：

```text id="2vv3m6"
同样的计算结果

现在：
较多时间 × 较低硬件并行度

优化模型后：
较少时间 × 较高硬件并行度
```

所以我们下一步**不应该只是继续砍参数**。

甚至：

> **C6 UltraFast 有可能是个错误方向。**

因为如果 C8 已经喂不饱 NPU，再缩到 C6：

```text id="xpk80d"
C8
↓
C6
```

可能只是让 NPU更吃不饱。

---

# 我现在反而想测试一个很有意思的结构

例如：

```text id="t22brt"
现在 Fast：

C8
8层
```

我们试：

```text id="wpc1pj"
GameSR NPU-WideFast：

C16
4～5层
```

看起来通道翻倍了，但层数减少。

它有可能出现：

```text id="h8xh2z"
C8 × 8层
NPU利用率 60%
NPU 17ms

↓

C16 × 4层
NPU利用率 90%
NPU 13ms
```

**参数甚至可能更多，但实际跑得更快。**

这就是做 NPU 模型和单纯看参数量最大的区别。

---

## 为什么 Wide + Shallow 可能更适合 XDNA

因为我们现在最吃亏的是：

```text id="yr5pts"
720p巨大 feature map
 ↓
Conv
 ↓
写回
 ↓
下一层再读
 ↓
Conv
 ↓
写回
 ↓
下一层……
```

每多一层，就多一次巨大的：

```text id="2lwcjr"
1280×720×feature channels
```

数据处理。

如果换成：

```text id="oydua6"
更宽
+
更少层
```

可能：

- 更好利用 NPU 并行计算单元
- 减少层间调度
- 减少中间 feature map 往返
- 提高 arithmetic intensity
- 减少 pipeline bubble

所以这很值得 benchmark。

---

# 另外我们现在还有第三个问题：每帧只提交一次同步推理

当前实际上大致是：

```text id="5ciyom"
准备 Frame N
↓
session.Run()
↓
等 NPU 完成
↓
拿 residual
↓
准备 Frame N+1
↓
session.Run()
```

这意味着即便 NPU内部某些阶段有空闲：

> 我们也没有第二帧立即塞进去填洞。

AMD 官方自己也强调，通过 **pipeline / overlap 不同 workload** 可以提高整体硬件利用率。([amd.com](https://www.amd.com/en/developer/resources/technical-articles/model-pipelining-on-npu-and-gpu-using-ryzen-ai-software.html?utm_source=chatgpt.com))

我们未来可以尝试：

```text id="1e68rb"
Frame N     → NPU
Frame N+1   → CPU prepare

NPU完成N
↓
马上提交N+1
↓
CPU准备N+2
```

不过这主要提高**吞吐量**，不一定降低单帧 latency。

对游戏而言：

> 吞吐量和延迟都重要。

所以不能为了100%利用率疯狂排队，否则：

```text id="jb1u59"
60 FPS
但延迟多两三帧
```

那玩起来反而更恶心。

---

# 你的日志现在给了我们一个非常有价值的 benchmark

Fast：

```text id="1io759"
NPU       ≈ 16.6–17.5 ms
CPU打包   ≈ 2.0–2.5 ms
上传       ≈ 0.9–1.4 ms
------------------
总计       ≈ 20 ms
≈ 50 FPS
```

Quality：

```text id="lbnrmn"
NPU       ≈ 24.7–25.7 ms
CPU打包   ≈ 2 ms
上传       ≈ 1 ms
------------------
总计       ≈ 28 ms
≈ 35 FPS
```

这个差距非常清楚。fileciteturn4file0

而两个模型都已经 **100% offload 到 VitisAI EP**。fileciteturn4file0

所以现在真正要研究的已经不是：

> “有没有跑 NPU？”

而是：

> **“我们的卷积形状是不是 NPU 喜欢的形状？”**

---

## 我现在最想做的不是一个模型，而是一组结构探针

下一版训练代码我建议直接增加一个 **NPU Architecture Benchmark**：

```text id="8kdy0u"
A：C8 × 8层      ← 当前 Fast

B：C12 × 6层

C：C16 × 4层

D：C16 × 6层

E：C24 × 4层
```

全部：

```text id="0qt5ly"
1280×720
INT8
相同输入输出
相同 residual
```

先不用认真训练。

随机权重都行。

我们要先在你的掌机上测：

```text id="o3sbno"
模型        NPU ms     NPU占用率
C8×8        16.7       例如 60%
C12×6       ?
C16×4       ?
C16×6       ?
C24×4       ?
```

然后找出一个很关键的点：

> **哪一种网络形状，在你这颗 NPU 上“每单位计算最划算”。**

再围绕那个结构认真训练。

这比我们现在凭感觉说：

> “参数越少越快”

高级很多，也靠谱很多。

### 所以我现在会修正之前的路线

之前我说：

> 做 C6 UltraFast。

我现在**不建议直接这么干了**。

你发现的“800 MHz一样，但利用率明显不同”非常关键。

下一步应该是：

> **先做 XDNA-friendly architecture benchmark，再确定真正的 Fast 模型。**

而不是继续盲目削模型。

而且 AMD 现在也明确区分了 ONNX Runtime + Vitis AI EP 与更直接的 NPU runtime：后者可以减少框架和 tensor-format conversion 的运行时开销；所以等网络结构选好后，我们还有可能继续优化 runtime，而不是永远卡在现在这套 ORT 数据路径。([docs.amd.com](https://docs.amd.com/r/en-US/vitis-video-analytics-sdk/VVAS-Pipeline-Performance-Considerations?contentId=QbtFVbuxoMzO28mfnTUuYA&utm_source=chatgpt.com))

**所以答案是：是的，你看到的低占用率很可能暴露了我们模型的 NPU 映射效率还有问题；这反而是个好消息，因为说明 17ms 未必已经是这颗 NPU 的极限。**
```

---

## `docs/30-deployment/magpie-sesr-v2.1/RYZEN-AI-RUNTIME.txt`

```text
AMD Ryzen AI Deployment runtime: 1.7.1
NuGet source package: RyzenAI_Deployment.1.7.1.nupkg
ONNX Runtime package label: 1.23.3-dev-20260320-0938-58b2075a54
ONNX Runtime commit: 58b2075a54578015d2c573c209b2ca462228672d

The ONNX Runtime commit exactly matches the commit recorded in the bundled
SESR-M7 precompiled X1 cache context.json.

Do not replace individual DLLs with files from another Ryzen AI release.
The files in onnx/bin must be deployed as one matched set.

```

---

## `docs/30-deployment/magpie-sesr-v2.1/SESR-v1.2-崩溃诊断说明.md`

# SESR-M7 NPU v1.2 崩溃诊断

这个版本不是最终性能版。它把默认 NPU 执行器数量从 4 降为 1，并在每个关键步骤后立即刷新日志；如果发生原生崩溃，还会尝试在 `logs` 文件夹生成 `magpie-crash-*.dmp`。

## 编译

在编译笔记本上解压到纯英文路径，例如：

`E:\Magpie-SESR-v1.2`

然后双击 `构建-SESR.cmd`。成功后使用：

`publish\x64`

把整个 `x64` 文件夹复制到掌机，不要只复制 `Magpie.exe`。

## 第一次测试（按顺序）

1. 在掌机上退出其他 Magpie，任务管理器中确认没有残留的 `Magpie.exe`。
2. 打开新版 Magpie 的“设置”，暂时关闭“始终以管理员身份运行”。
3. 打开一个普通的非管理员窗口，例如记事本。
