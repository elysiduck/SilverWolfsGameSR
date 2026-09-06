# GameSR training v1.7.4 — RAM Streaming Prefetch

这一版针对“笔记本 RTX 4060 训练、数据集放在掌机/SMB 共享盘”的场景。

你之前的实际日志：

- epoch: 128.4 s
- wait: 114.3 s
- gpu_stage: 12.6 s
- 网络约 394 Mbps / 49 MB/s

说明 GPU 大部分时间不是在算，而是在等下一批远程图片。

## v1.7.4 做了什么

### 1. RAM read-ahead（不占 NVMe）

训练顺序仍然随机，但程序会提前知道未来要用的图片，并在后台并发读取：

- 默认预取窗口：256 张
- SMB/文件读取线程：24
- 图片解码线程：6
- 已解码 batch 队列：4

这些内容只放在 RAM，训练结束就释放，不会把整个 Dataset 复制到笔记本 SSD。

### 2. 背景 batch 组装

网络读图、PIL 解码、crop、Tensor 组装不再全部卡在 GPU 主训练线程上。

### 3. CUDA 异步预取

已经准备好的 pinned-memory batch 会在独立 CUDA copy stream 中提前送到显存。
理论流水线变成：

SMB 读取 N+2/N+3 -> CPU 解码 N+1 -> H2D N+1 -> GPU 训练 N

### 4. 继续保留 v1.7.3 / v1.7.3.1 优化

- batch benchmark 使用小型复用 pool，不再每个 batch 重读完整数据集
- 训练 step 不再每一步 `torch.cuda.synchronize()`
- 优先使用 fused AdamW
- manifest 不再在 [2/8]/[3/8] 对几千张远程图逐个 `is_file()`
- C/L 自定义训练选择器继续可用

## 安装

把补丁内容覆盖到当前 GameSR 训练目录根目录即可，例如：

`GameSR-local-training-env-v1.1\gamesr-local-training\`

不要删除：

- `.venv`
- `.venv-quark`
- `dataset`
- `outputs`

不需要重新安装 Python / PyTorch / Quark。

之后继续双击：

`Train-Choose-C-L.cmd`

## 正常启动时会看到

类似：

```text
[STREAM] RAM read-ahead ON | remote=True | window=256 images | fetch_threads=24 | decode_threads=6 | batch_queue=4 | cuda_prefetch=True
[STREAM] Volatile RAM only: no dataset copy is written to NVMe.
```

如果显示 `RAM read-ahead OFF | remote=False`，说明自动检测认为数据在本地盘。

## 每个 epoch 新增诊断

例如：

```text
[STREAM] read=6100.0MiB avg_net=82.4MiB/s prefetch_hit=94.2% ready_q=133.5 batch_build=3.1ms
```

重点看：

- `avg_net`：本 epoch 从图片文件实际读取的数据量 / epoch 墙钟时间
- `prefetch_hit`：GPU 请求 batch 时，样本已经在 RAM 解码完成的比例；越接近 100% 越好
- `ready_q`：平均有多少未来样本已经准备好
- `batch_build`：主 batch 构造平均耗时
- 原来的 `wait=`：训练线程实际等数据的时间，目标是从 114 s 大幅下降

## 默认内存占用

256 张 read-ahead 并不是把 256 张 1080p 解码成全尺寸 float32 常驻。
网络原始压缩文件和 192x192 crop 会滚动释放。
对于你目前约 2 MB/张的图片，通常会使用几百 MB 到约 1 GB 左右的额外 RAM，具体取决于图片压缩大小和并发状态。

## 手动调参（可选）

生成的新 C/L YAML 会包含：

```yaml
streaming_prefetch:
  enabled: auto
  read_ahead_samples: 256
  fetch_threads: 24
  decode_threads: 6
  pin_memory: true
  batch_queue: 4
  cuda_prefetch: true
```

32 GB RAM 下如果网络仍然有很多空洞，可以试：

`read_ahead_samples: 512`

如果 SMB/CPU 线程太多反而抖动，可以把：

`fetch_threads: 16`

也可以临时通过环境变量覆盖：

- `GAMESR_STREAMING_PREFETCH=on/off/auto`
- `GAMESR_READ_AHEAD=512`
- `GAMESR_FETCH_THREADS=24`
- `GAMESR_DECODE_THREADS=6`

## 现实上限

RAM 预取主要解决的是 SMB open/read 往返、随机小文件访问、CPU 解码与 GPU 不能重叠的问题。
它不能突破 1GbE 的物理带宽上限。

如果当前 49 MB/s 能通过并发 read-ahead 提升到 90~110 MB/s，训练吞吐可能明显提高；但 RTX 4060 的纯 GPU 能力仍然比 1GbE 高很多，所以远程网络之后仍可能是最终瓶颈。
