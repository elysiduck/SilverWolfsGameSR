# GameSR training v1.7.5 — Persistent RAM Cache + 自动剩余内存检测

这一版是 v1.7.4 RAM Streaming Prefetch 的直接升级，专门解决两个现象：

1. 每个 epoch 都从 SMB/掌机重新读取同一批 RAW/PNG/JPG，网络重复搬运非常多；
2. 一个 epoch 结束后，下一轮又要重新填充 read-ahead，GPU 会再次等第一批数据。

你此前的实际日志：

```text
time=119.4s img/s=25.3 wait=101.7s gpu_stage=17.6s gpu_stream_busy~15%
[STREAM] read=5240.1MiB avg_net=43.9MiB/s prefetch_hit=6.8% ready_q=2.2 batch_build=1254.6ms
```

说明 v1.7.4 虽然可以并发读取，但缓存只活在单个 epoch 内，重复训练 100 epoch 时仍然会反复从网络取同一批源图。

## v1.7.5 的核心变化

### 1. Persistent Compressed RAM Cache（跨 Epoch 保留）

缓存的是**原始压缩图片字节**，不是解码后的 float32 全尺寸图片。

```text
掌机/SMB Dataset
       ↓ 第一次 miss 才走网络
Persistent RAM Cache
       ↓
随机 crop / decode
       ↓
Pinned batch queue
       ↓
CUDA async copy
       ↓
GPU
```

同一张源图第一次从 SMB 读到后，可以在后面的 epoch 直接从 RAM 命中。

训练结束/进程退出后 RAM 自动释放；**不会在 NVMe 上创建 Dataset 副本**。

### 2. 自动检测“当前剩余内存”

启动正式训练时会直接读取 Windows 当前物理内存状态，不需要安装 psutil。

例如：

```text
[RAM] total=31.72GiB | currently available=22.14GiB | cache=7.07GiB (auto) | reserve=8.00GiB
```

默认自动策略：

- 先给 Windows / DataLoader / 其它程序留安全余量；
- 从剩余可用内存中取约 50% 给压缩图缓存；
- 默认最多 12 GiB；
- 同时不超过总物理内存的约 35%；
- 如果安全可用空间不足 1 GiB，则自动关闭 persistent cache，但普通 streaming read-ahead 仍可工作。

因此：

- 现在 3000 多张、约 5 GiB：如果当前剩余 RAM 足够，很可能基本全缓存；
- 以后 10000 张、约 17 GiB：不会硬塞 17 GiB，只使用自动算出的有限缓存；
- 更大的 Dataset 也不会无限吃内存。

### 3. 两级缓存，避免大 Dataset 的 LRU 抖动

v1.7.5 默认把缓存分成：

- 80% Protected：跨 epoch 稳定保留，避免全量随机 shuffle 时被 LRU 反复冲掉；
- 20% Rolling LRU：给最近/下一 epoch 的预取使用。

这样 Dataset 大于 RAM cache 时，也能保持一部分稳定命中率，而不是每轮把缓存完全洗掉。

### 4. Next-Epoch Prewarm

Epoch N 训练结束后，会立刻根据 Epoch N+1 的确定性 shuffle 顺序，提前请求下一轮前 256 个样本。

这段预取会和：

- validation
- 保存 checkpoint
- preview

重叠执行。

所以你看到的“每轮结束后又等一会儿才开始下一轮”应该显著减少。

### 5. 内存压力自动退让

如果训练过程中别的软件突然占用大量 RAM，导致当前可用内存跌到安全阈值以下，缓存会：

1. 先清 Rolling 区；
2. 仍然不足时再释放 Protected 区；
3. 不会为了保持缓存命中率把 Windows 逼到换页。

默认紧急阈值为 4 GiB 可用物理内存。

## 安装

把这个补丁里的文件直接覆盖到你当前的 GameSR 训练目录，例如：

```text
GameSR-local-training-env-v1.1\gamesr-local-training\
```

不要删除：

```text
.venv
.venv-quark
dataset
outputs
models
```

不需要重新安装 Python / CUDA / PyTorch / AMD Quark。

之后仍然双击：

```text
Train-Choose-C-L.cmd
```

### 已经在训练的进程

不要在旧 Python 训练进程运行时覆盖。先正常结束或 Ctrl+C 停止，再覆盖补丁。

## 已有 C/L 配置不需要重建

即使你已有：

```text
configs\gamesr-c32-l3.yaml
```

里面还是 v1.7.4 的 streaming 字段，也没关系。

v1.7.5 会为缺失的新字段自动使用默认值，所以可以直接继续用原配置和原 checkpoint。

为了避免纯数据加载补丁导致已有 checkpoint 被判定“不兼容”，生成的新配置仍保持原模型 checkpoint ABI `version: 1.7.4`，并额外记录：

```yaml
trainer_version: 1.7.5
```

模型结构没有变化。

## 新配置默认字段

以后通过 C/L 选择器生成的新 YAML 会包含：

```yaml
streaming_prefetch:
  enabled: auto
  read_ahead_samples: 256
  fetch_threads: 24
  decode_threads: 6
  pin_memory: true
  batch_queue: 4
  cuda_prefetch: true

  persistent_cache: true
  cache_budget_gib: auto
  cache_reserve_gib: 8.0
  cache_cap_gib: 12.0
  cache_auto_fraction: 0.50
  cache_floor_gib: 1.0
  cache_protected_ratio: 0.80
  emergency_free_gib: 4.0
  next_epoch_prefetch_samples: 256
```

## 日志怎么看

v1.7.5 每轮会输出类似：

```text
[STREAM] net_read=450.2MiB avg_net=12.3MiB/s cache_hit=91.4% cache=5.08/6.00GiB prefetch_hit=97.6% ready_q=183.2 batch_build=2.8ms
[RAM] available_now=14.72GiB cache_entries=2890 evict=0 pressure_evict=0 next_epoch_prewarm=256
```

重点看：

- `net_read`：本 epoch 真正从 SMB 新读取了多少数据；Epoch 2+ 理想情况下应大幅下降；
- `cache_hit`：源图压缩字节在 RAM 中命中的比例；
- `cache=current/budget`：当前占用 / 自动预算；
- `prefetch_hit`：GPU 请求 batch 时，解码任务是否已准备好；
- `ready_q`：未来已完成的样本数量；
- `batch_build`：主训练线程等待/组 batch 的平均时间；
- `available_now`：当前 Windows 剩余可用物理 RAM；
- `pressure_evict`：因为系统内存压力主动释放了多少缓存条目；
- `next_epoch_prewarm`：已提前提交多少张下一 epoch 开头图片。

训练总结还会多显示：

```text
val_time=...s save_time=...s
```

这样以后如果“epoch 结束后有空档”，可以区分到底是 validation、checkpoint 保存，还是下一轮数据加载。

## 手动覆盖自动 RAM 预算（可选）

一般建议保持 `auto`。

如果你明确想固定 8 GiB：

```text
set GAMESR_RAM_CACHE_GIB=8
```

其它可选环境变量：

```text
GAMESR_RAM_CACHE_GIB=auto/4/8/12
GAMESR_RAM_CACHE_CAP_GIB=12
GAMESR_RAM_CACHE_RESERVE_GIB=8
GAMESR_READ_AHEAD=256
GAMESR_FETCH_THREADS=24
GAMESR_DECODE_THREADS=6
GAMESR_STREAMING_PREFETCH=auto/on/off
```

## 对 10000 张图片会怎样？

假设 10000 张源图约 17 GiB，而自动预算只有 7 GiB：

- 不会尝试缓存全部 17 GiB；
- Protected 区会稳定保留约 5.6 GiB；
- Rolling 区约 1.4 GiB 用于未来样本/下一轮预取；
- 未命中的样本继续通过 SMB read-ahead 读取；
- 训练仍然每个 epoch 全量 shuffle，不改变训练随机性。

因此 v1.7.5 不依赖“Dataset 必须小于内存”。
