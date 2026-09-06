# GameSR v1.8.7 数据集管理 + 稳定 Phase-Residual 训练完整补丁

适用基础版本：`gamesr-local-training-v1.8.5-CLO-phase3x3-residual-shared-env`

这是**完整覆盖式补丁**。不需要先装 v1.8.6；直接把 ZIP 解压到原始 v1.8.5 根目录并覆盖即可。补丁已经包含此前 v1.8.6 的所有稳定训练改动和 Epoch=100 修复。

## 安装

1. 停止当前训练。
2. 将本 ZIP 解压到 v1.8.5 训练目录根目录，选择“覆盖同名文件”。
3. 双击 `VERIFY-V1.8.7-DATASET-STABLE.cmd`。
4. 继续使用原来的 `Train-Choose-C-L-O.cmd` / `GameSR_Train.cmd`。

补丁**不会覆盖你的 `dataset/*.csv`、clean 图片、outputs/checkpoint 或模型文件**。数据集只有在训练开始时由你明确选择 Incremental / Full Rebuild 才会改写。

## 每次训练前新增 Dataset Manager

启动训练后会先显示当前 manifest 状态，并在已知 clean 源目录时做一次“只列文件、不解码图片”的快速盘点：

```text
Train: xxxx
Val  : xxxx
Test : xxxx
Clean files found: xxxx | Manifest files: xxxx | New: xxxx | Missing: xxxx

[1] Reuse existing manifests
[2] Incremental update
[3] Full rebuild
[4] Show detailed statistics
```

### 1. Reuse

完全复用当前 `train.csv / val.csv / test.csv`，不重新随机划分。适合没有新增数据时做模型 A/B 对比。

### 2. Incremental（健康数据集后推荐）

- 保留已有 Session 的 Train/Val/Test 归属，避免相邻帧泄漏。
- 新图片如果属于已有 Session，会保持该 Session 原来的归属。
- 如果当前 Val/Test 已健康，新的全新 Session 默认进入 Train，尽量保持 benchmark 稳定。
- 如果 Val/Test 不达最低门槛，才会用新的 Session 优先修补 Val/Test。
- 只校验新增图片，不重新解码全部旧数据。

### 3. Full Rebuild（你目前第一次升级推荐）

旧算法按“Session 个数”切 80/10/10，因此 1 张图片的 Session 可能成为整个 Test。v1.8.7 改为：

- **Session 绝不跨 Train/Val/Test**；
- 同时根据 **Session 内图片数量** 搜索尽量接近 `80% / 10% / 10%` 的组合；
- 默认要求 Val/Test 各至少约 `max(300 张, 总数据 5%)`，且数据允许时至少 2 个 Session；
- 极端 Session 分布无法满足目标时会明确 WARN，不再把 `Test=1` 当成正常数据集。

你现在旧 manifest 已知 `Test=1`，第一次运行建议直接选 `[3] Full Rebuild`，用目前约 6000 张 clean 重新建立基准。

### 4. Stats

显示 dataset 版本、Train/Val/Test 图片数和 Session 数，以及 clean 源目录中新增/缺失文件数量。

## 数据集版本与历史

每次 Incremental / Full Rebuild 会生成：

- `dataset/dataset_info.json`
- `dataset/history/...`：自动备份上一个版本的 manifest 和 metadata
- `manifest_fingerprint`：当前三份 manifest 的 SHA256 契约

例如：

```text
dataset v0001 -> Full Rebuild 6000 张
dataset v0002 -> Incremental 加入新采集
dataset v0003 -> 下一次 Incremental
```

训练 checkpoint 也会记录 `dataset_contract`。只要 manifest 改变，旧 `latest.pth` 就会被自动判定为不同数据集，**自动续训会跳过旧 checkpoint 并从新训练开始**，防止 3000 张旧数据的优化器状态混进 6000 张新实验。

## v1.8.6 稳定训练改进全部保留

- `Bilinear + PhaseAware3x3 RGB residual`；`base_skip=1.0`，decoder 零初始化语义不变。
- MS-SSIM `0.03`、Gradient `0.02`、HF `0.004`、Ringing `0.02`。
- detail ramp：15% 开始、55% 到峰值；70%~90% polish 回落。
- 修复 degradation 隐藏概率下限；最终保留约 75% clean anchor。
- resize kernel：bicubic/bilinear/area = `75/20/5`。
- hard mining 约 `2%~4%`，polish 降至约 `1%`。
- LR `1.5e-4` + plateau 自动 `x0.5` + Quality Brake。
- Early Stopping 仍保留。
- `best_quality.pth / best_psnr.pth / best.pth / latest.pth` 分离。
- 导出默认优先 `best_quality.pth`。
- Epoch 输入完全尊重用户：`Epochs [100]` 留空=100；输入100=100；输入120=120，不再有 100→80 隐藏覆盖。

## 新实验隔离

O>3 默认输出目录升级为：

```text
...-phase3x3r-v4
```

训练契约为：

```text
phase_residual_stable_v187
```

因此不会误续训 v1.8.5 的 v2 或旧 v1.8.6 的 v3 checkpoint。

## 建议你第一次运行

1. Dataset Manager 选择 `[3] Full Rebuild`。
2. 确认新 Train/Val/Test 数量和 Session 数合理；Test 不应再只有 1 张。
3. 继续先跑 `C16 / L4 / O4`，Epoch 可输入 100。
4. 后续采集增加后通常选 `[2] Incremental`。
