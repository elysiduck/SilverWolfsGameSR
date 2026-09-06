# GameSR v1.8.6 Stable Phase-Residual Training Patch

适用基础版本：`gamesr-local-training-v1.8.5-CLO-phase3x3-residual-shared-env`

## 安装

1. 停止正在运行的 v1.8.5 训练。
2. 备份当前训练目录（可选，但推荐）。
3. 将本补丁 ZIP **直接解压到 v1.8.5 训练目录根目录**，选择覆盖同名文件。
4. 双击 `VERIFY-V1.8.6-STABLE-TRAINING.cmd` 验证。
5. 继续使用 `Train-Choose-C-L-O.cmd`，建议先选 `C16 / L4 / O4`。

新版 O>3 默认训练目录使用 `-phase3x3r-v3`，不会自动续训旧 `-phase3x3r-v2` checkpoint。
即使手工指定旧 checkpoint，新加入的 `training_contract=phase_residual_stable_v186` 也会默认阻止误续训；除非显式使用不兼容续训开关。

## v1.8.6 集成修正：Epoch 输入

本补丁已经集成 epoch 覆盖逻辑修复，可直接覆盖原始 v1.8.5，不需要先安装旧版 v1.8.6。

- `Epochs [100]` 留空：最大 100 轮。
- 输入 `100`：最大 100 轮。
- 输入 `120`：最大 120 轮。
- 不再存在 `epochs == 100` 时强制改成 80 的隐藏逻辑。
- Early Stopping 属于独立质量控制：如果满足提前停止条件，仍可能在最大 epoch 之前结束。

## 主要改动

- 保持 `Bilinear + PhaseAware3x3 RGB residual` decoder 不变。
- phase loss：MS-SSIM `0.10 -> 0.03`，Gradient `0.05 -> 0.02`，HF `0.02 -> 0.004`，Ringing `0.01 -> 0.02`。
- detail ramp：`15%` 开始，`55%` 到满；`70% -> 90%` 进入真正的 polish downturn。
- 退化 curriculum：前期概率真正随 strength 从低到高，不再存在 60% 隐藏概率下限。
- 最终训练仍保留约 `75%` clean anchor，约 `25%` degraded samples，防止鲁棒性训练覆盖纯 SR 能力。
- resize kernel 权重：bicubic/bilinear/area = `75/20/5`，kernel diversity 也随 curriculum 渐入。
- hard mining：mix 约 `2% -> 4%`，polish 降到约 `1%`。
- Epoch 上限完全尊重用户输入：交互界面留空默认 `100`；输入 `100` 就最多训练 100 轮，输入 `120` 就最多训练 120 轮。不会再把 `100` 隐式改成 `80`。
- LR `1.5e-4`，3 epoch warmup，前段更快下降，并加入验证集 plateau `LR x0.5`。Early Stopping 仍可能在长期无提升时提前停止。
- Quality Brake：
  - `edge_ratio > 1.08`：Gradient/HF 压到 <= 50%。
  - `edge_ratio > 1.12`：Gradient/HF 暂停。
  - Ringing loss 不随此刹车关闭。
  - PSNR 明显掉出 best 或 oversharp 过高时也会压低细节项。
  - 指标恢复后会缓慢恢复细节项。
- Early stopping：训练进度 70% 后，连续 15 个可判定 epoch 未提升 >= 0.02 dB 时停止。
- checkpoint 分离：
  - `best_psnr.pth`：纯 PSNR 最佳。
  - `best_quality.pth`：满足 edge/detail/oversharp 约束后 PSNR 最佳。
  - `best.pth`：兼容旧流程，优先跟随 best_quality。
  - `latest.pth`：最新状态。
- pipeline 导出优先级：`best_quality.pth -> best_psnr.pth -> best.pth -> latest.pth`。

## C16/L4/O4 默认验收重点

不要只看最终 epoch。建议持续观察：

- `delta PSNR`：应长期 > 0，目标尽量保持 > +1.5 dB。
- `detail dDelta`：保持正值。
- `edge_ratio`：优先控制在约 `1.00 ~ 1.08`。
- `over`：不要持续上升。
- `[CONTROL]`：出现时表示质量刹车或 LR plateau 控制已触发。
- `[V1.8.6 PHASE]`：会打印本轮实际 MS/Grad/HF/Ring 权重及 detail scale。

## 说明

本补丁没有修改 decoder 输出语义，仍然是 `rgb_residual`，`base_skip=1.0`。因此本轮的核心目标是验证训练曲线是否从“约 epoch 9 达峰后持续过锐/掉 PSNR”变成“快速超过 bilinear 后稳定平台，并在完整 curriculum 下仍保持优势”。
