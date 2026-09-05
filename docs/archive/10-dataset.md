# GameSR 文档归档：10-dataset

> 本文件由项目现有文档自动归档生成；每个原始文件均保留路径标题与完整文本内容。

---

## `docs/10-dataset/collector/README_CN.md`

# GameSR Dataset Collector V3.2

V3.2 基于 V2.4 升级。采集端仍然只在游戏期间截图和写入 metadata；所有分析都在退出游戏后由 Cleaner 离线完成。

V3.2 新增实时清洗进度：显示完成百分比、已扫描/总数、KEEP/REVIEW/REJECT/ERROR/SKIP 计数、处理速度与 ETA。旧数据库指纹迁移也会显示单独进度。

## 一键使用

1. 首次解压后双击 `install.bat`。
2. 玩游戏前双击 `start_collector.bat`。
3. 退出游戏后双击 `run_cleaner.bat`。
4. 需要查看整体数据质量时双击 `run_audit.bat`，再打开 `GameSR_Dataset/reports/dataset_audit.html`。

## 新版 Cleaner

- 真正的增量处理：文件大小和修改时间未变化时直接跳过。
- 历史全局去重：新图片会与所有游戏、所有 Session 的历史 KEEP 指纹比较。
- 双感知指纹：一致的 64-bit dHash + 64-bit pHash；8 段索引在阈值不超过 7 时保证召回候选。
- V2.4 SQLite 自动迁移：首次运行会为历史 KEEP 图片补算新指纹和质量指标，不删除 RAW。
- 质量评分：综合曝光、对比度、熵、细节、高光/暗部截断和黑边，输出 0-100 分。
- 异常检测：软帧、严重截断、重黑边、极端偏色、低熵均会记录 flags。
- 新颖度/密度采样：不做强制场景平均；只稀疏高度密集且低 novelty 的视觉区域。
- 三层结果：`KEEP`、`REVIEW`、`REJECT`。不确定样本进入 REVIEW，而不是直接删除。

## 输出结构

```text
GameSR_Dataset/
├── raw/                         # 原始数据，Cleaner 永不删除
├── clean/                       # KEEP，训练候选
├── review/                      # 低质量/异常/密集低新颖度，供抽查
└── reports/
    ├── cleaner_state.sqlite3    # 持久化增量状态与全局指纹
    ├── last_run.csv             # 本轮处理结果
    ├── dataset_report.csv       # 全量逐图指标
    ├── summary.json             # Cleaner 汇总
    ├── dataset_audit.json       # 可机器读取的审计
    ├── dataset_audit.html       # 可直接打开的审计报告
    └── audit_contact_sheet.jpg  # REVIEW/低分样本抽查图
```

## 决策原则

- 错误尺寸、无法解码、黑屏/极低信息、历史近重复：REJECT。
- 指标可疑但仍可能是真实游戏画面：REVIEW。
- 高质量样本即使所在区域较密集，也可由 `quality_override` 保留。
- 密集区采样使用路径的稳定哈希，相同数据和配置重复重建可得到相同结果。
- RAW 永不自动删除；修改算法后可双击 `rebuild_clean.bat` 全量重建。

## Dataset Audit

Audit 与 Cleaner 分离，负责整体评价而不是逐图清理。报告包括：

- KEEP / REVIEW / REJECT / ERROR 数量；
- 各游戏数量、平均质量和平均新颖度；
- 质量分层、拒绝原因和异常 flag；
- 单游戏占比过高、重复率、REVIEW 积压等告警；
- Session 级亮度、对比度、偏色、质量的统计异常；
- 便于人工抽查的 contact sheet。

Dataset Audit 不会自动决定训练集是否“平均”，只报告风险。真实游玩分布仍是主分布。

## V2.4 数据迁移

如果已有 V2.4/V3.1 的 `GameSR_Dataset`，将它完整复制到 V3.2 根目录后运行 `run_cleaner.bat` 即可。旧 SQLite 会原位增加新列和新指纹表。历史 KEEP 数据只做一次指标回填；旧 RAW 不需要重新采集。

若希望所有历史 REJECT 也按 V3.2 新规则重新判断，使用 `rebuild_clean.bat`。完整重建会清空 clean/review 和 Cleaner 数据库，但不会改动或删除 raw。

## 配置提示

默认配置位于 `cleaner_config.json`。不建议随意提高两个 hash 阈值；它们必须小于等于 7。`dense_keep_probability` 只控制密集低 novelty 区域的软采样比例，不是全局保留率。

采集 HR 时仍建议：游戏输出 1920x1080、Render Scale 100%、关闭 DLSS/FSR/XeSS 超分和动态分辨率。光追可以正常开关。

---

## `docs/10-dataset/collector/VERSION.txt`

```text
3.2
```

---

## `docs/10-dataset/cleaner/覆盖安装说明.txt`

```text
GameSR Cleaner V3.2.1 空间统计小补丁
======================================

安装：

1. 先关闭 Collector 和 Cleaner 窗口。
2. 把本压缩包内的 cleaner.py、dataset_audit.py、test_pipeline.py
   复制到 GameSR_Dataset_Collector_V3_2 主目录。
3. Windows 询问时选择“替换目标中的文件”。
4. 不要复制到 GameSR_Dataset 数据文件夹里。
5. 双击 run_self_test.bat；看到 ALL TESTS PASSED 即安装成功。

补丁不会删除或重新复制已有图片，也不会改变 KEEP/REVIEW/REJECT 判断。

运行 run_cleaner.bat 后，终端和 reports/summary.json 会显示本轮：

- hardlinks_created：硬链接成功数量；
- copies_created：退回复制数量；
- hardlink_bytes_saved：估算节省的字节数；
- copied_bytes：普通复制额外占用的字节数。

运行 run_audit.bat 后，dataset_audit.json 和 dataset_audit.html 会检查
当前所有 KEEP/REVIEW 文件，显示硬链接、普通复制、节省空间和缺失文件。

如果 Cleaner 显示 SKIP，表示图片之前已经处理过，因此“本轮新建”统计可能为0；
run_audit.bat 仍会统计全部已有文件。
```

---

## `docs/10-dataset/dependencies/requirements-cleaner.txt`

```text
numpy>=2.0,<3
Pillow>=10,<13
```

---

## `docs/10-dataset/dependencies/requirements-collector.txt`

```text
dxcam==0.3.0
psutil>=6.0,<8
Pillow>=10,<13
```
