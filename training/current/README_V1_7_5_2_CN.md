# GameSR v1.7.5.2 Training Log Hotfix

这是针对 v1.7.5 / v1.7.5.1 续训时 `training_log.csv` 旧格式兼容问题的小型热修复。

## 修复的问题

如果一个模型从较旧 GameSR 版本一路续训，旧 `training_log.csv` 可能存在：

- 表头还是旧字段；
- 某些后来的行已经写入更多指标；
- Python `csv.DictReader` 会把这些“超出表头的值”放入特殊的 `None` 键；
- v1.7.5 扩展表头并重写旧行时因此报错：

`ValueError: dict contains fields not in fieldnames: None`

## v1.7.5.2 的处理

- 自动识别旧表头和多余列；
- 能按新增指标顺序推断的旧数据会自动映射到新字段；
- 无法可靠映射的额外值会保存在 `legacy_extra_N` 列，不静默丢弃；
- 先写临时文件，成功后才原子替换原 `training_log.csv`；
- 新训练和已经是新 schema 的日志继续直接追加，不额外重写。

## 安装

把补丁中的：

`gamesr\utils.py`

覆盖到现有 GameSR v1.7.5.1 训练目录的：

`gamesr\utils.py`

即可。不要删除 `outputs`、checkpoint、dataset、`.venv` 或 `.venv-quark`。

## 续训

原来的 `latest.pth` 可以继续使用。重新启动同一个 C/L 即可从原 checkpoint 继续。
