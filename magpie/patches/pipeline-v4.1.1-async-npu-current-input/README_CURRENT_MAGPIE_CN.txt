SilverWolfsGameSR - Magpie 唯一当前版本
====================================

当前版本：magpie-v2.1-pipeline-v4.1.1-async-npu-current-input

从这一版开始，本地只保留一个 Magpie 工作目录：
  SilverWolfsGameSR\\magpie\\current

不要再创建多个 v2.1 / fixed / patch / final 目录。
历史版本只通过 GitHub commit / tag / release 回退。

验证自己有没有跑对版本：
启动缩放后查看 magpie.log，必须出现：
  GameSR BUILD ID: magpie-v2.1-pipeline-v4.1.1-async-npu-current-input | contractv3 | canonical-current

如果日志没有这一行，说明运行的不是当前标准版本。

C16L4 模型运行时仍应存在：
  models\\GameSR\\gamesr-c16-l4-720p-to-1080p-int8-qdq.onnx
