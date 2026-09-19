# 历史研究档案

2026-09-19 起主线转为[分类器训练弱头](../classifier_guidance/README.md)。

| 内容 | 位置与约定 |
|---|---|
| 根目录散落的 20 组旧实验结果及总图册 | `results/`；原根目录路径是兼容软链接 |
| 历史研究代码 | [`experiments/`](../experiments/)；保持原模块名、源码和绝对路径，供复现及当前依赖使用 |
| 历史报告、协议、紧凑结果表 | [`docs/`](../docs/README.md)；分类索引见下方 |
| 归档清单、文件哈希及迁移记录 | [`manifests/20260919/`](manifests/20260919/) |
| 模型权重、采样包、数据集和下载论文 | `$HOME/data/eqvae/`，不新增进入 Git |

[分类目录](manifests/20260919/catalog.md)列出代码家族、报告和外部实验数据的位置。
数据没有删除，已有 Git 证据随目录迁移保留。历史源码不批量改写：很多旧任务按文件 SHA-256 校验，
并使用 `__file__` 推导根目录。随意搬动它们会破坏复现。

迁移脚本：[`tools/organize_workspace.py`](../tools/organize_workspace.py)。
清单是定位与完整性依据，不是外部数据备份；单独 clone 无法恢复未进 Git 的模型或图像。
