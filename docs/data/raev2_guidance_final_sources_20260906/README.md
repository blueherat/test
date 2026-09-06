# RAEv2 来源与复查脚本补充包

本包补齐 2026-09-06 研究材料中原先只保存在实验数据盘的轻量来源记录和辅助脚本。它补充 [终验数据包](../raev2_guidance_final_20260906/README.md)，不代表方法达到 ≥5% FID 目标，也不启动任何实验。

- [sources_plan.json](sources_plan.json)：一次冻结的精确来源选择及 SHA-256。211 条来源记录包含 155 条阅读来源/解析结果 JSON 和 56 条辅助 Python 源码；相同内容只复制一次，实际 203 份文件、1,367,202 字节。
- [manifest.json](manifest.json)：每条来源的原逻辑路径、解析后路径、文件大小、SHA-256 和包内位置。另有 45 个与现有仓库源码逐字节相同的来源副本只登记映射，避免重复导入。
- [supplement/manifest.json](supplement/manifest.json)：基础包冻结后最后两轮新增来源的独立追加清单，连接原 manifest SHA；包括 Moser / FP-Diffusion 阅读来源以及 temporal-score 的根线程独立审核，已在基础包的文件仅记录引用。
- [validation.json](validation.json)：可刷新的便携包字节校验与文档链接检查快照。它检查复制文件，不重新验证全部科学结论或大资产。
- [proposed_git_paths.txt](proposed_git_paths.txt)、[proposed_git_inventory.json](proposed_git_inventory.json)：当前改动中与此次研究范围相符的精确提交建议及身份。清单会随主线程最后几轮结果更新；生成器自身不执行 `git add`、commit 或 push。

从仓库根运行 `python experiments/prepare_raev2_final_git_manifest.py` 可验证固定来源包、原终验数据包并刷新提交建议。固定来源发生变化或旧复制文件遭修改时会失败，不会悄悄覆盖它们。来源包要求原实验目录仍在登记位置；复制脚本中的历史绝对路径保持原样，它们是归档源码，并非迁移后即可无条件重启的作业。

未复制权重、完整图像、latent/feature bank、完整 rollout、论文 PDF/HTML/文本或第三方代码快照。阅读 JSON 只保存来源与相应解析结果；它们引用的资产内容没有在本次重新全量校验。三个超过 100,000 字节的 GitHub 元数据 JSON 也明确列入 manifest 的 excluded 项。小文件来源选择上限为单份 100,000 字节、总复制内容 4,000,000 字节。

链接检查覆盖仓库 docs 内 Markdown 的内联本地链接，区分仓库内与外部路径；不访问网络、不检查标题锚点、引用式链接或代码中的裸路径。数学记号可能被轻量正则误识别，冻结文档的历史相对链接也可能在原位置有效、在复制目录下失效，因此须连同 [归档审核说明](../../RAEV2_FINAL_ARCHIVE_VALIDATION_20260906_ZH.md) 阅读。
