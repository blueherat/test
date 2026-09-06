# RAEv2 最终归档检查：来源、便携性与提交边界

本次整理保存理论、失败分支、实现和可复查的小型结果，不把归档完成改写为性能目标达成。当前终验结果与最终轮数由[轮次台账](RAEV2_FINAL_FIVE_ROUNDS_20260906_ZH.md)及[研究总索引](RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md)裁决；本文只检查归档边界。

## 已完成的字节校验

[终验数据包](data/raev2_guidance_final_20260906/README.md)当前为 revision 2：188 份原实验小型输出、3,366,966 字节。独立逐份读取包内文件，核对大小、SHA-256、计数和总字节数，并核对原导出脚本身份，全部相符。此次没有修改已绑定的导出脚本或旧复制文件。

该包 manifest 的 SHA-256 为 `4fd839826d6bb53ec612827e26e2e836c01650bda4fc066fcb32597bbdadb517`。这证明登记的小文件复制一致，不证明原大资产仍全部完整，不补齐历史准备成本缺口，也不代替 FID 或机制审核。

新建的[来源补充包](data/raev2_guidance_final_sources_20260906/README.md)有 211 条精确来源记录，去重复制 203 份、1,367,202 字节。其中包括 paired-bridge 的训练/恢复驱动、screen plan 生成器、实际速度单位分析、旧独立审计器，以及阅读原文与代码的来源 manifest、解析反例输出。另登记 45 个与现有仓库源码同 SHA 的副本而不重复复制。

选择范围已固定在 [sources_plan.json](data/raev2_guidance_final_sources_20260906/sources_plan.json)。包内不含模型权重、完整图像、feature/latent bank、PDF/HTML/论文全文或第三方代码快照；三个较大的 GitHub 元数据 JSON 明确排除。其余原资产仍由原地址和已有来源身份连接。本次没有将“路径有记录”写成“全部大文件重新校验过”。

## 文档链接与历史相对路径

[validation.json](data/raev2_guidance_final_sources_20260906/validation.json)是可刷新快照，覆盖仓库 `docs` 中 Markdown 内联本地链接并区分仓库内/外部目标。初次检查 309 份文档、701 个仓库内链接与 141 个外部链接；外部链接未发现缺失。三个需要解释的仓库内条目如下：

1. `PFR_OU_PROBABILITY_WAVELET_THEORY_ZH.md` 第 447 行的数学表达式被轻量正则误识别为指向 `docs/y` 的链接。它不是实际导航链接；保留旧文档，不作伪修复。
2. 终验包中 `files/query_mean_error_compatibility_v1/frozen_protocol.md` 第 3 行的相对链接来自原仓库 docs 根目录。冻结副本移动后按新目录解释会失效，而原目标 [decoder query-mean 结果](RAEV2_DECODER_QUERY_MEAN_RESULTS_20260906_ZH.md)存在。为保持冻结字节身份，不修改副本文字；读取时使用该原文档映射。
3. 初次检查时 `RAEV2_FINAL_CANDIDATE_REVIEW_20260906_ZH.md` 指向的 `data/raev2_temporal_score_pde_20260906/README.md` 尚未创建；后续检查已经存在，该缺失已解决。提交前状态以刷新后的 JSON 为准，不能以初次快照代替收尾检查。

来源补充包还完成了独立逐文件 SHA/大小核对：203 份唯一复制内容全部一致；其中 50 份 Python 使用 AST 解析确认语法有效，153 份 JSON 均可解析。没有执行这些归档脚本中的实验或恢复任务。这些检查证明保存与格式正确，不替代对旧原型正确性或性能的评价。

基础来源包冻结后，最后两轮新增的 Moser / FP-Diffusion 阅读来源与 temporal-score 独立审核通过[追加来源 manifest](data/raev2_guidance_final_sources_20260906/supplement/manifest.json)单独登记，连接基础 manifest SHA。Moser 阅读笔记的当前内容与阅读 manifest 中登记的大小/SHA 已直接核对；基础 manifest 及其旧复制文件保持不变。

检查器不访问网络，也未核验标题锚点、引用式 Markdown 链接、代码块中的裸路径或外部资产内容。理论文档里 `R/...` 的简写由总索引的目录定义解释，不计入上述链接统计。

## 提交建议与执行边界

[准备脚本](../experiments/prepare_raev2_final_git_manifest.py)只处理当前 Git 改动中明确限定的路径：`docs/RESEARCH_STATUS.md`、本次 `docs/RAEV2_*_20260906_ZH.md`、`docs/data/raev2_*20260906*`，以及 `experiments`、`tests` 中相关的 RAEv2 Python 文件。它输出[精确路径清单](data/raev2_guidance_final_sources_20260906/proposed_git_paths.txt)及[逐文件身份](data/raev2_guidance_final_sources_20260906/proposed_git_inventory.json)，并另列范围外改动。它不会按仓库根目录整体暂存，也不会自行提交或推送。

由于主线程的最终结果、索引与台账仍在更新，路径清单是待最终刷新的候选快照。完成最后一轮文档与数据后再次运行脚本，检查范围外列表、链接状态、文件总数与身份，再按精确路径执行获用户授权的本地 Git 提交。三个生成的清单/校验文件不递归登记自身 hash；其余候选文件均记录内容 SHA-256。源码、结果和便携 manifest 的 Git 提交不能代替大资产备份，也不构成端到端重现实验已经执行的声明。

## 主线程最终检查补记

第4轮后不再增加方法。最终文档补齐后，内联链接检查仅余上述两项已解释的历史例外。主线程按精确清单重读候选文件并验证每个已登记SHA/大小，154份Python通过AST解析、373份JSON可解析；没有借归档检查重新执行旧实验。

提交前发现根.gitignore的`*.npz`会漏掉两个已登记的小型证据数组。现在仅在各自数据包加入明确文件名例外：压力包的`channel_moments.npz`和Moser包的`trajectories.npz`；不放开模型、样本或外部大数组。两文件按原数据包manifest核对后随本地提交保存。新增.gitignore不改变原实验输出字节。

精确路径/身份清单保留的是**提交前快照**，其中`staged_or_committed:false`表示准备脚本不负责Git操作，不是提交后的仓库状态。实际本地commit、暂存一致性和工作树状态由主线程使用Git命令检查，提交身份以Git历史和最终交付消息为准。

Git默认空白检查会把原始CSV的CRLF标成尾部空白；启用`core.whitespace=cr-at-eol`后只剩一个既有测试文件末尾空行。为保留历史输出和源码身份，不统一格式化归档文件。最终以`cr-at-eol,-blank-at-eof`执行检查，其余空白错误为零；这两个格式例外不影响解析、公式或数据SHA。
