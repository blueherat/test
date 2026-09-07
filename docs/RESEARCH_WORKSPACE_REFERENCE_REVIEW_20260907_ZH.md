# 归档引用复核与历史资产缺口

2026-09-07。本次完整工作区索引对 Markdown 做字面路径检查，再逐项阅读被标记引用的上下文。没有为通过检查而改写旧冻结协议、伪造缺失数据或复制附近的 smoke 结果。逐项源文件摘要、原引用、解释及已确认的替代导航见[机器复核记录](../experiments/results/raev2_guidance_20260907/workspace_reference_review.json)，可用[复核脚本](../experiments/audit_research_workspace_references_20260907.py)重跑。

## 不能直接当作缺失资产的警告

| 来源 | 实际情况 | 正确导航 |
|---|---|---|
| AdvFD 官方实现审计 | 引用的是文首已指定官方仓库中的计划；原提交和文档都存在 | [官方 residual-RMS 文档](/data/users/zhoushunyu/research_repos/AdvFD/docs/plans/2026-07-28-fd-adv-shared-residual-rms.md) |
| Autoencoder Score 阅读 | 行内配置相对官方 RAEv2 仓库，而非工作区根 | [原配置](../external/RAEv2/configs/stage1/training/dinov3l-k7-imagenet.yaml) |
| Flow Matching 阅读 | 前文简写了 data 根下路径，同文末尾有完整路径 | [阅读资产 manifest](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_flow_matching_v1/manifest.json) |
| 旧 query-mean 冻结协议副本 | 归档复制后仍保留原相对链接起点 | [所引用结构诊断](RAEV2_DECODER_QUERY_MEAN_RESULTS_20260906_ZH.md) |
| DiT v1 锁定理论的五个链接 | 文本原来按 docs 相对路径写，锁定副本位置不同；同名实现均存在 | [校准](../experiments/calibrate_dit_blur_focused_eprocess.py)、[配置](../experiments/configs/dit_blur_focused_eprocess_v1.json)、[观察](../experiments/observe_dit_blur_focused_eprocess.py)、[重放](../experiments/replay_dit_blur_focused_eprocess_inputs.py)、[测试](../tests/test_observe_dit_blur_focused_eprocess.py) |
| 9月6日的归档验证文档 | 已明确解释一个数学表达式曾被正则误识别为链接 | [原解释](RAEV2_FINAL_ARCHIVE_VALIDATION_20260906_ZH.md)；不是数据缺失 |

实际代码行号后缀按文件路径解析，行号内容未据此重新验证。公式和代码块不会作为导航链接抽取；省略号、glob、模板及命令不冒报为确定的缺失文件。

## 当前本机不能认证存在的旧原始资产

以下是实际存在性缺口，不能归因于解析器。完整原路径逐项保存在上述 JSON 中。

- [RAEv2 LPL strict 续研](RAEV2_LPL_STRICT_CONTINUATION_ZH.md)列出的五个实验目录：lpl pilot、guidance-aware、flow-parallel、prediction-detach、raw 10step。
- [RAE decoder risk Phase 0](RAE_DECODER_RISK_PHASE0_RESULTS_ZH.md)的正式实验目录，以及[原实验台账](RAE_LPL_EXPERIMENT_LEDGER_ZH.md)中的正式校准/验证 cache。
- [确定性 LPL 复现](RAE_DETERMINISTIC_LPL_REPRODUCTION_ZH.md)引用的初始 adapter checkpoint。

已对原路径检查，并在本机 eqvae 数据根下到四层深度查找这些名称；只发现两个 decoder-risk smoke 目录，没有据此恢复或认证正式数据。此检查不是所有磁盘、远端备份或 Git 历史的穷尽搜索。旧代码、文档和现存紧凑结果仍在完整清单中；现有历史结论保留为报告中的结果，不能宣称这八项原始资产目前可直接重跑。

当前3%目标使用的 paired 1K/5K 并不依赖这八项旧资产，其全样本身份与独立指标审核单独见[当前研究索引](RAEV2_RESEARCH_ARCHIVE_INDEX_20260907_ZH.md)。全工作区枚举和旧研究导航见[完整归档](RESEARCH_WORKSPACE_INVENTORY_20260907_ZH.md)。
