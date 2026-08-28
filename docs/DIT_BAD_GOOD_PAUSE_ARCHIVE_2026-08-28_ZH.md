# DiT bad/good 研究暂停与归档快照（2026-08-28）

## 0. 当前状态

本主线按用户要求暂停。暂停意味着：

- 不再运行 GPU 实验；
- 不恢复 Self-Guidance / SD1.4 权重下载；
- 不继续 FPCV、B 干预或新指标搜索；
- 不把研究目标标成已完成或已失败；
- 在 Git 整理规则确认前，不删除、移动、暂存、提交或推送任何现有文件。

暂停时没有本项目的采样、下载或分析进程在运行。

Git 基线为 `main@4a58cc2`，与 `origin/main` 一致。初次盘点时工作树有 3 个已跟踪文件
被修改，以及 1,266 个未跟踪文件；新增本暂停文档后为 1,267 个。未跟踪文件的初次
盘点构成为：

| 类别 | 数量 | 说明 |
|---|---:|---|
| `experiments/locks/` | 931 | 冻结源码、协议、manifest 与历史 superseded lock；可追溯但重复很多 |
| `experiments/*.py` | 159 | 采样、提取、评估、审计与冻结脚本 |
| `experiments/annotations/` | 127 | 盲评草稿、封存评审与共识标签 |
| `experiments/configs/` | 19 | 冻结实验配置 |
| `docs/` | 18 | 理论、结果和方法账本 |
| `experiments/audits/` | 5 | 标签质量审计图与摘要 |
| `reports/` | 4 | 可再生成的静态分析报告 |
| `tests/` | 2 | e-process 单元测试 |
| 其他 | 1 | 小型配置/辅助文件 |

因此禁止直接执行 `git add .`。Git 必须按明确的路径清单分批整理。

### 0.1 归档结构 QA

对新增/未跟踪产物做了只读结构检查，没有执行模型或实验：

| 检查 | 范围 | 结果 |
|---|---:|---:|
| Python AST 解析 | 550 个 `.py`（含 lock 内 source snapshots） | 0 个语法错误 |
| JSON 完整解析 | 503 个 `.json` | 0 个解析错误 |
| CSV 行列一致性 | 157 个 `.csv`、14,442 个数据行 | 0 个结构错误 |
| SHA-256 内容去重 | 当前 1,267 个未跟踪文件 | 696 个唯一 blob；571 条为首份之后的完全重复路径 |

最大完全重复组含 50 条路径。Git 会按 blob 内容去重，所以重复 source snapshots 的实际
对象存储成本低于路径数所暗示的成本；但 931 个 lock 路径仍会增加审核和历史浏览负担，
因此依然应随对应实验分组提交，而不是一次性无语义地加入。

## 1. 暂停时真正保留下来的研究结论

### 1.1 有用、可继续复用

| 对象 | 当前证据 | 正确定位 |
|---|---|---|
| 盲化 bad/good 数据与标签协议 | 已形成 targeted、fresh 240、expansion 360、third-pool 等封存标签及审核链 | 最有价值的基础资产；以后换任何内部指标都可复用 |
| 独立后缀重采 | 在有修复机会的比较中严格修复 `27/71=38.0%` | 证明一部分 blur/fusion 是采样事故；它是动作证据，不是触发器 |
| 中途 decoded-`pred_xstart` 模糊量 `B` | replication AUC `0.9333`；后续标签敏感性 AUC `0.668--0.848`，最终口径 `0.6681` | 目前最强的 blur/fusion 窄候选；尚未成为可靠自动干预方法 |
| 冻结 DiT-XL/2 采样与轨迹管线 | 250-step ancestral DDPM、CFG=4、raw conditional/unconditional、`pred_xstart`、innovation 等均有严格复现与哈希 | 后续研究的主实验平台 |
| 外部裁判管线 | 盲评、Inception/FID、表征距离均已与内部方法隔离 | 只作评价，不进入在线触发或选枝 |

### 1.2 数学成立，但质量用途被证伪或明显降级

| 方法 | 关键结果 | 归档结论 |
|---|---|---|
| 跨尺度路径似然比 `E` | `Q*/P` 记账是精确 e-process；alarm clear-bad `1/26`，matched control `2/26`；匹配修复差 4 负、0 正、4 平 | 保留理论与反例；退出坏图检测、guidance 和 rollback |
| 大规模内部 metric zoo | candidate-v5 AUC `0.5398`；34 路 mixture 从 discovery `0.944` 跌到独立池 `0.441` | 不能继续在同一标签上拼特征或翻方向 |
| c3 结构量 `C` | 早期 replication `0.8631`；最终标签口径约 `0.4619` | 只作脆弱副量/反例，不是主候选 |
| medoid / high-`O` 多分支 | medoid `5/18=27.8%`，均匀随机 `37.5%`；high-`O` AUC `0.286` | `STOP_CURRENT_FORM` |
| `h=10` max-outlier 多分支 | post-hoc `10/18=55.6%`，但类别集中、外部安全略差；prospective 采样与内部选择已完成 | 用户已放弃多分支主线；不得把未完成的外部确认写成成功 |
| Doob consistency | 历史 discovery AUC `0.7014`、exact `p=0.0436`；leave-class795 约 `0.506` | 类别依赖的探索线索，不足以继续 |
| PTCV | generic discovery AUC `0.629`、`p=0.125`；事后 blur subgroup `0.889`；冻结 expansion AUC `0.4645`、`p=0.6060` | 模糊窄假设也被独立否定，退役 |
| Fisher-geodesic guidance | 1K 相对 baseline `-0.152`，但 time-only 已有 `-0.118`；正式 5K baseline `7.4938`、Fisher `7.9868` | 5K 明确变差 `+0.4930`，退役；不能保留 1K 的乐观叙事 |

### 1.3 未完成，不能写成结果

| 对象 | 已完成 | 缺少什么 |
|---|---|---|
| FPCV | 原 33 点 cross-polytope 在 seed 50--52 全零；128 点闭合多边形 label-free smoke 在 step 99/149 检出可重复的非零循环违反，step 199 回到零 | 尚未冻结唯一 redesign score、跑完整 cohort 或打开质量标签；现在只有“数值上不空洞”，没有质量结论 |
| B-triggered one-shot restart | 只完成了口头理论草图：不触发则保持原轨迹，触发则放弃 step149 路径并只完整独立重启一次 | 文档、冻结阈值、代码和配对试验均未落地 |
| Self-Guidance baseline | 官方 repo、COCO prompt lock、评估统计和配置已准备 | SD1.4 权重下载未完成；没有生成图，也没有 1K/5K FID |

## 2. Git 中建议保留的权威入口

### 2.1 结论与理论文档

优先保留并在提交前校正相互矛盾的状态描述：

- `docs/DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md`：主方法总账；目前仍有
  max-outlier “进行中”、Fisher 仅写 1K、Self-Guidance 权重已准备等过时文字，提交前必须修正；
- `docs/DIT_V22_INTERNAL_SIGNAL_REASSESSMENT_ZH.md`：`E`、后缀修复和多分支的详细证据；
- `docs/CROSS_SCALE_SEQUENTIAL_EVIDENCE_ZH.md`：路径似然比理论、Ville/TV/回滚边界；
- `docs/DIT_PTCV_DISCOVERY_RESULT_ZH.md`：PTCV discovery 与 expansion 的完整负结果；
- `docs/DIT_PROJECTED_TWEEDIE_CONE_VIOLATION_ZH.md`：PTCV 形式化；
- `docs/DIT_FINITE_POSTERIOR_CYCLIC_VIOLATION_ZH.md`：FPCV 形式化和原始冻结设计；
- `docs/BAD_GOOD_TRAJECTORY_METRICS_ZH.md` 与
  `docs/BAD_GOOD_METRIC_SCREEN_RESULTS_2026-08-27_ZH.md`：早期指标定义和筛查证据；
- 本文件：暂停状态、保留策略和恢复入口。

`docs/RESEARCH_STATUS.md` 跨越仓库内很多更早的 RAE/训练研究，不能用 bad/good 单线的
结论覆盖；只应追加一段短索引，详细内容由上述总账承载。

### 2.2 可复现源码

源码应保留，但提交时按作用分组而不是按创建时间堆在一起：

1. baseline 与轨迹采样：`reproduce_*`、`sample_*`、`trace_*`；
2. 内部量提取：`observe_*`、`extract_*`、`replay_*`、`derive_*`；
3. 冻结与防泄漏：`freeze_*`、`lock_*`、`experiments/configs/`；
4. 评价与独立审计：`analyze_*`、`evaluate_*`、`audit_*`、`summarize_*`；
5. 干预：`intervene_*`；
6. 当前最终几何负/未决线：Doob、PTCV、FPCV 对应的 core/runner/analyzer；
7. 单元测试：`tests/test_observe_dit_blur_focused_eprocess*.py`。

### 2.3 locks 与 annotations

这些文件虽然重复，但承担“先锁方法、后看标签”的证据链，不能随意删掉。Git 整理时：

- 保留所有被正式结果引用的最终 lock、manifest、source hash、completion 和 sealed labels；
- `superseded_*`、早期 v1/v2 lock 先不删除，只在本文件或专门 manifest 中标为 superseded；
- draft review 与 sealed/consensus 不能混称；正式分析只引用 sealed/adjudicated consensus；
- 任何裁剪都必须先验证后继 lock 已包含父版本的 source/hash/protocol，且需用户明确授权。

## 3. 外部数据盘登记

`/data/users/zhoushunyu/eqvae/cross_scale_evidence` 当前约 115 GB，共约 90 个顶层条目。
这些数据不应进入 Git；Git 只保存路径、schema、hash、结果摘要和重跑入口。

### 3.1 A 级：继续研究时不可替代，优先保留

| 路径（相对 `cross_scale_evidence/`） | 大小 | 作用 |
|---|---:|---|
| `dit_bad_good_custom_traces_cfg_locked` | 5.7 GB | 早期冻结轨迹与内部量 |
| `dit_bad_good_confirmation_v1_custom_traces_cfg_locked` | 11 GB | fresh 240 确认池原始轨迹 |
| `dit_bad_good_confirmation_expansion_v1_custom_traces_cfg_locked` | 13 GB | expansion 360 原始轨迹 |
| `dit_bad_good_third_pool_v1_custom_traces_cfg_locked` | 64 GB | 第三池与 label-free 产品的主原始轨迹 |
| `bad_good_metric_discovery`、`bad_good_metric_confirmation_*` | 约 1.2 GB | 已提取指标、阈值和历史结果 |
| 小型 analysis/result 目录 | 通常 KB--MB | 权威结果、identity、seal 与独立审计；体积小但证据价值高 |

上述四个 locked trace pool 合计约 93.7 GB，是 115 GB 的主体，也是恢复 B 或测试全新
内部量时最难重建的资产。

### 3.2 B 级：保留到论文/方向最终决策

| 路径 | 大小 | 说明 |
|---|---:|---|
| `dit_v22_repairability_pilot_v1_2_outputs` | 1.7 GB | 后缀可修复性动作证据 |
| `dit_v22_transient_escape_prospective_v1_outputs` | 5.5 GB | 已完成但主线被放弃的多分支 prospective 输出；先保留，避免丢失昂贵采样 |
| `cfg_rejection_edm2` | 4.9 GB | baseline 复现产物 |
| `dit_imagenet256_suffix_repairability` | 890 MB | 早期后缀重采素材 |
| PTCV / FPCV / Doob probe 与 analysis | 合计远小于 200 MB | 负结果与未决数值证据，保留成本低 |
| 盲评 delivery/private/public 包 | 数十 MB 到数百 MB | 在最终归档前保留；许多可由 mapping 重建 |

### 3.3 C 级：可再生或重复，但当前不删除

- `dit_bad_good_custom_traces_cfg`、`*_cfg_final`、`*_cfg_strict` 等 unlocked/中间版本；
- visualization、grid、delivery pack 与重复公开盲包；
- mechanics smoke、临时 shard、旧 summary 和 superseded protocol 输出；
- `.v22_topE.*` 等临时目录；
- baseline repo 内的 `__pycache__`。

它们是未来释放空间的候选，不是当前获准删除的清单。删除前必须生成逐路径、逐大小、
可重建来源和 hash 的确认表。

## 4. 外部 baseline 与评估资产

这些仓库位于 `/data/users/zhoushunyu/eqvae/baselines/`，总计约 5.4 GB，不进入主仓库：

| 仓库 | 固定 commit | 状态 |
|---|---|---|
| CFG-Rejection | `82dd2a50effd` | clean |
| DiT | `ed81ce222909` | 仅未跟踪 pretrained model |
| Radon-Nikodym-Estimator | `13ab89853610` | clean |
| Self-Guidance | `843bda799bb5` | repo clean；模型缓存不完整 |
| EDM2 | `4bf8162f601b` | 仅 checkpoint/cache 未跟踪 |
| Feynman--Kac Correctors | `aa6f5ed4a0eb` | 仅 checkpoint 未跟踪 |
| guided-diffusion | `22e0df818350` | 仅 checkpoint 未跟踪 |
| OpenAI CLIP | `d05afc436d78` | clean |
| text2image-benchmark | `532229f679d7` | clean |

COCO FID reference stats 位于
`/data/users/zhoushunyu/eqvae/eval_assets/t2ibenchmark_532229f/MS-COCO_val2014_fid_stats.npz`
（约 33 MB）。Fisher-geodesic 的完整代码与 1K/5K 结果在
`/home/zhoushunyu/AItest/ecsg_validation/fid_pivot/`，不应只在本仓库复制一半代码；本仓库
保留最终结论和外部路径即可。

## 5. 建议的 Git 整理顺序

在用户补全 Git 限制并确认后，建议用显式 pathspec 分成以下提交，每次提交前做
`git diff --cached --check`、敏感信息扫描和最小 self-test：

1. **采样与路径证据基础设施**：baseline、DiT/ADM trace、路径 LR 理论与测试；
2. **盲化数据集与标签协议**：annotations、review locks、cohort manifests；
3. **bad/good 指标与 B/C 结果**：metric extraction、conformal threshold、独立池评价；
4. **干预与反证**：suffix repair、`E` 退役、多分支及其失败/未完成边界；
5. **后验几何候选与负结果**：Doob、PTCV、FPCV；
6. **baseline 工程快照**：CFG-Rejection、Self-Guidance 配置和未完成状态；
7. **统一文档收口**：修正总账、README 和 `RESEARCH_STATUS.md` 的过时描述。

不要把 931 个 lock 文件单独做一个没有语义的“大垃圾提交”；它们应跟随首次引用它们的
实验组进入历史。也不要把 `/data` 软链接、模型权重、NPZ/PNG 生成池或外部 repo 纳入 Git。

## 6. 如果以后恢复，从哪里开始

恢复时不需要重走所有失败路线。当前最合理的窄入口仍只有：

1. 先把 `B` 作为 blur/fusion subtype detector 做事件充足、完全独立的确认；
2. 若确认 `TPR>FPR`，再测试一次性完整重启，而不是循环 suffix rejection 或多分支选优；
3. FPCV 只有在愿意支付一次严格、冻结、无 rescue 的 quality test 时才恢复；现在只证明
   dense geometry 数值不空洞；
4. Self-Guidance 只有在用户主动恢复权重下载后才继续。

其余 `E`、PTCV、C 主线、medoid/high-`O`、Fisher guidance 和同标签 metric mixture 不再
修补。
