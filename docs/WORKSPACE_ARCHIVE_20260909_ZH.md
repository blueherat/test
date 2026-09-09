# 全工作区研究归档

本入口整理截至本次提交的整个工作区，包括研究代码、实验协议、成功与失败记录、数据表、分析笔记和已有草稿。历史文件保留原貌；其中“正在运行”“下一步”等文字描述当时状态，当前状态以本入口及最新完成记录为准。已有草稿只是历史资产，不代表恢复论文写作或认可其中未验证的结论。

## 当前实验结论

| 研究线 | 已完成工作及主要结果 | 当前状态与证据范围 |
|---|---|---|
| JiT 内部读出训练 | 50,000 steps 完成，checkpoint SHA `23612abb6eae35b7c0386b336a117603df62663bb87035965d00088b22e864a2`；训练保存 depth4/8 读出 | 本轮正式方法比较使用 depth4；不代表 depth8 已完成同等扫参 |
| JiT IG | 前后段细扫，选中 early=.30、late=0；FID-1K 55.480326 | 相对同 Euler100 Full 的 63.674574 有收益；官方 CFG/Heun50 为 38.778979，不能宣称 IG 超过官方最佳生成方案 |
| JiT PFR | 原配置 110.230983；九个新变体最好 56.855438，全部差于 IG | 该工作点下未迁移成功；没有扩至 5K |
| JiT lifting | 最好 alpha=.30,m2 为 55.214220；m1 为 55.223581 | 仅约 .48% 探索性改善，m2 batch GPU 时间约 IG 的 3.59 倍；未达到预定扩展门槛 |
| 小 SiT 最强历史工作点 | 历史 IG64.851298、PFR61.859207；lifting 六组约65.27–67.35 | lifting 未超过对应 IG；历史原图已删除，保留指标/标签/预览，不声称像素重现 |
| 小 SiT 两阶段对照 | 完成常数 .35 工作点及提前结束、前段增强对照 | 是另一组采样配置；不能替代上面的历史最强基线 |
| RAEv2 持续 lifting | alpha=.50/.65/.78/.85/.90，FID-1K约38.738/38.452/38.468/38.457/38.497；原IG38.264239 | 稳定平台不等于质量改善，也缺少完整匹配的普通IG系数曲线来证明更稳健 |
| 官方 SiT PFR 5K | ordinary115 10.155902 → PFR100 9.291876，约8.51%改善 | 已有方法的迁移正结果；成本近似匹配，非官方最优配置胜出证明 |
| 历史小 SiT OU/PFR资产 | seed5/6保留特征重算FID约36.19015/35.75879；重新核验两组噪声独立 | 原像素包不在原位置；不能称为从像素端完整复现 |
| FSG、AG/IG载体、未来原型、FK、参考回读 | 保留全部协议、代码、正负检查与暂停记录 | 不以固定点/信息论恒等式替代生成质量；暂停任务未自动恢复 |
| 新 guidance 头脑风暴 | 已撤下两条早期提案；新入口检索与反例已记录 | 尚无成熟新方法；未因本次归档启动新实验 |

以上不同模型、数据集、采样规模、评测器和求解器之间的 FID 不可直接横比。1K 扫参的最优点有选择偏差；不得与独立 5K 确认混写。

## 阅读与复核入口

- JiT：[训练迁移](JIT_INTERNAL_GUIDANCE_TRANSFER_20260908_ZH.md)、[有序实验协议](JIT_ORDERED_METHOD_STUDY_20260909_ZH.md)、[细扫协议](JIT_FINE_SWEEP_20260909_ZH.md)、[九组PFR完整结果](JIT_PFR_VARIANTS_20260909_ZH.md)。
- 小 SiT：[最强工作点](SMALL_SIT_BEST_CONFIG_LIFTING_20260909_ZH.md)、[强度和迭代扩展](SMALL_SIT_BEST_LIFTING_STRENGTH_ITERATIONS_20260909_ZH.md)、[两阶段IG](SMALL_SIT_TWO_STAGE_IG_20260909_ZH.md)、[前段强度扩展](SMALL_SIT_TWO_STAGE_IG_STRENGTH_20260909_ZH.md)。
- RAEv2 lifting：[方法](IG_CAPACITY_LIFTING_20260908_ZH.md)、[持续策略](IG_CAPACITY_LIFTING_CONSTANT_20260908_ZH.md)、[低噪声对照协议](IG_CAPACITY_LIFTING_LOW_NOISE_CONTROLS_20260909_ZH.md)。协议存在不意味着相应所有任务已执行。
- SiT 正资产：[官方5K结果](OFFICIAL_SIT_PFR_5K_RESULTS_20260908_ZH.md)、[波动分析](OFFICIAL_SIT_PFR_5K_UNCERTAINTY_20260908_ZH.md)、[历史最强5K重新核验](PFR_BEST_5K_ASSET_REAUDIT_20260908_ZH.md)。
- 理论边界：[有限尺度OU勘误](OU_FINITE_SCALE_RATIO_CORRECTION_20260908_ZH.md)、[FSG往返与强度等价反例](AG_FSG_COMMUTING_FLOW_IDENTITY_20260908_ZH.md)、[历史参考影响](IG_GUIDANCE_HISTORY_20260908_ZH.md)、[新方向筛选](GUIDANCE_RETHINK_AUDIT_20260909_ZH.md)。
- 更早全家族：[工作区总索引](RESEARCH_WORKSPACE_INVENTORY_20260907_ZH.md)、[RAEv2研究归档](RAEV2_RESEARCH_ARCHIVE_INDEX_20260907_ZH.md)。

## 文件与数据保存方式

新增可移植数据位于 [docs/data/workspace_archive_20260909](data/workspace_archive_20260909)。

- `workspace_files.jsonl`：逐个记录工作区普通文件的路径、大小和实际计算的 SHA-256，以及符号链接的目标。它不遍历链接目标，不包含 `.git`、运行缓存和归档目录自身。
- `external_metadata.json`：复制指定近期实验目录的结果、配置、源码哈希所在的 request、验证与完成状态，保留原路径及原JSON哈希。快照可能含被后续任务取代的旧状态，须结合总入口阅读。
- `result_records.csv`：上述目录内逐组 `result.json` 的统一字段索引；缺失字段留空，不凭猜测补填。列表型结果另完整保存在元数据快照中。
- `symlinks.json`：兼容路径与外部数据位置的映射；`summary.json` 记录范围、计数及未找到的可选目录。

Git 保存全部正常跟踪/未跟踪文件，并补入被宽泛 ignore 规则隐藏的小型研究证据。2 MiB 以内的本地研究文件可补入，checkpoint 格式、缓存和第三方目录除外。较大的原始采样包、checkpoint 和外部依赖保留原位置；工作区内普通大文件同样计算 SHA-256 并进入清单。**清单不是原始数据备份**，单独 clone 仓库不能恢复未入 Git 的原始二进制文件。外部实验元数据内既有样本哈希为原运行记录，本次没有重读全部外部样本计算哈希。

归档脚本为 [tools/archive_workspace_20260909.py](../tools/archive_workspace_20260909.py)。脚本只读取原件并写归档目录，不移动、删除数据，不执行采样、训练或评测。它生成的临时小文件路径清单用于安全暂存；复跑会刷新归档快照。

## 运行约束

保留已有基线与失败原件。不同噪声 bank、数值精度、求解器和比较工作点必须明确区分。四卡协同同一候选是既定采样方式，不自动为闲置GPU创建实验。没有新的明确方法理由时，不继续扩大已失败变体的参数搜索。论文草稿归档与方法验证是两件事。
