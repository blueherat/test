# RAEv2 guidance：最终质量结论与归档

第6/8轮完成全部质量试验，**未达到 FID 改善至少3%的目标**。最后一个原设置的 semantic_add 5K也已完成独立审核。质量研究在此收束，剩余轮次仅用于最终核验与Git归档，不新增方法、系数、时间窗或seed。

## 结果

| 规模 | 原 official FID | 最佳候选 | 候选FID | 相对原official改善 | 推理成本比 |
|---|---:|---|---:|---:|---:|
| 1K | 38.486773927 | semantic_orthogonal | 37.704791770 | 2.031820% | 1.997326× |
| 5K | 6.949768478 | directional_variance | 6.910567613 | 0.564060% | 1.013986× |

1K更强的历史interval控制为38.335024032，最佳候选相对它仅改善1.644012%。5K方向候选相对旧全局方差只改善0.495737%。原official的3%门槛分别为37.332170709和6.741275423。

最后semantic_add5K的FID为7.250174076930，相对official改善-4.322527%，推理1.994621倍。全部5000张、625个输入batch、旧 .15 参数、原模型/decoder/统计和冻结源码均通过审核。独立FID为7.250174076932。由于质量未过，预定201调用Heun成本控制不执行；旧补丁也没有应用。

共17种候选完成1K，9种完成5K，连同4条控制共有30条完整质量结果。不同规模使用不同seed，不是嵌套样本；未做5K的其余候选仍标为未做。

| 方法 | 1K FID | 5K FID |
|---|---:|---:|
| official | 38.486773927 | 6.949768478 |
| piecewise | 38.335024032 | 7.011576542 |
| ancestral | 38.894423476 | 未做 |
| partial | 38.506604117 | 未做 |
| calibrated | 38.541199867 | 6.944996553 |
| velocity_projection | 38.478745998 | 未做 |
| noise_projection | 38.483935699 | 未做 |
| stochastic_weak | 38.270117888 | 7.027320234 |
| mean_weak | 38.566323254 | 未做 |
| critic_isotropic | 38.535460912 | 未做 |
| critic_exchangeable | 38.434217779 | 未做 |
| two_mode | 38.518130391 | 6.933352026 |
| semantic_add | 37.746545803 | 7.250174077 |
| semantic_orthogonal | 37.704791770 | 7.189964510 |
| paired_ratio | 38.442134865 | 未做 |
| paired_ratio_calibrated | 38.567599071 | 6.938002015 |
| prefix_ratio64k | 38.576569833 | 6.926132977 |
| conditional_variance | 38.547114655 | 6.957439858 |
| directional_variance | 38.610371195 | 6.910567613 |

## 保留的理论与证据

单一固定均值 Gaussian 风险、条件方差和分歧方向有明确推导及留出信号，但并未达到目标。精确反例进一步说明：Gaussian KL下降也可伴随FID上升。实际轨迹比值、分类风险与输入梯度、decoder空间和有限样本指标之间的缺口，见[理论结论](RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md)。

- [逐项目标审核与精确数值](../experiments/results/raev2_guidance_20260907/final_goal_requirement_audit.json)。
- [最后semantic5K完整审核](../experiments/results/raev2_guidance_20260907/final_semantic_add5k_audit.json)。
- [当前质量与历史方法入口](RAEV2_RESEARCH_ARCHIVE_INDEX_20260907_ZH.md)。
- [全工作区理论、代码和数据清单](RESEARCH_WORKSPACE_INVENTORY_20260907_ZH.md)。
- [历史资产缺口复核](RESEARCH_WORKSPACE_REFERENCE_REVIEW_20260907_ZH.md)：八项旧原始资产未在记录路径找到，不能宣称全部历史数据现可重跑。
- [既有52篇一手文献阅读档案](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md)与[各理论家族](RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md)。

大模型、latent、样本、特征及完整日志保留原址，Git保存代码、配置、理论、轻量结果与身份清单。训练和数据准备成本与在线推理成本分开；不把共享GPU的inclusive worker秒相加冒称独占算量。推理时间也在共享机器上实测，未认证GPU独占，百分之几的耗时差不能据此解释为速度改进。归档完成不等于FID目标完成。
