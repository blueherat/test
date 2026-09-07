# 研究工作区完整索引：理论、实验、代码与数据

2026-09-07。此索引覆盖整个 Git 已跟踪工作区，以及本次可见且未被忽略的新文件，包括根目录下的历史实验；没有只按 RAEv2 文件名前缀筛选。原文件保留原址和原结论。这是归档导航，枚举文档不等于重新读完或验证其中的论文、证明和实验。

当前 RAEv2 3% 目标的结论及可比质量表见[当前研究总入口](RAEV2_RESEARCH_ARCHIVE_INDEX_20260907_ZH.md)，剩余次数受[最多八轮台账](RAEV2_FINAL_EIGHT_ROUNDS_20260907_ZH.md)约束。旧 SiT/PFR 正结果不作为 RAEv2 成果；未采样候选不记为 FID 失败。

## 如何找回研究

- [2026-09-06 各理论家族与失败边界](RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md)。
- [52 篇一手文献阅读总索引](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md)：原文版本、实际阅读范围和外部 PDF/代码身份；本索引不增加已读篇数。
- [较早实验归档入口](EXPERIMENT_ARCHIVE_INDEX_ZH.md)：PFR、SiT、预测目标、latent/decoder、AdvFD 等旧结果的语境与限制。
- [RAEv2 当前机器证据索引](../experiments/results/raev2_guidance_20260907/research_evidence_index.json)：当前配对 1K/5K、独立审核、成本、样本路径与身份。
- [全工作区机器清单](../experiments/results/raev2_guidance_20260907/workspace_research_inventory.json)：逐文件大小、SHA256、类型、家族，以及 Markdown 的本地引用检查。
- [逐项引用复核与旧资产缺口](RESEARCH_WORKSPACE_REFERENCE_REVIEW_20260907_ZH.md)：区分外部仓库的相对路径、冻结副本链接和确实不在本机的原始资产。
- [最后八轮的理论结论](RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md)：Gaussian 风险与 FID 的严格反例、实际轨迹比值及 1K/5K 的解释边界。
- [构建脚本](../experiments/index_research_workspace_20260907.py)：运行 `python experiments/index_research_workspace_20260907.py` 更新清单。

## 覆盖范围与可验证边界

本快照覆盖 **6,214 个文件**；其中文档 **354 份**。家族按路径关键词划分，只用于导航，不推断科学状态。一个文件可涉及多个方向，这里按构建器中固定优先序归入一个家族；全部路径仍可在 JSON 中检索。

| 家族 | 全部文件 | 文档 | 代码/笔记本 | 结构化数据/证据 |
|---|---:|---:|---:|---:|
| RAEv2 guidance、理论与迁移 | 1521 | 143 | 439 | 923 |
| RAE、LPL、decoder 与 latent 几何 | 239 | 47 | 186 | 0 |
| PFR、反事实参考与半群 guidance | 135 | 12 | 51 | 72 |
| SiT / ImageNet-100 与内部头 | 1309 | 50 | 257 | 827 |
| DiT bad/good、事件与统计审核 | 1232 | 23 | 532 | 650 |
| AdvFD、Fréchet 与 score 后训练 | 179 | 15 | 120 | 43 |
| 预测目标、频谱与低维机制实验 | 1138 | 10 | 89 | 983 |
| 生成训练、评估与其他数据集 | 109 | 10 | 96 | 0 |
| 通用理论、协议、环境与总入口 | 352 | 44 | 200 | 69 |

所有普通文件本轮读取全部字节计算 SHA256。机器清单自身排除，避免循环摘要；本文先生成再纳入摘要。Git HEAD 记录的是构建时的父提交，文件摘要反映当时工作区，不把未提交修改冒称为该父提交的内容。若存在符号链接，只记录链接文本及目标存在性。

大模型、latent、原始样本、完整特征和外部代码库仍保留在原记录位置，不会被本脚本复制进 Git，也不宣称本轮重算了全部外部资产的 SHA。当前 RAEv2 数据根为 `/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/`；各 execution、request、summary、audit 提供冻结源码、输入、样本和成本身份。

Markdown 引用检查只解析明确的链接和单个行内本地路径，记录存在、缺失、歧义或模板表达式；历史绝对路径失效不会自动改写为另一个实验。原文引用中的 URL 只登记，不表示重新抓取或读过；动态路径、命令和 glob 不冒报为缺失文件。引用存在也不等于数据内容正确，质量结论仍以相应独立审核为准。

## 全部理论与说明文档

以下逐项连接工作区中的 Markdown，包括实验目录的 README、冻结协议和历史审核。文档标题为原首个一级标题，没有用自动生成摘要覆盖原论证。

### RAEv2 guidance、理论与迁移

- [实际轨迹密度比：分布诊断与小样本负结果](RAEV2_ACTUAL_RATIO_20260907_ZH.md) — `docs/RAEV2_ACTUAL_RATIO_20260907_ZH.md`
- [实际 rollout 的有限步缺陷与桥校正：一个条件性结构](RAEV2_ACTUAL_ROLLOUT_FINITE_TRANSPORT_20260906_ZH.md) — `docs/RAEV2_ACTUAL_ROLLOUT_FINITE_TRANSPORT_20260906_ZH.md`
- [RAEv2 仿射反射平均：固定配对 1K 协议](RAEV2_AFFINE_REFLECTION_PROTOCOL_20260906_ZH.md) — `docs/RAEV2_AFFINE_REFLECTION_PROTOCOL_20260906_ZH.md`
- [仿射反射 guidance：执行与结果](RAEV2_AFFINE_REFLECTION_RESULTS_20260906_ZH.md) — `docs/RAEV2_AFFINE_REFLECTION_RESULTS_20260906_ZH.md`
- [Query 对比的局部几何：两个不同的错误假设](RAEV2_ATTENTION_CONTRAST_GEOMETRY_20260906_ZH.md) — `docs/RAEV2_ATTENTION_CONTRAST_GEOMETRY_20260906_ZH.md`
- [RAEv2 raw latent 类内／类间矩审计](RAEV2_CLASS_MOMENT_AUDIT_20260906_ZH.md) — `docs/RAEV2_CLASS_MOMENT_AUDIT_20260906_ZH.md`
- [条件反向方差：最后阶段的一个固定候选](RAEV2_CONDITIONAL_VARIANCE_PROTOCOL_20260907_ZH.md) — `docs/RAEV2_CONDITIONAL_VARIANCE_PROTOCOL_20260907_ZH.md`
- [判别器方向上的去噪器 Jacobian：固定 CPU 检查](RAEV2_CRITIC_JACOBIAN_AUDIT_20260907_ZH.md) — `docs/RAEV2_CRITIC_JACOBIAN_AUDIT_20260907_ZH.md`
- [RAEv2 解码后类内／类间矩审计](RAEV2_DECODED_CLASS_MOMENT_AUDIT_20260906_ZH.md) — `docs/RAEV2_DECODED_CLASS_MOMENT_AUDIT_20260906_ZH.md`
- [RAEv2：解码协方差形状的固定跨类检验](RAEV2_DECODED_COVARIANCE_SHAPE_20260906_ZH.md) — `docs/RAEV2_DECODED_COVARIANCE_SHAPE_20260906_ZH.md`
- [RAEv2 decoder 一阶响应与有限非线性剩余](RAEV2_DECODER_LINEARIZATION_AUDIT_20260906_ZH.md) — `docs/RAEV2_DECODER_LINEARIZATION_AUDIT_20260906_ZH.md`
- [RAEv2：固定 DDT 解码器 query 均值弱分支的机制诊断](RAEV2_DECODER_QUERY_MEAN_PROTOCOL_20260906_ZH.md) — `docs/RAEV2_DECODER_QUERY_MEAN_PROTOCOL_20260906_ZH.md`
- [两层 DDT query 均值：结构诊断完成](RAEV2_DECODER_QUERY_MEAN_RESULTS_20260906_ZH.md) — `docs/RAEV2_DECODER_QUERY_MEAN_RESULTS_20260906_ZH.md`
- [RAEv2 线性密度去污染准入审计](RAEV2_DENSITY_DECONTAMINATION_AUDIT_20260906_ZH.md) — `docs/RAEV2_DENSITY_DECONTAMINATION_AUDIT_20260906_ZH.md`
- [RAEv2 深度 × 读出交叉审计](RAEV2_DEPTH_READOUT_AUDIT_20260906_ZH.md) — `docs/RAEV2_DEPTH_READOUT_AUDIT_20260906_ZH.md`
- [第4轮：沿原生双头分歧的单标量条件协方差](RAEV2_DIRECTIONAL_VARIANCE_PROTOCOL_20260907_ZH.md) — `docs/RAEV2_DIRECTIONAL_VARIANCE_PROTOCOL_20260907_ZH.md`
- [RAEv2 完整轨迹分布目标下的共享 gate guidance](RAEV2_DISTRIBUTIONAL_GATE_GUIDANCE_20260906_ZH.md) — `docs/RAEV2_DISTRIBUTIONAL_GATE_GUIDANCE_20260906_ZH.md`
- [RAEv2 Internal Guidance 分布 AUC 审计](RAEV2_DISTRIBUTION_AUC_AUDIT_ZH.md) — `docs/RAEV2_DISTRIBUTION_AUC_AUDIT_ZH.md`
- [RAEv2 完整终点伴随响应：8 图机制检验协议](RAEV2_ENDPOINT_ADJOINT_RESPONSE_PROTOCOL_20260906_ZH.md) — `docs/RAEV2_ENDPOINT_ADJOINT_RESPONSE_PROTOCOL_20260906_ZH.md`
- [RAEv2 完整终点伴随响应：8 图结果与独立 CPU 复核](RAEV2_ENDPOINT_ADJOINT_RESPONSE_RESULTS_20260906_ZH.md) — `docs/RAEV2_ENDPOINT_ADJOINT_RESPONSE_RESULTS_20260906_ZH.md`
- [终点对齐量与生成质量：还缺什么保证](RAEV2_ENDPOINT_ALIGNMENT_QUALITY_GAP_20260906_ZH.md) — `docs/RAEV2_ENDPOINT_ALIGNMENT_QUALITY_GAP_20260906_ZH.md`
- [RAEv2 跨原图原型的线性终点 observable](RAEV2_ENDPOINT_CROSSPROTOTYPE_WITNESS_20260906_ZH.md) — `docs/RAEV2_ENDPOINT_CROSSPROTOTYPE_WITNESS_20260906_ZH.md`
- [RAEv2：实际 rollout 的能量球投影](RAEV2_ENERGY_BALL_GUIDANCE_20260906_ZH.md) — `docs/RAEV2_ENERGY_BALL_GUIDANCE_20260906_ZH.md`
- [RAEv2 极值对称 Jacobian 曲率审计](RAEV2_EXTREMAL_CURVATURE_AUDIT_20260906_ZH.md) — `docs/RAEV2_EXTREMAL_CURVATURE_AUDIT_20260906_ZH.md`
- [RAEv2：FID 有限样本偏差、1K 筛查与历史特征复用审计](RAEV2_FID_FINITE_SAMPLE_READING_20260906_ZH.md) — `docs/RAEV2_FID_FINITE_SAMPLE_READING_20260906_ZH.md`
- [RAEv2 最终归档检查：来源、便携性与提交边界](RAEV2_FINAL_ARCHIVE_VALIDATION_20260906_ZH.md) — `docs/RAEV2_FINAL_ARCHIVE_VALIDATION_20260906_ZH.md`
- [第3轮：时间自洽性不能独自识别真实生成目标](RAEV2_FINAL_CANDIDATE_REVIEW_20260906_ZH.md) — `docs/RAEV2_FINAL_CANDIDATE_REVIEW_20260906_ZH.md`
- [RAEv2：最后最多八轮与收束台账](RAEV2_FINAL_EIGHT_ROUNDS_20260907_ZH.md) — `docs/RAEV2_FINAL_EIGHT_ROUNDS_20260907_ZH.md`
- [RAEv2 guidance：最后五轮与收束台账](RAEV2_FINAL_FIVE_ROUNDS_20260906_ZH.md) — `docs/RAEV2_FINAL_FIVE_ROUNDS_20260906_ZH.md`
- [第5轮：最后一个原设置的 semantic_add 5K](RAEV2_FINAL_SEMANTIC_ADD5K_20260907_ZH.md) — `docs/RAEV2_FINAL_SEMANTIC_ADD5K_20260907_ZH.md`
- [RAEv2 有限区间 Flow Pullback：理论、数值门槛与配对采样协议](RAEV2_FLOW_PULLBACK_20260905_ZH.md) — `docs/RAEV2_FLOW_PULLBACK_20260905_ZH.md`
- [RAEv2 Guidance 理论探索归档：从 PFR 迁移到半群 Value](RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md) — `docs/RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md`
- [RAEv2 guidance 本次研究结题：5%目标未实现](RAEV2_GUIDANCE_FINAL_CLOSEOUT_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_FINAL_CLOSEOUT_20260906_ZH.md`
- [RAEv2 guidance：当前研究目标](RAEV2_GUIDANCE_GOAL_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_GOAL_20260906_ZH.md`
- [RAEv2 guidance：重新启动的 3% 目标](RAEV2_GUIDANCE_GOAL_20260907_ZH.md) — `docs/RAEV2_GUIDANCE_GOAL_20260907_ZH.md`
- [RAEv2：固定 native BF16 heads 的 guidance 混合数值审计（2026-09-06）](RAEV2_GUIDANCE_MIX_NUMERICS_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_MIX_NUMERICS_20260906_ZH.md`
- [RAEv2：实际生成分布反馈与判别器梯度的可估计性](RAEV2_GUIDANCE_READING_ACTUAL_DISTRIBUTION_FEEDBACK_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_ACTUAL_DISTRIBUTION_FEEDBACK_20260906_ZH.md`
- [Guidance 扩展阅读：自适应控制的依据与保证对象](RAEV2_GUIDANCE_READING_ADAPTIVE_CONTROL_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_ADAPTIVE_CONTROL_20260906_ZH.md`
- [RAEv2 guidance 阅读：ADG 与范数约束的对象](RAEV2_GUIDANCE_READING_ANGLE_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_ANGLE_20260906_ZH.md`
- [RAEv2 guidance 阅读：PAG、SEG 与有结构的弱分支](RAEV2_GUIDANCE_READING_ATTENTION_WEAK_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_ATTENTION_WEAK_20260906_ZH.md`
- [RAEv2 基础阅读：去噪重建为何能给 score，何时不能](RAEV2_GUIDANCE_READING_AUTOENCODER_SCORE_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_AUTOENCODER_SCORE_20260906_ZH.md`
- [CFG++ 精读：重组、时间权重，以及迁移 RAEv2 时必须补上的机制](RAEV2_GUIDANCE_READING_CFGPLUSPLUS_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_CFGPLUSPLUS_20260906_ZH.md`
- [Characteristic Guidance：干净预测共识与 Gaussian 收敛条件](RAEV2_GUIDANCE_READING_CHARACTERISTIC_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_CHARACTERISTIC_20260906_ZH.md`
- [Covariance Mismatch：分量可辨识性与冻结模型的边界](RAEV2_GUIDANCE_READING_COVARIANCE_MISMATCH_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_COVARIANCE_MISMATCH_20260906_ZH.md`
- [Guidance 阅读：上界如何约束设计，评价如何识别新作用](RAEV2_GUIDANCE_READING_DISCREPANCY_EVALUATION_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_DISCREPANCY_EVALUATION_20260906_ZH.md`
- [RAEv2：判别器 guidance 的证明、残差等价性与轨迹边缘](RAEV2_GUIDANCE_READING_DISCRIMINATOR_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_DISCRIMINATOR_20260906_ZH.md`
- [强弱 × 条件双差：推理纠偏先例的有限检索](RAEV2_GUIDANCE_READING_DOUBLE_DIFFERENCE_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_DOUBLE_DIFFERENCE_20260906_ZH.md`
- [从 guidance 的局部近似追到输出偏差：DPS 反应项阅读](RAEV2_GUIDANCE_READING_DPS_REACTION_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_DPS_REACTION_20260906_ZH.md`
- [RAEv2 guidance 原文阅读：把终点质量变成采样中的控制目标](RAEV2_GUIDANCE_READING_ENDPOINT_CONTROL_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_ENDPOINT_CONTROL_20260906_ZH.md`
- [ERG 精读：检索熵有精确机制，guidance 的误差方向仍需证据](RAEV2_GUIDANCE_READING_ERG_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_ERG_20260906_ZH.md`
- [Guidance 的数值误差：终端精确系数与 ERK 误差方向](RAEV2_GUIDANCE_READING_FITTED_ERROR_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_FITTED_ERROR_20260906_ZH.md`
- [Flow Matching 原论文：局部配对桥的训练依据与有限实现](RAEV2_GUIDANCE_READING_FLOW_MATCHING_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_FLOW_MATCHING_20260906_ZH.md`
- [RAEv2 guidance 原文深读：分类边界、配对质量与随机校正器](RAEV2_GUIDANCE_READING_GEOMETRY_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_GEOMETRY_20260906_ZH.md`
- [ICG / TSG：条件参考的分布意义与随机扰动的机制](RAEV2_GUIDANCE_READING_INDEPENDENT_CONDITION_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_INDEPENDENT_CONDITION_20260906_ZH.md`
- [从分布一致性学习 guidance：Learn to Guide 与 Adversarial Schedules](RAEV2_GUIDANCE_READING_LEARNED_CONSISTENCY_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_LEARNED_CONSISTENCY_20260906_ZH.md`
- [RAEv2 阅读：线性 CFG 的均值与对比主成分机制](RAEV2_GUIDANCE_READING_LINEAR_CPC_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_LINEAR_CPC_20260906_ZH.md`
- [RAEv2 基础阅读：流形 score 的有限噪声分解与曲率耦合](RAEV2_GUIDANCE_READING_MANIFOLD_DECOMPOSITION_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_MANIFOLD_DECOMPOSITION_20260906_ZH.md`
- [Moser有限密度源：理论成立，当前可估计实现未闭合](RAEV2_GUIDANCE_READING_MOSER_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_MOSER_20260906_ZH.md`
- [Particle Guidance 与 EDDY：从联合目标理解轨迹内粒子相互作用](RAEV2_GUIDANCE_READING_PARTICLES_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_PARTICLES_20260906_ZH.md`
- [RAEv2：从实际分布与 flow 配对理解 guidance](RAEV2_GUIDANCE_READING_PATHS_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_PATHS_20260906_ZH.md`
- [RAEv2 guidance 阅读：APG 与 CFG-Zero⋆](RAEV2_GUIDANCE_READING_PROJECTION_ZERO_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_PROJECTION_ZERO_20260906_ZH.md`
- [Guidance 的有限精度误差：PTQD 与 Q-Diffusion](RAEV2_GUIDANCE_READING_QUANTIZATION_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_QUANTIZATION_20260906_ZH.md`
- [反射平均的文献入口：有限群投影与 Gaussian nuisance 积分](RAEV2_GUIDANCE_READING_REFLECTION_SYMMETRIZATION_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_REFLECTION_SYMMETRIZATION_20260906_ZH.md`
- [RAEv2 guidance 阅读：score 中的几何、密度与误差尺度](RAEV2_GUIDANCE_READING_SCORE_GEOMETRY_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_SCORE_GEOMETRY_20260906_ZH.md`
- [Sobolev Descent 原文核对：从实际采样分布学习输运方向](RAEV2_GUIDANCE_READING_SOBOLEV_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_SOBOLEV_20260906_ZH.md`
- [Stochastic Interpolants：配对桥、边缘保持与随机动力学的误差控制](RAEV2_GUIDANCE_READING_STOCHASTIC_INTERPOLANTS_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_STOCHASTIC_INTERPOLANTS_20260906_ZH.md`
- [S²-Guidance 原文深读：随机弱模型的均值、方差与迁移边界](RAEV2_GUIDANCE_READING_STOCHASTIC_WEAK_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_STOCHASTIC_WEAK_20260906_ZH.md`
- [RAEv2：strong/weak 原文阅读与目标冲突](RAEV2_GUIDANCE_READING_STRONG_WEAK_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_STRONG_WEAK_20260906_ZH.md`
- [SWG 阅读：可控信息缺失，不等于兼容的预测误差](RAEV2_GUIDANCE_READING_SWG_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_SWG_20260906_ZH.md`
- [RAEv2 guidance：扩展阅读与设计判断](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md`
- [RAEv2 guidance：开放机制探索，2026-09-06](RAEV2_GUIDANCE_RESTART_20260906_ZH.md) — `docs/RAEV2_GUIDANCE_RESTART_20260906_ZH.md`
- [2026-09-07 guidance 研究结果（持续更新）](RAEV2_GUIDANCE_RESULTS_20260907_ZH.md) — `docs/RAEV2_GUIDANCE_RESULTS_20260907_ZH.md`
- [最后八轮的理论结论：哪些缺口已经缩小，哪些仍在](RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md) — `docs/RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md`
- [固定 guided mean 的反向方差：第二个可实施检验](RAEV2_GUIDED_REVERSE_VARIANCE_20260907_ZH.md) — `docs/RAEV2_GUIDED_REVERSE_VARIANCE_20260907_ZH.md`
- [RAEv2 IG Decoder Pushforward 机制检查](RAEV2_IG_DECODER_PUSHFORWARD_MECHANISM_ZH.md) — `docs/RAEV2_IG_DECODER_PUSHFORWARD_MECHANISM_ZH.md`
- [RAEv2 Internal Guidance 的 Decoder 反转与参数化放大审计](RAEV2_IG_DECODER_REVERSAL_AUDIT_ZH.md) — `docs/RAEV2_IG_DECODER_REVERSAL_AUDIT_ZH.md`
- [RAEv2 Internal Guidance 终点机制验证协议](RAEV2_IG_ENDPOINT_MECHANISM_PROTOCOL_ZH.md) — `docs/RAEV2_IG_ENDPOINT_MECHANISM_PROTOCOL_ZH.md`
- [RAEv2 Internal Guidance 固定 Latent 机制审计](RAEV2_IG_FIXED_LATENT_MECHANISM_AUDIT_ZH.md) — `docs/RAEV2_IG_FIXED_LATENT_MECHANISM_AUDIT_ZH.md`
- [RAEv2 Internal Guidance Scale Response 实验记录](RAEV2_IG_SCALE_RESPONSE_RESULTS_ZH.md) — `docs/RAEV2_IG_SCALE_RESPONSE_RESULTS_ZH.md`
- [从图像密度信号到轨迹内的 Gaussian posterior tilt](RAEV2_IMAGE_CRITIC_GUIDANCE_20260907_ZH.md) — `docs/RAEV2_IMAGE_CRITIC_GUIDANCE_20260907_ZH.md`
- [RAEv2 可逆 latent adapter + LPL 试验记录](RAEV2_INVERTIBLE_LATENT_LPL_PILOT_ZH.md) — `docs/RAEV2_INVERTIBLE_LATENT_LPL_PILOT_ZH.md`
- [RAEv2 LPL 官方 Batch 严格续训协议](RAEV2_LPL_STRICT_CONTINUATION_ZH.md) — `docs/RAEV2_LPL_STRICT_CONTINUATION_ZH.md`
- [第4轮：有限密度差的Moser校正准入检查](RAEV2_MOSER_FINITE_SOURCE_PROTOCOL_20260906_ZH.md) — `docs/RAEV2_MOSER_FINITE_SOURCE_PROTOCOL_20260906_ZH.md`
- [RAEv2 噪声端点：零 velocity 是否更准确](RAEV2_NOISE_ENDPOINT_ZERO_WITNESS_20260906_ZH.md) — `docs/RAEV2_NOISE_ENDPOINT_ZERO_WITNESS_20260906_ZH.md`
- [RAEv2 已知法向噪声与方向曲率审计](RAEV2_NORMAL_CURVATURE_AUDIT_20260906_ZH.md) — `docs/RAEV2_NORMAL_CURVATURE_AUDIT_20260906_ZH.md`
- [RAEv2：由可观测连续性误差导出的最小 guidance](RAEV2_OBSERVABLE_ERROR_GUIDANCE_20260906_ZH.md) — `docs/RAEV2_OBSERVABLE_ERROR_GUIDANCE_20260906_ZH.md`
- [Observable potential：公平成本 1K 比较协议](RAEV2_OBSERVABLE_POTENTIAL_COST_PROTOCOL_20260906_ZH.md) — `docs/RAEV2_OBSERVABLE_POTENTIAL_COST_PROTOCOL_20260906_ZH.md`
- [固定势函数：纠正筛查规模后的配对 5K 评估](RAEV2_OBSERVABLE_POTENTIAL_SCALE_AUDIT_20260906_ZH.md) — `docs/RAEV2_OBSERVABLE_POTENTIAL_SCALE_AUDIT_20260906_ZH.md`
- [固定势函数：配对 5K 规模审计结果](RAEV2_OBSERVABLE_POTENTIAL_SCALE_RESULTS_20260906_ZH.md) — `docs/RAEV2_OBSERVABLE_POTENTIAL_SCALE_RESULTS_20260906_ZH.md`
- [Observable potential：配对 1K 筛查未达标](RAEV2_OBSERVABLE_POTENTIAL_SCREEN_20260906_ZH.md) — `docs/RAEV2_OBSERVABLE_POTENTIAL_SCREEN_20260906_ZH.md`
- [RAEv2 可观测误差的有限势函数求解器](RAEV2_OBSERVABLE_POTENTIAL_SOLVER_20260906_ZH.md) — `docs/RAEV2_OBSERVABLE_POTENTIAL_SOLVER_20260906_ZH.md`
- [RAEv2 配对桥流：固定可学习性实验](RAEV2_PAIRED_BRIDGE_PILOT_PROTOCOL_20260906_ZH.md) — `docs/RAEV2_PAIRED_BRIDGE_PILOT_PROTOCOL_20260906_ZH.md`
- [配对桥流固定机制实验结果](RAEV2_PAIRED_BRIDGE_PILOT_RESULTS_20260906_ZH.md) — `docs/RAEV2_PAIRED_BRIDGE_PILOT_RESULTS_20260906_ZH.md`
- [RAEv2 配对桥流：固定 1K 质量筛查](RAEV2_PAIRED_BRIDGE_SCREEN_PROTOCOL_20260906_ZH.md) — `docs/RAEV2_PAIRED_BRIDGE_SCREEN_PROTOCOL_20260906_ZH.md`
- [配对桥固定 1K：未达到目标，结束当前实现](RAEV2_PAIRED_BRIDGE_SCREEN_RESULTS_20260906_ZH.md) — `docs/RAEV2_PAIRED_BRIDGE_SCREEN_RESULTS_20260906_ZH.md`
- [共享噪声的边界一致判别器：推导与一次固定拟合](RAEV2_PAIRED_NOISE_RATIO_20260907_ZH.md) — `docs/RAEV2_PAIRED_NOISE_RATIO_20260907_ZH.md`
- [RAEv2 上的 PFR 迁移：指数重定时、正式协议与初步结果](RAEV2_PFR_TRANSFER_RESULTS_ZH.md) — `docs/RAEV2_PFR_TRANSFER_RESULTS_ZH.md`
- [RAEv2：具有独立 KL 证书的 posterior acceptance guidance](RAEV2_POSTERIOR_ACCEPTANCE_GUIDANCE_20260906_ZH.md) — `docs/RAEV2_POSTERIOR_ACCEPTANCE_GUIDANCE_20260906_ZH.md`
- [RAEv2：有限势函数补空间的可观测残差检验](RAEV2_POTENTIAL_OMITTED_WITNESS_20260906_ZH.md) — `docs/RAEV2_POTENTIAL_OMITTED_WITNESS_20260906_ZH.md`
- [RAEv2 predicted-clean 四格特征交互审计](RAEV2_PREDICTED_CLEAN_INTERACTION_AUDIT_20260906_ZH.md) — `docs/RAEV2_PREDICTED_CLEAN_INTERACTION_AUDIT_20260906_ZH.md`
- [固定预训练特征的实际轨迹密度比](RAEV2_PREFIX_RATIO_20260907_ZH.md) — `docs/RAEV2_PREFIX_RATIO_20260907_ZH.md`
- [固定前缀头的全局概率校准：一次可证伪的检查](RAEV2_PREFIX_RATIO_CALIBRATION_20260907_ZH.md) — `docs/RAEV2_PREFIX_RATIO_CALIBRATION_20260907_ZH.md`
- [最后第2轮：协方差压力与对称创新的结构裁决](RAEV2_PRESSURE_INNOVATION_PROTOCOL_20260906_ZH.md) — `docs/RAEV2_PRESSURE_INNOVATION_PROTOCOL_20260906_ZH.md`
- [最后第2轮结果：协方差压力与对称创新不进入训练](RAEV2_PRESSURE_INNOVATION_RESULTS_20260906_ZH.md) — `docs/RAEV2_PRESSURE_INNOVATION_RESULTS_20260906_ZH.md`
- [RAEv2：真实下一状态误差的凸 proximal 校正](RAEV2_PROXIMAL_ERROR_PROJECTION_20260906_ZH.md) — `docs/RAEV2_PROXIMAL_ERROR_PROJECTION_20260906_ZH.md`
- [RAEv2：全局零锚点 proximal 校准与实际单步 W₂ 保证](RAEV2_PROXIMAL_GLOBAL_CALIBRATION_20260906_ZH.md) — `docs/RAEV2_PROXIMAL_GLOBAL_CALIBRATION_20260906_ZH.md`
- [RAEv2 proximal 校准的 channel 共享结构审阅](RAEV2_PROXIMAL_SHARED_CALIBRATION_20260906_ZH.md) — `docs/RAEV2_PROXIMAL_SHARED_CALIBRATION_20260906_ZH.md`
- [固定 query 均值弱分支：同类误差放大机制检验](RAEV2_QUERY_MEAN_ERROR_COMPATIBILITY_PROTOCOL_20260906_ZH.md) — `docs/RAEV2_QUERY_MEAN_ERROR_COMPATIBILITY_PROTOCOL_20260906_ZH.md`
- [Query 均值弱分支：配对误差方向为正，但风险修复成分很小](RAEV2_QUERY_MEAN_ERROR_COMPATIBILITY_RESULTS_20260906_ZH.md) — `docs/RAEV2_QUERY_MEAN_ERROR_COMPATIBILITY_RESULTS_20260906_ZH.md`
- [RAEv2 clean-prediction 半径与方向：2026-09-05 机制筛查计划](RAEV2_RADIUS_DIRECTION_PLAN_20260905_ZH.md) — `docs/RAEV2_RADIUS_DIRECTION_PLAN_20260905_ZH.md`
- [RAEv2 raw token 确定性支撑界：固定缓存检查](RAEV2_RAW_TOKEN_SUPPORT_BOUND_20260906_ZH.md) — `docs/RAEV2_RAW_TOKEN_SUPPORT_BOUND_20260906_ZH.md`
- [RAEv2 Relative Transport Iteration：有限 Flow-Map 外推与严格对照](RAEV2_RELATIVE_TRANSPORT_ITERATION_20260905_ZH.md) — `docs/RAEV2_RELATIVE_TRANSPORT_ITERATION_20260905_ZH.md`
- [剩余轮次的机制筛选](RAEV2_REMAINING_MECHANISM_REVIEW_20260906_ZH.md) — `docs/RAEV2_REMAINING_MECHANISM_REVIEW_20260906_ZH.md`
- [RAEv2 guidance 研究档案导航](RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md) — `docs/RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md`
- [RAEv2 guidance：当前结果与新旧研究总入口](RAEV2_RESEARCH_ARCHIVE_INDEX_20260907_ZH.md) — `docs/RAEV2_RESEARCH_ARCHIVE_INDEX_20260907_ZH.md`
- [RAEv2 Self-Guidance 研究账本](RAEV2_SELF_GUIDANCE_RESEARCH_LEDGER_ZH.md) — `docs/RAEV2_SELF_GUIDANCE_RESEARCH_LEDGER_ZH.md`
- [官方 IG 上的固定类别信息补充](RAEV2_SEMANTIC_COMPLEMENT_20260907_ZH.md) — `docs/RAEV2_SEMANTIC_COMPLEMENT_20260907_ZH.md`
- [类别补充：固定CPU方向检查结果](RAEV2_SEMANTIC_COMPLEMENT_CPU_20260907_ZH.md) — `docs/RAEV2_SEMANTIC_COMPLEMENT_CPU_20260907_ZH.md`
- [固定类别补充：1K结果与独立5K确认](RAEV2_SEMANTIC_COMPLEMENT_RESULTS_20260907_ZH.md) — `docs/RAEV2_SEMANTIC_COMPLEMENT_RESULTS_20260907_ZH.md`
- [Actual-q 矩反馈与有限 Euler：独立理论审计](RAEV2_SOBOLEV_ACTUAL_Q_EULER_AUDIT_20260906_ZH.md) — `docs/RAEV2_SOBOLEV_ACTUAL_Q_EULER_AUDIT_20260906_ZH.md`
- [RAEv2：两个正交空间能量球的冻结 1K 试验](RAEV2_SPATIAL_ENERGY_BALLS_PROTOCOL_20260906_ZH.md) — `docs/RAEV2_SPATIAL_ENERGY_BALLS_PROTOCOL_20260906_ZH.md`
- [RAEv2 两个空间能量球：冻结 1K 结果](RAEV2_SPATIAL_ENERGY_BALLS_RESULTS_20260906_ZH.md) — `docs/RAEV2_SPATIAL_ENERGY_BALLS_RESULTS_20260906_ZH.md`
- [RAEv2 固定空间 DC/AC 二阶能量审计（2026-09-06）](RAEV2_SPECTRAL_ENERGY_AUDIT_20260906_ZH.md) — `docs/RAEV2_SPECTRAL_ENERGY_AUDIT_20260906_ZH.md`
- [RAEv2：以 full score 平衡 guidance 与随机扩散](RAEV2_STOCHASTIC_GUIDANCE_PLAN_20260906_ZH.md) — `docs/RAEV2_STOCHASTIC_GUIDANCE_PLAN_20260906_ZH.md`
- [固定删块弱参考：S²-Guidance 的 RAEv2 迁移](RAEV2_STOCHASTIC_WEAK_20260907_ZH.md) — `docs/RAEV2_STOCHASTIC_WEAK_20260907_ZH.md`
- [RAEv2 水平翻转平均：必要前提 CPU 审计](RAEV2_SYMMETRY_PREREQUISITE_20260906_ZH.md) — `docs/RAEV2_SYMMETRY_PREREQUISITE_20260906_ZH.md`
- [第3轮：时间 score PDE 的目标可辨识性审计](RAEV2_TEMPORAL_SCORE_PDE_PROTOCOL_20260906_ZH.md) — `docs/RAEV2_TEMPORAL_SCORE_PDE_PROTOCOL_20260906_ZH.md`
- [Guidance 的输运坐标投影：固定两种物理坐标](RAEV2_TRANSPORT_PROJECTION_20260907_ZH.md) — `docs/RAEV2_TRANSPORT_PROJECTION_20260907_ZH.md`
- [两个空间子空间上的 Gaussian 密度比 guidance](RAEV2_TWO_MODE_RATIO_20260907_ZH.md) — `docs/RAEV2_TWO_MODE_RATIO_20260907_ZH.md`
- [RAE / RAEv2 上 LPL 探索的深度综合](RAE_RAEV2_LPL_DEEP_SYNTHESIS_ZH.md) — `docs/RAE_RAEV2_LPL_DEEP_SYNTHESIS_ZH.md`
- [旧 RAE 与 RAEv2 的 LPL 分化机制](RAE_RAEV2_LPL_MECHANISM_ZH.md) — `docs/RAE_RAEV2_LPL_MECHANISM_ZH.md`
- [RAEv2 开放问题备忘](archive/RAEV2_OPEN_QUESTIONS_ZH.md) — `docs/archive/RAEV2_OPEN_QUESTIONS_ZH.md`
- [RAEv2 guidance exploration compact archive](data/raev2_guidance_exploration_20260905/README.md) — `docs/data/raev2_guidance_exploration_20260905/README.md`
- [RAEv2 guidance 最终轻量数据包](data/raev2_guidance_final_20260906/README.md) — `docs/data/raev2_guidance_final_20260906/README.md`
- [配对 5K FID：局部一阶误差分析](data/raev2_guidance_final_20260906/files/observable_potential_scale_audit_analysis_v1/README_ZH.md) — `docs/data/raev2_guidance_final_20260906/files/observable_potential_scale_audit_analysis_v1/README_ZH.md`
- [三臂配对 5K FID 一阶分析：执行结果](data/raev2_guidance_final_20260906/files/observable_potential_scale_audit_analysis_v1/RESULTS_ZH.md) — `docs/data/raev2_guidance_final_20260906/files/observable_potential_scale_audit_analysis_v1/RESULTS_ZH.md`
- [RAEv2 固定配对桥机制实验：CPU 汇总草案](data/raev2_guidance_final_20260906/files/paired_bridge_v1/analysis_v1/README_ZH.md) — `docs/data/raev2_guidance_final_20260906/files/paired_bridge_v1/analysis_v1/README_ZH.md`
- [固定 query 均值弱分支：同类误差放大机制检验](data/raev2_guidance_final_20260906/files/query_mean_error_compatibility_v1/frozen_protocol.md) — `docs/data/raev2_guidance_final_20260906/files/query_mean_error_compatibility_v1/frozen_protocol.md`
- [RAEv2 来源与复查脚本补充包](data/raev2_guidance_final_sources_20260906/README.md) — `docs/data/raev2_guidance_final_sources_20260906/README.md`
- [Final Round 4: finite-source Moser audit](data/raev2_moser_finite_source_20260906/README.md) — `docs/data/raev2_moser_finite_source_20260906/README.md`
- [RAEv2 PFR / OU 迁移的便携结果](data/raev2_pfr_ou_transfer_20260904/README.md) — `docs/data/raev2_pfr_ou_transfer_20260904/README.md`
- [RAEv2 final Round 2: pressure and symmetric innovation](data/raev2_pressure_innovation_20260906/README.md) — `docs/data/raev2_pressure_innovation_20260906/README.md`
- [第3轮：时间score PDE的目标可辨识性](data/raev2_temporal_score_pde_20260906/README.md) — `docs/data/raev2_temporal_score_pde_20260906/README.md`

### RAE、LPL、decoder 与 latent 几何

- [Prior-Decoder 机制的外部模型验证方案](EXTERNAL_PRIOR_DECODER_VALIDATION_ZH.md) — `docs/EXTERNAL_PRIOR_DECODER_VALIDATION_ZH.md`
- [LPL 路线结题审计与下一阶段研究议程](LPL_LINE_CLOSURE_AND_SOLID_RESEARCH_AGENDA_ZH.md) — `docs/LPL_LINE_CLOSURE_AND_SOLID_RESEARCH_AGENDA_ZH.md`
- [Prior-Decoder 对齐：文献定位、联合训练判断与实验路线](PRIOR_DECODER_ALIGNMENT_LITERATURE_AND_PLAN_ZH.md) — `docs/PRIOR_DECODER_ALIGNMENT_LITERATURE_AND_PLAN_ZH.md`
- [从两阶段生成到 Prior-Decoder 断层：实验与文献指南](PRIOR_DECODER_EXPERIMENTS_AND_LITERATURE_GUIDE_ZH.md) — `docs/PRIOR_DECODER_EXPERIMENTS_AND_LITERATURE_GUIDE_ZH.md`
- [RAE clean-estimate 轨迹探索：事前预测](RAE_CLEAN_ESTIMATE_TRAJECTORY_PREDICTIONS_ZH.md) — `docs/RAE_CLEAN_ESTIMATE_TRAJECTORY_PREDICTIONS_ZH.md`
- [RAE Cycle-Direction 因果干预协议](RAE_CYCLE_DIRECTION_PROTOCOL_ZH.md) — `docs/RAE_CYCLE_DIRECTION_PROTOCOL_ZH.md`
- [RAE Cycle-Direction 因果干预结果](RAE_CYCLE_DIRECTION_RESULTS_ZH.md) — `docs/RAE_CYCLE_DIRECTION_RESULTS_ZH.md`
- [RAE Decoder 噪声坐标几何审计](RAE_DECODER_NOISE_GEOMETRY_RESULTS_ZH.md) — `docs/RAE_DECODER_NOISE_GEOMETRY_RESULTS_ZH.md`
- [RAE Decoder-Aware Phase 0 结果](RAE_DECODER_RISK_PHASE0_RESULTS_ZH.md) — `docs/RAE_DECODER_RISK_PHASE0_RESULTS_ZH.md`
- [RAE 确定性 Decoder 上的严格 LPL 对照实验](RAE_DETERMINISTIC_LPL_REPRODUCTION_ZH.md) — `docs/RAE_DETERMINISTIC_LPL_REPRODUCTION_ZH.md`
- [DINOv2-L RAE 上的 Flow/LPL 误差机制 Pilot](RAE_DINOV2_L_LPL_PILOT_ZH.md) — `docs/RAE_DINOV2_L_LPL_PILOT_ZH.md`
- [RAE Encoder-Decoder 反向层级诊断结果](RAE_ENCODER_DECODER_ATLAS_RESULTS_ZH.md) — `docs/RAE_ENCODER_DECODER_ATLAS_RESULTS_ZH.md`
- [RAE 生成研究 Claim Matrix](RAE_GENERATION_CLAIM_MATRIX_ZH.md) — `docs/RAE_GENERATION_CLAIM_MATRIX_ZH.md`
- [RAE 生成机制文献审计与研究重定位](RAE_GENERATION_LITERATURE_AUDIT_ZH.md) — `docs/RAE_GENERATION_LITERATURE_AUDIT_ZH.md`
- [RAE Latent Trust 与 Decoder 语义轴对齐结果](RAE_LATENT_TRUST_DECODER_ALIGNMENT_RESULTS_ZH.md) — `docs/RAE_LATENT_TRUST_DECODER_ALIGNMENT_RESULTS_ZH.md`
- [RAE Latent Trust Spectrum：从失败的 SPC 到噪声依赖方向信任](RAE_LATENT_TRUST_SPECTRUM_RESULTS_ZH.md) — `docs/RAE_LATENT_TRUST_SPECTRUM_RESULTS_ZH.md`
- [RAE-DINOv2 Layerwise Correspondence 大样本结果](RAE_LAYERWISE_CORRESPONDENCE_RESULTS_ZH.md) — `docs/RAE_LAYERWISE_CORRESPONDENCE_RESULTS_ZH.md`
- [RAE Layerwise Representation Path 预注册协议](RAE_LAYERWISE_PATH_PROTOCOL_ZH.md) — `docs/RAE_LAYERWISE_PATH_PROTOCOL_ZH.md`
- [RAE LPL 跨 Tokenizer 生成验证协议](RAE_LPL_CROSS_TOKENIZER_PROTOCOL_ZH.md) — `docs/RAE_LPL_CROSS_TOKENIZER_PROTOCOL_ZH.md`
- [RAE LPL 跨 Tokenizer 生成验证](RAE_LPL_CROSS_TOKENIZER_RESULTS_ZH.md) — `docs/RAE_LPL_CROSS_TOKENIZER_RESULTS_ZH.md`
- [RAE-LPL prediction statistics detach 审计](RAE_LPL_DETACH_AUDIT_ZH.md) — `docs/RAE_LPL_DETACH_AUDIT_ZH.md`
- [RAE 上 LPL 实验全量台账](RAE_LPL_EXPERIMENT_LEDGER_ZH.md) — `docs/RAE_LPL_EXPERIMENT_LEDGER_ZH.md`
- [RAE 上 LPL 的有限半径机制实验](RAE_LPL_FINITE_RADIUS_MECHANISM_ZH.md) — `docs/RAE_LPL_FINITE_RADIUS_MECHANISM_ZH.md`
- [RAE 上 LPL 的改进研究](RAE_LPL_IMPROVEMENT_RESEARCH_ZH.md) — `docs/RAE_LPL_IMPROVEMENT_RESEARCH_ZH.md`
- [RAE-LPL 大规模真实性验证协议](RAE_LPL_LARGE_SCALE_AUTHENTICITY_PROTOCOL_ZH.md) — `docs/RAE_LPL_LARGE_SCALE_AUTHENTICITY_PROTOCOL_ZH.md`
- [RAE-adapted LPL 大规模真实性验证结果](RAE_LPL_LARGE_SCALE_AUTHENTICITY_RESULTS_ZH.md) — `docs/RAE_LPL_LARGE_SCALE_AUTHENTICITY_RESULTS_ZH.md`
- [RAE 上 LPL 后续研究审计](RAE_LPL_RESEARCH_AUDIT_ZH.md) — `docs/RAE_LPL_RESEARCH_AUDIT_ZH.md`
- [RAE layerwise path 逆条件数：修正后的事前预测](RAE_PATH_CONDITIONING_PREDICTIONS_ZH.md) — `docs/RAE_PATH_CONDITIONING_PREDICTIONS_ZH.md`
- [RAE layerwise path 逆条件数：探索结果](RAE_PATH_CONDITIONING_RESULTS_ZH.md) — `docs/RAE_PATH_CONDITIONING_RESULTS_ZH.md`
- [RAE path 2k->5k 交叉分叉：事前协议](RAE_PATH_CROSSOVER_PREREG_ZH.md) — `docs/RAE_PATH_CROSSOVER_PREREG_ZH.md`
- [RAE 子空间路径课程：2k→5k crossover 结果](RAE_PATH_CROSSOVER_RESULTS_ZH.md) — `docs/RAE_PATH_CROSSOVER_RESULTS_ZH.md`
- [RAE 生成路径差分方向：机制筛查协议](RAE_PATH_DIFFERENCE_PROTOCOL_ZH.md) — `docs/RAE_PATH_DIFFERENCE_PROTOCOL_ZH.md`
- [RAE 生成路径差分方向：机制筛查结果](RAE_PATH_DIFFERENCE_RESULTS_ZH.md) — `docs/RAE_PATH_DIFFERENCE_RESULTS_ZH.md`
- [RAE floor path：generated-latent closure 外推检验](RAE_PATH_FLOOR_CLOSURE_PREREG_ZH.md) — `docs/RAE_PATH_FLOOR_CLOSURE_PREREG_ZH.md`
- [RAE path：动态梯度翻转与 generated-latent closure 结果](RAE_PATH_GRADIENT_AND_CLOSURE_RESULTS_ZH.md) — `docs/RAE_PATH_GRADIENT_AND_CLOSURE_RESULTS_ZH.md`
- [RAE path 子空间梯度干扰：事前预测](RAE_PATH_GRADIENT_INTERFERENCE_PREREG_ZH.md) — `docs/RAE_PATH_GRADIENT_INTERFERENCE_PREREG_ZH.md`
- [RAE path 低噪声梯度翻转：独立确认协议](RAE_PATH_GRADIENT_REVERSAL_CONFIRM_PREREG_ZH.md) — `docs/RAE_PATH_GRADIENT_REVERSAL_CONFIRM_PREREG_ZH.md`
- [RAE floor path：5k checkpoint 持久性门控](RAE_PATH_SCHEDULE_5K_CHECKPOINT_PREREG_ZH.md) — `docs/RAE_PATH_SCHEDULE_5K_CHECKPOINT_PREREG_ZH.md`
- [RAE floor path：5k checkpoint 持久性门控结果](RAE_PATH_SCHEDULE_5K_CHECKPOINT_RESULTS_ZH.md) — `docs/RAE_PATH_SCHEDULE_5K_CHECKPOINT_RESULTS_ZH.md`
- [RAE floor path：held-out time/subspace error atlas 事前预测](RAE_PATH_SCHEDULE_ERROR_ATLAS_PREREG_ZH.md) — `docs/RAE_PATH_SCHEDULE_ERROR_ATLAS_PREREG_ZH.md`
- [RAE floor path：held-out error atlas 结果](RAE_PATH_SCHEDULE_ERROR_ATLAS_RESULTS_ZH.md) — `docs/RAE_PATH_SCHEDULE_ERROR_ATLAS_RESULTS_ZH.md`
- [RAE decoder 加权路径调度：离线筛选事前预测](RAE_PATH_SCHEDULE_SCREEN_PREDICTIONS_ZH.md) — `docs/RAE_PATH_SCHEDULE_SCREEN_PREDICTIONS_ZH.md`
- [RAE well-conditioned path：2k tiny gate 事前预测](RAE_PATH_SCHEDULE_TINY_GATE_PREREG_ZH.md) — `docs/RAE_PATH_SCHEDULE_TINY_GATE_PREREG_ZH.md`
- [RAE well-conditioned path：2k tiny gate 结果](RAE_PATH_SCHEDULE_TINY_GATE_RESULTS_ZH.md) — `docs/RAE_PATH_SCHEDULE_TINY_GATE_RESULTS_ZH.md`
- [RAE SPC 标准目标机制检查：看结果前的预测](RAE_SPC_MECHANISM_PREDICTIONS_ZH.md) — `docs/RAE_SPC_MECHANISM_PREDICTIONS_ZH.md`
- [RAE 子空间路径课程：五种子机制结果](RAE_SPC_MULTISEED_MECHANISM_ZH.md) — `docs/RAE_SPC_MULTISEED_MECHANISM_ZH.md`
- [RAE 子空间路径课程五 seed 验证：事前协议](RAE_SPC_MULTISEED_PREREG_ZH.md) — `docs/RAE_SPC_MULTISEED_PREREG_ZH.md`

### PFR、反事实参考与半群 guidance

- [仿射反事实密度比 Internal Guidance](AFFINE_COUNTERFACTUAL_RATIO_IG_THEORY_ZH.md) — `docs/AFFINE_COUNTERFACTUAL_RATIO_IG_THEORY_ZH.md`
- [AutoGuidance 的前瞻校准：分布、共轭算子与严格边界](AUTOGUIDANCE_FORESIGHT_DISTRIBUTION_THEORY_ZH.md) — `docs/AUTOGUIDANCE_FORESIGHT_DISTRIBUTION_THEORY_ZH.md`
- [PFR 的反事实参考残差理论：从有限弱响应到终点风险](PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md) — `docs/PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md`
- [PFR 的指数重定时理论与机制审计](PFR_EXPONENTIAL_RETIMING_THEORY_ZH.md) — `docs/PFR_EXPONENTIAL_RETIMING_THEORY_ZH.md`
- [PFR 机制审计与反例归档](PFR_MECHANISM_AUDIT_20260903_ZH.md) — `docs/PFR_MECHANISM_AUDIT_20260903_ZH.md`
- [PFR 的 OU 概率小波解释与谱认证方法](PFR_OU_PROBABILITY_WAVELET_THEORY_ZH.md) — `docs/PFR_OU_PROBABILITY_WAVELET_THEORY_ZH.md`
- [Projected Future-Reference Internal Guidance](PROJECTED_FUTURE_REFERENCE_IG_THEORY_ZH.md) — `docs/PROJECTED_FUTURE_REFERENCE_IG_THEORY_ZH.md`
- [Semigroup-Consistent Guidance：从端点目标推出整条引导路径](SEMIGROUP_CONSISTENT_GUIDANCE_THEORY_ZH.md) — `docs/SEMIGROUP_CONSISTENT_GUIDANCE_THEORY_ZH.md`
- [PFR counterfactual-residual theory compact artifacts](data/pfr_counterfactual_residual_theory_20260903/README.md) — `docs/data/pfr_counterfactual_residual_theory_20260903/README.md`
- [PFR mechanism audit data (2026-09-03)](data/pfr_mechanism_audit_20260903/README.md) — `docs/data/pfr_mechanism_audit_20260903/README.md`
- [PFR OU probability-wavelet compact results](data/pfr_ou_probability_wavelet_20260904/README.md) — `docs/data/pfr_ou_probability_wavelet_20260904/README.md`
- [Projected Future-Reference IG data](data/projected_future_reference_ig/README.md) — `docs/data/projected_future_reference_ig/README.md`

### SiT / ImageNet-100 与内部头

- [Checkpoint Reference Long Study v1](../checkpoint_reference_long_study_v1/summary/README.md) — `checkpoint_reference_long_study_v1/summary/README.md`
- [官方 SiT-S/2 近期冻结头实验复现](IMAGENET100_OFFICIAL_SIT_S2_REPLICATION_ZH.md) — `docs/IMAGENET100_OFFICIAL_SIT_S2_REPLICATION_ZH.md`
- [ImageNet-100 SiT 400K 有限强度 Guidance 动力学审计](IMAGENET100_SIT_400K_FINITE_GUIDANCE_DYNAMICS_ZH.md) — `docs/IMAGENET100_SIT_400K_FINITE_GUIDANCE_DYNAMICS_ZH.md`
- [ImageNet-100 SiT 400K 训练方向与共同分量实验记录](IMAGENET100_SIT_400K_FUTURE_COMMON_UNIQUE_RESULTS_ZH.md) — `docs/IMAGENET100_SIT_400K_FUTURE_COMMON_UNIQUE_RESULTS_ZH.md`
- [ImageNet-100 SiT 400K Guidance 对照实验记录](IMAGENET100_SIT_400K_GUIDANCE_MECHANISM_AUDIT_ZH.md) — `docs/IMAGENET100_SIT_400K_GUIDANCE_MECHANISM_AUDIT_ZH.md`
- [SiT 800K finite-guidance 紧凑复验](IMAGENET100_SIT_800K_COMPACT_REPLICATION_ZH.md) — `docs/IMAGENET100_SIT_800K_COMPACT_REPLICATION_ZH.md`
- [SiT 800K 强模型响应放大实验](IMAGENET100_SIT_800K_RESPONSE_AMPLIFICATION_RESULTS_ZH.md) — `docs/IMAGENET100_SIT_800K_RESPONSE_AMPLIFICATION_RESULTS_ZH.md`
- [SiT 800K Tangent Endpoint 投影实验](IMAGENET100_SIT_800K_TANGENT_PROJECTION_RESULTS_ZH.md) — `docs/IMAGENET100_SIT_800K_TANGENT_PROJECTION_RESULTS_ZH.md`
- [SiT 800K Tangent Transport 首轮实验分析](IMAGENET100_SIT_800K_TANGENT_TRANSPORT_RESULTS_ZH.md) — `docs/IMAGENET100_SIT_800K_TANGENT_TRANSPORT_RESULTS_ZH.md`
- [SiT 深度有限差分机制实验](IMAGENET100_SIT_DEPTH_DIFFERENCE_MECHANISM_RESULTS_ZH.md) — `docs/IMAGENET100_SIT_DEPTH_DIFFERENCE_MECHANISM_RESULTS_ZH.md`
- [ImageNet-100 SiT v800 Error-Triangulated Guidance 验证](IMAGENET100_SIT_ETG_V800_RESULTS_ZH.md) — `docs/IMAGENET100_SIT_ETG_V800_RESULTS_ZH.md`
- [ImageNet-100 SiT FID 实验总表](IMAGENET100_SIT_FID_INVENTORY_ZH.md) — `docs/IMAGENET100_SIT_FID_INVENTORY_ZH.md`
- [ImageNet-100 SiT flow baseline](IMAGENET100_SIT_FLOW_BASELINE_ZH.md) — `docs/IMAGENET100_SIT_FLOW_BASELINE_ZH.md`
- [ImageNet-100 SiT 前瞻物质导数 Internal Guidance](IMAGENET100_SIT_FORESIGHT_MATERIAL_IG_RESULTS_ZH.md) — `docs/IMAGENET100_SIT_FORESIGHT_MATERIAL_IG_RESULTS_ZH.md`
- [SiT 冻结 v800 后训练 clean head 的结果](IMAGENET100_SIT_FROZEN_V_CLEAN_HEAD_RESULTS_ZH.md) — `docs/IMAGENET100_SIT_FROZEN_V_CLEAN_HEAD_RESULTS_ZH.md`
- [ImageNet-100 SiT：JiT-style x Prediction 对照](IMAGENET100_SIT_JIT_STYLE_X_PROTOCOL_ZH.md) — `docs/IMAGENET100_SIT_JIT_STYLE_X_PROTOCOL_ZH.md`
- [ImageNet-100 SiT 多尺度 Guidance 实验报告](IMAGENET100_SIT_MULTISCALE_GUIDANCE_RESULTS_ZH.md) — `docs/IMAGENET100_SIT_MULTISCALE_GUIDANCE_RESULTS_ZH.md`
- [SiT 800K nominal-path frozen guidance 机制验证](IMAGENET100_SIT_NOMINAL_GUIDANCE_TRANSFER_800K_RESULTS_ZH.md) — `docs/IMAGENET100_SIT_NOMINAL_GUIDANCE_TRANSFER_800K_RESULTS_ZH.md`
- [ImageNet-100 SiT 单预测目标对照协议](IMAGENET100_SIT_SINGLE_TARGET_PROTOCOL_ZH.md) — `docs/IMAGENET100_SIT_SINGLE_TARGET_PROTOCOL_ZH.md`
- [SiT 终端分布控制审计](IMAGENET100_SIT_TERMINAL_DISTRIBUTION_CONTROL_AUDIT_ZH.md) — `docs/IMAGENET100_SIT_TERMINAL_DISTRIBUTION_CONTROL_AUDIT_ZH.md`
- [ImageNet-100 SiT-v 与 JiT-x 静态内插/外推报告](IMAGENET100_SIT_V_JIT_X_STATIC_SWEEP_ZH.md) — `docs/IMAGENET100_SIT_V_JIT_X_STATIC_SWEEP_ZH.md`
- [SiT 弱头差值与 x800 冻结读出实验](IMAGENET100_SIT_WEAK_DIFFERENCE_X800_READOUT_RESULTS_ZH.md) — `docs/IMAGENET100_SIT_WEAK_DIFFERENCE_X800_READOUT_RESULTS_ZH.md`
- [SiT EMA 权重外推实验](IMAGENET100_SIT_WEIGHT_EXTRAPOLATION_RESULTS_ZH.md) — `docs/IMAGENET100_SIT_WEIGHT_EXTRAPOLATION_RESULTS_ZH.md`
- [SiT 双输出端点与 NFE 机制审计](SIT_DUAL_OUTPUT_ENDPOINT_MECHANISM_AUDIT_ZH.md) — `docs/SIT_DUAL_OUTPUT_ENDPOINT_MECHANISM_AUDIT_ZH.md`
- [Official SiT-S/2 replication data](data/imagenet100_official_sit_s2_recent_replication/README.md) — `docs/data/imagenet100_official_sit_s2_recent_replication/README.md`
- [SiT 400K finite-guidance mechanism data](data/imagenet100_sit_400k_finite_guidance_dynamics/README.md) — `docs/data/imagenet100_sit_400k_finite_guidance_dynamics/README.md`
- [ImageNet-100 SiT 400K guidance 对照实验数据](data/imagenet100_sit_400k_guidance_mechanism/README.md) — `docs/data/imagenet100_sit_400k_guidance_mechanism/README.md`
- [ImageNet-100 SiT 800K 紧凑复验数据](data/imagenet100_sit_800k_compact_replication/README.md) — `docs/data/imagenet100_sit_800k_compact_replication/README.md`
- [ImageNet-100 SiT 800K nominal guidance transfer](data/imagenet100_sit_800k_nominal_guidance_transfer/README.md) — `docs/data/imagenet100_sit_800k_nominal_guidance_transfer/README.md`
- [SiT 800K strong-response amplification 便携数据](data/imagenet100_sit_800k_response_amplification/README.md) — `docs/data/imagenet100_sit_800k_response_amplification/README.md`
- [SiT 800K tangent endpoint 投影便携数据](data/imagenet100_sit_800k_tangent_projection/README.md) — `docs/data/imagenet100_sit_800k_tangent_projection/README.md`
- [SiT 800K tangent transport 便携数据](data/imagenet100_sit_800k_tangent_transport/README.md) — `docs/data/imagenet100_sit_800k_tangent_transport/README.md`
- [SiT 800K v500 gamma-rho 扫描便携数据](data/imagenet100_sit_800k_v500_gamma_rho/README.md) — `docs/data/imagenet100_sit_800k_v500_gamma_rho/README.md`
- [ImageNet-100 SiT CAFM tangent predictivity v1](data/imagenet100_sit_cafm_tangent_predictivity_v1/README.md) — `docs/data/imagenet100_sit_cafm_tangent_predictivity_v1/README.md`
- [Seed 0 DDP continuation to 5000 steps](data/imagenet100_sit_cafm_tangent_predictivity_v1/critic_training_ddp2_seed0_5000/README.md) — `docs/data/imagenet100_sit_cafm_tangent_predictivity_v1/critic_training_ddp2_seed0_5000/README.md`
- [SiT 深度有限差分机制便携数据](data/imagenet100_sit_depth_difference_mechanism_v1/README.md) — `docs/data/imagenet100_sit_depth_difference_mechanism_v1/README.md`
- [SiT v800 Error-Triangulated Guidance 验证数据](data/imagenet100_sit_etg_v800_depth8_v1/README.md) — `docs/data/imagenet100_sit_etg_v800_depth8_v1/README.md`
- [Foresight Material-Derivative IG 数据说明](data/imagenet100_sit_foresight_material_ig/README.md) — `docs/data/imagenet100_sit_foresight_material_ig/README.md`
- [SiT v800 冻结末层完整 x 预测头实验](data/imagenet100_sit_frozen_final_x_full_head_50k/README.md) — `docs/data/imagenet100_sit_frozen_final_x_full_head_50k/README.md`
- [SiT v800 冻结中间 epsilon 预测头实验](data/imagenet100_sit_frozen_internal_epsilon_head_50k/README.md) — `docs/data/imagenet100_sit_frozen_internal_epsilon_head_50k/README.md`
- [SiT v800 冻结中间 v 头实验](data/imagenet100_sit_frozen_internal_v_head_50k/README.md) — `docs/data/imagenet100_sit_frozen_internal_v_head_50k/README.md`
- [SiT v800 冻结中间 x 预测头实验](data/imagenet100_sit_frozen_internal_x_head_50k/README.md) — `docs/data/imagenet100_sit_frozen_internal_x_head_50k/README.md`
- [SiT frozen-v800 clean-head 便携数据](data/imagenet100_sit_frozen_v_clean_head_50k/README.md) — `docs/data/imagenet100_sit_frozen_v_clean_head_50k/README.md`
- [SiT v800 冻结 hidden-state 外推实验](data/imagenet100_sit_hidden_state_extrapolation/README.md) — `docs/data/imagenet100_sit_hidden_state_extrapolation/README.md`
- [SiT v800 强头、弱头及差值的频率分析](data/imagenet100_sit_internal_head_frequency/README.md) — `docs/data/imagenet100_sit_internal_head_frequency/README.md`
- [ImageNet-100 SiT 对角矩残差 20K 终验](data/imagenet100_sit_moment_residual_pilot20k/README.md) — `docs/data/imagenet100_sit_moment_residual_pilot20k/README.md`
- [ImageNet-100 SiT multiscale guidance study](data/imagenet100_sit_multiscale_guidance_study/README.md) — `docs/data/imagenet100_sit_multiscale_guidance_study/README.md`
- [SiT 800K 终端分布控制审计便携数据](data/imagenet100_sit_terminal_distribution_audit_800k_v1/README.md) — `docs/data/imagenet100_sit_terminal_distribution_audit_800k_v1/README.md`
- [数据说明](data/imagenet100_sit_weak_difference_x800_readouts_v1/README.md) — `docs/data/imagenet100_sit_weak_difference_x800_readouts_v1/README.md`
- [SiT v800-v500 权重外推便携数据](data/imagenet100_sit_weight_extrapolation_v800_v500/README.md) — `docs/data/imagenet100_sit_weight_extrapolation_v800_v500/README.md`

### DiT bad/good、事件与统计审核

- [DiT bad/good 内部方法总账（2026-08-28）](DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md) — `docs/DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md`
- [DiT bad/good 研究暂停与归档快照（2026-08-28）](DIT_BAD_GOOD_PAUSE_ARCHIVE_2026-08-28_ZH.md) — `docs/DIT_BAD_GOOD_PAUSE_ARCHIVE_2026-08-28_ZH.md`
- [用 B 状态门约束跨尺度路径证据：严格推导与下一独立池协议](DIT_BLUR_FOCUSED_EPROCESS_THEORY_ZH.md) — `docs/DIT_BLUR_FOCUSED_EPROCESS_THEORY_ZH.md`
- [DiT 跨尺度路径证据 v2.2：B 触发锁存的固定信息方向鞅](DIT_BLUR_LATCHED_DIRECTIONAL_EPROCESS_V2_THEORY_ZH.md) — `docs/DIT_BLUR_LATCHED_DIRECTIONAL_EPROCESS_V2_THEORY_ZH.md`
- [DiT B/C 事件富集确认方案（标签可靠性修订 v3）](DIT_EVENT_RICH_CONFIRMATION_PLAN_ZH.md) — `docs/DIT_EVENT_RICH_CONFIRMATION_PLAN_ZH.md`
- [DiT 事件富集动态确认流水线（v3 B/C）](DIT_EVENT_RICH_DYNAMIC_CONFIRMATION_PIPELINE_ZH.md) — `docs/DIT_EVENT_RICH_DYNAMIC_CONFIRMATION_PIPELINE_ZH.md`
- [Event-rich 端点盲评管线（canonical v7）](DIT_EVENT_RICH_REVIEW_PIPELINE_ZH.md) — `docs/DIT_EVENT_RICH_REVIEW_PIPELINE_ZH.md`
- [DiT event-rich scientific v4.2.1：收紧 G 的解释边界](DIT_EVENT_RICH_SCIENTIFIC_V4_2_1_ZH.md) — `docs/DIT_EVENT_RICH_SCIENTIFIC_V4_2_1_ZH.md`
- [DiT event-rich scientific v4.2：固定方法 v2.2 后的确认协议](DIT_EVENT_RICH_SCIENTIFIC_V4_2_ZH.md) — `docs/DIT_EVENT_RICH_SCIENTIFIC_V4_2_ZH.md`
- [DiT event-rich scientific v4.1：同一模糊评估总体上的 B / E 检验](DIT_EVENT_RICH_SCIENTIFIC_V4_ZH.md) — `docs/DIT_EVENT_RICH_SCIENTIFIC_V4_ZH.md`
- [Scientific v4.2.1：端点采样与外部盲评锁](DIT_EVENT_RICH_V4_2_1_ENDPOINT_REVIEW_ZH.md) — `docs/DIT_EVENT_RICH_V4_2_1_ENDPOINT_REVIEW_ZH.md`
- [DiT 有限尺度后验循环单调违反（FPCV）](DIT_FINITE_POSTERIOR_CYCLIC_VIOLATION_ZH.md) — `docs/DIT_FINITE_POSTERIOR_CYCLIC_VIOLATION_ZH.md`
- [DiT 单路径新候选：Projected Tweedie-cone Violation](DIT_PROJECTED_TWEEDIE_CONE_VIOLATION_ZH.md) — `docs/DIT_PROJECTED_TWEEDIE_CONE_VIOLATION_ZH.md`
- [DiT PTCV 冻结发现实验结果（2026-08-28）](DIT_PTCV_DISCOVERY_RESULT_ZH.md) — `docs/DIT_PTCV_DISCOVERY_RESULT_ZH.md`
- [DiT v2.2 内部信号复核：E 退役，分支共识首版未通过](DIT_V22_INTERNAL_SIGNAL_REASSESSMENT_ZH.md) — `docs/DIT_V22_INTERNAL_SIGNAL_REASSESSMENT_ZH.md`
- [Third-pool endpoint label reliability audit](../experiments/audits/dit_bad_good_third_pool_label_reliability_v1/AUDIT_REPORT.md) — `experiments/audits/dit_bad_good_third_pool_label_reliability_v1/AUDIT_REPORT.md`
- [用 B 状态门约束跨尺度路径证据：严格推导与下一独立池协议](../experiments/locks/dit_blur_focused_eprocess_protocol_lock_v1/theory_zh.md) — `experiments/locks/dit_blur_focused_eprocess_protocol_lock_v1/theory_zh.md`
- [DiT 跨尺度路径证据 v2：B 触发锁存的固定信息方向鞅](../experiments/locks/dit_blur_focused_eprocess_protocol_lock_v2/theory_zh.md) — `experiments/locks/dit_blur_focused_eprocess_protocol_lock_v2/theory_zh.md`
- [DiT 跨尺度路径证据 v2.1：B 触发锁存的固定信息方向鞅](../experiments/locks/dit_blur_focused_eprocess_protocol_lock_v2_1/theory_zh.md) — `experiments/locks/dit_blur_focused_eprocess_protocol_lock_v2_1/theory_zh.md`
- [DiT 跨尺度路径证据 v2.2：B 触发锁存的固定信息方向鞅](../experiments/locks/dit_blur_focused_eprocess_protocol_lock_v2_2/theory_zh.md) — `experiments/locks/dit_blur_focused_eprocess_protocol_lock_v2_2/theory_zh.md`
- [Third-pool endpoint label reliability audit](../experiments/locks/dit_event_rich_confirmation_protocol_lock_v3/label_audit/AUDIT_REPORT.md) — `experiments/locks/dit_event_rich_confirmation_protocol_lock_v3/label_audit/AUDIT_REPORT.md`
- [DiT event-rich scientific v4.2：固定方法 v2.2 后的确认协议](../experiments/locks/dit_event_rich_confirmation_protocol_lock_v4_2/scientific_amendment_zh.md) — `experiments/locks/dit_event_rich_confirmation_protocol_lock_v4_2/scientific_amendment_zh.md`
- [DiT event-rich scientific v4.2.1：收紧 G 的解释边界](../experiments/locks/dit_event_rich_confirmation_protocol_lock_v4_2_1/scientific_amendment_zh.md) — `experiments/locks/dit_event_rich_confirmation_protocol_lock_v4_2_1/scientific_amendment_zh.md`

### AdvFD、Fréchet 与 score 后训练

- [AdvFD paper-only 首轮结果冻结记录](ADVFD_CLEANROOM_PREOFFICIAL_FREEZE_ZH.md) — `docs/ADVFD_CLEANROOM_PREOFFICIAL_FREEZE_ZH.md`
- [AdvFD Paper-Only Clean-Room：阶段 A 结果](ADVFD_CLEANROOM_STAGE_A_RESULTS_ZH.md) — `docs/ADVFD_CLEANROOM_STAGE_A_RESULTS_ZH.md`
- [AdvFD 文献、理论与论文优先复现协议](ADVFD_LITERATURE_THEORY_CLEANROOM_PLAN_ZH.md) — `docs/ADVFD_LITERATURE_THEORY_CLEANROOM_PLAN_ZH.md`
- [AdvFD 官方实现审计](ADVFD_OFFICIAL_IMPLEMENTATION_AUDIT_ZH.md) — `docs/ADVFD_OFFICIAL_IMPLEMENTATION_AUDIT_ZH.md`
- [AdvFD 官方 pMF-B 10K 中程结果](ADVFD_OFFICIAL_PMF_B_10K_RESULTS_ZH.md) — `docs/ADVFD_OFFICIAL_PMF_B_10K_RESULTS_ZH.md`
- [AdvFD 论文优先复现：歧义与预先决策](ADVFD_PAPER_ONLY_AMBIGUITIES_ZH.md) — `docs/ADVFD_PAPER_ONLY_AMBIGUITIES_ZH.md`
- [AdvFD pMF-B 纯速度续训对照](ADVFD_PMF_B_VELOCITY_CONTROL_10K_RESULTS_ZH.md) — `docs/ADVFD_PMF_B_VELOCITY_CONTROL_10K_RESULTS_ZH.md`
- [AdvFD clean-room：pMF-B Inception-2048 前缀实验](ADVFD_PMF_INCEPTION2048_PREFIX_RESULTS_ZH.md) — `docs/ADVFD_PMF_INCEPTION2048_PREFIX_RESULTS_ZH.md`
- [AdvFD pMF-B 64维投影 Pilot 结果](ADVFD_PMF_PROJECTED_PILOT_RESULTS_ZH.md) — `docs/ADVFD_PMF_PROJECTED_PILOT_RESULTS_ZH.md`
- [AdvFD witness 梯度与扩散 score 反例审计](ADVFD_SCORE_COUNTEREXAMPLE_AUDIT_ZH.md) — `docs/ADVFD_SCORE_COUNTEREXAMPLE_AUDIT_ZH.md`
- [AdvFD 的 selective amplification 候选反例](ADVFD_SELECTIVE_AMPLIFICATION_HYPOTHESIS_ZH.md) — `docs/ADVFD_SELECTIVE_AMPLIFICATION_HYPOTHESIS_ZH.md`
- [AdvFD 时序坐标一致性审计](ADVFD_TEMPORAL_GAUGE_AUDIT_RESULTS_ZH.md) — `docs/ADVFD_TEMPORAL_GAUGE_AUDIT_RESULTS_ZH.md`
- [残差分数估计与 pMF 后训练实验整理](RESIDUAL_SCORE_POSTTRAIN_PILOT_RESULTS_ZH.md) — `docs/RESIDUAL_SCORE_POSTTRAIN_PILOT_RESULTS_ZH.md`
- [AdvFD 研究阶段归档（2026-08-25）](archive/ADVFD_RESEARCH_CHECKPOINT_2026-08-25_ZH.md) — `docs/archive/ADVFD_RESEARCH_CHECKPOINT_2026-08-25_ZH.md`
- [AdvFD clean-room reproduction](../experiments/advfd_cleanroom/README.md) — `experiments/advfd_cleanroom/README.md`

### 预测目标、频谱与低维机制实验

- [双预测目标闭环动力学：连续螺旋 Toy 实验报告](DUAL_TARGET_CLOSED_LOOP_SPIRAL_TOY_ZH.md) — `docs/DUAL_TARGET_CLOSED_LOOP_SPIRAL_TOY_ZH.md`
- [双目标 Flow 的 Teacher 与闭环生成诊断](DUAL_TARGET_CLOSED_LOOP_TOY_ZH.md) — `docs/DUAL_TARGET_CLOSED_LOOP_TOY_ZH.md`
- [频率外推与预测目标外推：代码、实验和研究判断](FREQUENCY_PREDICTION_EXTRAPOLATION_AUDIT_ZH.md) — `docs/FREQUENCY_PREDICTION_EXTRAPOLATION_AUDIT_ZH.md`
- [AutoGuidance、Internal Guidance 与预测目标外推的二维机制实验](GUIDANCE_TOY_AG_IG_PTG_RESULTS_ZH.md) — `docs/GUIDANCE_TOY_AG_IG_PTG_RESULTS_ZH.md`
- [Prediction Target、Projector 与多头实验归档](PREDICTION_TARGET_PROJECTOR_EXPERIMENT_ARCHIVE_ZH.md) — `docs/PREDICTION_TARGET_PROJECTOR_EXPERIMENT_ARCHIVE_ZH.md`
- [Prediction Target Toy v4 实验归档](PREDICTION_TARGET_TOY_V4_ARCHIVE_ZH.md) — `docs/PREDICTION_TARGET_TOY_V4_ARCHIVE_ZH.md`
- [Spectral preconditioning toy data](data/prediction_target_spectral_preconditioning_toy_v1/README.md) — `docs/data/prediction_target_spectral_preconditioning_toy_v1/README.md`
- [连续螺旋双目标闭环实验包](../dual_target_closed_loop_spiral_toy_v1/README_ZH.md) — `dual_target_closed_loop_spiral_toy_v1/README_ZH.md`
- [Prediction target、输出秩与算子目标：受控 toy 实验](../prediction_target_rank_operator_toy_v1/README_ZH.md) — `prediction_target_rank_operator_toy_v1/README_ZH.md`
- [Prediction-target toy v10 完整机制诊断归档](../prediction_target_toy_v10_final_full_mechanism/README_ZH.md) — `prediction_target_toy_v10_final_full_mechanism/README_ZH.md`

### 生成训练、评估与其他数据集

- [Imagenette-64 Decoder 放大效应诊断：预注册](IMAGENETTE_DECODER_AMPLIFICATION_PREREG_ZH.md) — `docs/IMAGENETTE_DECODER_AMPLIFICATION_PREREG_ZH.md`
- [Imagenette-64 Prior-Decoder 断层：机制诊断结果](IMAGENETTE_DECODER_AMPLIFICATION_RESULTS_ZH.md) — `docs/IMAGENETTE_DECODER_AMPLIFICATION_RESULTS_ZH.md`
- [Imagenette Decoder-Aware Prior 单 Seed 门槛结果](IMAGENETTE_DECODER_AWARE_PRIOR_GATE_ZH.md) — `docs/IMAGENETTE_DECODER_AWARE_PRIOR_GATE_ZH.md`
- [Imagenette-64 Frozen Decoder Response Atlas 预注册](IMAGENETTE_DECODER_RESPONSE_ATLAS_PREREG_ZH.md) — `docs/IMAGENETTE_DECODER_RESPONSE_ATLAS_PREREG_ZH.md`
- [Imagenette-64 Frozen Decoder Response Atlas：正式负结果](IMAGENETTE_DECODER_RESPONSE_ATLAS_RESULTS_ZH.md) — `docs/IMAGENETTE_DECODER_RESPONSE_ATLAS_RESULTS_ZH.md`
- [Imagenette-64 Latent Prior 与 Decoder 收益权衡：预注册协议](IMAGENETTE_LATENT_PRIOR_TRADEOFF_PREREG_ZH.md) — `docs/IMAGENETTE_LATENT_PRIOR_TRADEOFF_PREREG_ZH.md`
- [Imagenette-64 Latent Prior 与 Decoder 收益权衡：正式结果](IMAGENETTE_LATENT_PRIOR_TRADEOFF_RESULTS_ZH.md) — `docs/IMAGENETTE_LATENT_PRIOR_TRADEOFF_RESULTS_ZH.md`
- [Imagenette-64 噪声阶段责任曲线：受控实验预注册](IMAGENETTE_NOISE_RESPONSIBILITY_PREREG_ZH.md) — `docs/IMAGENETTE_NOISE_RESPONSIBILITY_PREREG_ZH.md`
- [Imagenette-64 噪声阶段责任曲线：正式结果](IMAGENETTE_NOISE_RESPONSIBILITY_RESULTS_ZH.md) — `docs/IMAGENETTE_NOISE_RESPONSIBILITY_RESULTS_ZH.md`
- [README.md](../train_eqvae/README.md) — `train_eqvae/README.md`

### 通用理论、协议、环境与总入口

- [README.md](../README.md) — `README.md`
- [Architecture-aware Gauge Exploration](ARCHITECTURE_GAUGE_EXPLORATION.md) — `docs/ARCHITECTURE_GAUGE_EXPLORATION.md`
- [Bad/Good 轨迹指标筛选：2026-08-27 结果与下一步](BAD_GOOD_METRIC_SCREEN_RESULTS_2026-08-27_ZH.md) — `docs/BAD_GOOD_METRIC_SCREEN_RESULTS_2026-08-27_ZH.md`
- [Bad/Good 轨迹指标：当前假设、严格定义与验证协议](BAD_GOOD_TRAJECTORY_METRICS_ZH.md) — `docs/BAD_GOOD_TRAJECTORY_METRICS_ZH.md`
- [CFG-Rejection / EDM2 / ADM64 视觉 bad-case 审计规约](CFG_REJECTION_VISUAL_AUDIT_ZH.md) — `docs/CFG_REJECTION_VISUAL_AUDIT_ZH.md`
- [紧凑语义 Tokenizer 与生成鲁棒 Decoder 文献调研](COMPACT_SEMANTIC_TOKENIZER_LITERATURE_REPORT_ZH.md) — `docs/COMPACT_SEMANTIC_TOKENIZER_LITERATURE_REPORT_ZH.md`
- [跨尺度序贯路径证据：理论边界与复现实验协议](CROSS_SCALE_SEQUENTIAL_EVIDENCE_ZH.md) — `docs/CROSS_SCALE_SEQUENTIAL_EVIDENCE_ZH.md`
- [跨信息时间速度差：精确分解、反例与后验压力假设](CROSS_TIME_VELOCITY_GEOMETRY_ZH.md) — `docs/CROSS_TIME_VELOCITY_GEOMETRY_ZH.md`
- [Depth-Information Difference-in-Differences Internal Guidance](DEPTH_INFORMATION_DIFFERENCE_IN_DIFFERENCES_IG_ZH.md) — `docs/DEPTH_INFORMATION_DIFFERENCE_IN_DIFFERENCES_IG_ZH.md`
- [Dynamic Dual-Output Diffusion Models 文献审计、SiT 迁移与正式结果](DYNAMIC_DUAL_OUTPUT_DIFFUSION_LITERATURE_REPORT_ZH.md) — `docs/DYNAMIC_DUAL_OUTPUT_DIFFUSION_LITERATURE_REPORT_ZH.md`
- [实验归档索引](EXPERIMENT_ARCHIVE_INDEX_ZH.md) — `docs/EXPERIMENT_ARCHIVE_INDEX_ZH.md`
- [Foresight Fixed Point：CFG 复验与 AutoGuidance 迁移实验](FORESIGHT_FIXED_POINT_CFG_AG_STUDY_ZH.md) — `docs/FORESIGHT_FIXED_POINT_CFG_AG_STUDY_ZH.md`
- [生成时间瓶颈小型受控实验：预注册](GENERATION_TIME_BOTTLENECK_PREREG_ZH.md) — `docs/GENERATION_TIME_BOTTLENECK_PREREG_ZH.md`
- [生成时间瓶颈研究结果](GENERATION_TIME_BOTTLENECK_RESULTS_ZH.md) — `docs/GENERATION_TIME_BOTTLENECK_RESULTS_ZH.md`
- [生成时间瓶颈：Latent Prior 预注册](GENERATION_TIME_LATENT_PRIOR_PREREG_ZH.md) — `docs/GENERATION_TIME_LATENT_PRIOR_PREREG_ZH.md`
- [Information-Time Posterior Revision for Internal Guidance](INFORMATION_TIME_POSTERIOR_REVISION_IG_ZH.md) — `docs/INFORMATION_TIME_POSTERIOR_REVISION_IG_ZH.md`
- [Internal Guidance、层间动力学与自适应控制研究报告](INTERNAL_GUIDANCE_CONTROL_RESEARCH_REPORT_ZH.md) — `docs/INTERNAL_GUIDANCE_CONTROL_RESEARCH_REPORT_ZH.md`
- [Internal Guidance 文献、代码与研究路线审计](INTERNAL_GUIDANCE_LITERATURE_REPO_AUDIT_ZH.md) — `docs/INTERNAL_GUIDANCE_LITERATURE_REPO_AUDIT_ZH.md`
- [Internal Guidance 机制研究：第一阶段结果](INTERNAL_GUIDANCE_MECHANISM_PHASE1_RESULTS_ZH.md) — `docs/INTERNAL_GUIDANCE_MECHANISM_PHASE1_RESULTS_ZH.md`
- [Latent Transport 阶段 0 审计](LATENT_TRANSPORT_PHASE0_AUDIT_ZH.md) — `docs/LATENT_TRANSPORT_PHASE0_AUDIT_ZH.md`
- [Latent Transport 阶段 2：无训练路径审计](LATENT_TRANSPORT_PHASE2_RESULTS_ZH.md) — `docs/LATENT_TRANSPORT_PHASE2_RESULTS_ZH.md`
- [Latent Transport 阶段 3A：2D 四路径因果实验预注册](LATENT_TRANSPORT_PHASE3_PREREG_ZH.md) — `docs/LATENT_TRANSPORT_PHASE3_PREREG_ZH.md`
- [Latent Transport 阶段 3A：四路径 toy 结果与停止决定](LATENT_TRANSPORT_PHASE3_RESULTS_ZH.md) — `docs/LATENT_TRANSPORT_PHASE3_RESULTS_ZH.md`
- [Latent Transport Compatibility 研究协议](LATENT_TRANSPORT_RESEARCH_PROTOCOL_ZH.md) — `docs/LATENT_TRANSPORT_RESEARCH_PROTOCOL_ZH.md`
- [从失败机制到生成质量：P22-P24 结论](MECHANISM_TO_QUALITY_STUDY_ZH.md) — `docs/MECHANISM_TO_QUALITY_STUDY_ZH.md`
- [矩精确残差 Flow Matching：理论、实现与泄露审计](MOMENT_RESIDUAL_FLOW_THEORY_AND_LEAKAGE_AUDIT_ZH.md) — `docs/MOMENT_RESIDUAL_FLOW_THEORY_AND_LEAKAGE_AUDIT_ZH.md`
- [多阶段语义退火生成：文献边界、研究价值与低成本验证路线](MULTISTAGE_SEMANTIC_ANNEALING_LITERATURE_REPORT_ZH.md) — `docs/MULTISTAGE_SEMANTIC_ANNEALING_LITERATURE_REPORT_ZH.md`
- [噪声分辨的生成责任曲线：预注册协议](NOISE_RESPONSIBILITY_PROFILE_PREREG_ZH.md) — `docs/NOISE_RESPONSIBILITY_PROFILE_PREREG_ZH.md`
- [Research Status](RESEARCH_STATUS.md) — `docs/RESEARCH_STATUS.md`
- [研究工作区完整索引：理论、实验、代码与数据](RESEARCH_WORKSPACE_INVENTORY_20260907_ZH.md) — `docs/RESEARCH_WORKSPACE_INVENTORY_20260907_ZH.md`
- [归档引用复核与历史资产缺口](RESEARCH_WORKSPACE_REFERENCE_REVIEW_20260907_ZH.md) — `docs/RESEARCH_WORKSPACE_REFERENCE_REVIEW_20260907_ZH.md`
- [Train-only Rollout Checkpoint Selection 前瞻协议](ROLLOUT_CHECKPOINT_SELECTION_PREREG_ZH.md) — `docs/ROLLOUT_CHECKPOINT_SELECTION_PREREG_ZH.md`
- [Rollout Checkpoint Selection 前瞻结果：频谱方法线的最终否定](ROLLOUT_CHECKPOINT_SELECTION_RESULTS_ZH.md) — `docs/ROLLOUT_CHECKPOINT_SELECTION_RESULTS_ZH.md`
- [小型 Flow Transport Gap：机制预测预注册](SMALL_TRANSPORT_GAP_PREREG_ZH.md) — `docs/SMALL_TRANSPORT_GAP_PREREG_ZH.md`
- [频谱退化自引导：文献边界、严格形式化与研究计划](SPECTRAL_SELF_GUIDANCE_LITERATURE_THEORY_PLAN_ZH.md) — `docs/SPECTRAL_SELF_GUIDANCE_LITERATURE_THEORY_PLAN_ZH.md`
- [谱加权、随机输运与生成质量：理论解释、因果复现和剩余 Gap](SPECTRAL_THEORY_EXPLANATION_AND_GAPS_ZH.md) — `docs/SPECTRAL_THEORY_EXPLANATION_AND_GAPS_ZH.md`
- [Teacher MSE 改善但 rollout 变差：机制结论](TEACHER_ROLLOUT_MECHANISM_ZH.md) — `docs/TEACHER_ROLLOUT_MECHANISM_ZH.md`
- [Teacher 指标与自生成输运断层：Research Gap 与低成本验证路线](TEACHER_ROLLOUT_RESEARCH_GAP_ZH.md) — `docs/TEACHER_ROLLOUT_RESEARCH_GAP_ZH.md`
- [Telescoping Scale-Space Internal Guidance](TELESCOPING_SCALE_SPACE_INTERNAL_GUIDANCE_ZH.md) — `docs/TELESCOPING_SCALE_SPACE_INTERNAL_GUIDANCE_ZH.md`
- [方向加权为何改善 Teacher MSE 却损害生成：机制研究结论](TRANSPORT_REVERSAL_MECHANISM_STUDY_ZH.md) — `docs/TRANSPORT_REVERSAL_MECHANISM_STUDY_ZH.md`
- [Transport Risk Atlas：回顾性校准与前瞻验证协议](TRANSPORT_RISK_ATLAS_PROTOCOL_ZH.md) — `docs/TRANSPORT_RISK_ATLAS_PROTOCOL_ZH.md`
- [Transport Risk Atlas 与训练路径反转：结果报告](TRANSPORT_RISK_ATLAS_RESULTS_ZH.md) — `docs/TRANSPORT_RISK_ATLAS_RESULTS_ZH.md`
- [Information-time query semantics](data/information_time_posterior_revision/README.md) — `docs/data/information_time_posterior_revision/README.md`
- [Research experiments](../experiments/README.md) — `experiments/README.md`
