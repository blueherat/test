# RAEv2 guidance 研究档案导航

本索引响应 2026-09-06 用户要求：再做最多五轮有明确判别力的研究，当前 paired-bridge 配对 1K 计作第一轮；届时无论是否达标，都收束并保存理论、想法、实验与可校验数据。它是档案导航，不重写已冻结协议，不把局部代理改善认作 FID 收益。当前目标仍以 [目标文件](RAEV2_GUIDANCE_GOAL_20260906_ZH.md) 为准；最新执行状态见 [研究状态](RESEARCH_STATUS.md)。

**最终补记：四轮已结束，公平总成本≥5%目标未实现，第4轮后收束。** [结题记录](RAEV2_GUIDANCE_FINAL_CLOSEOUT_20260906_ZH.md)给出本次真实结果和归档状态。原导航盘点日期为2026-09-06；现已补齐最后四轮与实际便携数据包。 已阅读现行目标、状态、阅读总索引、前一轮研究归档及关键早期结题记录，完成下面的有界导航盘点；没有逐一重算全部历史实验或校验所有大文件。下面的“失败”仅指已实施的固定方法或已做的实验，不证明整个理论家族不可能有效。数值优先链接相应结果记录；不同模型、bank、样本数和评价器的绝对 FID 不横比。

## 读档顺序

1. [当前目标](RAEV2_GUIDANCE_GOAL_20260906_ZH.md) → [当前状态](RESEARCH_STATUS.md) → 本索引中的机制家族。
2. [2026-09-06 阅读综述](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md)：论文原文、假设、实现边界和设计判断。
3. [2026-09-05 guidance 档案](RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md)、[较早实验总索引](EXPERIMENT_ARCHIVE_INDEX_ZH.md)、[RAEv2 self-guidance 台账](RAEV2_SELF_GUIDANCE_RESEARCH_LEDGER_ZH.md)。这些保留历史目标与判断，不能覆盖最新用户约束。

## 机制家族

数据根目录简写如下。表中的相对目录均接在明确根目录后，不是新的数据副本。

- **R**：[2026-09-06 RAEv2 原始实验根目录](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/)。
- **E**：[较早外部实验根目录](/home/zhoushunyu/data/eqvae/experiments/)。
- **S**：[ImageNet-100 SiT 原始实验根目录](/home/zhoushunyu/data/eqvae/imagenet_sit_flow/)。
- **D**：[仓库内小型数据目录](data/)。它已有大量历史 CSV/JSON 与 manifest；当前新增结果并非全部复制到 D。

### 一、当前有限步运输候选及紧邻的已完成反例

| 家族 / 理论入口 | 实现或关键验证 | 已知结果与状态 | 可复查数据入口 |
|---|---|---|---|
| 配对桥有限运输：[有限 source 与条件 covariance 散度](RAEV2_ACTUAL_ROLLOUT_FINITE_TRANSPORT_20260906_ZH.md)，[Flow Matching 原文与本地扩展边界](RAEV2_GUIDANCE_READING_FLOW_MATCHING_20260906_ZH.md) | [辅助场](../experiments/raev2_paired_bridge.py)、[固定训练/验证 runner](../experiments/run_raev2_paired_bridge_pilot.py)、[采样器](../experiments/sample_raev2_paired_bridge.py)、[训练协议](RAEV2_PAIRED_BRIDGE_PILOT_PROTOCOL_20260906_ZH.md)、[1K 协议](RAEV2_PAIRED_BRIDGE_SCREEN_PROTOCOL_20260906_ZH.md) | **已结束，未达标**。固定2048更新后的配对1K candidate38.316312对成本匹配official107的38.372179仅改善0.145592%；对均值场优势0.068449%。714项独立复核通过。[筛查结果](RAEV2_PAIRED_BRIDGE_SCREEN_RESULTS_20260906_ZH.md)、[pilot](RAEV2_PAIRED_BRIDGE_PILOT_RESULTS_20260906_ZH.md)。不扩大当前实现。 | `R/paired_bridge_v1/`：`pilot/`、`train/`、`validate/`、`rollout/`、`analysis_v1/`、`screen_v1/`；`train/final.pt` 为固定 checkpoint，`analysis_v1/summary.json` 为独立 CPU 汇总。 |
| 最小能量 observable-error 势：[理论](RAEV2_OBSERVABLE_ERROR_GUIDANCE_20260906_ZH.md)、[有限求解器](RAEV2_OBSERVABLE_POTENTIAL_SOLVER_20260906_ZH.md) | [势函数模块](../experiments/raev2_observable_potential.py)、[训练](../experiments/train_raev2_observable_potential.py)、[采样](../experiments/sample_raev2_observable_potential.py)、[成本协议](RAEV2_OBSERVABLE_POTENTIAL_COST_PROTOCOL_20260906_ZH.md) | **固定方法已退役**。局部加权 coupling MSE 改善 1.0968%，但 5K official100 / candidate100 / official105 为 6.97489785 / 6.97464743 / 6.96041703；相对更贵对照反而差约 0.2044%。[1K](RAEV2_OBSERVABLE_POTENTIAL_SCREEN_20260906_ZH.md)、[5K](RAEV2_OBSERVABLE_POTENTIAL_SCALE_RESULTS_20260906_ZH.md)。 | `R/observable_potential_{train,validation,screen}_v1/`、`R/observable_potential_scale_audit_v1/`、`R/observable_potential_scale_audit_analysis_v1/`；D 中的 [solver 审计](data/raev2_guidance_restart_20260906/observable_potential_solver_audit.json)。 |
| 势空间遗漏与实际 q 反馈：[遗漏 witness](RAEV2_POTENTIAL_OMITTED_WITNESS_20260906_ZH.md)、[Sobolev 有限 Euler 审计](RAEV2_SOBOLEV_ACTUAL_Q_EULER_AUDIT_20260906_ZH.md) | [遗漏方向审计](../experiments/audit_raev2_potential_omitted_witness.py)、[两篇实际分布反馈阅读](RAEV2_GUIDANCE_READING_ACTUAL_DISTRIBUTION_FEEDBACK_20260906_ZH.md) | **诊断/理论，未形成新质量方法**。预设两个遗漏 witness 的同 bank 插值收益仅占旧代理收益约 0.4002%，不是总遗漏误差上界；实际 q 与 teacher-p 可以不同，但还需说明具体应纠正的可达缺口。 | `R/potential_omitted_witness_v1/`、`R/potential_omitted_witness_summary_v1/`、`R/reading_actual_distribution_feedback_v1/`；[小型审计](data/raev2_guidance_restart_20260906/potential_omitted_witness_audit.json)。 |
| affine support 反射平均：[协议与数学边界](RAEV2_AFFINE_REFLECTION_PROTOCOL_20260906_ZH.md)、[群平均阅读](RAEV2_GUIDANCE_READING_REFLECTION_SYMMETRIZATION_20260906_ZH.md) | [采样器](../experiments/sample_raev2_affine_reflection.py)、[独立审核](../experiments/review_raev2_affine_reflection_screen.py) | **固定方法已退役**。配对 1K official100 / reflection100 / official201 为 38.25159189 / 38.28864082 / 38.55812173；比便宜原基线差 0.096856%，不能凭较高步数基线变差宣称成功。[完整结果](RAEV2_AFFINE_REFLECTION_RESULTS_20260906_ZH.md)。 | `R/affine_reflection_v1/`：冻结请求、完整样本、成本选择、FID 与 463 项身份核验；先试 cost-only200 未覆盖成本的样本仍保留并计费。 |
| proximal / 总能量 / 两子空间能量：[原始逐坐标理论](RAEV2_PROXIMAL_ERROR_PROJECTION_20260906_ZH.md)、[共享结构](RAEV2_PROXIMAL_SHARED_CALIBRATION_20260906_ZH.md)、[全局单步 W₂](RAEV2_PROXIMAL_GLOBAL_CALIBRATION_20260906_ZH.md)、[energy ball](RAEV2_ENERGY_BALL_GUIDANCE_20260906_ZH.md) | [global 校准](../experiments/raev2_proximal_global_calibration.py)、[energy sampler](../experiments/sample_raev2_energy_ball_guidance.py)、[spatial sampler](../experiments/sample_raev2_spatial_energy_balls.py)、[空间能量证据](RAEV2_SPECTRAL_ENERGY_AUDIT_20260906_ZH.md) | **逐坐标 held-out 失败；channel-only 留理论；两个 global 固定方法及 spatial 固定方法 FID 阴性**。global/energy 的 1K 均约 38.408，劣于 official 38.3978；另一 bank spatial 38.31697 劣于 official 38.15988。[spatial 结果与停止决定](RAEV2_SPATIAL_ENERGY_BALLS_RESULTS_20260906_ZH.md)。 | `R/proximal_calibration_seed202609062/`、`R/proximal_global_c_seed202609065/`、`R/proximal_seed202609066/`、`R/energy_ball_seed202609066/`、`R/spatial_energy_balls_v1/`；[global FID](data/raev2_guidance_restart_20260906/global_fid_comparison.json)、[energy FID](data/raev2_guidance_restart_20260906/energy_ball_fid_comparison.json)。 |

### 二、当前模型的误差语义与机制诊断

| 家族 / 理论入口 | 实现或关键验证 | 已知结果与状态 | 可复查数据入口 |
|---|---|---|---|
| decoder 终点 adjoint 控制：[机制/协议](RAEV2_ENDPOINT_ADJOINT_RESPONSE_PROTOCOL_20260906_ZH.md)、[alignment 与质量缺口](RAEV2_ENDPOINT_ALIGNMENT_QUALITY_GAP_20260906_ZH.md)、[cross-prototype witness](RAEV2_ENDPOINT_CROSSPROTOTYPE_WITNESS_20260906_ZH.md) | [完整 suffix 响应审计](../experiments/audit_raev2_endpoint_adjoint_response.py)、[终点控制阅读](RAEV2_GUIDANCE_READING_ENDPOINT_CONTROL_20260906_ZH.md) | **固定实现停止**。8 图 FP32 有限响应约为线性预测 23.55%，native BF16/uint8 则反号至 −15.17%；3992 次主模型 forward 与 792 次输入 VJP 只是机制成本，未做 FID。[结果](RAEV2_ENDPOINT_ADJOINT_RESPONSE_RESULTS_20260906_ZH.md)。 | `R/endpoint_adjoint_response_v1/`、`R/endpoint_alignment_quality_gap_v1/`、`R/endpoint_crossprototype_witness_v1/`。 |
| decoder/query information deletion：[attention 几何](RAEV2_ATTENTION_CONTRAST_GEOMETRY_20260906_ZH.md)、[PAG/SEG 阅读](RAEV2_GUIDANCE_READING_ATTENTION_WEAK_20260906_ZH.md) | [query-mean 审计](../experiments/audit_raev2_decoder_query_mean.py)、[paired-error 审计](../experiments/audit_raev2_query_mean_error_compatibility.py) | **结构响应成立，质量候选未放行**。新响应 95.3291% 能量正交于旧 gap，但 80 条 teacher 记录的共同标量最佳风险收益仅 0.00204468%；没有部署 oracle 系数。[结构结果](RAEV2_DECODER_QUERY_MEAN_RESULTS_20260906_ZH.md)、[误差兼容性结果](RAEV2_QUERY_MEAN_ERROR_COMPATIBILITY_RESULTS_20260906_ZH.md)。 | `R/decoder_query_mean_v2/`、`R/decoder_query_mean_review_v1/`、`R/query_mean_error_compatibility_v1/`、`R/query_mean_error_compatibility_review_v1/`。 |
| Full/Base 深度与 readout；密度去混合：[crossed-head](RAEV2_DEPTH_READOUT_AUDIT_20260906_ZH.md)、[decontamination](RAEV2_DENSITY_DECONTAMINATION_AUDIT_20260906_ZH.md) | [深度审计](../experiments/audit_raev2_depth_readout.py)、[实际 density transport 审计](../experiments/audit_raev2_density_transport.py)、[去混合模块](../experiments/raev2_density_decontamination.py) | **前提不足/固定前提失败，无新 FID**。crossed 分解大项约为原 gap 13–31 倍且抵消；density 近似出现 82 个负 posterior covariance margin，不能区分全部误差来源，也不能靠温度/clip 救场。 | `R/depth_readout_audit_seed202609071/`、`R/density_transport_audit_seed202609074_v2/`；[深度小型审计](data/raev2_guidance_restart_20260906/depth_readout_audit.json)、[去混合小型审计](data/raev2_guidance_restart_20260906/density_decontamination_audit.json)。 |
| 法向噪声、曲率、support 与数值：[法向审计](RAEV2_NORMAL_CURVATURE_AUDIT_20260906_ZH.md)、[极值曲率](RAEV2_EXTREMAL_CURVATURE_AUDIT_20260906_ZH.md)、[support 半径](RAEV2_RAW_TOKEN_SUPPORT_BOUND_20260906_ZH.md)、[混合算术](RAEV2_GUIDANCE_MIX_NUMERICS_20260906_ZH.md)、[symmetry 前提](RAEV2_SYMMETRY_PREREQUISITE_20260906_ZH.md) | [normal-noise](../experiments/audit_raev2_guidance_normal_noise.py)、[Krylov](../experiments/raev2_symmetric_krylov.py)、[极值审计](../experiments/audit_raev2_extremal_curvature.py)、[noise endpoint witness](RAEV2_NOISE_ENDPOINT_ZERO_WITNESS_20260906_ZH.md) | **诊断，不能直接指定质量修正方向**。随机方向未发现正曲率而 Krylov 检出正向，并不矛盾；gap 法向能量约 8e−6；184320 个 raw token 预测无一超过结构上界 64；BF16 混合误差约为 gap 范数 1.06–1.07%，未支持“相减灾难”解释。 | `R/normal_noise_audit_seed202609071/`、`R/curvature_audit_seed202609072/`、`R/extremal_curvature_seed202609081/`、`R/raw_token_support_bound_v1/`、`R/numerical_guidance_mix_v1/`、`R/noise_endpoint_zero_witness_v1/`。 |
| 实际分布与 decoder 反转：[raw class moments](RAEV2_CLASS_MOMENT_AUDIT_20260906_ZH.md)、[decoded moments](RAEV2_DECODED_CLASS_MOMENT_AUDIT_20260906_ZH.md)、[covariance shape](RAEV2_DECODED_COVARIANCE_SHAPE_20260906_ZH.md)、[predicted-clean 交互](RAEV2_PREDICTED_CLEAN_INTERACTION_AUDIT_20260906_ZH.md)、[JVP 剩余](RAEV2_DECODER_LINEARIZATION_AUDIT_20260906_ZH.md) | [raw moments](../experiments/audit_raev2_raw_latent_class_moments.py)、[decoded moments](../experiments/audit_raev2_decoded_class_moments.py)、[decoder JVP](../experiments/audit_raev2_decoder_linearization.py)、[旧实际 q AUC](RAEV2_DISTRIBUTION_AUC_AUDIT_ZH.md) | **历史 paired-bank 诊断**。IG 的 raw 与 decoded 类内/类间 trace 变化方向相反；JVP 非线性剩余大，但 block-mean 投影不跨 seed 复现。旧 q AUC 结果否定“IG 全程把 q 拉近 p”的强解释；AUC 本身不是严格距离。 | `R/raw_latent_class_moments_v1/`、`R/decoded_class_moments_v1/`、`R/decoded_covariance_shape_v1/`、`R/decoder_fullbank_blockmeans_v1/`、`R/predicted_clean_interaction_n5000x2_v1/`；D 中同名 audit JSON。 |
| 未有充分机制的学习 gate；人工 semantic/routing；随机弱化：[gate](RAEV2_DISTRIBUTIONAL_GATE_GUIDANCE_20260906_ZH.md)、[重启记录](RAEV2_GUIDANCE_RESTART_20260906_ZH.md)、[随机 guidance 计划](RAEV2_STOCHASTIC_GUIDANCE_PLAN_20260906_ZH.md) | [distribution 模块](../experiments/raev2_distribution_guidance.py)、[semantic 模块](../experiments/raev2_semantic_quality_guidance.py)、[stochastic 模块](../experiments/raev2_stochastic_guidance.py) | **暂停/未放行**。可训练 gate + 终点损失不自动构成机制；人工调度的 semantic/routing screen 已停止且未评价。随机构造的 Gaussian 恒等式不填补 Full/Base 误差语义。保留代码不是授权重启。 | `R/semantic_seed202609061/`、`R/routing_seed202609061/`；[随机 Gaussian 小型审计](data/raev2_stochastic_guidance_20260906/gaussian_audit.json)。 |
| 生成后 posterior accept-D：[旧数学/代码记录](RAEV2_POSTERIOR_ACCEPTANCE_GUIDANCE_20260906_ZH.md) | [旧 sampler，仅档案](../experiments/sample_raev2_accept_d_guidance.py)、[probe](../experiments/raev2_posterior_probe.py) | **被用户明确排除，已终止**。832 个完整 proposal、321 个接受结果，没有完整 1K 或 FID；条件 KL 证书不能使完整图像拒绝采样成为本任务允许的 guidance。 | `R/posterior_acceptance_seed202609068/`、`R/posterior_cls_features_v1/`、`R/posterior_cls_probe_v1/`；结果不得并入方法性能总表。 |

### 三、PFR、fixed point、SiT 与 9 月 5 日 RAEv2 路线

| 家族 / 理论入口 | 实现或关键验证 | 已知结果与状态 | 可复查数据入口 |
|---|---|---|---|
| Foresight Fixed Point，CFG 到 AG 的语义迁移：[原论文复验与理论边界](FORESIGHT_FIXED_POINT_CFG_AG_STUDY_ZH.md)、[autoguidance 分布理论](AUTOGUIDANCE_FORESIGHT_DISTRIBUTION_THEORY_ZH.md) | [fixed-point flow](../experiments/foresight_fixed_point_flow.py)、[SiT study](../experiments/run_imagenet100_sit_foresight_fixed_point_study.py) | **CFG 内部配对 1K 正结果；AG 迁移阴性**。等调用数 CFG50 / FSG40 为 60.8032 / 57.8860；固定点收敛与 gap 缩小不提供 AG 质量语义。不是公开 50K SOTA。 | [小型 FID 表](data/foresight_fixed_point_cfg_ag_fid1k.csv)；原始 run 路径和配置见对应研究文档。 |
| PFR 反事实残差、指数重定时、OU 小波：[机制反例](PFR_MECHANISM_AUDIT_20260903_ZH.md)、[counterfactual](PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md)、[retiming](PFR_EXPONENTIAL_RETIMING_THEORY_ZH.md)、[OU](PFR_OU_PROBABILITY_WAVELET_THEORY_ZH.md)、[原始 PFR](PROJECTED_FUTURE_REFERENCE_IG_THEORY_ZH.md) | [反事实代数](../experiments/pfr_counterfactual_residual_theory.py)、[OU 谱](../experiments/pfr_ou_semigroup_spectrum.py)、[等计算 runner](../experiments/run_imagenet100_sit_pfr_equal_compute.py)、[终点审计](../experiments/analyze_pfr_terminal_distribution.py) | **SiT 有可靠内部正结果，当前不绑定此路线**。等计算 5K ordinary / PFR 为 40.912 / 37.530；OU strong direction + raw norm 在两个独立 5K bank 相对 ordinary 改善 11.54% / 9.82%。必须保留早期 nominal seed 高重叠勘误；within-bank 对比与真正独立复验分开。 | D：`pfr_counterfactual_residual_theory_20260903/`、`pfr_mechanism_audit_20260903/`、`pfr_ou_probability_wavelet_20260904/`、`projected_future_reference_ig/`；S：`pfr_counterfactual_residual_theory_v1/`、`pfr_stage_reuse_v1/` 等，完整路径见原记录。 |
| RAEv2 PFR/OU 跨表示迁移与 retiming 救援：[迁移结果](RAEV2_PFR_TRANSFER_RESULTS_ZH.md) | [RAEv2 代数](../experiments/raev2_pfr_retiming.py)、[采样](../experiments/sample_raev2_pfr_retiming.py)、[公式审核](../experiments/audit_raev2_pfr_retiming.py) | **正式 5K 阴性**。两个 1K bank 的正信号未维持：official IG 7.034546，raw PFR 7.224213；OU、angular、information-matched 等没有恢复正式收益。不是 SiT 正结果造假，而是不能直接迁移。 | [D 中迁移包](data/raev2_pfr_ou_transfer_20260904/README.md)、[正式 5K 表](data/raev2_pfr_ou_transfer_20260904/official_fid5k_raw_pfr.csv)；[9 月 5 日总档案](RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md)。 |
| 半群一致 value / Feynman–Kac：[半群理论](SEMIGROUP_CONSISTENT_GUIDANCE_THEORY_ZH.md)、[9 月 5 日第 9–10 节](RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md) | [value 模块](../experiments/raev2_semigroup_value.py)、[训练](../experiments/train_raev2_semigroup_value.py)、[采样](../experiments/sample_raev2_semigroup_value.py)、[解析 toy](../experiments/run_semigroup_consistent_guidance_toy.py) | **解析对象成立，固定真实方法阴性**。balanced value 1K 从 39.3702 恶化到 40.6671；最早 128 类 bank 无效已废弃。full-dimensional plug-in FKC 权重 ESS 接近 1，不支持此固定低成本路线。 | D：`raev2_guidance_exploration_20260905/raw/raev2_semigroup_value_beta1p78_v2_balanced/`、`raw/semigroup_consistent_guidance_toy_v1/`、`raw/semigroup_consistent_guidance_toy_v2/`；E 中相应完整目录。 |
| posterior 投影、consensus、innovation、characteristic query | [posterior sampler](../experiments/sample_raev2_posterior_iprojection.py)、[consensus runner](../experiments/run_raev2_bayes_consensus_screen.sh)、[innovation runner](../experiments/run_raev2_orthogonal_innovation_screen.sh)、[characteristic 模块](../experiments/raev2_characteristic_guidance.py)、[新版 characteristic 理论阅读](RAEV2_GUIDANCE_READING_CHARACTERISTIC_20260906_ZH.md) | **旧固定采样大多阴性或双 bank 符号反转；新阅读/解析检查未放行新 sampler**。I-projection 1K 45.7511 劣于 ordinary 39.3242；minimum-norm consensus 双 bank 更差；innovation 未建立正确符号的稳定性。 | [9 月 5 日总档案第 6–8 节](RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md)、D 中 `raev2_guidance_exploration_20260905/raw/`；新解析检查 `R/characteristic_gaussian_audit_v1/`。 |
| radial/tangential、未来 flow pullback、relative flow composition：[radius 计划](RAEV2_RADIUS_DIRECTION_PLAN_20260905_ZH.md)、[pullback](RAEV2_FLOW_PULLBACK_20260905_ZH.md)、[relative iteration](RAEV2_RELATIVE_TRANSPORT_ITERATION_20260905_ZH.md) | [radius 模块](../experiments/raev2_radius_guidance.py)、[pullback 模块](../experiments/raev2_flow_pullback.py)、[relative sampler](../experiments/sample_raev2_relative_transport.py) | **固定方法已阴性归档**。无增益 radial 双 bank 符号翻转；pullback 38.7351 → 39.2757；relative composition 38.8281 劣于 piecewise IG 38.1266。原历史手工 piecewise 窗口约 0.5% 的 1K 正信号保留为历史对照，不是当前允许的新方法。 | D：`raev2_guidance_exploration_20260905/raw/raev2_relative_transport_20260905/` 及该总包的逐 run metadata；[全包 manifest](data/raev2_guidance_exploration_20260905/archive_manifest.csv)。 |
| SiT prediction-target、depth difference、future/common/unique 与 moment residual | [SiT FID 总表](IMAGENET100_SIT_FID_INVENTORY_ZH.md)、[depth mechanism](IMAGENET100_SIT_DEPTH_DIFFERENCE_MECHANISM_RESULTS_ZH.md)、[finite guidance](IMAGENET100_SIT_400K_FINITE_GUIDANCE_DYNAMICS_ZH.md)、[future/common/unique](IMAGENET100_SIT_400K_FUTURE_COMMON_UNIQUE_RESULTS_ZH.md)、[moment 终验](MOMENT_RESIDUAL_FLOW_THEORY_AND_LEAKAGE_AUDIT_ZH.md)、[双目标归档](PREDICTION_TARGET_PROJECTOR_EXPERIMENT_ARCHIVE_ZH.md) | **有机制素材与模型内结果，逐项以总表为准**。diagonal moment residual 等预算 5K 155.6983 → 161.0571，已终止。预测目标外推不是普遍有用方向；800K/400K 与单 seed/多 seed 不能混用。 | D 中 `imagenet100_sit_*` 小型结果包；[早期实验总索引](EXPERIMENT_ARCHIVE_INDEX_ZH.md) 链接源码/测试；完整 checkpoint 和 tensor 留在 S。 |

### 四、较早背景研究：保留洞见，不冒充当前 RAEv2 guidance 成果

| 家族 | 权威文档与实现导航 | 已知状态与档案意义 |
|---|---|---|
| LPL / decoder-aware 后训练、contrast-preserving common adapter | [结题审计](LPL_LINE_CLOSURE_AND_SOLID_RESEARCH_AGENDA_ZH.md)、[RAE/RAEv2 综合](RAE_RAEV2_LPL_DEEP_SYNTHESIS_ZH.md)、[strict RAEv2](RAEV2_LPL_STRICT_CONTINUATION_ZH.md)、[self-guidance 台账](RAEV2_SELF_GUIDANCE_RESEARCH_LEDGER_ZH.md)、[旧 RAE 实验台账](RAE_LPL_EXPERIMENT_LEDGER_ZH.md) | **普通 LPL 的 RAEv2 方法线已停止；旧 RAE 正结果保留**。旧 DINOv2-B 50K FID 13.5043 → 11.1027；RAEv2 strict LPL 在 IG=1 也坏于 Flow，对照消除了“仅外推损伤”的单因解释。common adapter 三 seed × 5K 增量约 0.01，无实用收益。源码与真实 checkpoint 原址由这些既有总账登记。 |
| 表征等变性、latent trust、prior-decoder 接口、transport/路径顺序、容量与噪声责任 | [prior-decoder 总指南](PRIOR_DECODER_EXPERIMENTS_AND_LITERATURE_GUIDE_ZH.md)、[transport 风险 atlas](TRANSPORT_RISK_ATLAS_RESULTS_ZH.md)、[latent trust spectrum](RAE_LATENT_TRUST_SPECTRUM_RESULTS_ZH.md)、[path conditioning](RAE_PATH_CONDITIONING_RESULTS_ZH.md)、[谱理论边界](SPECTRAL_THEORY_EXPLANATION_AND_GAPS_ZH.md)、[生成声明矩阵](RAE_GENERATION_CLAIM_MATRIX_ZH.md) | **背景/受控系统研究，非当前采样方法**。弱几何响应不等于全局群表示；静态局部 proxy 无法稳定预测真实完整 rollout。Imagenette 从头训练的小两阶段系统、旧 RAE、RAEv2 必须分别登记，不能混用模型或评价数值。更早入口仍在 RESEARCH_STATUS 历史主体与各预注册/结果对文档。 |
| AdvFD / static FD / residual-score 后训练 | [官方实现审计](ADVFD_OFFICIAL_IMPLEMENTATION_AUDIT_ZH.md)、[10K 真实结果](ADVFD_OFFICIAL_PMF_B_10K_RESULTS_ZH.md)、[temporal gauge](ADVFD_TEMPORAL_GAUGE_AUDIT_RESULTS_ZH.md)、[score 反例](ADVFD_SCORE_COUNTEREXAMPLE_AUDIT_ZH.md)、[residual-score 结果](RESIDUAL_SCORE_POSTTRAIN_PILOT_RESULTS_ZH.md)、[实现 README](../experiments/advfd_cleanroom/README.md) | **短预算复现与机制研究已归档**。static 后训练有效；adaptive 增量依赖评价 reference，严格配对 held-out 三表示 FD 均变差。不能把共同 feature gauge 放大当作选择性 artifact 放大。D 有 `advfd_score_counterexample_v1/` 与 `residual_score_posttrain_pilot_v1/`；大资产原址见各结果记录。 |
| DiT bad/good、e-process、PTCV/FPCV、Doob 与 Fisher 路径 | [2026-08-28 暂停总档案](DIT_BAD_GOOD_PAUSE_ARCHIVE_2026-08-28_ZH.md)、[方法台账](DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md)、[PTCV 反证](DIT_PTCV_DISCOVERY_RESULT_ZH.md)、[FPCV 未决形式化](DIT_FINITE_POSTERIOR_CYCLIC_VIOLATION_ZH.md)、[跨尺度顺序证据](CROSS_SCALE_SEQUENTIAL_EVIDENCE_ZH.md) | **暂停，部分阴性、部分未完成**。Fisher 的 1K 乐观信号在正式 5K 反转（7.4938 → 7.9868）；PTCV 冻结 expansion AUC 0.4645。FPCV 只有数值非空洞，不能补写质量结论；曾设想的 one-shot restart/筛选不进入当前方法空间。旧锁、sealed labels、manifest 不删除；以暂停总档案纠正历史台账过时文字。 |

## 最后新增的三个机制裁决

| 机制 | 保留理论与缺口 | 结果和数据 |
|---|---|---|
| 二阶压力/对称innovation | Stein压力存在有条件的标签方差动机；有限IG漏掉均值项，posterior uncertainty不等于IG仍缺的covariance | [结果](RAEV2_PRESSURE_INNOVATION_RESULTS_20260906_ZH.md)：全100时刻缓存及严格Gaussian重复补偿反例；[数据](data/raev2_pressure_innovation_20260906/README.md)，不训练 |
| 时间score PDE自洽性 | 时间方程有结构约束，但奇异完全噪声边界不能独自识别数据目标；有真实边界的原FP-Diffusion仍有价值 | [结果](RAEV2_FINAL_CANDIDATE_REVIEW_20260906_ZH.md)：一般Gaussian符号恒等式和独立完整ODE复核；[数据](data/raev2_temporal_score_pde_20260906/README.md)，无新增FID |
| Moser有限密度source | 精确正密度流运输正确；有限Galerkin与任意经验Gram不能自动继承全路径保证，与旧Sobolev估计存在重叠 | [结果](RAEV2_GUIDANCE_READING_MOSER_20260906_ZH.md)：正密度圆例的准确解和有限投影、经验不可识别性；[数据](data/raev2_moser_finite_source_20260906/README.md)，不扩基调参 |

## 文献导航与跨家族证据

[阅读总索引](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md) 是当前 52 篇原始论文的主入口，已经逐篇区分定理、假设、官方代码、质量证据与成本缺口。具体专题包括 strong/weak error compatibility、实际 guided path、CFG++/APG/CFG-Zero*、attention/PAG/SEG/ERG、Characteristic/linear CPC、SWG、独立 condition、learned consistency、Sobolev/Discriminator Flow、群对称、量化、manifold/DAE、endpoint control、particles/Doob、Flow Matching、Stochastic Interpolants 等；不要把“论文读过”当作“该方法在此模型上已验证”。原始 PDF/HTML/官方代码快照及 SHA manifest 留在 `R/reading_*_v1/`。新增论文应进入总索引及来源 manifest，不在此另建互相冲突的论文计数。

这些档案共同留下了五条可直接影响后续判断的证据：

1. **配对 teacher 风险与实际生成质量不能互相替代。** PFR、LPL、observable potential、proximal 分别从不同侧面说明保证对象必须写清楚；“更接近真实 bridge”也可能消除已经有用的质量倾斜。当前paired bridge的速度风险为负收益，固定矩差改善而1K FID仅微小改善；不把它写成“teacher loss改善但质量恶化”的证据。
2. **Raw latent 与 decoded 分布方向可以相反。** class-moment/JVP/LPL 记录要求把 decoder 和有限位移放进解释，而不是把单一 latent trace 当作质量。
3. **1K 的排序会反转，但不能因此随意追种子。** RAEv2 PFR、Fisher 及旧 1K 门槛修正应一起阅读；固定方法、配对输入、独立确认与相应样本规模必须共同保留。
4. **相同形式不等于相同误差语义。** CFG 的 conditional/unconditional、SiT 后训练 weak head、RAEv2 联训 Full/Base，不能靠共用 fixed point、幂密度或 posterior 记号就相互替代。
5. **失败后不靠 gain、窗口、采样器精度或表示宽度扫描保留旧故事。** 除非出现新的可证伪机制，新解释只归档，不自动触发下一轮 GPU 实验。

## 数据、复现与 Git 边界

- 小型理论文档、源码、测试、配置、汇总 CSV/JSON 与可复查 manifest 适合纳入 Git。
- 模型权重、完整样本、latent/feature bank、论文 PDF 和大日志保留原址，用相对/绝对位置、文件大小和 SHA-256 连接，不把路径存在等同于内容已逐一验证。
- 运行中的结果以执行记录和真实进程为准；文件存在本身不能证明作业完成。

### 建议纳入 Git 的路径范围

本次盘点的 HEAD 为 `2531f17d2734015d0368539ca9e53c2cdea92e59`（“归档RAEv2引导理论探索与负结果”）。9 月 5 日及更早材料多数已经在 Git；不需要搬迁或重复导入。快照时约 210 个改动/未跟踪文件，docs 约 3.35 MB、experiments 约 1.06 MB、tests 约 0.18 MB；最大约 437 KB，**这些是当时工作树元数据，不是最终提交清单**，后续运行会增加文件。

| 范围 | 纳入内容 | 提交前需做的具体收尾 |
|---|---|---|
| `docs/RESEARCH_STATUS.md`、当前 GOAL、reading synthesis、本索引、各 `docs/RAEV2_*_20260906_ZH.md` | 目标、机制、协议、阅读、结果、负结果和成本限制 | 保留冻结协议；用新的结果文档和清晰日期覆盖历史状态，不改旧实验含义。最终轮数/停止原因需由主研究记录补齐。 |
| `experiments/` 中本次相关新增分析/校准/训练/采样/汇总脚本；`tests/test_*raev2*` | 能复查公式、数据流、配对与实际成本的源码和必要测试 | 按机制分组暂存，检查导入所依赖的配置/辅助脚本已跟踪；未执行原型明确标档案，不写成成功复现。避免直接 `git add .` 混入后续新资产。 |
| 已存在的 `docs/data/raev2_guidance_restart_20260906/`、`docs/data/raev2_proximal_error_projection_20260906/`、`docs/data/raev2_stochastic_guidance_20260906/`、`docs/data/raev2_fid_finite_sample_20260906/` | 小型 CSV/JSON 与 toy 结果 | 把每个表关联至结果文档；保留原始 precision、评价 reference 和样本数，不抹去失败分支。 |
| [最终轻量结果包](data/raev2_guidance_final_20260906/README.md)、[来源补充包](data/raev2_guidance_final_sources_20260906/README.md) | 前者已复制188份终验/机制小文件，3,366,966字节；后者保存论文获取manifest及仓库外独有分析/恢复脚本 | 已逐份核对复制SHA；大资产原件未复制进Git。另有最后第2/3/4轮各自数据包。 |
| 模型、样本、latent/feature bank、PDF、完整日志与缓存 | **不加入 Git** | 留在 R/E/S 或现有 source cache；manifest 记录逻辑路径与解析后真实路径、大小和内容 SHA。完整大目录本次未重新 hash，不能宣称已备份或全量审核。 |

建议最终提交至少按“理论/文献与导航”“实验实现与验证”“轻量结果/身份与成本表”三个可审查块准备；是否分 commit 由主线程实际收束决定。本盘点未执行 `git add`、commit 或 push，也没有启动、停止或重启任何实验。

## 尚未确认的档案项

1. **最后四轮均已完成。** [轮次台账](RAEV2_FINAL_FIVE_ROUNDS_20260906_ZH.md)记录固定假设、设计、证据与停止决定；≥5%目标未达成。第5轮未使用，不恢复旧候选。
2. **完整终验的小型输出已复制并核验。** [主数据包](data/raev2_guidance_final_20260906/README.md)保留188份原件，[压力](data/raev2_pressure_innovation_20260906/README.md)、[时间PDE](data/raev2_temporal_score_pde_20260906/README.md)、[Moser](data/raev2_moser_finite_source_20260906/README.md)分别保存最后三轮；论文和大模型仍由来源manifest连接原址。
3. **总成本仍有明确缺口。** paired-bridge 原训练父进程消失造成外层精确耗时未捕获；后续验证/rollout 没有重训，记录了恢复链。旧 real-bank 选取/编码及历史准备成本边界也不都闭合。已有耗时下界和区间必须原样保存，不能把推理匹配筛查改称总成本达标。
4. **旧 bank 并非都独立。** PFR nominal seed 重叠勘误、9 月 5 日 invalid 128 类 bank、历史已看过的 1K/5K 子集、当前 train-heldout 与整个研究历史之间的区别，应随对应表一起入档。
5. **较早 RAE/Imagenette/AdvFD/DiT 全量大资产未在本次重新归属或重算。** 本索引用既有结题总账连接它们，未证明每个 checkpoint、annotation、锁和巨大原始目录都仍完整可读；恢复具体家族前先依其 manifest 核验。FPCV、Self-Guidance 权重下载、历史 one-shot restart 草图等未完项不能当作结果。
6. **历史状态文字有时间层次。** 旧 EXPERIMENT_ARCHIVE_INDEX 的“当前 PFR 主线”、旧 RESEARCH_STATUS 主体的“active”与老实验计划只陈述当时状态。当前 GOAL、最新结果和本索引的显式裁决优先；不要据旧默认值重启已排除路线。
7. **环境指引与非研究文件边界。** 本次在 `/`、`/home`、用户目录和仓库祖先链未发现 `AGENTS.md`，仓库/用户目录的限定文件搜索也未找到；未据此声称机器所有目录不存在指引。未改动已有训练资产、外部仓库或非本任务文件。
