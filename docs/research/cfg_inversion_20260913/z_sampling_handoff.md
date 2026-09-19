# Z-Sampling 后续准备：已做内容、时间约定与两个有限候选

2026-09-13。仅查阅原文、作者源码与仓库记录；没有启动 GPU、Z 采样或修改公共代码。按用户安排，须先完成当前 inverse-prior 实验；本文件不表示已批准跳过其结果。

**当前最有价值的下一步是补齐当前强基线上的 Z 复测，再检验一个利用缓存、同成本的离散逆改动。** 第二个候选是较贵的终点同场偏差中心化。两者改善的是 Z 的 no-op 结构契约，均没有证明“真实好图一定是其固定点”，也不把已有数值思想称为新方法。

## 1. 重新核验的原始来源

[Zigzag Diffusion Sampling: Diffusion Models Can Self-Improve via Self-Reflection](https://arxiv.org/abs/2412.10891) 初版 2024-12-14，当前 arXiv 仍为 2024-12-17 的 v2。本文重新读取 [Algorithm 1、Eq. (2–5) 与 §3](https://arxiv.org/html/2412.10891v2)：每个被选中的步执行强前向、弱反演、强前向。其累计 latent 位移解释不等于 FID 改善定理。

2026-09-13 通过 GitHub API 重新取得作者仓库 HEAD：`eef8bb265deb8f33efd47c53e6ee5506606de360`，提交日期 `2026-05-20T12:13:41Z`。本地阅读副本在 `/tmp/cfg_inversion_prior_20260913/Zigzag-Diffusion-Sampling/`；它不是当前项目的生成实现。

[作者参数入口](https://github.com/xie-lab-ml/Zigzag-Diffusion-Sampling/blob/eef8bb265deb8f33efd47c53e6ee5506606de360/infer.py#L10) 默认 SDXL base、DDIM50、`lambda_step=49`、`T_max=1`、前向 guidance 5.5、逆向 0。[实际循环](https://github.com/xie-lab-ml/Zigzag-Diffusion-Sampling/blob/eef8bb265deb8f33efd47c53e6ee5506606de360/utils/pipeline_stable_diffusion_xl.py#L1421) 的逆网络输入为已推进的 `next_latents`，查询时间仍为旧 `t`；inverse scheduler 接收 `timesteps[i+1]`。逆腿虽然权重为 0，源码仍预测 conditional/null 两个分支。

统一采用 `v_w=v_u+w(v_c-v_u)`：代码逆权重 0 是 null，1 是 conditional。论文 Eq. (5) 使用附加系数 `(1+gamma)v_c-gamma*v_u`，与正文把 `gamma=0` 称为 unconditional 存在表述冲突；实现时必须报告实际场。增加 `T_max`、换 conditional 逆腿均已在作者参数空间内。[W2SD/FSG 的相关先例和固定提交](fsg_w2sd_prior_art.md)不再重复综述。

## 2. 仓库哪些已经做过

|记录|实际操作与结果|不能据此声称什么|
|---|---|---|
|[旧 SiT FSG/AG](../../FORESIGHT_FIXED_POINT_CFG_AG_STUDY_ZH.md)|Euler40/50；FSG 事件 `0:5:2,5:5:2,15:5:1`；CFG 1K FID 60.8032→57.8860，调用配平；strong800K/weak500K 的 AG 往返显著变差|不是 Z 默认逐步 reflection，也不是当前 Heun64 调优 CFG 基线|
|[通用实现](../../../experiments/foresight_fixed_point_flow.py:191)与[主采样入口](../../../experiments/sample_imagenet100_sit_foresight_fixed_point.py:1183)|`foresight_round_trip` 在未来真实时间查询 inverse；可表示逐步强前弱逆再强前，但旧实际配置使用稀疏较长事件|存在可表达的代码不等于某个 Z 配置已有真实 FID|
|[今日算子探针](../../../experiments/cfg_transport_search_20260913/operator_probe.py)及[结果](../../CFG_TRANSPORT_SEARCH_RESULTS_20260913_ZH.md)|8 状态，t=.25/.5/.75，h=1/32、1/16、1/8，Heun8/16，另做16/32；同场误差小、末腿抵消成立、匹配有效 w 后尚有小的有限步差异|只测算子距离，没有 Z 质量结果；当前 gap 正交比例不是质量证据|
|[transport 监督筛选](../../research/cfg_transport_search_20260913/mechanism_debate.md)|扣除同场往返并去除已有方向后，独立留出 FM 风险改变量区间跨零；未进入质量确认|不能把几何非零当有用监督方向，也不能外推为全部 Z 失败|
|[旧离散逆审计](../../FSG_ANCHORED_INVERSE_SOLVER_20260908_ZH.md)、[多步审计](../../FSG_ANCHORED_INVERSE_MULTISTEP_20260908_ZH.md)|RAE 上 Picard/Anderson 能改变求根精度，但离散化和初期不稳定明显；没有生成质量突破|求根、加速、冻结目标本身不能重新命名为创新|
|[FSG follow-up 第6项](../../FSG_FOLLOWUP_RESEARCH_20_IDEAS_20260910_ZH.md)|已提出并实现 `L_CU-L_UU` 的输入端同场缺陷扣除，另有真实 residual acceptance；[源码](../../../experiments/sit_fsg_followup_20260910/core.py:202)|下述第二项复用该思想，不能称首次提出 bias subtraction|

本次对 docs/experiments 的命名与算子检索，**尚未定位到当前 tuned SiT-S CFG64 下独立标记、按作者旧时间逆腿约定运行的 Z FID 结果**。这是有界检索结论，不是声称历史上不存在任何等价参数组合。

当前直接对照为 SiT-S/2 EMA800K、同 SD-VAE、ImageNet100 balanced1K，Heun64，附加 alpha1.25（总 w2.25），accepted-left t≥.75 关闭额外 guidance：FID44.910314，224 单分支调用/图。已有 APG 同预算43.464164可作强参考，但不再拆解或混合其模块。输入/结果定位：[冻结 bank](../../../../data/eqvae/experiments/cfg_transport_search_20260913/baseline_1k/request.json)、[汇总](../../CFG_TRANSPORT_SEARCH_RESULTS_20260913_ZH.md)。不同 bank 的绝对 FID 不能直接作配对差值。

## 3. 三种“逆”不能混成一个

以下均在 physical FM 时间 t→t+h，令 A=v_s，B=v_b，`E_A(x)=x+h A(x,t)`，y=E_A(x)。

|定义|一次逆操作|实际意义|
|---|---|---|
|作者旧时间操作的 FM Euler analogue|`q=y-h B(y,t)`|保留其“advanced state + old model time”约定；仍不是 SDXL DDIM 逐数值复现|
|物理时间对齐的反向 Euler|`q=y-h B(y,t+h)`|对逆 ODE 做显式一步；时间正确不等于精确逆前向 Euler|
|前向 Euler 的真正离散逆|求 `q+h B(q,t)=y`|未知 q 的隐式方程；这里旧 t 正是被逆映射的时间，不能误改 t+h|

三者最后都必须再做强前 `E_A(q)` 才是完整 Z。若最后改为精确弱前 `E_B(E_B^{-1}(E_A(x)))`，结果只是原来的强前，不能把抵消后的端点改善归因于反演。

同场 A=B 的理想契约是完整 Z 等于 E_A，而非净推进为零。对于不同场，一阶有效权重为 `W=2w_s-w_b`；K 轮为 `W=w_s+K(w_s-w_b)`。减少 same-field defect 不等于保持异场图像，更不意味着朝真实分布移动。

## 4. 候选一：从已知原状态启动、复用缓存的离散逆

**优先级最高：与优化过分支调用的 Euler Z 同成本。** 第一次 strong CFG 已获得 x 处的 conditional/null 预测，缓存 A(x,t) 与 B(x,t)。求解上述离散逆时从原状态 x 启动：

\[
q_1=x+h[A(x,t)-B(x,t)],\qquad
q_2=x+h[A(x,t)-B(q_1,t)],\qquad
Z_{\rm cached}(x)=q_2+hA(q_2,t).
\]

这里 q1 不增加模型查询；q2 只新增一次 weak 分支，最后一次 strong pair。因此 strong 非端点、weak 为0或1时，每个事件仍为 `2+1+2=5` 单分支调用，与作者式 Euler 三腿优化版完全相同。它是两次 Picard 更新，但第一次的 weak 求值由原始 CFG 缓存提供。

**不变性来自具体构造：** A=B 时，q1=x、q2=x，每个有限迭代即满足同场 no-op，最终等于原 Euler 强前。用 `x+h*(A_cached-B_query)` 的差分表达避免先形成 y 再减去大项的额外舍入；实现仍须测逐样本端点。A≠B 时不会退化为 `G G^{-1}=I`，末腿保留 strong，所以仍是真正的 Z 改动。

对于光滑 B，真离散逆为 `q*=x+h(A-B)-h² J_B(A-B)+O(h³)`，q2 已匹配该展开；原旧时间显式逆为 `x+h(A-B)-h² J_B A+O(h³)`。这不是 Richardson，也不是只改 CFG 标量，但主要消掉的是数值自循环项。Picard 不保证大步收敛；q2 不宣称已经精确求根。

**已有工作边界：** [今日离散逆推导](operator_debate.md:218)及旧 RAE anchored inverse 已包含相应方程和 Picard 思路。本次可验证的新问题仅是“用已付费 CFG 预测初始化，能否在 Z 同成本下改善 FID”，不是新逆理论。它也可能去掉原 Z 偶然有利的离散扰动，因此必须同时保留原 Z，不先按 residual 挑胜者。

## 5. 候选二：在 Z 最终读出上扣除同场自循环

**次优先，成本较高。** 令 S 是固定的一步 strong 前向、B_b 是某种固定 inverse 近似；两个逆分支均从相同 y=S(x) 出发，并保持同一时间约定：

\[
Z_{\rm centered}(x)=y+
\big[S(B_b(y))-S(B_s(y))\big].
\]

当 b=s，两项在相同输入下相同，端点严格为 y，即使使用粗 inverse；若 B_s 是所选 S 的精确离散逆，则 `S(B_s(y))=y`，恢复原 Z。与在输入端减去缺陷后再非线性 S 推进不同，此处抵消的是最终读出上的同场漂移；不要求错误先线性相加再输运。它保留由弱参考切换引起的有限响应，没有投影回当前 gap。

Euler 下共享 x 的初次 pair、y 的逆 pair，以及两个不同逆状态的 strong pair，总计8单分支调用/事件；原来为5。不能称“免费”。这仍是已有 defect-control 思想的终点变体；[FSG follow-up #6](../../FSG_FOLLOWUP_RESEARCH_20_IDEAS_20260910_ZH.md)和今日 transport q 已给出相邻方案，不能重复讲成全新方向。未定位到本式在当前 Z 完整采样中的质量记录；其优先级低于候选一，不能仅因同场残差为零就投入大扫参。

## 6. 如当前工作阴性后的最小交接

先固定输入 bank、FP32/TF32约定、SD-VAE、cutoff和实际单分支计数，保存普通 CFG、作者旧时间 Euler analogue、物理时间反向 Euler三者。强系数若沿用 w_s2.25、b0，reflection 区间首阶 W4.5；必须另有相应有效强度控制。若目标是匹配当前 W2.25，则 reflection 内 `w_s=(2.25+w_b)/2`：b0用1.125、b1用1.625；非 reflection 步仍用2.25。有限区间不保证完全匹配，另报实际位移投影与终点统计。

一个有用的预算例子是 Euler64、cutoff.75、最前37步各做一次 reflection：基础112单分支调用，37×3额外调用，总223，与当前 Heun64的224接近。这是预算构造，不是已验证的最佳时间窗口。原作者代码在逆端点仍算两分支，因此未优化时是260，不能把它与223混报。候选一仍223；候选二同37事件变334，可与336调用对照，但首轮不应同时扩大到该成本。

仅保留 K=1；候选一与原 Z 共用 w_s、w_b、事件、noise/labels及预算，之后才谈质量。若它只减小 cycle error却没有独立 FID/图像改善，应记录为数值验证；不要重新打开 Anderson、APG/CTRL混合、全新schedule或learned终点refiner。五轮保存如果用于诊断，应明确是固定时间层的 Z 重复操作，每轮实际执行和保存，不能把五个采样时间点写成五轮原图回灌。

最重要的范围界线是：这里的合法不变性是“没有改变动力学时，往返不能额外改变端点”。它尚未提供原始用户观察中的真实图像质量固定点。两个候选只有质量实验能决定去留；数值恒等本身不补足外部数据目标。
