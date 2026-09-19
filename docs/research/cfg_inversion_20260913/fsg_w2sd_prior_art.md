# 强 CFG 前进与普通条件反演：已有方法、算子区别与研究余地

**“高强度 CFG 前进一步，再用普通 conditional 模型反演”有直接前例。** 它属于 Z-Sampling / W2SD 的强前向、弱反演构造；FSG 也采用这一类操作，但默认反演分支、前瞻区间与普通推进不同。仅把逆腿从 null 换为 conditional、增加反复次数，或将往返解释为“信息写入”，不足以成为新方法。更值得研究的问题是：扣除等效 CFG 强度、求解器误差和精确抵消后，有限区间往返剩下什么可测作用。

## 1. 先固定分支、时间和最终读出

以下统一用噪声→数据的时间方向，定义

\[
v_w=v_u+w(v_c-v_u),\qquad w=0\text{ 为 null},\quad w=1\text{ 为普通 conditional}.
\]

在同一个区间 \([t,t+h]\) 上，记 \(C=\Phi_c^{t+h\leftarrow t}\)、\(S=\Phi_w^{t+h\leftarrow t}\)。若反演使用相同 conditional 场的准确逆流，则所说的回到原时间操作是

\[
L=C^{-1}\circ S,\qquad x_t\longmapsto x'_{t}=L(x_t).
\]

这里没有重新加入独立高斯噪声；返回原来的**时间层**，不要求返回原来的**状态**。两腿不同，故它是相对流映射，不是同一模型的 identity 测试。\(w=1\) 时两腿相同，才有理想 \(L=I\)；数值采样器只能近似满足此关系。

最终从哪里继续生成会改变算法，不能省略：

|完整操作|精确流恒等式|含义|
|---|---|---|
|只做强前→条件逆|\(L=C^{-1}S\)|状态校正，还没有净前进|
|一次校正后，沿相同区间 conditional 前进|\(CL=S\)|准确反演的作用被随后 conditional 流准确抵消；等于原来的一次强 CFG 前进|
|两次校正后 conditional 前进|\(CL^2=SC^{-1}S\)|等于一次强前→条件逆→强前的 reflection|
|做 K 次校正后强 CFG 前进|\(SL^K\)|Z-Sampling/W2SD 式净推进，通常不等于一段原强流|

这些是已明确定义的流映射的代数关系，不依赖模型质量。若后续 conditional 继续到终点，且起点仍为 t，同样有 \(\Phi_c^{1\leftarrow t}C^{-1}S=\Phi_c^{1\leftarrow t+h}S\)：只是一段强 CFG 后接 conditional 后缀。部分写入、不同区间、不同逆场、缓存或离散误差会打破抵消，必须逐项记录。

## 2. 最接近的三篇前例

### Z-Sampling：直接前例，注意论文与代码系数不一致

Z-Sampling 的 Algorithm 1 明确先强 guidance 去噪，再弱 guidance 反演，最后强 guidance 重新去噪；在选定的早期时间步重复。它也讨论完整生成后反演与逐步 reflection 的区别。[^zpaper]

其 Eq.(5) 写成 \(\epsilon=(1+\gamma)\epsilon_c-\gamma\epsilon_u\)，所以按这条公式，\(\gamma_2=0\) 对应普通 conditional。但正文又把 \(\gamma_2=0\) 称为 unconditional；**不能仅按论文数字判断实际分支。** 原作者公开实现用 \(\epsilon_u+w(\epsilon_c-\epsilon_u)\)：默认逆腿参数 0 实际为 null，将 `inv_guidance_scale=1` 就是普通 conditional。[^zpaper][^zcode]

因此，本构造既已包含在论文的一般强弱 guidance 参数化中，也可由公开代码直接配置。准确表述是“研究 conditional 作为 reflection 的逆流基准”，而非首次提出强前弱后。

### W2SD：将该操作推广到模型、条件和 pipeline 差异

W2SD Algorithm 1 用 \(\mathcal M_{\rm inv}^{w}\mathcal M^s\) 校正后再用 strong 推进；§4.2 明确将 guidance 高低作为强弱配对。其他配对包括模型权重、LoRA、MoE 路由、提示词和 ControlNet。[^wpaper]

作者当前 LoRA 示例默认 strong/weak guidance 为 5.5/1.0，但同时切换 strong/weak LoRA 权重，所以不是“同一权重仅改 CFG”的干净验证。论文附录对纯权重差异采用两腿同 guidance 的控制，也对纯 guidance 差异单独研究；这些设置不能混为一行。[^wcode][^wpaper]

其 Theorem 1 的 score 差公式在证明中忽略了反演近似误差；Appendix E.2 显式恢复误差项。进一步把 weak→strong 方向解释为 strong→ideal 的修正，还需要两种误差方向相近，**不是高 CFG 自动更接近真实分布的证明**。[^wpaper]

### FSG：相同算子家族，前瞻调度与主推进不同

FSG Algorithm 1 在选定时间对较长前瞻区间做 guided 前向→null 反演，可以多轮；随后执行 CFG++ 校正与普通采样推进。它的目标是 conditional/null 路径一致，但实用算法并不计算完整未来残差后再接受或拒绝更新。其区间优化定理依赖收缩、平滑和有界假设，约束的是校准轨迹上的预测差，不直接给出 FID 保证。[^fpaper]

这与 ordinary conditional 逆腿有明确区别：FSG 默认 \(w_b=0\)，本构造为 \(w_b=1\)。但 Z-Sampling / W2SD 已允许此选择，所以该差别主要是实验变量，不足以支持独立原创性。

## 3. 作者代码的执行顺序与时间核对

核对的是三个固定提交；均只阅读源码，未运行 GPU。

|实现|真实分支与顺序|循环与普通推进|时间处理|
|---|---|---|---|
|Z 原作者 release|先 `scheduler.step` 强去噪，再弱 inverse，随后强去噪|每一步 `T_max` 次逆→强前，默认1；最后将 `next_latents` 作为下一时间状态|逆腿输入是已去噪状态，但 UNet 仍查询旧 t；inverse scheduler 参数为 `timesteps[i+1]`|
|W2SD LoRA release|`reflection_operation` 切 strong LoRA/CFG 前进，再 weak LoRA/CFG 反演|每个非最后时间步一次 reflection，随后恢复主 LoRA 并普通推进|同样在已去噪状态上查询旧 t，inverse scheduler 接收 `timesteps[i+1]`|
|FSG release|选定索引循环 guided 前向→inverse；随后 CFG++ 修正 latent|NFE50示例：40主步，索引0/5/15各2/2/1次，名义前瞻5索引；每腿 `foresight_steps=1`|替换 scheduler 的 `timesteps`，不更新 `num_inference_steps`；需按具体 scheduler 版本验证实际系数区间|

Z/W2SD 的“模型查询时间仍为旧 t”是源码事实，不等同于准确积分逆 ODE。它与论文将弱预测在两个相邻状态近似相等的处理相容；但对较大 h，把它当成准确逆流会混入时间与离散误差。[^zcode][^wcode]

FSG 的特殊时钟问题，本仓库已有在本地 diffusers 0.30.0、合成常量预测上的 scheduler 重放：名义长前瞻并不一定对应真实长区间系数。该结论限定在已核软件组合，不能推断论文指标错误，也不应静默改代码后仍称严格复现。详见[作者代码审计](../../SIT_FSG_AUTHOR_CODE_AUDIT_20260911_ZH.md)。[^fcode]

还有一个容易制造假新颖性的差别：FSG Table 1 将 Z-Sampling 标为不支持多轮，但 Z 原作者当前代码明确有 `for step in range(T_max)`。因此，“每个时间步多做几次 reflection”已有公开实现。FSG 自己对 Z 的 backward→forward 解释也从已推进的中间状态切入；比较时应先写出完整箭头序列，不能仅按“forward-backward”名称判定不同。

## 4. 扣除普通 CFG 后，剩下的是什么

令 \(g=v_c-v_u\)。光滑、小区间下，单次回到原时间的位移为

\[
L(x)-x=h(w-1)g(t,x)+O(h^2).
\]

如果随后 strong 推进一步，K 次完整校正的首阶有效场是

\[
v_w+K(v_w-v_c)=v_u+[w+K(w-1)]g.
\]

如果最后 conditional 推进，则首阶有效系数为 \(1+K(w-1)\)。因此，不匹配有效强度的“比原 CFG 更强语义”比较，很可能只测到了更大的 CFG。常向量场中这些结论可以精确成立，所有复杂往返只是强度重参数化，构成反对“reflection 必有独立机制”的直接算子反例。

有限区间中，两场在不同状态上查询、非线性和流输运会产生普通标量 CFG 未必表达的变化；但这仍不决定质量的方向。本仓库早已用

\[
S,W,\quad L=\Phi_W^{-1}\Phi_S,\quad
M_{\alpha,m}=\Phi_S\bigl[I+(\alpha/m)(L-I)\bigr]^{\circ m}
\]

研究相同构造，推导了二阶差异，并给出一维线性情况下能完全重标定为普通 guidance 的反例。此次只是令 \(S=v_w,W=v_c\)，不能重命名为新的 lifting。详见[局部流与强度重标定](../../LIFTING_MODIFIED_FLOW_AND_SCALE_20260909_ZH.md)。

W2SD 作者对 autoguidance 的比较使用了一档固定额外系数；它可支持该设置的指标差异，不能排除“普通 guidance 更充分调参后等效”的竞争解释。其“避免高引导伪影”表述也不构成任意模型、任意区间的保证。[^wpaper]

## 5. 有研究价值的收窄问题

**优先问题：conditional 逆流能否提供超过标量强度重标定的稳定收益？** 固定高前向分支，把逆腿 conditional 与 null 比较，同时匹配实际动作半径、普通 CFG 的有效强度和总模型查询。先保留 \(CL=S\) 的准确抵消对照，再比较 \(SC^{-1}S\) 与最充分调参的普通 CFG；否则很难知道收益来自方向、更多计算还是名义系数变化。这个问题是对仓库旧研究的收窄和验证，不应当作新算法命名。

**第二个问题：什么时候高 CFG 已经不是可信教师？** 高 guidance 是更强控制，不是独立更好模型。在普通 conditional 已准确的理想情形，继续外推可以偏离目标条件分布；当其目标本来就是更高偏好的子分布，又需要独立偏好定义。因而可研究的是在固定外部语义/质量读出下，哪些状态、哪些 inverse 基准允许收益和代价有利交换，而非不断压低 cycle residual。本仓库已讨论教师正确性、未来价值与类内扭曲，不应重新包装为“未来一致性必然带来质量”。见[基本假设与反例](../../GUIDANCE_FUNDAMENTAL_HYPOTHESIS_20260911_ZH.md)。

**数值层面的问题：有限时间的输运作用在误差可控后是否仍存在？** 需要并列准确对齐的 inverse ODE、冻结旧 t 的发布版、同场往返控制；如果收益只在不准确 inverse 时出现，研究对象就是特定离散扰动，而不是精确反演的语义运输。仓库已有时钟与 Jacobian 审计，这应是解释旧实验的必要步骤。

以上是可证伪的研究问题，尚不是未被文献覆盖的贡献。仅新增自适应回退或 reflection 蒸馏也有近例：Ctrl-Z 按代理奖励停滞触发回退、重采样并筛选；2026-09-03 的 RA-GRPO 在同一 FM 模型两档 guidance 下做强前→弱逆→强前，随机一次 reflection，再用偏好训练吸收修正轨迹。[^ctrlz][^ragrpo] 后者把“强 guidance 更接近目标”列为假设，不能拿其经验机制解释作普遍定理。

## 6. 可核查来源

[^zpaper]: Bai et al., [Zigzag Diffusion Sampling](https://arxiv.org/html/2412.10891v2)，2024-12，Algorithm 1、Eq.(2–5)、§3；ICLR 2025。论文以 latent 位移及累计量解释信息，不能自动等同于图像语义收益。
[^zcode]: [Z-Sampling 原作者 pipeline，提交 eef8bb265deb8f33efd47c53e6ee5506606de360](https://github.com/xie-lab-ml/Zigzag-Diffusion-Sampling/blob/eef8bb265deb8f33efd47c53e6ee5506606de360/utils/pipeline_stable_diffusion_xl.py#L1421)，L1421–1520；[参数入口](https://github.com/xie-lab-ml/Zigzag-Diffusion-Sampling/blob/eef8bb265deb8f33efd47c53e6ee5506606de360/infer.py#L10)。核查 HEAD 日期为2026-05-20；pipeline SHA256 `a5ddcf26f80ef052a365ed1408615b8c5d2afd0cfa7e8243f40423f02a476cfd`。
[^wpaper]: Bai, Sugiyama, Xie, [Weak-to-Strong Diffusion with Reflection](https://arxiv.org/html/2502.00473v3)，2025-04-24 v3，Algorithm 1、§4.2、Appendix B.1/D.1/E.2/E.4；本文读取 arXiv v3，不将仓库 ICLR2026 标签误作另一个已核正文版本。
[^wcode]: [W2SD 原作者 reflection_operation，提交 c1c160d634837e92dd98851a74b52aa5415f5da5](https://github.com/xie-lab-ml/Weak-to-Strong-Diffusion-with-Reflection/blob/c1c160d634837e92dd98851a74b52aa5415f5da5/utils/pipeline_stable_diffusion_xl.py#L1099)，主循环 L1491；[示例参数入口](https://github.com/xie-lab-ml/Weak-to-Strong-Diffusion-with-Reflection/blob/c1c160d634837e92dd98851a74b52aa5415f5da5/w2sd_lora.py#L22)。核查 HEAD 日期2026-01-28。
[^fpaper]: Wang et al., [Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](https://arxiv.org/abs/2510.21512)，2025-10-24，NeurIPS2025，Table1、Algorithm1、Theorem1及Appendix C。
[^fcode]: [FSG 原作者 pipeline，提交012398fae56912f88fd8fec588b4ceb92800d9d6](https://github.com/Ka1b0/Foresight-Guidance/blob/012398fae56912f88fd8fec588b4ceb92800d9d6/utils/pipeline_stable_diffusion_xl.py#L995)，L995–1027、L1114、L1156；[NFE50配置](https://github.com/Ka1b0/Foresight-Guidance/blob/012398fae56912f88fd8fec588b4ceb92800d9d6/configs/NFE-50.yaml)。HEAD 与2026-09-11仓库审计一致。
[^ctrlz]: Mao et al., [Ctrl-Z Sampling: Scaling Diffusion Sampling with Controlled Random Zigzag Explorations](https://arxiv.org/abs/2506.20294)，原始2025-06，当前公开摘要明确奖励停滞检测、自适应更深回退与候选接受。此处仅用于覆盖已有研究方向，不将它视为同一确定性 inverse 算子。
[^ragrpo]: Wu et al., [Step Back to Move Forward: Reflection-Aware Preference Optimization for Visual Generation](https://arxiv.org/html/2609.04282v1)，2026-09-03，§3.2–3.3。原文包含一阶经验方向假设；未核其训练复现代码，不据此作严格质量保证。

检索与代码核查截至2026-09-13。下载源文件位于 `/tmp/cfg_inversion_prior_20260913`；正式链接固定到论文版本与作者提交。未新增图像实验，未改动共享 runtime。
