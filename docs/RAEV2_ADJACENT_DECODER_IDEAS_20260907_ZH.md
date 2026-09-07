# RAEv2 邻近方向：闭式 decoder readout 与条件随机 residual

**判断：目前没有足够证据把 decoder 路线推为新的主方向。** 只保留一个成本很低的判伪候选：冻结 decoder 的非线性部分，闭式重估最后的像素读出层。条件随机 residual 的生成机制更完整，但 RAEv2 高维表示可能近似可逆，不能先假定存在值得补充的条件熵；先做可辨识性与容量检查，不立即训练随机 decoder。

本文是新增并行思考，不修改正在运行的四种方法 1K→5K 补量、不改任何冻结采样/合并/评价代码，也没有新增 GPU 调用、训练或 FID。只有一个低维 CPU 线性代数核对。两个方向都不改变 latent prior、轨迹或 guidance：它们改变终点的解码规则。

## 1. 阅读范围与真正有约束力的事实

本次定向查阅下列 **8 篇原论文**。这是相关部分的精读，不宣称逐页复核整篇证明，也不因此重算仓库原有论文总数。

| 原论文与本次范围 | 对当前选择的约束 |
| --- | --- |
| [RAE，2510.11690v1](https://arxiv.org/html/2510.11690v1)，§3、4.3 与训练目标 | 真实 latent 重构好不能保证生成好；原文已经训练带噪 decoder。普通 noise fine-tuning 不是新方法。 |
| [RAEv2，2605.18324v1](https://arxiv.org/html/2605.18324v1)，§2.1、3.3、附录实现与广义表示 | 多层聚合增加可重建信息。K7 不是单一 LayerNorm 后的球面，不能套固定 token 半径。论文中的广义表示也不能替代本地实现审计。 |
| [Scaling RAE，2601.16208v1](https://arxiv.org/html/2601.16208v1)，§3.1–3.3 | 大规模 T2I 中，decoder noise augmentation 的收益主要在早期，后期弱化；不能把小模型鲁棒性叙事视为所有模型的共同瓶颈。 |
| [Improving the Diffusability of Autoencoders，2502.14831v1](https://arxiv.org/html/2502.14831v1)，频谱机制、§4 设置与对照 | scale equivariance 已是强先例；其生成器训练成本不能忽略，不能把它包装成固定 prior 上的免费修改。 |
| [Image Tokenizer Needs Post-Training，2509.12474v1](https://arxiv.org/html/2509.12474v1)，§4.3、A.5–A.6 | 对生成 latent 适配 decoder 已被直接研究。扩散版本用 SDEdit 构造对应关系，结果随 preservation ratio 和训练时长变化；表中 200K 图、10/20 epochs 不属于这里的极小训练预算。 |
| [LV-RAE，2602.08620v1](https://arxiv.org/html/2602.08620v1)，§3.1–3.3 | 低层信息补足、off-manifold 增益、噪声微调与终点噪声注入均已有先例。其玩具 decoder 有人工指定法向响应；没有识别本模型的真实法空间。 |
| [The Perception-Distortion Tradeoff，CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.pdf)，问题定义与 tradeoff 论述 | paired distortion 与输出分布不是同一目标；不能因 MSE 下降就推断生成质量提升。 |
| [A Theory of the Distortion-Perception Tradeoff in Wasserstein Space，2107.02555v1](https://arxiv.org/html/2107.02555v1)，§2–4、Gaussian 构造 | 在其假设下，最优 distortion–perception 点可由传输描述。即使联合 Gaussian，满秩时也可以是确定性映射；不能从“确定性 decoder”推出“缺少随机性”。 |

没有依赖二手摘要作为理论证据。下面的有限矩读出优化是本次独立推导；它不是上述论文已经证明了 RAEv2 改善，更不是 FID 定理。

## 2. 与仓库旧工作明确分开

已查阅 [总索引](RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md)、[cycle 方向实验](RAE_CYCLE_DIRECTION_RESULTS_ZH.md)、[decoder risk Phase 0](RAE_DECODER_RISK_PHASE0_RESULTS_ZH.md)、[噪声坐标审计](RAE_DECODER_NOISE_GEOMETRY_RESULTS_ZH.md)、[latent trust 与 decoder alignment](RAE_LATENT_TRUST_DECODER_ALIGNMENT_RESULTS_ZH.md)、[IG pushforward](RAEV2_IG_DECODER_PUSHFORWARD_MECHANISM_ZH.md)、[可逆 latent LPL adapter](RAEV2_INVERTIBLE_LATENT_LPL_PILOT_ZH.md) 及 [prior–decoder 文献总指南](PRIOR_DECODER_EXPERIMENTS_AND_LITERATURE_GUIDE_ZH.md)。

另已补读更新的 [9/7 八轮理论教训](RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md) 与 [9/7 结题质量表](RAEV2_GUIDANCE_FINAL_CLOSEOUT_20260907_ZH.md)。其中状态条件方差 5K FID 为 `6.957439858`，方向方差为 `6.910567613`，对应 official 为 `6.949768478`；前者恶化，后者只改善约 `0.5641%`。这些是另外一批已完成的历史实验；不与正在进行的嵌套补量或其 seed 混用。它们进一步降低了下面候选 B 的优先级。

- 不再把 `E(D(z))` 当作最近点投影、score 或可直接减小的质量误差。旧 cycle/idempotence 及跨空间 Bayes 反例已有明确约束。
- 不再训练静态 channel/DCT metric 去近似完整 decoder 梯度；旧 proxy 的失败包含很强的 oracle 对照。
- 不再用“raw latent 更像真实”解释质量；旧 IG 在 decoder 后发生方向旋转与排序反转，且没有简单抵消重构 bias。
- 本地 K7 表示是多层归一化特征的聚合并包含空间均值项，已知通道和约束可以保留，固定球面约束不成立。[既有说明](RAEV2_AFFINE_REFLECTION_PROTOCOL_20260906_ZH.md)。
- 普通 decoder post-training、随机 decoder、可逆 latent adapter 都不是空白。下面的差别必须落实到目标、可证对象与成本，不能靠重新命名。

## 3. 候选 A：最后读出层的矩约束重建，优先级为“廉价判伪”

### 3.1 具体机制与设计

冻结现有 decoder 到最后一次 LayerNorm 的全部计算。对一个真实图像的 patch，记最后 hidden 为 `H∈R^d`、目标像素 patch 为 `X∈R^p`。当前架构的最后一步是

`Y = A₀H + b₀`，随后 unpatchify、clamp、量化。

本地 [decoder](../external/RAEv2/src/stage1/decoders/decoder.py) 与 [ViTXL 配置](../external/RAEv2/configs/decoder/ViTXL/config.json) 给出 `d=1152`、`p=16×16×3=768`，因此只涉及 `1152×768+768=885,504` 个现有参数。没有新模块、额外 main forward、VJP 或 decoder forward。推理时直接替换原读出权重。

真实动机不是“最后一层一定有错”，而是检验一个小而明确的问题：**当前读出的 paired 重建与输出二阶统计之间，是否存在可以解析修正的失衡？** 不要求生成 latent 有真实逐图标签，不伪造 generated↔real 配对，也不把 final hidden 当作 decoder Jacobian proxy。

用独立真实训练图的配对 `(H,X)` 估计均值及中心矩 `C_h,C_x,C_xh`。考虑完整确定性仿射头中的约束优化：

\[
\min_{A,b}\ E\|AH+b-X\|^2,
\qquad E[AH+b]=\mu_x,\quad A C_h A^\top=C_x.
\tag{A1}
\]

取满足均值的 `b=μ_x−Aμ_h`。先在 `C_h` 的真实 range 上工作：

\[
C_h=U_r\Lambda_rU_r^\top,\quad V=U_r\Lambda_r^{-1/2},
\quad S=C_x^{1/2},\quad B=S C_{xh}V.
\]

当 `C_x≻0` 且 `r≥p`，写 `A=S Q Vᵀ`，约束化为 `QQᵀ=I_p`。因为 `tr(A C_h Aᵀ)=tr(C_x)` 为常量，最小 MSE 等价于最大化 `tr(QBᵀ)`。若 `B=L diag(s) Rᵀ` 为薄 SVD，一个全局解是

\[
\boxed{A_*=S L R^\top V^\top,\qquad b_*=\mu_x-A_*\mu_h.}
\tag{A2}
\]

证明只使用二阶矩和矩阵 Procrustes，不要求 `(X,H)` 联合 Gaussian。若进一步联合 Gaussian，矩匹配还匹配该 patch 的 Gaussian 边缘；一般自然图像则**只匹配 patch 一二阶矩**。

`C_h` 不能不加检查地假定可逆：末端 LayerNorm 的仿射归一化产生至少一个结构零方向。应先根据其已知仿射约束移除该方向，再对真实 range 求解。若条件数、有效 rank 或跨图统计不稳定，停止，不能加一串 ridge/rank 超参数去挽救。`rank(C_h)<rank(C_x)` 时确定性矩约束不可行；这不是自动添加随机噪声的许可证。

### 3.2 理论能接到实际生成分布的哪一步

训练只使用真实配对。设固定官方生成 endpoint 经同一个冻结 decoder 前缀后，hidden 的矩为 `μ_q,C_q`。任何新读出在实际生成输入上的**量化前**像素矩都可精确计算：

\[
\mu_{Y,q}=A_*\mu_q+b_*,\qquad C_{Y,q}=A_*C_qA_*^\top.
\tag{A3}
\]

因此能在不修改 prior 的前提下，直接检查 A2 是否把真实配对训练上的修正转移到了实际 q；不必假设 `q=p_h`，也不必把 decoder 线性化。非线性部分是精确冻结并完整执行的。

但 A2 的最优性仅针对真实配对下的 A1。若 `μ_q,C_q` 与真实 hidden 矩不同，A3 完全可能更坏；clamp/uint8 也会改变其边缘。更根本地，任意重新排列图像内的 patches 都可以保留 pooled patch 矩，却破坏对象结构。**patch Gaussian 距离为零也不蕴含 Inception FID、语义或图像全局密度正确。**

### 3.3 最小判伪与预算

建议只允许一个矩约束头、一个普通 least-squares 头和原头三个确定分支；不插值、不搜索 gain。

1. 固定 2K 真实训练图拟合统计，另 1K 真实图检查 paired MSE、量化后重构、矩约束迁移和数值稳定性。按图划分，不能把同图 256 个 patches 当独立验证样本。
2. 固定已有两个独立生成 bank 的 endpoint，各取预先确定的 1K；完整运行 decoder 前缀并计算 A3。它只用于检验跨分布迁移，不能重新拟合 A2 或选频带。若两组生成矩都反向，或新头只能靠严重增大 clipping 满足量化前矩，停止。
3. 若必要条件通过，再对所有固定 endpoint 应用三个头并做同样本数量的真实图像 FID/另一冻结表征评价。此处才出现生成质量证据。一次失败就降级，不追加头宽度、目标 band 或系数搜索。

预算以调用数表达：最多 3K 真实图的 E/D 前缀（已有缓存可少算）；2K 生成 endpoint 的 D 前缀；零 Stage-2 调用、零反向传播、零 SGD。流式累积约 `O(d²+p²+dp)` 的 FP64 统计，核心约 23 MB；不必保存全部 token hidden。部署参数数目和浮点运算形状不增加。统计提取与 CPU 分解仍是校准成本，不能写成完全零训练成本。

低维 CPU 核对仅确认代数实现思路：5K 合成配对、`d=7,p=3,rank(C_h)=6`，约束相对误差 `2.02e−15`；解析解 MSE `0.26717`，100 个随机可行正交头中最好为 `17.63110`。这不是图像实验，也不构成高维稳定性证据。

### 3.4 新意与失败条件

与旧 latent inverse adapter 不同，它不学习坐标变换或 LPL；与普通 decoder 微调不同，它有固定可解的读出优化，没有优化器、学习率、GAN 权重或噪声调度。**矩约束回归、Procrustes 和 Gaussian 最优传输本身都是经典工具；应用到 RAE 读出层不自动够成论文。** 若最终主要改变颜色/纹理统计、只改善某一指标，这应记为校准工程结果。

最可能失败的原因是：当前头已经由大量 L1/LPIPS/GAN 训练充分优化；global/semantic 错误在冻结的上游 hidden 中；patch 矩对真实 image FID 的约束太弱；或真实 hidden→生成 hidden 的分布变化把干净域校准转为过敏。这些风险足以把它限定为廉价判伪项，而不是目前“最强的新理论方向”。

## 4. 候选 B：极小条件随机 residual，机制成立但暂不放行训练

### 4.1 正确的生成机制

令 `Z=E(X)`、`M(Z)=E[X|Z]`。真正的条件随机 decoder 目标是核 `K(z,dx)=P(X∈dx|Z=z)`，而不是给输出任意加噪声。若 `Z∼p_Z` 且核准确，则 `p_Z K=p_X`。对 `q_Z` 和近似核 `K_θ`，数据处理与核误差三角界给出

\[
\mathrm{TV}(q_ZK_\theta,p_X)
\leq \mathrm{TV}(q_Z,p_Z)
+E_{q_Z}\mathrm{TV}(K_\theta(Z),K(Z)).
\tag{B1}
\]

这说明准确核不会放大输入 TV，且区分了 prior 误差与 decoder 误差；它没有证明当前 q 的差距能由随机 decoder 消除。还有一个识别边界：数据只确定 `K` 在 `p_Z` 几乎处处的取值；若 q 落在其支撑外，那里如何延拓并没有真实条件标签。特别是支撑互斥时 TV 上界可直接变得无信息。连续密度/支撑和绝对连续等条件满足时还可用 KL 链式分解，但不能对离流形 q 直接声称有限 KL。

对现有 decoder `D` 定义 residual `R=X−D(Z)`。精确条件均值为 `b(Z)=M(Z)−D(Z)`，条件协方差为 `V(Z)=Cov(X|Z)`。若小模型真能表达条件 Gaussian，结构自然导出

\[
X'=D(Z)+b_\theta(Z)+L_\theta(Z)\xi,
\quad\xi\sim N(0,I),\quad L_\theta L_\theta^\top=V_\theta.
\tag{B2}
\]

均值/协方差由适当的条件 likelihood 学习，不再手调推理 noise scale。即使条件分布非 Gaussian，正确均值和协方差也只给条件前两矩，不给真实图像核。条件 Gaussian、空间因子化、低秩近似都必须列为模型假设。

### 4.2 为什么还不能据此训练

关键恒等式是

\[
E\|X-D(Z)\|^2
=E\operatorname{tr}V(Z)+E\|M(Z)-D(Z)\|^2.
\tag{B3}
\]

**重建 residual 的能量不是条件方差。** RAEv2 latent 维数高于像素维数，编码器在自然图像支撑上可能近似单射；若是单射，`V(Z)=0`，全部残差都可能是确定性 decoder 的近似误差。额外噪声会把可学习信号当作不可约随机性。维数比较本身也不能证明单射或非单射。

同样，只有一个真实 X 对应每个高维 Z 时，几乎不存在完全重复的条件观测。一个小回归器的 held-out residual 方差混合了有限模型容量、训练误差和真实条件方差，不能直接拿来宣称“发现了 decoder 缺少的熵”。在 Gaussian distortion–perception 理论中，满秩条件甚至允许确定性最优解；随机性不是普遍更好。

这里还必须继承仓库已经给出的更强反例：目标 `P=N(0,1)`、模型固定均值为 1 时，把方差从 1 改成 NLL 最优的 2，`KL(P||Q)` 从 `.5` 降到 `.5 log2`，FID 却从 1 升到 `4−2√2≈1.171573`。这不是 decoder 非线性、有限样本或路径分布偏移造成的。对有限容量均值头也有同样问题：固定 `b_θ(z)` 时，Gaussian NLL 的最优 covariance 是

\[
V^*_\theta(z)=V(z)+[b(z)-b_\theta(z)][b(z)-b_\theta(z)]^\top.
\tag{B4}
\]

所以 NLL 学到的随机幅度可能在掩盖确定性 bias。旧 conditional/directional variance 在采样过程内改变 latent 转移，B 在终点改变 pixel 条件核，两者实现对象不同；**代理风险不能推出 FID 的逻辑缺口完全相同**。仅把方差预测器搬到 decoder 后面不构成机制突破。

### 4.3 最小判伪及暂定硬预算

第一阶段应当只问：同样冻结特征、同样数据与参数预算，确定性 residual 是否已解释几乎全部可修复误差？若它已解释清楚，随机头不应继续。

只有得到独立的 residual 分布证据后，才值得固定一次极小比较：一个确定性均值 residual、一个共享相同均值网络的条件 covariance residual；不使用 unconditional Gaussian，也不使用推理 variance multiplier。可把一次配置限定在 **≤1M 参数、≤1024 次更新、global B32**，冻结 E、D 和 Stage-2。训练预算至多 32,768 次标签呈现，数据与 split 预先固定，是否够用未知。

完整 `196608×196608` 像素条件协方差不可行。小头必须选择例如固定正交多尺度变换下的局部块分布，这是一项真实容量约束；它可能只能产生颗粒噪声，无法补足跨区域纹理。不能因为使用 Haar/DCT 就称其结构受理论保证。对量化像素应使用相应离散观测 likelihood 或明确的观测模型，不能事后调 variance floor 得到好看的 NLL。

在共享生成 endpoint 上，每幅图只抽一次预定随机数、只输出一张图；零 Stage-2 增量，部署为原 decoder 加一次小 residual forward。要单独报告训练成本、取样成本、clamp/uint8 后图像分布，以及确定性控制。若只降低 NLL/重构误差而不能改善新增独立图像质量，停止。

### 4.4 新意判断

条件生成 decoder、DiffAE、SODA、SWYCC、DiTo 等的基本路线已在 [旧总指南](PRIOR_DECODER_EXPERIMENTS_AND_LITERATURE_GUIDE_ZH.md) 登记。这里没有新增这些论文的精读，也不把 `p(z)p(x|z)` 重新声称为新理论。

可能值得研究的部分只能是：**现有近可逆的高维 RAE 中，能否识别一个小而稳定、不会被确定性小头解释的条件残差，并以自然 likelihood 在一次极小预算内补足？** 目前还没有这项证据。若实际条件熵极小、残差依赖长程结构、或生成 q 不在训练条件覆盖域，这条路不成立。

## 5. 最后排序

保留 A 做低成本判伪，但不把闭式形式当作品味或新意的替代品。B 只保留机制问题和停止条件，暂不分配随机 decoder 训练预算。普通噪声微调、cycle/retraction、固定 K7 球面投影和再次训练 decoder proxy 已明确排除。

对这两个方向，5% FID 改善均没有现有证据；所有局部最优性、核恒等式与代理矩检查都不承担该性能结论。
