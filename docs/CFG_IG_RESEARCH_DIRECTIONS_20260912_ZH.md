# CFG 与 IG 的两条研究主线

目前更值得优先推进的是纯 CFG 中的一个具体问题：**条件分支与 null 分支的差，包含多少真正区分类别的变化，又包含多少对所有类别都相同的变化？这两部分是否需要不同的引导强度？** 完整类别集合给出了一个必要的概率关系，可以把其中一部分共同变化实际分离出来。本轮小规模真实模型诊断已经发现这一分量，质量作用尚待验证。

IG 可以保留独立主线：**弱头是否延迟了某些结构的分辨，而同时引入了另一种去噪标定偏差？** 新增对照表明，低噪声的 Gaussian 平移拟合并不具有唯一解释力；简单的 score 幅度与径向项能达到接近的拟合效果。因而，研究重心应落在“哪种结构在深浅头之间发生变化”，而不预设弱头一定是强头的 Gaussian 平滑。

两条路线不需要共享一个成功条件。CFG 有真实类别与 null 的概率关系；IG 有同条件下计算深度不同的预测。一个在 CFG 上成立的类别机制不必迁移到 IG，IG 的深度机制也不必解释 CFG。

| 主线 | 具体问题 | 当前证据 | 首个候选 | 主要难点 |
|---|---|---|---|---|
| 纯 CFG | 类别差异与共同分量被同一系数放大，是否限制质量？ | 完整 100 类诊断中，中、低噪声的共同正交分量约占目标 gap 能量的 5%–10% | 保留条件 / null 模型，分别控制两种分量 | 共同分量可能有益；完整类别查询需离线压缩 |
| IG | 有用差异是否来自结构分辨的延迟，晚期差异是否主要是标定？ | 低噪声的两参数标定能接近热平移的解释力；结构分辨假说尚未在真实模型确认 | 先比较标定项与剩余项，再检验“刚好未分辨结构”的弱深度 | 结构读出与生成质量之间仍需实际证据 |

这两项属于研究候选，不是已经取得 FID 优势的方法。此前多种局部一致性、去噪误差和几何修正没有稳定转化为质量收益，说明机制需要产生不同于“多加一点 guidance”的实际预测。[已有实验与边界](GUIDANCE_RETHINK_AUDIT_20260909_ZH.md)

**纯 CFG：从完整类别集合识别共同分量。**

固定状态 $x$ 和噪声层 $t$，记各类别 score 为 $s_c$，null score 为 $s_u$。如果这些场来自同一个自洽联合分布，则

$$
p_t(x)=\sum_{c=1}^{K}\pi_c p_t(x\mid c),\qquad
s_u(x,t)=\sum_{c=1}^{K}p_t(c\mid x)s_c(x,t).
\tag{1}
$$

这不仅给出通常的 $s_c-s_u=\nabla_x\log p_t(c\mid x)$，还要求 null score 落在所有类别 score 的凸包中。权重随后验变化，不能用固定的均匀类别平均代替。CFG 的 Bayes 解释与这一混合恒等式来自同一概率关系。[^1]

记

$$
\bar s=\frac1K\sum_c s_c,\qquad
\mathcal L=\operatorname{span}\{s_c-\bar s:c=1,\ldots,K\},
$$

其中 $P_{\mathcal L}$ 是在指定 latent 欧氏度量下的正交投影。对目标类别 $c$，定义

$$
\begin{aligned}
g_c&=s_c-s_u,\\
r&=(I-P_{\mathcal L})(\bar s-s_u),\\
d_c&=P_{\mathcal L}g_c.
\end{aligned}
\qquad g_c=d_c+r.
\tag{2}
$$

这里的 $r$ 对同一状态的所有类别完全相同，而且与所有两两类别差 $s_i-s_j$ 正交。这是可直接验证的线性代数事实，不需要先知道类别后验。若 Eq. (1) 精确成立，则必有 $r=0$。

因此，非零 $r$ 见证了网络条件族与 null 分支之间的一种局部不相容。它不能完整解释为该联合模型的类别选择 score；否则，后验概率之和为 1 所要求的加权梯度平衡无法成立。

需要保留两个识别边界。第一，$r$ 只识别类别差异空间之外的共同分量，并没有恢复全部“语义”与“质量”成分；空间之内仍可能混合模型误差与类别信号。第二，不相容可以来自条件分支、null 分支，或两者；这不是 null 分支错误的单独证明。共同分量也可能恰好承担有益的质量修正，不能直接规定删除它。

**真实模型中的新读数。**

模型为 ImageNet100 SiT-S/2、800K EMA，使用合法 null 标签 100。固定四个已有 validation 图像 ID，在三个 teacher 噪声状态上查询完整 100 类与 null，共 12 个状态。时间采用 $z_t=(1-t)\epsilon+tX$，因此 $t=.15$ 为高噪声，$t=.90$ 为低噪声。

原始数组保存模型速度 $v$。在加性噪声坐标 $y=z_t/t$ 下，$s(y,q)=t(tv-z_t)/(1-t)$。同一状态与时间上的所有类别和 null 场经历相同的标量缩放与平移，因此凸包关系、类别差异空间及下述能量比例均保持不变；速度输出没有被直接当作 score 使用。

下表的共同分量能量为

$$
R_{\rm affine}=
\frac{\sum_i\|r_i\|^2}{\sum_i\|s_{c_i,i}-s_{u,i}\|^2},
\tag{3}
$$

即先合计能量再求比。凸包残差则最小化非负且和为 1 的类别权重；它包含了仿射关系之外的额外约束，不等同于共同正交分量。

| 时间 | 共同正交分量能量 / 目标 CFG gap 能量 | 四个状态的范围 | 凸包拟合剩余能量 / 目标 gap 能量 |
|---|---:|---:|---:|
| $t=.15$ | 0.44% | 0.27%–0.84% | 0.75% |
| $t=.50$ | 7.29% | 5.42%–10.47% | 13.09% |
| $t=.90$ | 7.72% | 5.69%–9.15% | 8.73% |

这个分量在中、低噪声可测，但不占 gap 的大多数。四个图像、一次噪声与相关时间点不能支持总体比例；teacher 上的结果也不能直接代表 guided rollout。当前最有价值的结论是：**有一个事先由概率必要关系定义、并且可以分离的分量，值得检验其质量作用。** [逐状态数据](data/cfg_ig_research_20260912/cfg_bayes_geometry.csv)、[汇总](data/cfg_ig_research_20260912/cfg_pooled_summary.csv)

计算使用 FP32、CPU，48 次批量前向，等价于 1,212 个单样本模型求值，耗时 44.10 秒，CUDA 未初始化。三个 SVD 相对截断 $10^{-6},10^{-5},10^{-4}$ 都得到 99 维类别差异空间，结论不依赖本次截断选择。凸包优化全部正常结束，归一化目标的 Frank–Wolfe 对偶间隙均小于 $3.6\times10^{-7}$；仿射正交残差本身已提供无需依赖凸包最优性的非零见证。[数值验证](data/cfg_ig_research_20260912/validation.json)

解析正对照也通过：精确三类 Gaussian 混合的凸包残差约为 $2.2\times10^{-33}$；给 null 加入类别差异空间之外的偏差后能检出。尤其需要完整类别集合：只保留精确三类中的两类，也会得到约 5.69% 的假阳性残差。随机选几个负类别不能替代这个必要关系检查。[解析结果](data/cfg_ig_research_20260912/analytic_checks.json)

**CFG 的具体方法候选与决定性比较。**

在明确 Eq. (2) 的分解之后，候选形式是

$$
s_{\rm new}=s_c+\alpha d_c+\beta r.
\tag{4}
$$

普通 CFG 对应 $\alpha=\beta=\gamma$，这里的 $\gamma$ 是相对条件场的额外强度。这个候选提出一个具体限制：普通 CFG 把两种有不同概率身份的变化绑定在一条系数对角线上。若共同分量主要导致晚期失真，$\beta<\alpha$ 应当更好；若它主要负责质量修正，减弱它反而会损害质量。这两种结果都能改变后续方法设计。

第一组有解释力的比较是：原始 gap、只保留 $d_c$、将 $d_c$ 恢复到原 gap 长度，以及仅保留原幅度的 $r$。恢复长度的分支只用于判断方向变化是否超出减弱总强度的作用，不能把它当作天然正确的算法。还需比较一个按时间匹配平均作用大小的普通 CFG，以及当前已经调优的 CFG / APG 基线。

如果删除 $r$ 的收益完全被普通 CFG 的强度或窗口调整覆盖，那么它只是另一种调强度方式，论文价值有限。如果保留相同总作用量时，分量选择仍稳定改变独立质量和类内变化，才支持 Eq. (4) 中的区别有实际意义。不能只用类别识别率上升作为质量证明。

完整 100 类投影适合定义离线监督，不适合直接在每个采样步执行。一个可部署版本是在已有 null 前向的中间特征上增加小读出 $\hat r(H_u,t)$，监督目标来自 Eq. (2)，推理仍使用原来的条件 / null 两路主干。首先应确认真实 rollout 上该分量的作用，再做读出；读出误差、离线查询与训练成本均需要单列。没有额外主干前向不等于没有额外延迟。

这仍属于纯 CFG 分支，不要求内部弱 denoiser。若研究限定完全免训练，则应把完整投影视为机制诊断，另外寻找廉价近似；目前没有证据说明少量随机类别或一个标量 rescale 足以准确估计 $r$。

**与已有 CFG 工作的实际差别。**

ICG 用随机条件构造参考；固定类别平均对应几何平均密度的 score，而不是 Eq. (1) 的后验加权混合。它不能直接测出这里的必要关系违例。仓库第 36 项已经尝试独立类别参考，不应再次把换几个负类别当成新主线。[^2] [既有实现与结果](SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md)

APG 对 guidance 作投影与动量修正；本候选投影空间由完整类别响应确定。真正需要建立的区别是这个空间是否比原有 clean / denoising 方向更有解释力，而不是换一个投影公式。[^3]

CDG 已明确提出通过语义退化条件消除共有变化，并用于文本组合生成。因此，“抑制共同变化”这个宽泛动机已有直接先例。这里可能形成的区别是完整类别混合关系给出的必要条件、可识别的残差，以及对共同分量应保留还是去除的实证判断；不能宣称首次分离语义与其他变化。[^4]

针对凸包关系、共同残差与 CFG 的检索尚未找到直接同型采样方法，但有限检索不构成新颖性证明。这个方向的价值应由概率身份、廉价近似与质量结果共同支撑。

**IG：Gaussian 平滑解释需要加入更简单的对照。**

原有诊断比较

$$
s_W(y,q)\approx s_S(y,q+\tau),\qquad
y=z_t/t,\quad q=((1-t)/t)^2.
\tag{5}
$$

在低噪声、浅层弱头上，有限热平移确实降低了很多拟合误差。但同一批数据上的新对照是

$$
s_W-s_S\approx a(t)s_S+b(t)y.
\tag{6}
$$

每个弱头、每个时间只拟合两个全局标量；仍使用前 16 个图像拟合，后 16 个图像评价。没有逐图选择参数，也没有新增模型查询。下表的“解释率”均为相对原始 gap 的平方残差减少 $1-R$，并非已经确认的因果分量占比。

| $t=.90$ 的弱参考 | Gaussian 热平移解释率 | 幅度 + 径向两参数解释率 |
|---|---:|---:|
| depth4 | 68.38% | 65.50% |
| depth6 | 60.00% | 58.74% |
| depth8 | 50.00% | 48.64% |
| depth10 | 26.98% | 24.61% |
| 外部 500K | 0.83% | 0.35% |

depth4 在 $t=.70$ 的两种解释率为 41.11% 与 44.63%，在 $t=.50$ 为 5.74% 与 18.85%。热平移并没有在这些对照中显示出唯一的结构解释。两个参数与一个经过候选选择的非线性平移模型并不完全等复杂；这里也没有证明简单模型优于热平移，只证明此前的拟合成功不能单独鉴定机制。[完整 75 行结果](data/cfg_ig_research_20260912/ig_nuisance_summary.csv)

这个结果需要收紧原结论：**可以说弱头在低噪声具有与额外噪声尺度相似的场特征，不能说已经识别了其生成分布的 Gaussian 平滑机制。** Gaussian 分布本身的热平移就可表现为 score 幅度缩放，因此这类混淆不是意外。它也没有否定更一般的模式合并或分辨率损失。

该控制是在热平移结果可见之后增加的，属于探索性再分析。相同评价输入被再次使用；置信区间只覆盖固定拟合选择下 16 个输入的抽样变动，不包含候选选择与重新拟合的不确定性。独立数据与真实 rollout 才能确认其可推广性。[协议](CFG_IG_RESEARCH_DIAGNOSTIC_PROTOCOL_20260912_ZH.md)

Self-Guidance 已经利用更高噪声参照实现显式平滑式 guidance，AG 也已有强弱密度比的解释。这两个宽泛原理不是待提出的新贡献。真实 IG 的新问题是深度约束究竟改变了哪种结构，以及这种改变为何在特定时间有效。[^5][^6]

**IG 的便宜起点：先分离标定，再判断结构。**

令 $g=s_S-s_W$，把 Eq. (6) 对应的部分定义为

$$
g_{\rm cal}=-a(t)s_S-b(t)y,\qquad
g_{\rm rem}=g-g_{\rm cal}.
\tag{7}
$$

可以直接比较原 IG、只保留 $g_{\rm rem}$、只保留 $g_{\rm cal}$，以及作用大小匹配的剩余项。这个计算只使用现有输出和状态，适合先做低成本验证。这里的“剩余”不能直接命名为有效结构；它仍包含未解释误差。

这个分解本身不是足够的新方法。仓库已经做过从浅层特征预测 gap 后保留残差的构造，并在一个固定 small-SiT 配置上得到约 1.27% 的 5K FID 改善，尚未建立超过调优 IG 强度曲线的优势。Eq. (7) 的新增用途是提供一个很小、解释清楚的标定对照，帮助判断深浅差真正值得研究的部分，而不是把旧残差化路线重新命名。[既有确认结果](SMALL_SIT_PREDICTABLE_GAP_CONFIRMATION_RESULTS_20260909_ZH.md)

如果分离后的收益完全等价于重新选择速度幅度、径向漂移与时间窗口，就应将其视为校准效果。只有在匹配这些因素之后，结构性差异仍能解释不同弱头的作用，IG 才有更强的机制主线。

**IG 的机制假说：弱头延迟了结构的分辨。**

一个具体模型是强分布仍保留双峰，弱分布把它们合并，但保持整体均值与方差：

$$
p_S(x,q)=\tfrac12\mathcal N(-a,v)+\tfrac12\mathcal N(a,v),
\qquad p_W(x,q)=\mathcal N(0,a^2+v),
\qquad v=v_0+q.
\tag{8}
$$

它们不需要误差共线，也不满足非平凡的独立加性噪声卷积关系：后者会增加方差，而 Eq. (8) 保持方差。弱模型的缺陷是对多峰结构的表达不足。密度比可以区分强分布的谷与峰，但沿各噪声层组合 score 并不自动采样端点幂密度。

这个模型给出比“弱模型更平滑”更具体的可测量量。对于双峰强分布，

$$
s_S(x,q)=-\frac{x}{v}+\frac{a}{v}\tanh\frac{ax}{v},
\qquad
\partial_xs_S(0,q)=\frac{a^2-v}{v^2}.
\tag{9}
$$

弱 Gaussian 在所有位置都有负 score 曲率。高噪声 $v>a^2$ 时，两者都把中心视为一个峰；当 $v<a^2$ 时，强模型在中心出现两个分支之间的谷，弱模型仍然给出单峰。进入某一个真正峰的邻域后，强模型局部曲率又可变为负。

因此值得检验的不是预先规定的指数衰减，而是一个状态与深度相关的区别：**当前结构在强头中已经可分辨，在某个弱头中仍被合并。** 它允许高噪声暂时没有明确结构、过渡阶段出现深浅分歧、完成局部结构选择后不再需要同一种引导。但低噪声时仍停留在谷附近的状态可能继续需要修正，不能把该假说等同于全局晚期归零。

在理想 Gaussian 加噪模型中，Tweedie 恒等式进一步给出

$$
\nabla_y^2\log p(y,q)=
\frac{\operatorname{Cov}(X\mid y,q)}{q^2}-\frac{I}{q}.
\tag{10}
$$

正曲率方向对应后验方差超过噪声方差的方向。这为“仍有多个竞争解释”提供一个局部代理。实际网络可能非保守，此时只能解释为对称 Jacobian 的局部响应，不能自动视为真实后验协方差。

**一个有区分力的 IG 预测是：最佳弱头可能是当前结构上刚好还没有完成分辨的那个深度。** 它未必永远最浅，也未必有最大的 gap。这个预测额外假设更浅的头会引入更多无关标定偏差；只有实际比较才可确定它是否成立，Eq. (8) 本身不证明最优深度。

IG 论文已有内部深度与噪声窗口的结果；其 depth4 优于 depth2 的例子本身就不支持普遍的“越浅越好”。真实头的辅助训练也会改变强模型，研究时应尽量固定主干，避免把训练变化混入深度解释。[^7]

**怎样把分辨假说与已有曲率工作区分。**

Saddle-Free Guidance 已经根据强模型正曲率引导采样；Dynamic Guidance 已利用类别划分减少不合理的模式插值。只提出“离开谷部”或“沿正曲率方向走”，新意不足。这里有待验证的增量是强弱之间的结构分辨差，及它对弱深度选择的预测。[^8][^9]

一个具体局部候选是在强场的正曲率特征方向 $u_j$ 中，再检查相同方向上的弱场曲率：

$$
\kappa_{S,j}=u_j^T\operatorname{sym}(J_{s_S})u_j>0,
\qquad
\kappa_{W,j}=u_j^T\operatorname{sym}(J_{s_W})u_j\le0.
\tag{11}
$$

只有满足两者分辨差的方向才进入候选投影 $P_\Delta$，然后比较 $s_S+\gamma P_\Delta g$ 与原 IG、只按强场正曲率投影的对照。各 $u_j$ 取自强场的一组正交方向；不能把两个不同 Hessian 的正负特征空间默认当作可同时对角化。

这是一个可执行的研究探针，尚不适合直接作为低成本采样器。完整曲率估计很贵，少量 gap 或随机方向又可能漏掉目标方向；已有 RAEv2 审计恰好说明这一点：40 个状态均找到强场正方向，而相同状态上的随机方向全为负。另一方面，深浅头在不少正方向上都已为正，这对 Eq. (11) 是真实挑战，不能只保留支持假说的状态。[已有极值曲率审计](RAEV2_EXTREMAL_CURVATURE_AUDIT_20260906_ZH.md)

如果在 IG 有效区间找不到有作用的分辨差，或其质量效果与普通强场正曲率方法相同，就不应继续把“临界分辨深度”作为新机制。正曲率也会来自合法多峰结构或强模型新增的错误峰，必须通过生成结果判断；它不是缺陷标签。一般的类别分辨还可能依赖高阶结构，单个 Hessian 代理并不完备。[^10]

若分辨差能预测跨状态、跨弱深度的效果，再考虑从已计算的中间特征预测深度选择或局部门控，避免每步昂贵 Jacobian 查询。理论首先用于给出方向和深度的预测，不应提前承诺免训练、零额外开销或通用最优窗口。

**研究取舍。**

优先级最高的是 CFG 的共同分量：它有独立于强弱质量顺序的必要概率关系，有新的真实读数，也允许明确的保留 / 去除干预。第一项需要回答的质量问题很窄：同样大小的 guidance，类别差异空间内外的分量是否应当区别对待。

IG 的两参数标定对照适合作为便宜的并行验证。它能决定现有平滑解释还有多少独立内容，也可能得到直接的校准改进；即使有小幅收益，也不能单凭该分解构成新的核心贡献。

IG 的结构分辨假说具有更大的机制空间，但证据更弱。它值得发展的前提是对“哪个弱深度、哪个状态、哪个方向有效”作出超过幅度和时间窗口的预测，而不是仅在拟合图上看起来像 coarse-to-fine。

两条主线都应分别对本身的强基线负责。纯 CFG 的候选应与调优 CFG / APG 比较；IG 的候选应与同主干、同成本的调优 IG 比较。不能用 CFG 与 IG 之间原本很大的基线差异充当方法贡献，也不能把多点 1K 选择结果当作独立确认。

**数据与可复核范围。**

本轮新增的是 CFG 的 12 个 teacher 状态机制诊断、IG 原始数组的替代模型拟合，以及解析正反例；没有新增图像质量或训练结果。IG 保留全部 75 行汇总与 2,400 行逐图记录，CFG 保留全部 12 个状态与 1,200 个类别权重。主要数据、实现与验证入口如下：

- [诊断协议](CFG_IG_RESEARCH_DIAGNOSTIC_PROTOCOL_20260912_ZH.md)
- [可运行诊断](../experiments/audit_cfg_ig_research_20260912.py)
- [独立算术验证](../experiments/validate_cfg_ig_research_20260912.py)
- [IG 逐图数据](data/cfg_ig_research_20260912/ig_nuisance_per_image.csv)
- [CFG 全类别原始场](/home/zhoushunyu/data/eqvae/imagenet_sit_flow/cfg_bayes_compatibility_cpu_20260912/manifest.json)
- [新增文献归档](../readings/cfg_ig_research_20260912/manifest.json)

**文献来源。**

[^1]: Jonathan Ho and Tim Salimans. [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). 2022. 条件 / 无条件 score 的 Bayes 对比；本文完整类别凸包与正交残差分解为独立推导。

[^2]: Seyedmorteza Sadat, Manuel Kansy, Otmar Hilliges, and Romann M. Weber. [No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models](https://arxiv.org/html/2407.02687v2). ICLR 2025. ICG / TSG 与独立条件参考；本地已核对正式稿与附录。

[^3]: Seyedmorteza Sadat et al. [Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models](https://arxiv.org/html/2410.02416v2). ICLR 2025. APG 的投影、范数约束与动量方法；并非本文完整类别分解。

[^4]: Shilong Han, Yuming Zhang, and Hongxia Wang. [Guiding Diffusion Models with Semantically Degraded Conditions](https://arxiv.org/html/2603.10780v1). 2026-03-11. 尤其 §4–5 的共有变化解释与条件退化构造。

[^5]: Tiancheng Li, Weijian Luo, Zhiyang Chen, Liyuan Ma, and Guo-Jun Qi. [Self-Guidance: Boosting Flow and Diffusion Generation on Their Own](https://arxiv.org/html/2412.05827v3). 引用版本 v3，2025-05-25；后续 v5 日期为 2025-09-26。本报告的更高噪声参照与 heat equation 分析对应所归档的 v3。

[^6]: Tero Karras et al. [Guiding a Diffusion Model with a Bad Version of Itself](https://arxiv.org/html/2406.02507v3). NeurIPS 2024. 强弱同目标模型与密度比解释，尤其 Eq. (4)–(5)。

[^7]: Xingyu Zhou et al. [Guiding a Diffusion Transformer with the Internal Dynamics of Itself](https://arxiv.org/html/2512.24176v2). CVPR 2026. 深度辅助读出、Table 1 与 §4.2 的噪声区间。

[^8]: Eric Yeats, Darryl Hannan, Wilson Fearn, Tim Doster, Henry Kvinge, and Scott Mahan. [Saddle-Free Guidance: Improved On-Manifold Sampling without Labels or Additional Training](https://arxiv.org/html/2511.21863v1). 2025-11-26. §3 的正曲率探针与采样方法。其 SFG 缩写与 Foresight Diffusion Guidance 的 FSG 不同。

[^9]: Kostas Triaridis, Alexandros Graikos, Aggelina Chatziagapi, Grigorios G. Chrysos, and Dimitris Samaras. [Mitigating Diffusion Model Hallucinations with Dynamic Guidance](https://arxiv.org/html/2510.05356v1). 2025-10-06. 对不合理模式插值的动态类别引导；不是一般插值都应删除的论证。

[^10]: Beatrice Achilli, Marco Benedetti, Giulio Biroli, and Marc Mézard. [Theory of Speciation Transitions in Diffusion Models with General Class Structure](https://arxiv.org/html/2602.04404v1). 2026-02-04. 一般类别结构与分辨转变，尤其 §2–3；不提供真实 IG 最优深度的现成定理。

上述十项为外部来源的完整清单。实际 SiT / RAEv2 数字来自正文所链接的本地实验与原始数据；Eq. (2)–(4)、本轮诊断与候选取舍属于独立分析。新增四篇原文的版本与 SHA256 保存在文献归档，其余文献的既有阅读入口分别见 [平滑报告](AG_IG_SMOOTHING_HYPOTHESIS_20260912_ZH.md)、[ICG 精读](RAEV2_GUIDANCE_READING_INDEPENDENT_CONDITION_20260906_ZH.md)与 [APG 精读](RAEV2_GUIDANCE_READING_PROJECTION_ZERO_20260906_ZH.md)。
