# Signed guidance schedule：CFG 文献、边界渐近与可验证机制

2026-09-23。只做 primary literature 核查和解析推导；未改训练、未运行 GPU。采样时间统一为噪声 t=0、数据 t=1。真实 SiT/JiT 曲线与实现由主分支审计；本笔记不把一般 CFG 的结论直接套到 internal strong−weak 上。

**结论：负 guidance 窗口已有明确理论先例，但没有找到“早高正—中深负—末端再强正”是 CFG 普遍最优规律的证据。对简单 Gaussian 的准确 rectified-flow ODE，可证明负段能补偿正 guidance 的方差收缩；同一个最小模型却只支持一次正到负的转换。末端上抬需要额外机制，例如模型误差、控制方向退化或离散修正，不能由负窗论文自动推出。**

## 1. 先统一系数约定

标准 CFG 有两种等价写法：

\[
v_g=v_u+w(v_c-v_u)
=v_c+a(v_c-v_u),\qquad w=1+a.
\]

这里使用额外系数 a：a=0 是原 conditional field，a=−1 是 unconditional field，−1<a<0 是二者的凸组合，a<−1 才越过 unconditional 继续外推。

项目的 internal guidance 是

\[
v_g=S+a(S-W).
\]

相同的代数意味着：a=0 用 strong；a=−1 用 weak；−1<a<0 是 strong/weak 凸组合；a<−1 是越过 weak。**但 W 不是 unconditional model，因此负 a 不能直接翻译成减弱条件，更不能直接叫反向语义 guidance。**

比较文献图还必须检查时间方向。常见 diffusion 文献以 t=0 表示数据，采样从大 t 向 0 行进；此时“早高”曲线按横轴 t 正向看可能恰好是上升的。

## 2. 文献实际上支持哪些形状

| Primary source | 已支持的事实 | 不应推出的结论 |
|---|---|---|
| [Applying Guidance in a Limited Interval, 2024](https://arxiv.org/html/2404.07724v1) | 多个图像设置中，高噪阶段 guidance 有害，低噪阶段常不必要，主要在中间使用改善结果。 | 不是 signed +−+ 先例；也不是所有模型的强制形状。 |
| [Analysis of CFG Weight Schedulers, 2024](https://arxiv.org/html/2404.13040v1) | 系统实验发现若干模型上沿采样过程递增的正 guidance 很有效；复杂最优参数不跨模型通用。 | 文中的 negative perturbation 是某段关 guidance，不是负 scale。 |
| [Stage-wise Dynamics of CFG, 2025/ICLR 2026](https://arxiv.org/html/2509.22007v1) | 多模态条件模型中区分早期均值偏置、中期模式选择、晚期类内收缩；提出中间高、两端低 schedule。 | 三个动力学阶段不等于 scale 应有 +−+ 三个符号阶段。 |
| [Emergence of Distortions, v4, 2026-05](https://arxiv.org/html/2602.00716v4) | 明确提出真正的 negative-guidance window；最新版本还在 SD1.5 中验证负窗的 diversity/condition-separation 改善。 | 方案是早正晚负，不是末端再正；有限实验不是跨模型普适最优性的证明。 |
| [Information-Theoretic CFG, 2026-06](https://arxiv.org/html/2606.24025v1) | 直接优化 actual sampler distribution 的一致性/覆盖目标，学得高噪近 conditional、中间增强、部分低噪更强的 schedule。 | 图中负的 update direction 意味增加 scale，不是负 scale；其 fixed-trajectory proposal 也不是完整参数梯度。 |
| [Analytic Distribution of CFG, 2026-07](https://arxiv.org/html/2607.19725v1) | 得到确定性 guided density 的路径积分表达；DG-CFG 对噪声系数、信号量及低噪误差做校准，两端弱、中间强。 | 该具体公式在 VPSDE/DDIM 下设计，不能原样搬到直线 flow interpolation。 |
| [Adversarial Learning of CFG Schedules, 2026-08](https://arxiv.org/html/2608.14038v1) | 用 GAN/marginal consistency 学 time/state/condition-dependent CFG，直接接近本项目动机。 | Eq.6 用额外系数 ω，输出 softplus 强制 ω>0；结构上不允许我们的负中段。 |

这些文献的差异本身就是证据：schedule 取决于目标、模型误差、噪声参数化和采样器，并没有一个普适曲线。

特别需要区分 [Dynamic Negative Guidance, ICLR 2025](https://arxiv.org/abs/2410.14398)：它研究避免负面条件、估计 posterior class probability 的动态负提示 guidance，不能仅因标题有 negative 就拿来证明对同一个正条件采取负 CFG scale 最优。

## 3. 最直接的负窗先例及其限制

Ventura et al. 最新 v4 的 [§4.2](https://arxiv.org/html/2602.00716v4#S4.SS2) 使用

\[
\tilde s=s_c+w(s_c-s_u),
\]

因此其 w 与本笔记的额外 a 同约定。论文的 VE forward 时间从数据出发，提出 w(t)=w0+ωt，w0≥−1、ω>0。沿反向采样是早期正、晚期负；在选定的 Gaussian/高维 mixture 条件下，可以扩大均值同时避免方差收缩。

这为“负控制不一定无意义”提供了真实理论依据。最新 v4 的 Appendix F.3.2 已加入真实模型实验：SD1.5、Euler Ancestral Discrete、30 steps、50 prompts×20 samples；先用正的额外 w，之后切到 w=−1，即末段使用 unconditional 场。作者报告合适负窗可增加 feature diversity，同时保留较好的 condition separation。

但其负窗仍在 w≥−1，即只在 conditional/unconditional 之间插值，且晚期不再回到正值。不能据此解释项目中可能低于 −1 的 strong/weak 外推，更不能把它当作真实图像的 +−+ 已被证实。作者亦说明该实验旨在隔离负 guidance 的效应，并非全面的最优 schedule 评估。

版本说明：本轮初读 v1 时“负窗尚未在 learned scores 验证”的判断已被 v4 新增实验改变，此处已以 2026-05-08 最新版本修正。

## 4. 准确线性插值 CFG 的边界渐近

令 X∼q_c 或 q_u，Z∼N(0,I) 独立，

\[
Y_t=(1-t)Z+tX.
\]

canonical velocity 是

\[
v_j(y,t)=\mathbb E[X-Z\mid Y_t=y,j]
=\frac{\mathbb E[X\mid Y_t=y,j]-y}{1-t}
=\frac yt+\frac{1-t}{t}\nabla_y\log\rho_{j,t}(y).
\tag{1}
\]

于是准确 conditional/unconditional 差为

\[
\boxed{
\Delta_t=v_c-v_u
=\frac{1-t}{t}\nabla_y\log\frac{\rho_{c,t}}{\rho_{u,t}}.
}
\tag{2}
\]

最后一个 ratio 也可写成 noisy classifier posterior 的 log-gradient，但仅适用于真正 cond/uncond 分布，不适用于任意 S−W。

### 高噪端 t→0

在固定有限 y、充分矩与可交换渐近条件下，记两终点分布的均值和 covariance 为 μj、Σj，则

\[
\mathbb E[X\mid Y_t=y,j]=\mu_j+t\Sigma_jy+O(t^2).
\]

因此

\[
\Delta_t=\Delta\mu+t(\Delta\mu+\Delta\Sigma\,y)+O(t^2).
\tag{3}
\]

结论：一般 Δt 有有限非零极限 Δμ；若两均值相同，则 Δt=O(t)，若更多低阶矩相同则可能更快消失。不能泛称“初始 CFG 差一定为零”。但高噪处最先暴露的只是低阶全局统计，强早期控制容易首先改变布局/均值，而不是细节。

### 低噪端 t→1

令 ε=1−t。若 qc、qu 在考察区域是平滑正密度，且 score 差收敛有界，则

\[
\Delta_t=\epsilon\nabla\log(q_c/q_u)(y)+O(\epsilon^2),
\qquad v_j(y,1)=y.
\tag{4}
\]

因此 bounded a(t) 的实际 guidance 速度 aΔ 线性消失。若 a≈const/ε，系数可以上抬而 aΔ 保持有限。这是一个**可检验的坐标补偿机制**，不是正号的理论来源。

Eq. (4) 不适用于未加正则的奇异数据流形，也未自动适用于网络误差主导的尾部；在不同支撑、决策边界或 off-manifold 点上，score 差可能不有界。实际 S、W 即使都为同一目标训练，误差差 S−W 也未必满足 O(ε)。必须测量，不能仅引用 canonical 渐近。

对于 strong/weak 两个准确估计同一 canonical field 的极限，S−W 反而在所有时刻都为零。项目中的有效方向主要来自有限容量、训练状态或架构差异，与 cond−uncond 的统计对象根本不同。

## 5. 一个可证的 flow-ODE 方差补偿机制

把负窗论文的直觉直接放到 rectified-flow ODE。考虑一维 jointly Gaussian conditional 结构：

\[
q_c=N(\mu,c),\quad q_u=N(0,u),\qquad 0<c<u.
\]

其中 c、u 是方差。定义

\[
D_j(t)=(1-t)^2+t^2j,
\quad A_j(t)=\frac{tj-(1-t)}{D_j(t)}.
\]

两 canonical 场分别为

\[
v_c=A_c x+\frac{1-t}{D_c}\mu,
\qquad v_u=A_u x,
\]

且

\[
\boxed{A_c-A_u=
\frac{t(1-t)(c-u)}{D_cD_u}<0\quad(0<t<1).}
\tag{5}
\]

令实际采样场 vg=vc+a(t)(vc−vu)。它是线性 ODE，所以**任意可积 schedule 的实际终点方差**为

\[
\boxed{
\operatorname{Var}(X_1)=
c\exp\left[2\int_0^1a(t)(A_c-A_u)dt\right].
}
\tag{6}
\]

因此非零、全非负 a 必然使类内方差小于 c。若想保留正 guidance 的其它效应，同时让方差恢复 c，必须让负区间对这个加权积分作补偿（或改变控制方向、噪声、模型）。这不依赖“CFG等于温度”近似，计算的是实际 ODE 输出。

这个模型足以说明 signed schedule 有原则上的用途；但只给出积分约束，不指定负窗必须在哪里，更不指定必须有第二次正向反弹。

## 6. 更细的一阶控制结果：最简模型不要求 +−+

在 a=0 的 conditional 基线上，终点均值和 log-variance 的一阶响应是

\[
\delta\mu_1=\int_0^1K_\mu(t)a(t)dt,
\quad K_\mu(t)=\frac{\mu(1-t)\sqrt{c/D_c(t)}}{D_u(t)},
\]

\[
\delta\log\operatorname{Var}(X_1)
=\int_0^1K_\Sigma(t)a(t)dt,
\quad K_\Sigma(t)=2(A_c-A_u).
\tag{7}
\]

Kμ 用条件基线的未来传播因子 √(c/Dc) 推出，不是只看即时速度。对 μ>0，

\[
\frac{K_\mu(t)}{-K_\Sigma(t)}
=\frac{\mu\sqrt{cD_c(t)}}{2t(u-c)}
\tag{8}
\]

随 t 严格下降，因为 Dc/t²=((1−t)/t)²+c 严格下降。

于是，最大化一阶均值增益、约束一阶方差不变、并加入正权二次控制代价 \(\lambda\int G(t)a(t)^2dt/2\) 的最优解具有

\[
a^*(t)=\frac{K_\mu(t)-\nu[-K_\Sigma(t)]}{\lambda G(t)}.
\tag{9}
\]

ν 由方差约束确定，在相应积分有限、解非退化时恰为 Eq. (8) 的加权均值。分母为正，故最优 schedule 只有一次正到负过零：早期主要推均值、晚期恢复方差。这是经典线性二次约束控制的直接计算，不是新算法。

**所以不能拿“先收缩再放松”直接解释晚期重新强正。** 真实 +−+ 若可靠，则意味着至少要加入这个最小模型未包含的因素：非 Gaussian 模式结构、S−W 的模型误差几何、终点奖励不只依赖均值方差、跨时间耦合约束、或离散误差。

独立 CPU 核验：c=.6、u=1.7、μ=.9、a(t)=.9−1.7t+.4sin(7t)。直接积分均值/covariance ODE 的终点方差为 .5999755992716，Eq. (6) 为 .5999755992704。将 schedule 乘 ±1e−4 做中心差分，均值一阶响应为 .2404585328797，Eq. (7) 为 .2404585328730；方差响应分别为 −.000024401226595 和 −.000024401225806。该核验未使用图像模型或 GPU。

## 7. 终点上抬的三个区分性预测

### 7.1 方向退化导致系数补偿

若 ||S−W|| 近终点按 ε 衰减而 a 按 1/ε 上升，则 ||a(S−W)|| 不应同步爆发。应画实际控制速度、相对速度范数，以及每个 Heun interval 对终点的敏感度，而不仅是 raw a。

若“猛抬”只存在于 raw a，而实际作用平滑，这是控制基底退化的证据。系数正号仍需要未来目标响应解释。

### 7.2 网格/边界效应

对标准 canonical CFG 且末步长度 h，若 Δ=O(1−t)，最后区间的单位 scale 终点响应通常为 O(h²)。若差由 O(1) 模型误差主导，则可能为 O(h)。保持一个集中在最后单个 cell 的有限冲量，会相应要求 scale 随网格细化放大；但这仅适用于“固定冲量集中到收缩 cell”的假设。

因此改 NFE 后：峰的位置若固定在物理 t/log-SNR，而形状平滑，倾向于真正的时间区域机制；若峰追随最后几格、幅值强烈随 h 改变，则更像离散或基底退化补偿。这个区分不需要将最终 Euler、输出 floor 单独当作跨模型共因。

### 7.3 正负段承担不同统计任务

如果负段真在补偿收缩，消掉它应降低类内变化量或扰动传播增益；如果末正段真在做质量修复，则消掉末正段应主要改变局部细节/终点误差，而非简单恢复方差。用相同 seeds 的分段 ablation、生成路径中注入小扰动后的响应、以及 feature covariance 区分这些预测。

“negative=a更接近weak”与“negative=提高diversity”不是等价陈述。应先验证 current S−W 对相应 feature 的收缩/扩张方向，才能把 CFG Gaussian 机制迁移过来。

## 8. 与现有研究路线的连接

最稳妥的理论对象是各 interval 的实际终点响应及其交叉相关，而不是仅凭 scale 曲线给阶段命名。连续时间有

\[
\frac{\delta J}{\delta a(t)}
=E[\nabla V_t(X_t)\cdot(S-W)(X_t,t)].
\]

它的符号取决于未来价值敏感度与当前控制方向的配合，没有来自 CFG 定义的一般符号限制。若后期需要恢复前期损伤，也可以出现符号反转；但 +−+ 必须由响应核或受限模型几何推出，而不是由“动力学三阶段”标签推出。

文献还提醒我们：确定性 CFG 的输出通常不是简单的 powered endpoint density。[Classifier-Free Guidance is a Predictor-Corrector](https://arxiv.org/abs/2408.09000) 已给出反例并区分 ODE/SDE；2026 年的路径积分与信息论 schedule 工作进一步直接追踪诱导分布。因而把正负 a 等同“降温/升温”只能是特定 Gaussian 例子中的有限类比。

值得追的结论是有条件的：**signed control 可能解耦语义/质量增益与过度收缩，但对 internal guidance，还需识别 weak direction 在时间上的几何角色。真实曲线的中段负窗和末端反弹可能恰好提供了这个辨识机会。**
