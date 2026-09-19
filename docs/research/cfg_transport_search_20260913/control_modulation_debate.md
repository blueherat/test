# CTRL 两周期：平均修正、有限步调制与最小反证实验

**保留一个备用构造：将 CTRL 在同一 gap 上的两次代数更新拆成平均项与正负调制项，再用现有两步 Heun 执行。它不增加模型查询，能形成“只保留平均／完整调制／反相调制”三个明确对照；目前没有证据认定调制是 CTRL 收益来源，也没有质量改进保证。**

新配对 1K、seed 2026091397 的已提交结果为：标准 CFG alpha=1.25 的 FID 44.910314；APG alpha=2 为 43.464164；CTRL alpha=2.75、lambda=5、K=.2 为 43.481451。三者均 224 full/0 extra prefix；APG/CTRL 的 sFID 分别 210.747747 / 208.712332，IS 分别 63.177006 / 62.198566。两个强对照的 FID 几乎相当，不能把它们之间 .017 的差值解释成可靠排序。这是继续辨识 CTRL 的理由，不是已经鉴定其内部机制。[原始新 1K](</home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/baseline_1k>)。

## 已有分析与新问题的边界

[Sep11 报告](/home/zhoushunyu/eqvae/docs/SIT_FSG_CTRL_HYPOTHESIS_RESULTS_20260911_ZH.md:83)已经写出小正 gap 的两周期、均值为原 gap，并测到第 1 步约 68.17% 坐标的历史修正符号不同于即时 sign(g)。因此“发现两周期”本身不能认领新意。旧库也已有 [Lie-bracket 分解](/home/zhoushunyu/eqvae/experiments/internal_guidance_path_extrapolation.py:867)、[平滑时钟补偿](/home/zhoushunyu/eqvae/docs/SIT_CLOCK_COMPENSATOR_PROTOCOL_20260908_ZH.md)，以及 [50 方向中的 #32 tanh 边界层输出反馈](/home/zhoushunyu/eqvae/docs/FSG_CONTROL_50_IDEAS_20260910_ZH.md:475)。后者修改完整未来输出的反馈，未完成的配置也不能算已经被结果排除。

相关源码/报告中未检到明确的 Sigma–Delta 累计误差控制、同 NFE 的 CTRL 两步平均/反相分解，或通过目标函数解调的 extremum seeking 实验。这是有限范围检索结果，不是全球文献新颖性结论。剩余问题很窄：**CTRL 的正信号是否需要交替分量，还是其分段平均修正已经足够？**

## 1. 必须先纠正“CTRL 是零均值调制”的说法

记条件/null 场为 c、u，gap 为 g=c−u，额外 guidance 为 alpha，w=1+alpha。仓库实际输出为

\[
v_{\mathrm{CTRL},n}=u_n+w m_n,
\qquad m_n=g_n-K\operatorname{sign}\{g_n+(\lambda-1)m_{n-1}\}.
\tag{1}
\]

首次历史取当前 g；每个 Heun 步的两 stage 共用旧历史，接受后提交左 stage 的修改 gap。式 (1) 是离散递推，不能自动当成某个连续滑模微分方程。

对单坐标恒定 g、lambda>1，令

\[
\theta=\frac{\lambda-1}{\lambda}K.
\]

写 `m_n=g−K*s_n`，则二值状态满足 `s_n=sign(lambda*g−(lambda−1)*K*s_(n−1))`。避开阈值等号及 sign(0) 的特例，可直接得到：

| 恒定 gap 区域 | 长期修改 gap | 长期平均 |
|---|---|---|
| 0<abs(g)<theta | g−K sign(g)、g+K sign(g) 交替 | g |
| abs(g)>theta | 固定 g−K sign(g) | g−K sign(g) |

lambda=5、K=.2 时 theta=.16。g=.05 的两周期为 −.15 / .25，平均 .05；g=.199 则固定 −.001，并非平均保留 .199。还存在 theta<g<K 的窄区间，正 gap 被持续反向。g=0 且历史初始化为 0 时固定在 0；阈值等号要单独处理，不能把浮点 tie 当作通常周期。

因此 CTRL 同时包含**大 gap 持续修正、小 gap 避免持续反转、周期相位、状态变化下的切换**。旧同 bank 的 Instant K=.2 FID122.9979、soft threshold111.9753、scalar norm62.9550 都比 CTRL44.7256 差，但它们没有保持上表的平均项；不能把这些失败直接归因于“去掉振荡”。更重要的替代解释是：它们误伤了大量小 gap，而 CTRL 的历史抵消避免了该问题。[旧完整对照](/home/zhoushunyu/eqvae/docs/SIT_FSG_CTRL_HYPOTHESIS_RESULTS_20260911_ZH.md:11)。

## 2. 相邻正负扰动留下什么：先是 Jacobian 响应，未必是曲率

令 H_f^h 是对光滑自治场 f 的一个 Heun 步。一个局部两步块先用 v+b，再用 v−b。由二阶 Taylor 展开，或标准流组合的 BCH 展开，可得[^split]

\[
H_{v-b}^h\circ H_{v+b}^h(x)
-H_v^h\circ H_v^h(x)
=h^2\{J_v b-J_b v\}(x)+O(h^3).
\tag{2}
\]

这里规定 `[b,v]=J_v b−J_b v`，避免不同文献的 bracket 符号口径混淆。两边都用相同 Heun，因此这个领先项不是仅仅相减了两套不同精度的求解器误差；精确光滑流也有同阶项。非自治且 b 随物理时间变化时，右侧再有 `−h²*partial_t b`。若把一个两步块的 b 真正冻结为常向量，便简化为

\[
\Delta_+=h^2J_v b+O(h^3),\qquad
\Delta_-=-h^2J_v b+O(h^3).
\tag{3}
\]

这带来四个限制：

- **线性场已经有作用。** v(x)=Ax+c 的 Hessian 为 0，式 (3) 仍有 A b。因此“曲率激发”不是领先阶的准确名称。将正/反相两条局部结果平均后，首个相位奇项抵消，剩余才从更高阶考察；不能只观察终点变化就断言是 Hessian 效应。
- **零平均输入不代表零终点作用。** 不同时间的扰动会经过不同的后续 Jacobian 运输。一般终点一阶响应为 `integral P(T,t)b(t)dt`，P 是后续状态敏感度；即使 `integral b(t)dt=0`，加权结果也未必为 0。
- **fixed K 与高频极限不同。** 每块长 2h，式 (3) 对应的每单位时间修正是 O(h)。在光滑有界、固定 K 和固定总时间条件下加密，效应趋于 0。把 K 随 1/h 或其他幂次放大以保留效应会改变算法与预算，不能继承当前 CTRL 的成功。
- **没有下降方向保证。** 对一个外部终点损失 L，领先变化取决于 `grad L` 与运输后的 `J_v b` 的内积，符号未知。翻相可翻转领先位移，却不能事先说明哪一相图像更好。

最短严格反例：真实基础场 v(x)=x、x0=0，精确推进 h 时加 +b、再精确推进 h 时加 −b，则

\[
x_{2h}=b(e^h-1)^2;
\tag{4}
\]

纯基础流的正确终点为 0。正/反相都产生非零误差，没有 Hessian、没有离散求解误差。另一个极端 v(x)=常量 时，两次常量正负扰动完全抵消。因此“振动天然改善”“零均值天然无害”都不成立。

式 (2) 还要求两条光滑场在块内成立。原 CTRL 会在每次 RHS 对新的 g 判断 sign，并更新隐藏状态；跨 relay 切换面时，不能直接把它套成同一光滑 b(x)。

## 3. 三类原始文献能借什么，不能借什么

**振动控制与 extremum seeking。** Bullo 的机械系统平均化定理研究特定二阶机械结构，以及随高频参数一起增大的 forcing；其有效力与控制场的 symmetric product 有关，稳定性还需要额外矩阵/势条件。它不证明固定 K 的 SiT 一阶速度场会被振荡改善。[^bullo] Dürr 等的基本极值搜索例子为 `xdot=alpha*sqrt(omega)*cos(omega*t)+f(x)*sqrt(omega)*sin(omega*t)`，相位配合与含目标 f 的反馈给出约化梯度方向；CTRL 没有这个 f 测量/解调结构，不能把任意 sign 振荡称作质量 extremum seeking。[^durr]

**Sigma–Delta。** Daubechies–DeVore 第 2 节式 (8) 的一阶递推是 `r_n=r_(n−1)+f_n−q_n`，`q_n=sign(r_(n−1)+f_n)`。因此 `sum(q_n−f_n)=r_0−r_N`，有界内部误差给出平均重构控制；进一步的频带重构保证还需要信号/重构核假设。[^sigma] CTRL 的历史是修改后的 g，而不是累计的量化误差；大 gap 的累计差甚至线性增长 `sum(m_n−g)=−N*K*sign(g)`。所以没有依据直接给它贴“Sigma–Delta noise shaping”的机制标签。

**数值抖振。** Acary–Brogliato 证明/展示某些滑模系统的隐式 Euler 可消除显式离散的寄生抖振，同时保持其滑模目标。[^acary] 这里提供的反向提醒是：周期可能属于离散实现，而非期望连续控制本身。它也不能推出“去掉 CTRL 周期必然更好”，因为 SiT 的真实质量目标与那些滑模面不同。简单把 sign 换 tanh 或 implicit/prox 并未保留上述分段平均，且边界层本身已有充分先例，不作为本轮独立新候选。

**与 APG 互补的反方核查。** 既有 APG 的线性动量部分为 `M_n=d_n−.5*M_(n−1)`，忽略 cap 和时变投影时，常量输入增益是 2/3，交替输入增益是 2，AC/DC 比放大 3 倍。因此把 CTRL 修改 gap 直接送进 APG 不能称为低通平滑；后续 cap/投影还可能把交替分量整流成平均分量。由此只得到“不应在三臂机制检查里同时叠加 APG”的结论，未得到 APG/CTRL 必然互补或互相破坏的质量结论。

## 4. 唯一备用构造：固定两步的均值与调制分量

使用已保留的 alpha=2.75、lambda=5、K=.2、Heun64、left cutoff=.75。基础场始终是 **`v=c+alpha*g`**，不是 conditional 单支。设每个两步块开始的状态/时间为 `(x_j,t_j)`，只用原本第一 RHS 所需的 c/u 查询得到 g_j。

定义纯代数控制器 `C(g,p)=g−K sign(g+(lambda−1)p)`。第一次 p=g_j；之后使用上一块储存的虚拟第二次结果。做两次代数迭代：

\[
m_j^{(0)}=C(g_j,p_j),\qquad
m_j^{(1)}=C(g_j,m_j^{(0)}),
\]
\[
\mu_j=\tfrac12(m_j^{(0)}+m_j^{(1)})-g_j,\qquad
a_j=\tfrac12(m_j^{(0)}-m_j^{(1)}).
\tag{5}
\]

**只冻结控制分量 mu_j、a_j，不冻结图像状态、真实时间或 conditional/null 模型预测。** 两个普通 Heun 步分别使用

\[
v_{j,0}(x,t)=v(x,t)+w(\mu_j+\rho a_j),
\qquad
v_{j,1}(x,t)=v(x,t)+w(\mu_j-\rho a_j).
\tag{6}
\]

每个步内两个 Heun stage 使用同一个加性向量；四次 RHS 都按各自真实 x/t 查询 c/u。块结束后将 **`p_(j+1)=m_j^(1)`** 保存为虚拟控制历史，三个臂使用相同规则；不把实际第二步输出作为 p，以免 rho 直接改写历史定义。这不是原 CTRL 在时变 gap 上的逐项等价实现，应使用新名称。

| 最小三个臂 | rho | 作用 |
|---|---:|---|
| mean_only | 0 | 同一虚拟递推的平均修正，只去掉交替分量 |
| mean_plus_ac | +1 | 原次序执行完整两次代数修正；唯一主候选 |
| mean_reversed_ac | −1 | 保留同一局部平均、振幅和能量，交换交替相位 |

固定两步块的加性控制积分都是 `2h*w*mu`。rho=±1 还具有相同局部能量 `2h*w²*(||mu||²+||a||²)`；rho=0 的能量更低，因此“完整优于 mean_only”本身仍不能唯一归于 bracket，而非额外控制能量或高阶响应。必须同时看反相。

在恒定 gap、各 stage 无切换时，rho=+1 完整保留原 CTRL 的持续修正与两周期；大 gap 会自动给出 a=0，小 gap 会给出 mu=0。它比在每次 RHS 对反馈不加区分地解释 sign 更容易审查，但没有消除 sign，也没有声称更平滑或更稳定。

**成本与初始化必须固定。** 64 步的 48 个活跃区间恰好组成 24 个两步块，96 步则是 72 区间/36 块；不能留下末端半块。截止后恢复 conditional，并停止控制历史更新。每次代数迭代不查询模型，64 步仍 224 full/0 prefix；不额外生成路径。零额外 guidance 按既有接口直接返回 conditional。新实现须置于独立文件/阶段，不能改冻结 baselines.py 或 runner.py。

### 主要混杂与不能省略的反证

1. **历史已经改了。** 原 CTRL 的历史来自实际第二步左 stage 的 gap；这里是同一块起点 g 上的虚拟 m^(1)。恒定 gap 的等价不能证明时变等价，更不能把该候选的质量差全部归于振荡。
2. **慢变本身不够。** CPU 反例：g0=.159999、下一步 g1=.160010，仅变化 .000011；原 CTRL 下一 m=−.03999，固定两步构造为 .36001，相差 **.4=2K**。原因是接近 relay 切换面；必须测到切换裕量及每步符号翻转，不能仅报平均 gap 变化小。
3. **完整轨迹的 mu 并不严格相同。** 即使三个臂的虚拟历史规则完全相同，它们走到的 x 不同，下个块的 g、mu 也会不同。因此三个独立生成臂是算法级消融，不能称为严格固定整条控制序列的因果实验。额外的共同起点局部两步检查可以区分这件事，不需要先接受全局机制解释。
4. **相位会与模型时间相关。** 只测一个固定步数/块起点，可能把有效相位与 particular grid 的误差混在一起。首轮不展开新调参；若有明显质量正信号，再以已有较高步数强对照检查，不可一开始改 K 随 h 缩放来追拟合。

## 5. 已完成的 CPU 证伪与实际 1K 的最低判据

[CPU 结果](control_modulation_cpu.json)使用 NumPy FP64 的原始递推、仿射场与二维非线性场，无 GPU、无学习或图像质量推断：

- 恒 gap 的完整分段均值已逐点核对；不是只测试小 gap。
- 正负两步相对普通 Heun 两步的领先项符合式 (3)。h=1/512 时仿射/非线性场的相对展开误差分别约 `.0834% / .1273%`；仿射场正反相平均后的差在 `5.6e-17`，非线性相位偶项为 O(h³)。
- 固定总时间、固定扰动振幅，把步数从 32 依次加倍到 512，终点作用量依次约减半；相邻比为 1.999961、1.999990、1.999998、1.999999。这验证“本组固定振幅平均效应随 h 消失”，不是新的高频稳定化定理。
- 恒 gap 时完整两步构造与原递推一致；上面的阈值反例说明这种等价不均匀地延续到时变 gap。
- 式 (4) 的精确仿射反例在 h=1/64、b=1 得到非零终点 .0002479903；没有基础模型错误可被它纠正。

**下一项有意义的工作是固定三臂各 1K 真实图像，而非再扩大 toy 或先写质量结论。** 最小比较必须同时保留新 bank 已跑过的原 CTRL、APG、标准 CFG；只有三臂都用相同模型、bank、labels、64 Heun、alpha/lambda/K/window 与真实调用数，才可判定是否值得独立 5K。这三臂本身不调宽度、频率、lambda、K、相位偏移或归一化，也不先与 APG 叠加以免增加解释维度。

预先约定读法：

- 若 mean_only 已接近或超过完整臂，**不支持交替分量是必要收益来源**；优先考虑分段平均/状态切换解释。
- 若完整和反相都优于 mean_only，支持交替分量或其能量有用，但**不支持特定带符号的一阶 bracket 是唯一机制**。
- 若完整优于反相且共同起点的局部差符合预测相位，才得到有限步调制的方向性证据；仍必须赢原 CTRL/APG，不能只赢新造的弱对照。
- 若完整构造输原 CTRL，说明两步冻结/虚拟历史没有保住有效行为；不应靠重搜 K、再换边界层或混 APG 来把此次机制失败掩盖掉。
- 三者都没有超过原 CTRL/APG 时，直接结束这个备用构造；不把公式清楚当成应继续放大预算的理由。若胜出，固定配置进入新的 5K，报告 FID/sFID/IS、同卡成本及均值/相位诊断，再讨论论文级贡献。

## 实现与检查记录

私有实现为 [control_modulation.py](/home/zhoushunyu/eqvae/experiments/cfg_transport_search_20260913/control_modulation.py)，冻结 SHA256 `a2cda5038333b401db345f46e4816dd243b4b1d5b45303653f078ad5cd10dcf8`；未修改已有 baselines.py、runner.py 或历史实验。接口为 `sample(rt,noise,labels,config,snapshots=False)`，config 使用 `kind='ctrl_modulation'`、`rho=0/+1/-1`、`alpha=2.75`、`lambda_ctrl=5`、`K=.2`、`steps=64`、`cutoff=.75`；返回 latents、full/prefix counts 及可选五个状态快照/逐步读数，SOURCE_FILES 声明运行依赖。

与数学 toy 分开的 [实现检查](control_modulation_implementation_check.json)已完成：FP32/FP64、恒 gap 的三个臂终点均与原 CTRL 相符（最大浮点差分别约 3.73e−8 / 1.11e−16）；时变、状态依赖假模型中 K=0 的整条轨迹逐元素回到普通 CFG，alpha=0 逐元素回到 conditional；64/96 步实际计数、局部均值/能量、五快照及末尾半块拒绝均通过。独立只读源码复核未发现必须修复项。这里没有运行新 GPU 或提供三臂 FID；真实 8 图与固定 1K 由主实验阶段另行执行，不能用这些 CPU 身份代替质量结果。

## 文献

[^split]: Sergio Blanes, Fernando Casas, Ander Murua. *Splitting and composition methods in the numerical integration of differential equations*. arXiv:0812.0377, 2008，第 3.1 节 BCH 组合展开。[原文](https://arxiv.org/pdf/0812.0377)。式 (2)–(3) 是在本问题约定下的独立二阶展开，不是声称论文讨论了 CFG。

[^bullo]: Francesco Bullo. *Averaging and Vibrational Control of Mechanical Systems*. SIAM J. Control Optim. 41(2), 542–562, 2002，第 4 节，尤其 (4.1)、Theorem 4.1。作者托管版本标注修订 2003-02-14。[作者 PDF](https://motion.me.ucsb.edu/pdf/1999b-b.pdf)，[期刊](https://epubs.siam.org/doi/10.1137/S0363012999364176)。

[^durr]: Hans-Bernd Dürr, Miloš S. Stanković, Christian Ebenbauer, Karl H. Johansson. *Lie Bracket Approximation of Extremum Seeking Systems*. Automatica 49(6), 1538–1552, 2013，第 2 节 (1)–(3)。[作者 PDF](https://people.kth.se/~kallej/papers/extr_auto13durr.pdf)，[原始预印本](https://arxiv.org/abs/1109.6129)。

[^sigma]: Ingrid Daubechies, Ron DeVore. *Approximating a bandlimited function using very coarsely quantized data: A family of stable sigma-delta modulators of arbitrary order*. Annals of Mathematics 158(2), 679–710, 2003，第 2 节式 (8)、Lemma 2.1。[作者 PDF](https://sites.math.duke.edu/~ingrid/publications/annals-v158-n2-p09.pdf)，[期刊页](https://annals.math.princeton.edu/2003/158-2/p09)。

[^acary]: Vincent Acary, Bernard Brogliato. *Implicit Euler numerical scheme and chattering-free implementation of sliding mode systems*. Systems & Control Letters 59(5), 284–293, 2010。[作者 PDF](https://tripop.inrialpes.fr/people/acary/publications/Acary.Brogliato_SCL2010.pdf)，[期刊页](https://www.sciencedirect.com/science/article/pii/S0167691110000332)。
