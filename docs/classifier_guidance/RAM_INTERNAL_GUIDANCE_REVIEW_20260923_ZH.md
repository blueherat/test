# RAM 与 internal self-guidance：有价值的计算替代，尚非等价的优化替代

2026-09-23。针对用户贴文的独立复核，结合仓库当前实现、RAM 原论文与官方代码，以及本轮解析推导和 CPU 核验。未修改正式训练，未启动 GPU 实验。

后续理论推进见[局部密度响应与概率通量保持](LOCAL_RESPONSE_AND_FLUX_THEORY_20260923_ZH.md)：在实际 Heun 上用局部反事实 Riesz 回归构造双稳健反馈，以及 normalized ITM 如何在特定桥一致条件下运输保分布流。后续同时补充了 Tilt Matching、Newton Matching 和自动 Riesz 回归的直接文献近邻；因此本文所识别的问题不能单独作为新颖性声明。

**这篇分析值得吸收，并改变候选方法的优先级：在实现一个学习未来敏感度的 critic 之前，应先评估 RAM 提供的便宜局部反馈。但“可以回归当前 composite field”与“等价优化当前确定性终点目标”是两件事。最适合接续上一轮的方案，是把 RAM 梯度作为预测，以少量现有精确反传做随机校正。**

上一轮 RL 笔记没有充分覆盖 RAM，这是文献覆盖的缺口。本篇补上，并保留此前关于完整离散反传和随机校正的结论。

## 1. 贴文真正推进了什么

当前场是

\[
v_\theta(x,t,c)=S(x,t,c)+a_\theta(t,c)\{S(x,t,c)-W_\theta(x,t,c)\}.
\]

在当前 SiT native weak head 设置中，strong backbone 冻结，训练 weak 与分段 scale。局部状态一旦固定，就可以将 strong 特征 detach，只反传 weak head 和 scale。RAM 因而与这个架构有实际的计算互补性：它把终点奖励变成局部场回归所用的数值，不要求穿过整个生成轨迹，也不要求奖励网络的输入梯度。

这也不等于“把 weak 重新训练成普通 FM 模型”。RAM 拟合的对象是整个 v，weak 只承担其中的可训练修正。不能因为损失形式是平方误差，就断言它回到了 standalone weak FM。

另一方面，当前实现已经通过逐场保存 state、逆序重算与 VJP，避免同时保存全部 backbone 激活。SiT 的 64 Heun interval 仍需 128 次采样场前向、128 次反向重算场前向和 128 次场 VJP；RAM 主要有望省掉后两者，而不是首次解决整条激活图常驻的问题。细节见[现有反传审计](../research/self_guidance_rl_20260923/current_backward_cost.md)。

贴文中“原 RAM 训练整个大模型”也需修正：官方实现冻结基座，训练分布在 Transformer 层中的 LoRA。它仍需要经过相关大模型层的局部反传；我们的独立输出头可以更便宜，但重新加噪状态上的 frozen backbone 前向仍要支付。[官方训练代码](https://github.com/AndreasBergmeister/ram/blob/main/scripts/training_sd3.py)

论文报告的最高约 50 倍是其任务上达到 Flow-GRPO 指标所需更新步数的比较，不能直接转换成当前仓库的预期加速。其每次更新使用的终点数、局部训练状态数及奖励任务均不同。[作者项目页](https://bergmeister.ai/ram/)

## 2. 精确写出适配后的更新

本节及下文除特别说明外采用仓库方向：t=0 为噪声，t=1 为数据。省略类别条件。

从实际 sampler 得到终点 X，另取独立 Gaussian Z：

\[
X\sim q_\theta,\qquad Y_t=(1-t)Z+tX,\qquad U=X-Z.
\]

令 R 是数值奖励，强场 S 为 reference。RAM 式局部损失为

\[
L_{\rm RAM}
=\frac12\,\mathbb E\left\|
v_\theta(Y_t,t)-
\operatorname{sg}\left[S(Y_t,t)+R(X)\{U-v_\theta(Y_t,t)\}\right]
\right\|^2.
\tag{1}
\]

终点采样、重加噪输入、奖励和整个 target 均停止梯度；只对 leading v 求导。因此以下所谓梯度是这种局部半梯度，不包含分布 qθ 随参数变化的导数。

Eq. (1) 是将原论文 Eq. 17 统一时间方向后的形式。官方训练还使用 old/EMA policy 采样和构造 target，奖励做 group centering、pooled std 归一化并乘系数。分析当前 θ 的简化版与实际 old/current 版本时需分开；后者还有滞后差异。[原始方法](https://arxiv.org/html/2605.10759v1#S4)、[实现](https://github.com/AndreasBergmeister/ram/blob/main/scripts/training_sd3.py)

若 a 是当前 64 个 interval 参数，随机 t 上必须定义同一分段映射。把它改为新连续网络会改变可训练函数族，不能当作纯损失替换。JiT 若仍冻结 weak，则局部训练只涉及 schedule。

## 3. RAM 原理论不能自动继承到当前终点训练

需区分以下三层：

1. 对某个指定随机扩散参考过程，带路径控制成本的最优控制问题。
2. 在其解析加噪桥和 memoryless 条件下，相应的终点 KL 指数倾斜。
3. 当前受限 guided ODE，用固定 Heun 求解器优化 non-saturating GAN 终点损失。

第一与第二层有条件下的等价关系。第三层没有因此自动等价。特别是论文里的归一化 SDE 控制满足 σu=2(v−vref)，它不是贴文直接定义的速度残差 v−S；一般 ODE 的平方速度能量也不能直接改名为终点 KL。

RAM 的基本形式进一步省略了路径成本梯度，采用终点重加噪构造联合样本，并以网络当前速度替代该重加噪分布的 canonical velocity。原文 §4.1 将其固定点与准确 KL optimum 的差异解释为路径积分的求积近似，表 1 也明确标注基本 RAM 有偏。[论文 §4–5](https://arxiv.org/html/2605.10759v1#S4)

独立核验可以把这个区别具体化。此段暂用论文方向：数据 t=0，噪声 t=1。取

\[
p_{\rm ref}=\mathcal N(0,1),\qquad R(x)=-x^2/2.
\]

准确的终点 KL 最优分布为 q*=N(0,1/2)。在 t=1/2、状态 x=1：

\[
v_{\rm ref}=0,\qquad v^*=2/3,\qquad
\mathbb E[R(X)\{(\epsilon-X)-v^*\}\mid X_t=1]=4/9.
\]

因此准确最优场并不是基本 RAM 的固定点，残差为 2/9。这里不存在容量不足、采样轨迹不一致或有限 batch 噪声。

原因很直观。对于指数倾斜族 qλ∝pref exp(λR)，其 canonical velocity 满足

\[
\partial_\lambda m_\lambda
=\operatorname{Cov}_{q_\lambda}(R,\epsilon-X\mid X_t).
\]

准确的 m1−m0 是这个量从 0 到 1 的积分；RAM 固定点使用 λ=1 的值。线性近似良好时可能有效，但两者通常不相等。CPU Gaussian 求积分别得到 2/3 和 4/9，并复核积分确为 2/3。

这个反例验证的是原论文明确保留的近似，而非声称原论文定理被推翻。

## 4. 对我们的 weak head 更关键的是 velocity 自洽性

回到噪声到数据方向。定义自身终点重加噪所对应的 canonical velocity：

\[
m_\theta(y,t)=\mathbb E[U\mid Y_t=y],\qquad
d_\theta(y,t)=v_\theta(y,t)-m_\theta(y,t).
\]

一个场能生成 qθ，不代表它等于 mθ。甚至所有时刻的密度都一致，仍可能存在保分布旋转等速度差异。

令 u=v−S，μR=E[R|Yt]，CR=Cov(R,U|Yt)。对 Eq. (1) 的残差取条件期望，准确得到

\[
\boxed{
H_R=u-C_R+\mu_Rd_\theta,\qquad
g_{\rm RAM}=\mathbb E[(\partial_\theta v_\theta)^\top H_R].
}
\tag{2}
\]

三部分分别是 reference anchoring、reward 与速度 target 的条件协方差、以及 reward 均值乘上速度自洽性缺陷。

最后一项是迁移到当前方法时值得专门研究的对象：我们的 weak head 由终点表现训练，并没有保证 composite v 成为其生成分布的 canonical FM velocity。

因此奖励增加常数 C 时，有

\[
\boxed{
g_{\rm RAM}(R+C)-g_{\rm RAM}(R)
=C\,k_\theta,\qquad
k_\theta=\mathbb E[(\partial_\theta v_\theta)^\top d_\theta].
}
\tag{3}
\]

真实终点奖励或终点奖励减 KL 的梯度对这个常数不变。RAM 的局部半梯度则未必如此。

而 kθ 不需要另训练 score teacher：

\[
k_\theta
=\mathbb E[(\partial_\theta v_\theta)^\top(v_\theta-U)].
\tag{4}
\]

右边就是用自身生成终点做 stop-sampling FM 所得的局部参数梯度。因此可以直接测量奖励平移引入的更新方向，检查它是否恰好落在现有 weak+scale 的可训练切空间内。

这不说明所有 reward centering 都不好，而是说明它在此处可能兼具改变更新原则的作用，不能只解释为 REINFORCE baseline 降方差。对同一条件下 N 个 iid 终点，使用包含自身的 group mean 还会给 reward 项带来 1−1/N 系数；leave-one-out 能消掉该有限样本系数，但不消除 Eq. (3) 的自洽性问题。随机 std 归一化也有额外影响。

这些推导针对 Eq. (1)。实际 old/EMA target 版本对应 old 场的缺陷及 current Jacobian，不能不加说明地照搬 current/current 等式。

## 5. 两个解析例子：从“不同代表”到“受限优化冲突”

令二维 Gaussian 的 canonical 场为

\[
s(t)=(1-t)^2+t^2,\quad b(t)=\frac{2t-1}{s(t)},\quad S(x,t)=b(t)x,
\quad J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
\]

### 5.1 所有密度都正确，RAM 仍可因奖励常数而改变方向

取

\[
v_\omega=S+\omega Jx.
\]

其流为 √s(t) exp(ωtJ)z。因此所有 ω 都有相同的完整边缘密度族 N(0,s(t)I)，终点均为 N(0,I)。但 vω−m=ωJx。

均匀时间采样、常数 reward R≡C 时，半平方 RAM 梯度为

\[
g_\omega=\frac43(1+C)\omega.
\]

取 ω=.4，C=0、1、−1、−2，梯度分别为 .533333、1.066667、0、−.533333；真实终点奖励和终点 KL 的梯度全部为零。

这说明 RAM 除了改变终点分布，还可能选择一个特定的流。不能仅检查两种状态分布是否接近，就排除这种误差。

这里的保分布自由度，与此前 a 和 S−W 互相缩放、保持 v 本身不变的参数 gauge 不同。前者改变 v 但保留密度；后者连 v 都不改变。

### 5.2 当容量受限，选择流的代表可能损伤终点

考虑只含一个参数、且包含 reference 的族：

\[
v_\theta=[b(t)+\theta(\theta-1)]x+\omega\theta Jx.
\]

终点为 N(0,e^{2h(θ)}I)，h(θ)=θ(θ−1)。θ=0 和 θ=1 都生成准确 reference，是常数奖励加终点 KL 目标的全局最优；但 θ=1 处带有旋转。

在 θ=1，RAM 梯度为

\[
g_\theta=\frac43(1+C)\omega^2.
\]

取 C=0、ω=.7、学习率 .1：θ 从 1 变为 .9346667，终点 KL 从 0 升至 .00716328。

局部回归试图消除旋转，但这个受限族把旋转和终点缩放绑在同一个参数上，因而离开了终点最优。取 a=1、Wθ=2S−vθ，就能把这个例子嵌入 internal-guidance 形式。

这不是当前图像模型一定失败的证据，也不反驳原始路径控制目标对旋转的惩罚。它说明一个确实存在的理论障碍：**在受限函数族内，先选择 canonical 场再做局部投影，不保证等于直接优化终点。** 多个时刻共享 weak 参数时，时间权重也改变这种投影的折衷。

## 6. 固定判别器建议有理论价值，但要说清楚校正什么

设 base 分布 b，真实数据 d，判别器只看固定特征 Φ(x)。理想 logistic 判别器的 logit 是

\[
R(x)=\log\frac{d_\Phi(\Phi(x))}{b_\Phi(\Phi(x))}.
\]

以下假设数据特征分布被 base 特征分布覆盖，即 dΦ≪bΦ，且相应权重可归一化、所用 KL 有限；否则不能保证恢复全部数据特征边缘。

若准确优化 E_q R−β KL(q∥b)，则

\[
q_\beta(x)\propto b(x)
\left[\frac{d_\Phi(\Phi(x))}{b_\Phi(\Phi(x))}\right]^{1/\beta}.
\]

在 β=1 时，

\[
\boxed{q^*(dx)=d_\Phi(dz)\,b(dx\mid z),\quad z=\Phi(x).}
\tag{5}
\]

它校正特征边缘分布，保留 base 在同一特征纤维内的条件分布。这是明确的受约束 KL 投影，但不保证恢复像素空间的全部数据分布。DRL 原文也明确讨论了 feature-space 密度比的限制，不能把这一点当作我们新发现的问题。[DRL §4](https://arxiv.org/html/2606.19162v1#S4)

独立的 2×2 概率表核验中，特征边缘完全匹配后，完整 KL(d∥q*) 仍为 .9809945；它恰等于数据特征边缘加权的 base/data 条件 KL。

对当前项目还需区分 D logit 与 −softplus(−D)。后者才是现有 non-saturating generator loss 的负值；二者虽对单张图保持排序，期望目标并不相同。要用 Eq. (5)，就已经更换了奖励与正则目标。

“跑远了再刷新 D”还有一个容易忽略的问题。若新的 Dk 学的是 log(d/qk)，而 KL reference 始终固定为 b，理想最优响应是

\[
q_{k+1}\propto b(d/q_k)^{1/\beta}.
\]

其自洽固定点为 b^{β/(β+1)}d^{1/(β+1)}；β=1 从 b 出发，精确响应甚至在 b→d→b 之间循环。

所以刷新必须约定：继续学习 data/base 密度比，或将 KL reference 也更新为 current 做近端改进。后者可写 qk+1∝qk(d/qk)^η。这不是禁止更新判别器，而是要求 reward 的分母与所声明的 reference 一致。

## 7. 两条都值得保留、但保证不同的路线

### 7.1 直接 RAM：接受新 surrogate，以终点质量和时间评价

它的优点是简单、无需 critic、无需可微奖励，也不必给确定性 sampler 额外添加 SDE 噪声。

实现上可以保留 current guidance 参数化：no-grad 采样，计算数值奖励，端点重新加噪，strong 前向提取特征并 detach，仅回归 weak+scale。reference、奖励尺度、时间分布和 old/current policy 是显式方法选择。

这条路线可能是最佳工程方案，即使它不是当前目标的精确梯度。需要证据的是相同墙钟时间与采样预算下的质量，而不是仅证明公式能够代入。

现有 a=.75、W≠S。如果 reference 取 S，初始化不是零控制；若改为当前 guided 作为 reference，则初始控制归零，但新 reference 未必 canonical。直接把 a 设零又会暂时消掉 weak 的局部梯度。这些选择都应承认，而不是套用零控制初始化论证。

### 7.2 RAM 作为便宜预测，随机精确反传作为校正

这是本轮相对上一轮最直接的推进：先不学习 critic，用 RAM 提供廉价向量。

固定当前参数 θ、固定本次更新的 D、同一实际采样 batch B，记现有完整反传的生成器梯度为 gB。用该 batch 的端点和独立重加噪 ξ 得到便宜预测

\[
\widetilde g_{B,\xi}=\alpha\,g_{{\rm RAM},B,\xi}.
\]

α 可由之前的精确探针拟合，也可分 weak/scale 两块；它只用于对齐数值尺度，并没有把 RAM 变成准确梯度。随后独立抽取 I∼Bernoulli(p)，p>0：

\[
\boxed{
\widehat g=\widetilde g+\frac Ip(g_B-\widetilde g).
}
\tag{6}
\]

只有 I=1 才运行完整反向。条件于 B、ξ、θ、D 及已经固定的便宜预测，

\[
\mathbb E_I[\widehat g]=g_B,\qquad
\mathbb E_I\|\widehat g-g_B\|^2
=\frac{1-p}{p}\|g_B-\widetilde g\|^2.
\tag{7}
\]

这是代数恒等式，不需要 canonical velocity、reward baseline 合法性、RAM 固定点准确性，也不要求便宜预测本身是某个全局目标的梯度。它保留的是当前固定 D 下真实离散 sampler 的生成器梯度。

这条混合路线利用了当前奖励网络可微、精确反传已存在的条件。对于没有可用精确梯度的黑盒奖励，不能直接执行此校正；直接 RAM 的适用范围在这一点上更广。

反过来，它也不自动保证更快收敛：若 RAM 与真实梯度不一致，校正方差会很大。至少应比较完全不用代理的预测 g̃=0；实际需求是缩小梯度残差，而非让 RAM regression loss 更低。

在同一个原目标已趋于驻点时，RAM 还可能有非零代表选择梯度，因此此时应降低 α 或提高 p。否则“准确梯度很小、便宜预测很大”会造成额外噪声。

实施顺序有几个必要细节：

- g̃ 必须在抽取 I 之前固定；本次精确标签可以用于以后更新 α 或代理，不能只在 I=1 时重定义本次 g̃。
- I=1 上的 gB 必须对应当前 θ、当前 D 和同一批实际轨迹；若端点由 old policy 生成，需重新定义基准，不能声称校正了 current on-policy 梯度。
- p 可以依赖抽取前的历史/便宜信息，但校正必须使用对应概率。周期性“偶尔做一次精确更新”不具有 Eq. (7) 的逐步无偏性。
- 当前 norm probe 等便宜原目标项可直接准确计算并另外相加。原有 batch log-gap penalty 不是已证明的纯 gauge fixing。
- 梯度无偏不等于经过 clipping、Adam 或有限学习率以后，参数步和最终模型也与完整反传训练完全相同。

若无梯度采样和奖励前向成本为 F，完整终点及轨迹反向增量为 B，RAM 局部成本为 CR，则每更新期望成本约为

\[
F+C_R+pB,\qquad\text{完整反传基准为 }F+B.
\]

即使 p 很小，F 也不会消失；CR 还包括额外重加噪状态的 backbone 前向。需要用实际测量决定能否获益。

这一方法是经典控制变量/随机差值校正思想在当前问题中的具体使用，不应仅凭 Eq. (6) 宣称新的 RL 原理。研究价值在于找到特别适合 internal guidance 的便宜预测，并证明或实证解释它为什么能以小残差代替大部分未来反传。

## 8. 进一步的两个有用判断

**多次重加噪不等于多次独立终点探索。** 设 M 个 iid 终点，每个 K 个条件独立局部状态，单个局部梯度 G 的条件均值为 m(X)。对协方差的迹，有

\[
\operatorname{trVar}(\overline G)
=\frac1M\operatorname{trVar}_X m(X)
+\frac1{MK}\mathbb E_X\operatorname{trVar}(G\mid X).
\tag{8}
\]

增大 K 只减少第二项。若奖励差异和 endpoint 差异主导，继续增大 K 不能替代更多终点。若每终点成本 F、每局部状态成本 C，两个方差分量分别为 A、B，在这些量为正的固定计算预算简化模型下，连续最优 K 为 √(FB/(CA))，而不是一律越多越好。实际 K 至少为 1，并需取整数；group normalization、batch 共享会使 iid 分解需要修正。

**指导强度通常不能精确解释成 KL 温度。** 简单 Gaussian 例子即可检验：reference N(0,1)，单位二次 reward 的 KL optimum N(0,1/2)。将两者 canonical velocity 线性外推，权重 w 得到终点方差 2^(−w)；把 reward 放大 w 倍，准确 KL tilt 的方差却是 1/(1+w)。w=2 时分别为 1/4 和 1/3。RLG 的推断时场组合有实际价值，但不能直接据此将我们任意弱头的 scale 识别为准确 KL 温度。[RLG 原文](https://arxiv.org/html/2508.21016v1)

## 9. 应该怎样推进，而不把方法比较弄混

第一步应是复用少量已有完整反传能力，对同一 θ、D、终点 batch 比较 RAM 局部预测与精确梯度。除了 cosine，要测相对残差、weak/scale 两块的尺度，以及 Eq. (4) 的 reward-offset 敏感方向。仅有正 cosine 不保证随机校正的方差足够小。

随后有三种清晰的比较对象：原有完整反传；直接 RAM surrogate；RAM 预测加随机精确校正。比较保持 head 容量、实际 sampler 和评估流程一致，并报告终点数、field forward/VJP 数与墙钟时间。此处是研究设计，尚未启动这些图像实验。

若另做“固定 data/base D logit＋KL”任务，应单独命名目标并配对应基准，不能用原 non-saturating GAN 的梯度作为新目标真值。可以研究，但这是目标改变。

还可用理想指数倾斜的 weighted FM 帮助判断机制：

\[
\mathbb E_{X\sim b,Z,t}
\left[e^{R(X)/\beta}\|v_\theta((1-t)Z+tX,t)-(X-Z)\|^2\right].
\]

无限容量下其最优 canonical 场对应 b exp(R/β)，但实际受权重退化、base 覆盖和受限投影影响。若奖励是精确像素 log(data/base) 且 β=1，它退化为 composite field 上的数据 FM。仓库此前已有固定 scale 的 guided_weak composite FM 路线，不能将这一点作为新 idea 重复提出。

真正值得发展的理论问题是：**能否用自身端点重加噪构造的廉价方向，解释并预测当前小容量控制空间中的终点梯度，从而只为其残差支付完整动力学反传？** RAM 给出了非常简单的首个候选；速度自洽性缺陷、分布自由度与受限投影则给出了它可能失效的具体机制。

## 10. 本轮可复核材料

- [RAM 原文与官方实现审计](../research/self_guidance_ram_20260923/ram_primary_audit.md)。
- [受限回归、baseline 与 gauge 完整推导](../research/self_guidance_ram_20260923/restricted_theory.md)。
- [CPU 审计脚本](../../experiments/theory_self_guidance_20260922/ram_transfer_audit.py)。
- [数值结果](../research/self_guidance_ram_20260923/ram_transfer_audit.json)。
- [图 PNG](../research/self_guidance_ram_20260923/ram_transfer_audit.png) / [PDF](../research/self_guidance_ram_20260923/ram_transfer_audit.pdf)。
- [上一轮 RL 与反传分析](SELF_GUIDANCE_RL_AND_BACKWARD_20260923_ZH.md)。

复现命令：

    /home/zhoushunyu/miniconda3/envs/myenv/bin/python experiments/theory_self_guidance_20260922/ram_transfer_audit.py

本轮验证了 Gaussian 条件期望、指数倾斜积分、保分布旋转、受限族更新、feature KL 分解、reference 刷新循环和 velocity 外推的解析结果。它们不是图像质量或训练加速实验；RAM 在当前 SiT/JiT 上是否更好仍需上述实证。
