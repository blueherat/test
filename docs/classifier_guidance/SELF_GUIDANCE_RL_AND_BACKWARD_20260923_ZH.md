# Self-guidance 的 RL 与反传：从完整伴随到可校正的局部反馈

2026-09-23。依据当前源码、运行日志只读快照、此前“查找并实现 Self-Guidance”的 RL 讨论、原始论文及本轮 CPU 核验。本文没有修改训练或启动 GPU 实验。

**后续补充：** 用户提供的 RAM 分析补上了本文文献覆盖的缺口。新的优先候选是先用 RAM 局部回归提供便宜梯度，再接本文的随机精确校正，不必先训练未来敏感度网络。RAM 本身不等价于当前终点梯度；适配中的自洽性缺陷、受限族反例和具体方案见 [RAM 独立复核](RAM_INTERNAL_GUIDANCE_REVIEW_20260923_ZH.md)。

**主要判断：可以用 RL。对 frozen-weak scale-only，低维随机动作是直接可用的接口；对 joint weak+scale，更值得研究的是学习并复用未来敏感度。最稳妥的候选是在便宜局部反馈之外保留随机抽取的精确伴随校正，使当前确定性 ODE 和固定 D 的生成器目标不变。**

这不是把 PPO/GRPO 换一个名字。需要明确动作覆盖哪些参数、反馈估计的偏差和方差、仍需支付的 forward 成本，以及与已有 AdaGen、DRL/AM、SVG/Q-Prop 的关系。

## 1. 当前具体怎样处理“大反传”

[sampler.py](../../classifier_guidance/sampler.py)的前向逐步保存 latent/pixel state 与 Heun predictor，不保存各时刻整套 backbone 激活。反向逆序重新运行每个场，立即计算该场的输入与参数 VJP。CUDA graphs 重放固定算子，降低调度开销。

这是完整离散链式法则：没有截断时间，没有删 strong 输入 Jacobian，没有以连续 ODE adjoint 近似离散 Heun 梯度。模型参数冻结仅免去其参数梯度，状态导数依然存在。

| 每条轨迹 | SiT joint | JiT schedule |
|---|---:|---:|
| 求解器 | 64 Heun | 49 Heun + 1 Euler |
| 实际采样 field forward | 128 | 99 |
| 反向重算 field forward | 128 | 99 |
| field VJP | 128 | 99 |
| 生成侧训练参数 | weak + 64 scales | 50 scales |

一次 field forward 同时获得 strong 与挂在其中的 weak 输出，不是两套完整 backbone 各跑一次。D 与 G 共用一次实际采样的端点；G 重算端点 decode/Inception 以获得输入梯度，然后进入 sampler 反向。

上述计数不含端点 VAE/Inception、D/R1、norm probe、预热及诊断。激活 checkpoint 还会在对应网络内部增加重算。

当前显存约为

\[
O(NB_{\rm local}d)
+O(\text{单场激活}(b))
+O(\text{单批反馈激活}(b))
+\text{常驻参数/优化器/图内存池}.
\]

它不是 N 倍 backbone 激活，也不是严格常数内存。SiT 的全部 saved states/predictors 约 64 MiB；JiT 每卡约 1.172 GiB，像素轨迹本身较大。

2026-09-23 00:11 北京时间只读快照：SiT global step1067，最近更新8.906秒，peak allocated6.593GiB；JiT global step3483，最近更新6.832秒，每卡最大allocated2.452GiB。step 含128次D预热；最近一步耗时不是长期吞吐均值。来源与细节见[成本审计](../research/self_guidance_rl_20260923/current_backward_cost.md)。

所以目前的主要问题是**仍需反复计算完整未来导数**。普通 checkpoint 已处理了主要内存问题，不能再把 checkpoint 本身当成新的数量级加速方案。

## 2. 这是一个控制问题，但“是动力学系统”不等于 model-free RL 必然更省

固定一次生成器更新使用的 D。定义终点奖励

\[
R_D(x_N,c)
=-\operatorname{softplus}\{-D(F(\operatorname{Decode}(x_N)),c)\}.
\tag{1}
\]

这正是当前 non-saturating generator loss 的负值。若改成 D logit，则是另一奖励；虽然两者对单张图都随 D 增大而增大，其期望优化一般不同。SiT 的独立 norm probe 可继续直接求梯度，RL 不要求将这项也改成环境奖励。

固定步数与类别，设

\[
x_{i+1}=F_i(x_i,a_i),\qquad a_i=\mu_\theta(x_i,i,c),
\qquad J(\theta)=\mathbb E R_D(x_N,c).
\]

其中 F 是实际 Heun interval，环境必须在给定动作后不再依赖需要用 policy-gradient 更新的参数。

定义当前策略的未来价值

\[
V_i^\theta(x)=\mathbb E[R_D(x_N)\mid x_i=x],\qquad
Q_i^\theta(x,a)=V_{i+1}^\theta(F_i(x,a)).
\]

在光滑与可积条件下，有限时域 deterministic policy gradient 为

\[
\nabla_\theta J
=\mathbb E\sum_i
(\partial_\theta\mu_\theta(x_i))^T
\nabla_aQ_i^\theta(x_i,a_i).
\tag{2}
\]

且

\[
\nabla_aQ_i
=(\partial_aF_i)^T\nabla V_{i+1}.
\tag{3}
\]

完整 BPTT 通过链式法则精确计算右侧；actor–critic 学习右侧的未来效应。两者服务同一个控制目标。区别在于昂贵的未来导数是每次重新计算，还是用一个学到的函数近似并复用。[DPG](https://proceedings.mlr.press/v32/silver14.html)、[SVG](https://arxiv.org/abs/1510.09142)是这条连接的标准基础。

因此本问题最有价值的压缩对象是**允许动作的边际收益**。状态可能有几万维，但当动作只有一个 guidance 系数时，∇aQ 只是一个标量；不必先学出整个未来图像或完整状态 Hessian。

这份低维性仅适用于低维动作。若 weak 的完整向量输出也是可训练动作，所需反馈维度就会扩大，不能把 scale 的优势自动推广到整个 joint。

一个更几何化的连接是固定方向字典 U，写 dx/dt=S+Ub。若另行定义带动作能量的代价 E[ℓ_D(x_T)+½∫bᵀMb dt]，M正定，则光滑HJB条件下最优控制为 b*=−M⁻¹Uᵀ∇V。它只需要沿允许方向的价值导数。这个闭式公式依赖新加入的动作代价或相应约束，不能把M当作当前gap锚点，也不能把无约束线性Hamiltonian说成总有有限最优控制。详见[独立actor–critic推导](../research/self_guidance_rl_20260923/actor_critic_theory.md)。

## 3. 最直接的 RL：在系数上探索，不必把图像 ODE 改成 SDE

冻结 S、W，令

\[
a_i=\mu_\theta(s_i)+\sigma_i\epsilon_i,\qquad
\epsilon_i\sim N(0,I),\qquad
x_{i+1}=F_i(x_i,a_i).
\tag{4}
\]

环境转移可以完全确定；只要动作分布具有可微 likelihood，就可使用

\[
\nabla_\theta J_\sigma
=\mathbb E\sum_i
(R_D-b_i(s_i))\nabla_\theta\log\pi_\theta(a_i\mid s_i).
\tag{5}
\]

在正确 on-policy 分布、动作无关 baseline 及适当可积条件下，(5) 是随机动作目标的无偏梯度。给定动作，Heun 两次场查询使用同一个系数，保留当前离散动作定义。

这条路线不需要 strong、weak、VAE、Inception 的输入梯度。先 no-grad 生成，更新 D 后可复用保存的终点特征重新算 reward。策略学习只对小 policy 求参数梯度，强特征可以 detach。

当前所有样本共享的 a_i，可先用 μ_i=a_i 的 Gaussian policy 表示；也可以另行引入状态条件 policy。后者改变了模型能力，不能把它的收益全归因于 RL。

**JiT frozen-weak scale-only 是最直接的合法对照。** 相反，SiT 中若 W_φ 仍在 F_φ(x,a) 内，单独计算 (5) 会漏掉 ∂φF_φ；不能既把 weak 当环境，又说 REINFORCE 训练了 weak。要么固定 weak，要么把 weak 输出或全部需要训练的控制坐标纳入 action，要么另给 weak 有效的局部梯度。

这正是 [AdaGen](https://arxiv.org/html/2603.06993v1) 的直接邻近设置：冻结生成器、低维采样参数、确定性 ODE 环境、随机 policy 与终点对抗奖励。该组合不是新增贡献。

### 3.1 为什么不直接全面采用 GRPO

- **forward 仍在。** 每张终点图仍需99/128次场求值。额外组采样可能抵消省掉 backward 的收益。
- **探索目标变化。** J_σ 一般不等于确定性均值策略 J_0。推理时去掉探索噪声，不保证保留训练分布。
- **PPO clipping、组内标准差归一化、KL regularization** 都引入额外近似或改变目标；不能沿用原始 (5) 的全部等价性。
- **高维 action 的标量反馈昂贵。** 将 weak 的全部输出作为 Gaussian action 在数学上可行，但与只探索一个系数的统计难度不同。
- **时序状态不可随意压缩。** 用 frozen feature 作观测是合理近似，但不证明其是充分 Markov state；训练 critic 时尤其重要。

一个精确目标变化例：R(a)=−(a²−1)²，a~N(μ,σ²)。随机目标最优的正均值为 √max(1−3σ²,0)，确定性最优为1。σ=.5时，随机训练最优均值为.5，其确定性奖励−.5625，低于确定性最优0。这不是对实际图像策略的定量预言，只证明随机化不是无损替换。

另一个线性例：R(a)=gᵀa，a=μ+σε，使用最优常数 baseline R(μ) 后，均值参数的 score estimator 为 (gᵀε)ε，具有

\[
\mathbb E\widehat g=g,\qquad
\mathbb E\|\widehat g-g\|^2=(d+1)\|g\|^2.
\tag{6}
\]

即使极简单的奖励也有随动作维数增长的梯度方差。真实模型的结构和 baseline 可改变实际情况；(6) 不等于任意 RL 方法的下界。

Flow-GRPO 的高斯转移则需 ODE→SDE 构造；保边缘需要正确的 marginal score。任意学习后的 S+a(S−W) 不自动满足相应 FM velocity-score 恒等式，不能在当前系统上直接继承理想保边缘结论。[Flow-GRPO](https://arxiv.org/html/2505.05470v5)。

## 4. 更适合 joint 的路线：学习每次场查询的未来反馈

把实际求解器看成计算图。记第 j 次 field 查询的输出为 v_j、完整终点奖励对它的 cotangent 为 b_j=∂R/∂v_j。固定查询状态后的局部参数 Jacobian 为 K_j=∂θv_θ(x_j)。则

\[
g=\nabla_\theta R=\sum_jK_j^Tb_j.
\tag{7}
\]

在当前 SiT 参数化下，

\[
\partial_\phi v=-a_i\partial_\phi W_\phi,\qquad
\partial_{a_i}v=S-W_\phi.
\tag{8}
\]

只要 b_j 已知，(7) 可以在 detach 的状态和 strong 特征上，仅重算小头及 scale 的局部导数。strong 的状态 Jacobian 已包含在 b_j 中，不能再把 b_j 随便换成终点 ∇x_NR。

本轮非线性5步Heun、正负零系数的CPU核验：完整autograd与全部精确b_j注入后的局部weak+scale梯度完全一致；把b_j直接替换成统一终点梯度乘步长，梯度相对误差14.74%。这是链式法则核验，不是学习critic已经成功。

因此可用少量完整反传产生高质量 b_j 标签，训练一个便宜的反馈模型，在其余新轨迹上预测这些标签。已有缓存伴随多轮训练与 [DRL/Adjoint Matching](https://arxiv.org/html/2606.19162v1)直接相关；新增候选应研究**对新轨迹的反馈泛化、guidance 参数真正需要的投影，以及精确校正**，不能只重复缓存。

### 4.1 保持 Heun 精确性，需要给 critic 正确的阶段状态

一整个 Heun interval 的局部反传仍需要 strong 输入 Jacobian：

\[
y=x+h v_\theta(x),\quad
x'=\tfrac12(x+y)+\tfrac h2v_\theta(y).
\]

∂θx' 包含 h²J_xv(y)∂θv(x)/2。截断 predictor 导数就改变了梯度。

但可将求解器精确拆成两个受控阶段。令动作是实际修正 d=a(S−W)，环境 strong 冻结：

\[
\text{阶段1： }y=x+h[S(x)+d_1],
\]
\[
\text{阶段2： }x'=\tfrac12(x+y)+\tfrac h2[S(y)+d_2].
\tag{9}
\]

阶段1 critic 的未来状态包含 (x,y)，阶段2终点价值是 V_{i+1}(x')。相应 action derivative 分别为

\[
h\,\partial_yV_{\rm stage2}(x,y),
\qquad
\tfrac h2\nabla V_{i+1}(x').
\tag{10}
\]

若这些导数精确，就将所有 strong Jacobian 吸收入未来价值，而 actor 本地只求 ∂θd。阶段2只看 y、遗漏 x，一般不是 Markov；同一 y 可能来自不同 x，最终 x' 不同。时间、类别、阶段索引也必须保留。

直接预测向量反馈可以把 frozen features detach；若改成 scalar V(H_S(x)) 再求 ∇xV，仍需 J_H_S(x)^T，不能声称完全没有 backbone VJP。低维动作 Q(s,a) 只对 a 求导，则不需对状态特征反传。

## 5. critic 的正确性应看动作梯度，而不是 value MSE

确定性轨迹只观察 a=μ(s)。对任意 c(s)，

\[
\widehat Q(s,a)
=Q(s,\mu(s))+c(s)^T[a-\mu(s)]
\]

都能完美拟合这些 on-policy value，但动作导数可以任意。这意味着没有动作探索或导数标签时，普通 return regression 不识别需要的 actor gradient。

即使函数值处处接近，也未必足够：Q(a)=a，

\[
\widehat Q(a)=a-\epsilon\sin(a/\epsilon^2)
\]

的最大值误差≤ε，a=0处值完全正确，却有 Q'(0)=1、Qhat'(0)=1−1/ε。ε=.01时为−99，更新方向相反。

因此检验应针对有效梯度：

\[
\widetilde g-g
=\sum_jK_j^T(\widehat b_j-b_j).
\tag{11}
\]

全状态 costate 的某些误差可能落在所有 K_j^T 的零空间里，对当前更新无害；少量但与可训练方向对齐的误差则可能主导更新。这与 DPG 的 compatible approximation 思路一致，不是新定理。

只学系数时，可以直接监督每区间标量 ∂R/∂a_i；联合weak时，仅有这些标量不够恢复weak梯度。应保留向量反馈、适当投影，或用局部扰动/完整梯度检查训练所需方向。

短段精确rollout+tail value是另一种中间档，属于SVG思路。它避免完全依赖critic，但仍支付段内VJP；tail导数误差还会被段内Jacobian放大，长段不无条件更稳定。

## 6. 更稳妥的候选：便宜反馈加随机精确反传校正

这条可以保持**当前确定性采样器和固定D的生成器目标**，无需为每一步添加探索噪声。

对同一新batch、当前θ和固定D，先定义便宜近似梯度 gtilde，例如 (7) 中使用预测b。独立抽 I~Bernoulli(p)，0<p≤1；只在I=1时计算当前代码的完整精确梯度g。使用

\[
\boxed{\widehat g
=\widetilde g+\frac Ip(g-\widetilde g).}
\tag{12}
\]

条件于该batch和已固定的反馈模型，

\[
\mathbb E_I[\widehat g]=g,\qquad
\mathbb E_I\|\widehat g-g\|^2
=\left(\frac1p-1\right)\|g-\widetilde g\|^2.
\tag{13}
\]

这是标准控制变量/多保真随机估计思想；相关RL先例包括 [Multi-Fidelity Policy Gradient](https://arxiv.org/abs/2503.05696)。此处的具体使用是同一真实生成轨迹上的便宜actor–critic梯度与精确离散adjoint，而不是换一个低保真环境。

它提供清楚的取舍：

- p=1时回到原精确梯度。
- 反馈预测越准，可以在相同附加方差预算下减少完整反传。
- 只要p>0且采样/补偿正确，critic有误也不会改变固定D下的梯度期望。
- p=0且直接使用gtilde，则失去这个保证，退为有偏actor–critic。

不能把它说成比每batch完整反传方差更小。总体有

\[
\operatorname{trVar}(\widehat g)
=\operatorname{trVar}(g)
+\mathbb E\left[\left(\frac1p-1\right)\|g-\widetilde g\|^2\right].
\tag{14}
\]

它以附加方差换计算节省；同墙钟可增加样本/更新才可能获得效率优势。预测很差时不应强行把p压小。CPU用全部Bernoulli分支精确求和核验(12)–(13)，均值误差≤6.94e−18。

### 6.1 实现上哪些条件不能省

1. 精确g与近似gtilde必须对应同一batch、同一θ、同一固定D与目标。不能拿旧checkpoint梯度直接代替当前g。
2. gtilde与p必须在本次I的精确标签揭示前确定；不能仅在I=1分支先用该标签拟合critic，再反过来改变当前估计的gtilde。标签可训练下一次更新使用的模型。
3. 可以依据历史误差/当前廉价特征选择p，但要使用实际纳入概率，且所有需纠偏情形保持p>0。
4. 上式保证的是裁剪/Adam之前的梯度估计，非线性梯度裁剪与Adam不保持“期望更新等于原更新”。
5. I可在rollout前抽取：I=0无需建立可回放的完整反传上下文；I=1走现有精确路径。
6. norm probe可始终直接求导。若它被纳入总g，也必须在两路中用一致定义。

若允许自适应p，在预计精确计算成本B(z)、固定平均反传预算下，控制附加均方误差的理想抽样率具有

\[
p(z)\propto \frac{\|g(z)-\widetilde g(z)\|}{\sqrt{B(z)}},
\quad p\le1.
\tag{15}
\]

来自带约束最小化 E[‖g−gtilde‖²/p]。真实残差在未反传样本上未知，只能用历史/不确定性估计决定抽样率；这个规则不是免费知道真实误差。它把“何时值得付完整反传的钱”变成可研究的统计分配问题。

## 7. 如果选择随机动作，也可用critic而保留无偏纠偏

低维动作Gaussian policy，均值μθ，协方差固定。设h(s)是stop-gradient的动作收益导数估计，b(s)是动作无关baseline。可用

\[
\widehat{\nabla_\theta J_\sigma}
=\sum_i\nabla_\theta\log\pi(a_i\mid s_i)
\{R-b(s_i)-h(s_i)^T[a_i-\mu_\theta(s_i)]\}
+\sum_i(\partial_\theta\mu_\theta(s_i))^Th(s_i).
\tag{16}
\]

正确on-policy期望下，减去线性控制变量的score贡献恰由最后一项补回。它不要求h准确才无偏；h改善可能降方差，也可能因失准而增加方差。[Q-Prop](https://arxiv.org/abs/1611.02247)与[LAX/RELAX](https://arxiv.org/abs/1711.00123)提供直接基础。

在线性例(6)中，该估计方差变为(d+1)‖g−h‖²。h=.8g时，为原来的4%。这是对动作梯度预测为何有用的精确说明，不是实际GAN的25倍样本效率承诺。

(12)和(16)解决的问题不同：(12)保持当前确定性目标并随机购买精确导数；(16)取消环境导数，用随机动作的终点回报校正critic，但仍优化随机动作目标。不能把两者的无偏性混用。

## 8. 动态GAN、缓存与可达到的加速上限

当前D与actor交替变化。一次actor估计中固定D是必要的局部条件。旧端点特征可以用新D重新打分，修复reward陈旧；但旧轨迹仍来自旧policy，重打分不能修复state occupancy和future policy的变化。

直接在旧replay state上更新actor一般是带分布偏差的semigradient，即使Q准确也不能直接称为当前endpoint目标的无偏梯度。fresh no-grad rollout、短的固定D/target-policy阶段以及独立精确校准可控制问题，但不构成一般GAN收敛证明。

缓存也有内存代价：early feature通常比SiT latent大，缓存128次全部feature可能比当前只存state更昂贵。可以只保存随机选择的查询、转CPU或重新计算小量特征；不能把“可缓存”写成无成本。若gtilde使用随机查询子集，(12)只需两分支共用同一定义的gtilde；条件无偏性不要求gtilde本身无偏。

设F为一次完整no-grad采样与奖励成本，B为额外精确反向成本，C为局部actor/critic等额外成本。原方法约F+B，(12)约

\[
F+pB+C.
\tag{17}
\]

单次更新要变快至少需C<(1−p)B；p→0也不能省掉F。若改RL后每有效更新要更多终点样本，必须计入额外F。不能把“少做90%反传”直接说成10倍训练加速。

最终应比较相同生成器初值、目标定义、求解器/部署策略、真实walltime和strong NFE下的独立质量。optimizer步数和显存峰值不足以证明达成相同FID更快。当前工程的实现优化已经生效，理论加速仍需单独测量。

## 9. 已有文献划定的边界

| 来源 | 最相关之处 | 当前应区别什么 |
|---|---|---|
| [AdaGen](https://arxiv.org/html/2603.06993v1) | 冻结generator、低维采样动作、PPO、终点对抗奖励 | scale RL本身已有直接先例 |
| [Adv-GRPO](https://arxiv.org/html/2511.20256v1) | 对抗奖励与Flow-GRPO，更新generator | GAN作reward不是新点；组采样要付forward成本 |
| [DRL](https://arxiv.org/html/2606.19162v1) | SiT/JiT/REPA/RAE，固定D，buffered AM | 缓存state/adjoint多轮训练已有；仍付buffer刷新伴随成本 |
| [Adjoint Matching](https://arxiv.org/html/2409.08861v5) | KL正则SOC与lean adjoint回归 | 其目标、SDE与memoryless条件不能自动转移到当前ODE joint |
| [EAM](https://arxiv.org/html/2605.11480v2) | 重构base drift与terminal cost，实现闭式adjoint | 取消反向伴随有条件，受限self-guidance族中非自动等价 |
| [Explicit Critic Guidance](https://arxiv.org/html/2605.27736v1) | noisy-state value critic、backbone特征与PPO | “学一个中间value网络”也不是新点 |
| [Neighbor GRPO](https://arxiv.org/html/2511.16955v1) | ODE邻域轨迹与训练用代理policy | 代理likelihood不是原确定性ODE真实转移密度 |
| [DPG](https://proceedings.mlr.press/v32/silver14.html)、[SVG](https://arxiv.org/abs/1510.09142) | 局部Q梯度、短rollout+tail value | 本笔记actor–critic与伴随连接的经典基础 |
| [Q-Prop](https://arxiv.org/abs/1611.02247)、[MFPG](https://arxiv.org/abs/2503.05696) | 不精确critic/低保真信息作可校正控制变量 | 无偏校正原则已有；需证明当前具体实现的效率 |

不将“memoryless SDE”误译为无需保存计算图：它是有关基过程与初始噪声记忆的条件。DRaFT-K式只反传最后K步也不是合适的自动替代：当前每时段scale独立，早期scale会失去全部梯度；共享weak虽然还有晚期梯度，目标同样被改变。

## 10. 本轮建议与已完成核验

**工程直接对照：** frozen-weak的JiT使用低维随机scale动作，对照AdaGen式RL；若引入状态条件策略，单独记录表达能力变化。它检验不用环境导数是否能同成本更快改善，但未承担方法创新。

**更有研究价值的主候选：** 利用现有精确离散反传，监督guidance所需的局部未来反馈；用其生成便宜actor更新，再以(12)随机校正。先从scale的低维导数开始检查可预测性，之后才处理joint weak的高维反馈。其价值若成立，应来自反馈可学习、校准成本低及同walltime质量改善，而不是理论公式看起来复杂。

**与上一轮结构理论的连接：** 少量互补参考方向配低维state gates，使允许动作的未来收益导数维数小，可能同时有利于表达与信用分配。方向本身仍需训练时，不能把它们藏进环境后漏掉梯度；可用精确/近似伴随更新方向，用低维critic或RL更新幅度。两种更新都针对同一个组合终点效用。

[CPU脚本](../../experiments/theory_self_guidance_20260922/rl_adjoint_audit.py)与[数值原件](../research/self_guidance_rl_20260923/rl_adjoint_audit.json)核验了：

1. 非线性Heun中，精确field cotangent局部回传与完整weak+scale autograd相同。
2. 直接终点梯度替代中间伴随会产生明确误差。
3. value近似误差小甚至on-policy为零，动作梯度仍可反向。
4. Gaussian线性奖励的score方差与线性critic控制变量的方差公式。
5. 探索噪声可改变最优部署均值。
6. 随机精确校正的无偏性与附加方差恒等式。

这些是数学与CPU证据；尚未训练任何RL/价值网络，也没有图像质量或加速倍数结果。当前正式训练流程保持原状。

配套：[当前反传成本审计](../research/self_guidance_rl_20260923/current_backward_cost.md)、[DPG/Heun/随机校正的独立理论推导](../research/self_guidance_rl_20260923/actor_critic_theory.md)、[上一轮方向与密度可控性分析](SELF_GUIDANCE_DEEP_THEORY_20260922_ZH.md)。
