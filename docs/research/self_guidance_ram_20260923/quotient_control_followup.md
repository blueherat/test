# 从完整伴随到局部密度响应：internal guidance 的受限控制理论

2026-09-23。独立后续推导；与本轮主研究的 Stein/双稳健推导交叉核验。只写研究笔记，未修改训练、未启动 GPU。

核心判断：对于冻结 weak、只学习少量 schedule/gate 的情形，可以把未来导数换成当前边缘上、沿实际控制方向的一个标量密度响应。这个响应可以直接通过 Riesz 弱回归学习，不要求学习整个高维 score，也不要求计算 strong 的输入散度。更具体地，可以对实际 Heun 的单步参数做正负扰动，以两个局部前向建立有限反事实的 Riesz 目标；配合 value 回归后，有限反事实估计的误差恰为两个 nuisance 误差的乘积，无需任何输入导数、完整未来重采样或整轨迹伴随。它相对真实梯度另有有限差分偏差。以下推导给出连续起点与正确离散版本。

这条路线不需要将当前组合速度规范化为重新加噪的 canonical FM 场。它可能保留 RAM 难以区分的生成分布自由度。但是基本 Stein identity、Riesz 回归、双稳健都是已有工具；潜在研究贡献在于把它们组成适合受限内部控制、实际离散采样器的低成本估计器，并验证误差和成本。

## 1. 准确对象与假设

连续动力学先写作

\[
\dot X_t=v_\theta(X_t,t),\quad X_0\sim q_0,\quad
J(\theta)=\mathbb E[R(X_T)].
\]

q0 不依赖 θ，本次生成器更新内 R 固定；例如 R 可取当前冻结 D 的 non-saturating generator loss 的负值。若 R、初始分布或显式正则另含 θ，需要另外加入相应直接导数。

记 q_t 为当前实际 rollout 边缘，而不是终点独立重加噪边缘。假设流光滑可逆、q_t 正、相关积分有限、分部积分边界通量为零。选定参数方向 δθ，局部速度方向为

\[
b_t(x)=D_\theta v_\theta(x,t)\,\delta\theta.
\]

终点回报从当前状态看的 value 为

\[
V_t(x)=\mathbb E[R(X_T)\mid X_t=x].
\]

确定性流和固定 reward 下 V_t(x)=R(\Phi_{t,T}(x))；写成条件期望也允许独立 reward 测量噪声。标准伴随形式是

\[
DJ[\delta\theta]=\int_0^T\mathbb E[b_t(X_t)\cdot\nabla V_t(X_t)]dt.
\tag{1}
\]

## 2. 不通过未来 Jacobian 的准确反馈

定义 actual-marginal score s_t=∇log q_t，以及标量局部密度响应

\[
\eta_{b,t}(x)=-\operatorname{div}b_t(x)-b_t(x)\cdot s_t(x)
=-q_t(x)^{-1}\operatorname{div}(q_tb_t)(x).
\tag{2}
\]

它是把当前粒子作 x→x+εb_t(x) 时，密度对 ε 的对数导数。分部积分直接给

\[
\mathbb E_{q_t}[b_t\cdot\nabla f]
=\mathbb E_{q_t}[\eta_{b,t}f].
\tag{3}
\]

因此

\[
\boxed{DJ[\delta\theta]=\int_0^T
\mathbb E[R(X_T)\eta_{b,t}(X_t)]dt.}
\tag{4}
\]

这里只出现已有轨迹的标量 reward 与局部密度响应，不出现未来轨迹的输入 Jacobian。若 t∼π(t)>0，可用 Rη/π(t) 随机时间估计积分；这增加时间抽样方差，并不自动比完整伴随更好。

同一等式也能从连续性方程看清。设 h_t(x)=Dθ log q_t(x)[δθ]，则

\[
(\partial_t+v\cdot\nabla)h_t=\eta_{b,t},\quad h_0=0,
\qquad h_T(X_T)=\int_0^T\eta_{b,t}(X_t)dt.
\tag{5}
\]

所以 Eq. (4) 就是终点分布的 likelihood-ratio 梯度，不是人为向 action 添加噪声后才存在的 policy score。

### 2.1 gauge 与常数奖励

若 div(q_tb_t)=0，则 η_b=0：该局部方向保持当前边缘，不论未来 reward 是什么都没有这一阶密度效应。这与 canonical FM 回归可能消除保分布旋转不同。

若各时刻的 η 沿粒子积分互相抵消，终点也可能不变。因此瞬时加权无散只是终点等价的充分条件，未穷尽全部终点 gauge。

由于 Eη_b=0，准确 Eq. (4) 对 R→R+C 不变。用近似 η 时 Eηhat 不必为零；可显式校准均值，但均值为零不足以证明整个响应正确。

Newton Matching §8.3.1 Eq. (60) 已给出 current/reference 密度比沿流演化的有限变化形式；对速度扰动求一阶导数即得到 Eq. (5)。不能把此基础恒等式当作新发现。[Newton Matching](https://arxiv.org/pdf/2609.05727)

## 3. 真正有用的省计算步骤：直接学 η，不先学完整 score

直接计算 Eq. (2) 并不便宜。b 若通过 frozen strong features 依赖 x，div b 仍需 strong 的输入导数。学习全部高维 s_t 也可能比原任务更难。

Eq. (3) 允许直接学习标量 η_b。对候选小网络 e(x,t)，定义

\[
\mathcal L_{\rm Riesz}(e)
=\mathbb E\left[\frac12e(X_t,t)^2
-b_t(X_t)\cdot\nabla_xe(X_t,t)\right].
\tag{6}
\]

由 Eq. (3)，

\[
\mathcal L_{\rm Riesz}(e)
=\tfrac12\|e-\eta_b\|^2_{L^2(q)}
-\tfrac12\|\eta_b\|^2_{L^2(q)}.
\tag{7}
\]

因此无限函数类最优解准确等于 η_b；受限类则拟合其 L² 投影。训练时将 b 的数值 detach，只对小网络 e 反传。需要 e 的输入方向导数与其参数梯度，但不需要 b 的输入导数。

这个目标是平均方向导数泛函的标准 Riesz regression。自动 Riesz 回归和神经网络实现已有成熟文献，含有限样本分析与双稳健估计。[Automatic Debiased ML via Riesz Regression](https://arxiv.org/abs/2104.14737)、[RieszNet](https://arxiv.org/abs/2110.03031)

实现边界：若为了表达能力又让 e 的输入导数穿过 frozen strong 特征提取器，成本会回来。低成本设计需要让 e 直接接收 x 与便宜状态编码，或明确计算已缓存特征的所需方向导数。将 features detach 再把得到的梯度误称为 ∇xe 是错误的。

## 4. 与 value 回归结合，得到乘积误差而非一阶 surrogate 偏差

训练另一个小网络 h_t≈V_t，用真实 rollout 的 (X_t,R_T) 做回归。对固定 h 和 e 定义

\[
\boxed{
G_{\rm DR}[b]=\int_0^T\mathbb E\left[
b_t\cdot\nabla h_t+(R_T-h_t)e_t
\right]dt.}
\tag{8}
\]

条件期望与分部积分给出准确误差等式

\[
\boxed{
G_{\rm DR}[b]-DJ[b]
=\int_0^T\mathbb E[(V_t-h_t)(e_t-\eta_{b,t})]dt.}
\tag{9}
\]

因此任意一个 nuisance 精确即可得到正确方向导数。两者都不精确时，

\[
|G_{\rm DR}-DJ|
\le\|V-h\|_{L^2(dt\,q_t)}
\,\|e-\eta_b\|_{L^2(dt\,q_t)}.
\tag{10}
\]

这里的可取之处是最终误差由 value 的 L² 误差控制，而不单独要求 value 的输入导数接近真实 costate。输入导数项与密度响应残差一起校正了偏差。边界项、可积性、小网络过拟合、参数更新引起分布变化仍需控制。

在样本上同时拟合 nuisance 并评估 Eq. (8)，不能无条件宣称有限样本无偏。可使用 held-out split/cross-fitting、上一轮网络或严格独立数据；训练自适应还需另行分析。

若 e=−div b−b·shat，则 Eq. (9) 化为

\[
G_{\rm DR}-DJ
=-\int\mathbb E[(V-h)b\cdot(\widehat s-s)]dt.
\]

DR policy gradient 本身也已有直接文献，包括确定性策略离策略估计；那些工作与这里的 on-policy actual-marginal directional response 场景不同，不能因此省略 prior。[Huang/Jiang 2020](https://proceedings.mlr.press/v119/huang20b.html)、[Kallus/Uehara 2020](https://arxiv.org/abs/2006.03900)

## 5. 为什么 frozen weak 的时间 schedule 是最匹配的子问题

令 B(x,t)=S(x,t)−W(x,t)，weak 暂固定：

\[
v=S+a(t)B.
\]

若 scale 的变化方向是 δa(t)，则 b=δa(t)B，且

\[
\eta_b=\delta a(t)\eta_B.
\tag{11}
\]

只需一个共享时间的 scalar response network e(x,t)≈η_B(x,t)，就能服务全部 time bins 的系数梯度；不用为每个 latent 维度学习一份 score，也不用为每个 scale 学一个独立的完整 critic。B 的数值已由 strong/weak 前向得到，Riesz loss 不对它求输入导数。

若使用少量冻结方向 B_k，v=S+Σa_k(t)B_k，可训练 m 个 η_{B_k}。线性组合律准确成立。

状态 gate 的正确乘积律则是

\[
\boxed{\eta_{aB}=a\eta_B-B\cdot\nabla a.}
\tag{12}
\]

不能只将 η_B 乘以 a(x,t)。额外项正是 state gate 创造密度压缩的机制。若 a 依赖 strong features，B·∇a 可能再次要求局部 strong JVP；time-only gate 没有这项。也可直接对 aB 拟合新的 Riesz response，将计算压力转到 scalar nuisance 的拟合。

对于 joint weak 的所有参数，b=−aDφWδφ+D_a vδa，方向空间可能很大。每个参数一个 scalar nuisance 并不合理；低成本结论只覆盖选定 m 维方向子空间，例如 recent/RAM update directions、少量 adapters，或冻结 weak 的 schedule。不能用小 action 空间的推导宣称整个 weak head 无代价训练。

## 6. 实际 Heun：准确单步映射版本

写实际采样器为

\[
X_{i+1}=F_{i,\theta}(X_i),\quad i=0,\ldots,N-1.
\]

假设每一步为局部/全局可逆的适当光滑映射。对一个参数方向 δθ，在一步输入 x 固定时定义

\[
c_i(x)=D_\theta F_{i,\theta}(x)\delta\theta,
\qquad b_i(y)=c_i(F_{i,\theta}^{-1}(y)).
\tag{13}
\]

若 V_{i+1}(y) 是从该下一状态出发的终点 value，则准确离散梯度为

\[
DJ[\delta\theta]=\sum_i\mathbb E[c_i(X_i)\cdot\nabla V_{i+1}(X_{i+1})]
=\sum_i\mathbb E[R_T\eta_i(X_{i+1})],
\tag{14}
\]

其中 η_i=−q_{i+1}^{−1}div(q_{i+1}b_i)。对参数密度 score，准确递推为

\[
h_{i+1}(y)=h_i(F_i^{-1}(y))+\eta_i(y).
\tag{15}
\]

这与连续式相同，但是在下一状态边缘 q_{i+1} 上，而不是随手取 q_i 或独立重加噪分布。

关键是 Riesz 拟合本身不需要显式 F_i^{-1}：

\[
L_i(e)=\mathbb E\left[\tfrac12e(X_{i+1})^2
-c_i(X_i)\cdot\nabla e(X_{i+1})\right].
\tag{16}
\]

直接使用已配对的 (X_i,X_{i+1},c_i) 即可。离散双稳健估计相应为

\[
G_{\rm DR}=\sum_i\mathbb E[c_i\cdot\nabla h_{i+1}
+(R_T-h_{i+1})e_i],
\]

偏差仍准确为 ΣE[(V_{i+1}−h_{i+1})(e_i−η_i)]。

### 6.1 Heun 的 c_i 不是步长乘上 weak gap

令 predictor y=x+h vθ(x,t)，Fθ(x)=x+h[vθ(x,t)+vθ(y,t+h)]/2。固定 x 时

\[
c_i=\frac h2\left[b(x,t)+b(y,t+h)
+hD_xv(y,t+h)b(x,t)\right].
\tag{17}
\]

最后一项包括 strong 的局部输入 Jacobian。省略它就是更换了梯度，而非消除了伴随。若两次场评估用不同 scale index，b 还必须按真实代码分别取参数方向。

这个局部强场 JVP 只涉及一步，不涉及整个未来链条；随机挑 interval 可避免所有 interval 都计算。但自动微分支持、实际前向/反向成本与方差要测量，不能仅按公式的层数推算速度。

### 6.2 完全前向的局部反事实 Riesz 拟合

如果也不想计算 Eq. (17)，对冻结的一步输入 X_i 构造

\[
Y_i^{\pm}=F_{i,\theta\pm\epsilon\delta\theta}(X_i),\quad
M_{i,\epsilon}(f)=\frac{f(Y_i^+)−f(Y_i^-)}{2\epsilon}.
\tag{18}
\]

然后最小化

\[
L_{i,\epsilon}(e)=\mathbb E[\tfrac12e(X_{i+1})^2-M_{i,\epsilon}(e)].
\tag{19}
\]

此处只做两个局部 Heun 映射的 no-grad 前向，并对小 e 网络的输出反传；没有未来 rollout、strong 输入导数、divergence 或逆映射。

这里有一个比 ε→0 更强的准确表述。定义 q_{i,±}=(F_{i,θ±εδθ})#q_i，q_{i,0}=q_{i+1}。只要两种扰动边缘相对于 q_{i,0} 绝对连续、相应密度响应平方可积，Eq. (19) 在任意有限 ε 的 population root 准确等于

\[
\eta_{i,\epsilon}(y)=\frac{q_{i,+}(y)−q_{i,-}(y)}{2\epsilon q_{i,0}(y)}.
\tag{19a}
\]

这个有限反事实结论不要求 Heun 映射可逆，也不要求知道各密度值，只需要正负一步映射样本与 base 下一状态样本。初始噪声可使边缘具有密度，即使条件转移本身是确定性的；但绝对连续、支撑覆盖与 L² 条件仍需验证。

q_{i,±} 是固定当前输入边缘 q_i，只扰动当前一步所得的下一状态边缘。它并不是整个参数变化后的完整 q_{i+1,θ±εδθ}，后者还包括之前所有步骤的变化。下游 value V_{i+1} 也固定为当前其余所有步骤继续运行所得的终点回报。

将 DR 中的 c_i·∇h 同样换成 Mε(h)，在任意 h 下，都满足

\[
G_\epsilon-\sum_i E[M_{i,\epsilon}(V_{i+1})]
=\sum_i E[(V_{i+1}−h_{i+1})(e_i−\eta_{i,\epsilon})].
\tag{20}
\]

因此若 value 或有限密度响应中的任何一个精确，估计就精确等于该有限单步反事实 contrast，而非仅在 ε→0 才成立。进一步若相应参数方向的三阶导数、换序与可积性成立，Mε(V)=c_i·∇V+O(ε²)，才可将此有限 contrast 与原参数梯度比较。surrogate fitting 有乘积误差，中心差分另有 O(ε²) 偏差；不能把有限 ε 说成原梯度严格无偏。噪声、有限精度、小 ε 放大、有限支持及 Riesz representer 平方可积性也都需要处理。

对 frozen-weak 的每个时间 bin，只有一个 scale 方向，因而此方案尤为自然。对高维 weak 参数，全方向成本仍然存在，必须选低维子空间或接受随机方向估计的方差。

## 7. 商空间度量：区分粒子移动与分布改变

取 m 个选定参数方向，令 H_j(X_T)=Σ_iη_{i,j}(X_{i+1})（连续时取积分）。在可逆确定性流下它是终点参数 score；若仅观测 feature Z=Φ(X_T)，可见 score 为 E[H_j|Z]。

终点 Fisher 为

\[
\mathcal I_{jk}=E[H_jH_k].
\tag{21}
\]

它的零空间正是这 m 维子空间内的一阶终点密度零方向。对应的粒子位置 Gramian E[(DθX_T)^T(DθX_T)] 可能把保分布旋转计为非零，二者不是同一种 endpoint geometry。仅基于 Inception 的 reward 更进一步只看 feature Fisher，其零空间通常更大。

一个可测有用比率是选定参数方向的 density-response 能量与局部 field 能量之比。分母很大而分子很小的方向，可能主要改变粒子标签/流的代表，或不同时间效应互相抵消。这个比率不能证明方向永远无用：二阶效应和离开当前参数后的激活仍可重要。

自然梯度、Wasserstein/Fisher pullback 和冗余参数的不变性不是新概念；这里的作用是避免用局部场 L² 几何去代替终点密度几何。[Natural-gradient invariance](https://doi.org/10.1007/s41884-022-00067-9)、[Natural gradient via optimal transport](https://arxiv.org/abs/1803.07033)

## 8. 最小可证、可诊断的研究问题

最适合先验证的是 frozen-weak schedule，不是立刻全 weak 联合训练：

1. 将现有完整离散反传作为方向导数真值。
2. 对同一 batch 的真实 (X_i,X_{i+1},R_T)，拟合少量方向 η 的 Riesz 网络与 value 网络。
3. 比较局部 Riesz reward 估计与 DR 的梯度偏差、方差、cosine，以及相同成本下的有效信号。
4. 分开测 actual-marginal 与 re-noise 边缘替代误差、Heun c_i 与 hB 近似误差、以及有限 ε 误差。
5. 检查常数 reward invariance、真实保分布旋转 null、有限容量导致 gauge/有用方向耦合时的表现。

值得保留的研究假设是：**对少量实际控制方向，局部密度响应比整个未来输入梯度更易学习；结合回报残差校正后，可以低成本获得准确 enough 的终点梯度。** 它尚未由图像实验确认，也不由一般 Riesz/DR 理论自动保证。

这比“在无穷函数空间先求一个 canonical transport，再让小头拟合它”更直接地尊重当前受限生成动力学；但能否成为实际改进，取决于 nuisance 学习难度、局部前向成本与梯度方差三者的平衡。
