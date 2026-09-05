# PFR 的 OU 概率小波解释与谱认证方法

日期：2026-09-04

状态：**坐标变换、OU 条件分数恒等式、后验 martingale、OU 半群谱和消去器
唯一性是精确结果；score 的 Hermite 分阶解释只在高噪声近 Gaussian 区域作
一阶展开；degree-1 特异性、location/shape 审计、投影粒度、方向/幅值因果分解和
等计算 FID 已有配对实验支持；RAEv2 跨模型迁移给出明确反例；尚无“必然改善
FID”的定理。**

对应实现：

- OU 坐标和谱算子：`experiments/pfr_ou_semigroup_spectrum.py`；
- 因果分量与采样场：`experiments/pfr_ou_semigroup_controls.py`；
- 固定 Heun、调用计数和配对 FID：
  `experiments/run_imagenet100_sit_pfr_equal_compute.py`；
- 100 类 location/shape energy 审计：
  `experiments/audit_pfr_ou_location_shape_energy.py`；
- 投影粒度与跨尺度秩审计：`experiments/audit_pfr_ou_location_shape_energy.py`、
  `experiments/audit_pfr_ou_multiscale_rank.py`；
- 可解 Gaussian-mixture 条件分数审计：
  `experiments/audit_pfr_ou_conditional_score_identity.py`；
- 可解 Gaussian-mixture 有限尺度 score-FPE 审计：
  `experiments/audit_pfr_ou_score_shape_identity.py`；
- posterior-mean closure 反例：`experiments/audit_pfr_ou_score_closure.py`；
- 单元与端到端测试：`tests/test_pfr_ou_semigroup_spectrum.py`、
  `tests/test_pfr_ou_semigroup_controls.py`、
  `tests/test_imagenet100_sit_pfr_equal_compute.py`。

## 一、结论先行

当前最符合公式和实验的解释是：

> **线性 flow 的噪声时间可以精确改写为 Ornstein--Uhlenbeck 概率
> scale-space。跨尺度 relative score 满足一个精确的条件期望恒等式；在固定
> 观测坐标上比较两个尺度，得到的是一次“反事实后验分数创新”。它同时可以看成
> 一个两抽头、带一个 vanishing moment 的概率小波：严格消掉纯 location family，
> 并在高噪声谱展开下保留 covariance、多模态和更高阶分布形状。**

这里的“谱”不是图片 Fourier 频率，而是 Gaussian 空间中 Hermite degree 的
**概率谱**。这一区分不能省略。

由此得到的采样方法也不是再调一个系数：

1. 先构造 PFR 已有的弱头时间修订 $R$；
2. 从 OU 半群唯一导出 degree-1 消去方向 $D_1$；
3. 使用 $R$ 在 $\operatorname{span}(D_1)$ 上的最小二乘投影，保留有概率谱
   依据的部分；
4. 只在原 IG 已经存在的第一时间段使用该投影，第二时间段仍使用原始时间修订。

它没有训练新网络，没有新增 guidance gain，也没有用 FID 选择新的时间边界。
在严格配对、低于 ordinary IG 计算量的 FID-5K 中：

| 条件 | FID | FID mean | FID covariance | sFID | IS | FFE |
|---|---:|---:|---:|---:|---:|---:|
| ordinary IG, Heun-32 | 40.9122 | 7.8470 | 33.0652 | 68.8750 | 37.8545 | 80,000.0 |
| 原 projected PFR, Heun-27 | 37.5299 | 5.8926 | 31.6373 | 68.9555 | 40.8774 | 80,696.3 |
| weak OU degree-1 common-first, Heun-25 | 37.2123 | 5.9054 | 31.3069 | **67.5400** | 40.9181 | 79,117.5 |
| strong OU degree-1 common-first, Heun-22 | 37.0827 | 5.9277 | 31.1550 | 67.5420 | 41.1940 | 79,991.3 |
| strong OU degree-1 direction + raw norm, Heun-22 | **36.1902** | **5.4434** | **30.7467** | 67.9882 | **41.5645** | 79,991.3 |
| OU degree-1 energy-adaptive, Heun-23 | 37.3211 | 5.8771 | 31.4440 | **67.3602** | 40.6593 | 79,982.5 |

当前最好 norm-preserving strong certificate 在这组 bank 上相对 ordinary IG 的
FID 改善为 $4.7221$，即 **11.54%**，同时 full-forward equivalent 略少
**0.011%**。第二个独立 balanced 5K bank 给出同样排序：

| 5K bank | ordinary IG | strong common | strong direction + raw norm | 相对 ordinary |
|---|---:|---:|---:|---:|
| seed 5 | 40.9122 | 37.0827 | **36.1902** | **-11.54%** |
| seed 6 | 39.6538 | 36.3598 | **35.7588** | **-9.82%** |
| 两组均值 | 40.2830 | 36.7212 | **35.9745** | **-10.70%** |

这是当前仓库内部、该配对协议下的最好结果；它不是对外部 benchmark 的 SOTA
声明。

一个额外的方向/幅值交换实验排除了最直接的“投影只是缩小 correction”解释：
在相同 5K bank 上，只把 raw revision 缩到 certificate 的范数，FID 仅从
`37.7279` 到 `37.6576`；保留 certificate 方向但恢复 raw 范数，则达到
`37.2113`，与完整投影的 `37.2082` 几乎相同。因而当前收益的判别信息主要在
OU 证书选择的**方向**，而不在它附带的幅值衰减。

使用 strong $D_1$ 轴的第二组交换实验进一步把两者分开：普通投影为 `37.0827`；
保留 raw 方向、只复制投影范数反而是 `37.7666`；保留证书方向并恢复 raw 范数则
达到 **`36.1902`**。三者模型查询数、FFE、noise 和 label hash 完全一致。

## 二、先把 linear flow 精确变成 OU channel

SiT 使用的线性 bridge 是

\[
Z_t=(1-t)E+tX,
\qquad E\sim\mathcal N(0,I),\quad 0<t<1.
\]

定义

\[
c_t=\sqrt{t^2+(1-t)^2},
\qquad
Y_t=\frac{Z_t}{c_t},
\]

以及

\[
a_t=\frac{t}{c_t},
\qquad
b_t=\frac{1-t}{c_t}.
\]

于是

\[
a_t^2+b_t^2=1,
\qquad
Y_t=a_tX+b_tE.
\]

令 OU 时间为

\[
s_t=-\log a_t.
\]

那么 $Y_t$ 正好是对 clean distribution 施加 OU/Gaussian channel 后的
随机变量。这里没有近似；只是对原 linear bridge 作了确定性归一化。

这个变换回答了“跨时间应该在哪个空间比较”的问题。相同 raw latent $Z$ 在
不同时间并不代表相同的标准化位置。固定 OU 坐标 $Y=y$ 时，未来 raw query
应当取

\[
Z_\tau=c_\tau y
=\frac{c_\tau}{c_t}Z_t.
\]

### 这个结构不依赖 linear timestep 参数化

更一般地，任意仿射 Gaussian bridge

\[
Z_t=\alpha_tX+\sigma_tE,
\qquad E\sim\mathcal N(0,I)
\]

只要 $\alpha_t,\sigma_t>0$，都可定义

\[
c_t=\sqrt{\alpha_t^2+\sigma_t^2},
\qquad
a_t=\frac{\alpha_t}{c_t},
\qquad
Y_t=\frac{Z_t}{c_t}.
\]

于是精确地有

\[
\boxed{
Y_t=a_tX+\sqrt{1-a_t^2}\,E,
\qquad s_t=-\log a_t.
}
\]

因此只要 signal-to-noise ratio 沿采样方向单调，任意 VP/VE/flow 风格的仿射
Gaussian path 在这个归一化坐标中都由同一 OU semigroup 参数化；原始时间表只是在
重新参数化 $s_t$。

velocity 到 score 的转换也有闭式。令

\[
\lambda_t=\frac{\dot\alpha_t}{\alpha_t},
\qquad
\kappa_t=\sigma_t(\lambda_t\sigma_t-\dot\sigma_t),
\]

并令 $v_t(z)=\mathbb E[\dot\alpha_tX+\dot\sigma_tE\mid Z_t=z]$。由 Tweedie
恒等式 $\mathbb E[E\mid Z_t=z]=-\sigma_t\nabla_z\log p_t(z)$ 得

\[
\boxed{
\nabla_z\log p_t(z)
=\frac{v_t(z)-\lambda_tz}{\kappa_t},
\qquad
r_t(y)=c_t\nabla_z\log p_t(z)+y.
}
\]

当 $\alpha_t=t,\sigma_t=1-t$ 时，它正好退化为本文实现使用的公式。因而 OU
certificate 的理论对象是 **Gaussian-channel/SNR canonical** 的；当前代码和正式
实验只实现了 linear-flow 特例，尚未把跨 schedule 迁移当作已有实验结论。

## 三、OU 半群给出精确的概率谱

记 $\phi$ 为标准 Gaussian 密度，$\mu_t$ 为 $Y_t$ 的边缘分布，并定义

\[
f_t=\frac{d\mu_t}{d\phi}.
\]

OU 半群 $P_s$ 满足

\[
f_t=P_{s_t}f_0.
\]

在 $L^2(\phi)$ 中把 clean density ratio 展开成 Hermite degree：

\[
f_0=1+\sum_{k\ge1} f^{(k)},
\]

则 OU 半群逐阶对角化：

\[
\boxed{
f_t
=1+\sum_{k\ge1}a_t^k f^{(k)}.
}
\]

因此 diffusion time 不只是采样日程，它也是一条具有已知特征值
$a_t^k$ 的概率 scale axis：

- $k=1$：location/mean shift；
- $k=2$：covariance 和二次形状；
- $k\ge3$：更高阶、非 Gaussian 或多模态形状；
- 高 degree 在强噪声下更快衰减。

这就是本文所说的“概率频率”。它与图片边缘、纹理的 Fourier 高频没有恒等关系。
这里的 location 也只是**标准化 latent 分布的 degree-1/均值模态**，不是图中物体的
二维位置；方法只从额外 revision 中消去该模态，并不从基础 strong/IG 场中删除
类别语义或全局 transport。

## 四、从 density ratio 到模型能提供的 relative score

模型没有直接输出 $f_t$，但 velocity 可以精确转换成标准化空间的 relative
score：

\[
r_t(y)
=\nabla_y\log\mu_t(y)-\nabla_y\log\phi(y)
=\nabla_y\log f_t(y).
\]

若 raw velocity 是 $v_t(z)$，则

\[
\boxed{
r_t(y)
=c_t\frac{t v_t(z)-z}{1-t}+\frac{z}{c_t},
\qquad y=\frac{z}{c_t}.
}
\]

这个 velocity-score 变换是精确的。

真正需要近似的是下一步。高噪声时 $f_t$ 接近 1，写成

\[
f_t=1+\varepsilon h_t,
\]

则

\[
r_t=\nabla\log(1+\varepsilon h_t)
=\varepsilon\nabla h_t+O(\varepsilon^2).
\]

代入 Hermite 展开：

\[
\boxed{
r_t(y)
=\varepsilon\sum_{k\ge1}a_t^k\nabla f^{(k)}(y)
+O(\varepsilon^2).
}
\]

所以 relative score 在近 Gaussian 区域一阶继承了 OU 的 degree-wise 衰减；
到了低噪声，$\log$ 的非线性会耦合不同 degree，这个简单谱解释不再可靠。

## 五、一个由不变量唯一决定的两抽头概率小波

取 $t<\tau$，并始终在相同 OU 坐标 $y$ 查询。考虑归一化为当前系数 1 的
两时间线性滤波器

\[
D(y)=r_t(y)+B r_\tau(y).
\]

我们提出一个独立于 FID 的语义要求：

> **若概率路径只有 location/mean shift，额外的“细节修订”应当严格为零。**

在 degree-1 路径上

\[
r_t^{(1)}=a_t m,
\qquad
r_\tau^{(1)}=a_\tau m.
\]

要求对任意 $m$ 都有 $D=0$，唯一得到

\[
B=-\frac{a_t}{a_\tau}.
\]

因此唯一的归一化 degree-1 消去器是

\[
\boxed{
D_1(y;t,\tau)
=r_t(y)-\frac{a_t}{a_\tau}r_\tau(y).
}
\]

这与 wavelet 的 vanishing moment 完全同构：两抽头滤波器不响应最粗的
location mode，只响应跨概率尺度出现的形状变化。

更重要的是，$D_1$ 不只在 Hermite 线性化里有意义。令

\[
a=\frac{a_t}{a_\tau}<1,
\]

并把转移相关系数为 $a$ 的 OU kernel 记为 $Q_a$。因为

\[
f_t=Q_a f_\tau,
\qquad
\nabla Q_a h=aQ_a(\nabla h),
\]

所以

\[
\nabla f_t=aQ_a(f_\tau r_\tau).
\]

除以 $f_t=Q_a f_\tau$，由 Bayes 公式得到对任意足够光滑分布都成立的
**精确条件分数恒等式**：

\[
\boxed{
r_t(y)
=a\,\mathbb E\!\left[r_\tau(Y_\tau)\mid Y_t=y\right].
}
\]

因此

\[
\boxed{
D_1(y;t,\tau)
=a\left(
\mathbb E[r_\tau(Y_\tau)\mid Y_t=y]-r_\tau(y)
\right).
}
\]

右边比较的是两件不同的事：第一项是所有与当前 noisy observation 相容的 cleaner
states 的合法后验平均；第二项则把 cleaner score **反事实地查询在同一个数值坐标
$y$ 上**。所以 $D_1$ 可以被理解为一个 frozen-coordinate posterior-score
innovation。这里“反事实”很重要：$Y_\tau=y$ 通常不是 OU coupling 下真实发生的
cleaner state。

若 clean density ratio 可微，并记

\[
q(x)=\nabla\log f_0(x),
\qquad
M_t(y)=\frac{r_t(y)}{a_t},
\]

则同样由梯度交换律得到

\[
\boxed{
M_t(y)=\mathbb E[q(X)\mid Y_t=y].
}
\]

沿真实 OU Markov coupling，它满足

\[
\boxed{
\mathbb E[M_\tau(Y_\tau)\mid Y_t]=M_t(Y_t),
}
\]

即一个随信息增加而更新的后验 martingale。与此同时

\[
\frac{D_1(y;t,\tau)}{a_t}=M_t(y)-M_\tau(y)
\]

不是实际 martingale increment，而是把两个后验估计固定在同一坐标后得到的
**反事实信息创新**。这给 $D_1$ 一个不依赖近 Gaussian 展开的精确定义；Hermite
degree 只是它在高噪声端的谱解释。

同一个对象还有一个更直接的动力学解释。用 $s=-\log a$ 表示 OU semigroup
time，记标准 Gaussian 为 $\phi$，并定义

\[
f_s=\frac{p_s}{\phi},\qquad
\ell_s=\log f_s,\qquad
r_s=\nabla\ell_s.
\]

OU generator 为

\[
L=\Delta-y^\top\nabla,
\]

且 $\partial_s f_s=Lf_s$。由链式法则

\[
\partial_s\ell_s=L\ell_s+\|r_s\|^2.
\]

再取梯度，并使用 $\nabla L\ell=L\nabla\ell-\nabla\ell$ 以及
$J_{r_s}=\nabla^2\ell_s$ 的对称性，得到 OU 特例下的 score Fokker--Planck
方程：

\[
\boxed{
(\partial_s+1)r_s
=Lr_s+2J_{r_s}r_s
\equiv\mathcal C[r_s].
}
\]

这里 $-r_s$ 是所有 location score 都服从的平凡指数衰减；
$\mathcal C[r_s]$ 则是扣除该衰减后剩余的概率形状动力学。令
$s_t=s_\tau+\delta$，其中 $t$ 比 $\tau$ 更 noisy，则乘以积分因子 $e^s$ 后
可精确积分为

\[
\boxed{
D_1(y;t,\tau)
=r_{s_t}(y)-e^{-\delta}r_{s_\tau}(y)
=\int_{s_\tau}^{s_t}
e^{-(s_t-u)}\mathcal C[r_u](y)\,du.
}
\]

这个式子对有限 $\delta$ 精确成立，不是假设 horizon 很小的 Taylor 展开。等价地，
定义去掉平凡 location 衰减的 interaction-coordinate score

\[
\widetilde r_s=e^s r_s,
\]

就有

\[
\boxed{
e^{s_t}D_1
=\widetilde r_{s_t}-\widetilde r_{s_\tau}.
}
\]

因此 $D_1$ 的最短动力学解释是：**它是在先除去 OU 的通用 location 衰减后，
分数随概率尺度发生的有限变化。** 对纯平移该变化严格为零；协方差、偏斜和多峰
结构都会触发它。计算上，$D_1$ 用两次前向差分实现了这个含空间二阶导数的
score-FPE 积分，不需要显式 Jacobian 或 Hessian。

特别地，若数据只是标准 Gaussian 的平移

\[
X\sim\mathcal N(m,I),
\]

则 $r_t(y)=a_t m$ 对 $y$ 为常数，因而

\[
D_1\equiv0.
\]

所以“纯 location 不应触发 detail revision”不仅是一阶近似要求，而是对整个
translated-Gaussian family 的精确不变量。

这个结论还在不含神经网络、不含 FID 的一维可解模型中作了数值审计。对单 Gaussian
和平滑 Gaussian mixtures，右侧条件期望由 80 点 Gauss--Hermite quadrature 独立
计算；所有时间和分布上的 identity residual 最大值不超过 `1.3e-15`。在
$t=.05,\tau=.08125$ 时：

| 分布 | $\|D_1\|_{\rm RMS}$ | $\|D_2\|_{\rm RMS}$ | $\|D_1/a_t\|_{\rm RMS}$ |
|---|---:|---:|---:|
| translated $\mathcal N(m,I)$ | $4.8\times10^{-17}$ | $3.18\times10^{-2}$ | $9.2\times10^{-16}$ |
| covariance-shifted Gaussian | $2.76\times10^{-3}$ | $3.08\times10^{-5}$ | $5.25\times10^{-2}$ |
| symmetric bimodal mixture | $5.80\times10^{-3}$ | $2.03\times10^{-4}$ | $1.10\times10^{-1}$ |
| skewed mixture | $1.88\times10^{-3}$ | $7.98\times10^{-4}$ | $3.58\times10^{-2}$ |

这张表也给出一个与 ImageNet control 独立的判别：$D_1$ 精确忽略 location、响应
shape；$D_2$ 则会响应 location，并在高噪声端消去以 covariance 为首项的 shape。

另一个独立解析审计直接计算了 $\mathcal C[r_s]$，并用 48 点
Gauss--Legendre quadrature 计算上面的有限区间积分。对 covariance Gaussian、
symmetric bimodal 和 skewed mixture，在五个时间点上积分与 $D_1$ 的 cosine 均为
`1.0`，最大绝对残差不超过 `3.3e-15`；translated Gaussian 两侧都处于
`1e-16` 数值零量级。这验证的是有限尺度恒等式，而不是小 horizon 近似。

在线性化下，density degree $k$ 的响应系数是

\[
\boxed{
H_k(t,\tau)
=a_t^k-\frac{a_t}{a_\tau}a_\tau^k
=a_t\left(a_t^{k-1}-a_\tau^{k-1}\right).
}
\]

于是 $H_1=0$，而 $k\ge2$ 一般非零。由于两个时间的 OU smoothing 都会
压低极高 degree，它严格说是**概率 band-pass**，不是无限单调的 high-pass。

## 六、为什么不是直接把 $D_1$ 加进 sampler

真实 PFR 的 time revision 是

\[
R(z,t)=W(z,t)-W(z,t+h),
\]

其中两个弱头在相同 raw $z$ 上查询。它同时混合了：

1. 概率尺度变化；
2. raw/normalized 坐标变化；
3. 有限神经网络的 approximation bias；
4. 低噪声下的 nonlinear mode coupling。

$D_1$ 则是在固定 OU 坐标上构造的理论 scale detail。在一个配对银行上直接用
RMS-matched $D_1$ 的 FID-1K 为 `65.8544`，明显不如原 PFR；所以理论算子本身不是
一个足够的 guidance vector。

更稳健的做法是把它当成**证书**，而不是 teacher。给定每个样本的 $R,D_1$，
求与原修订最接近、但必须位于理论方向上的向量：

\[
\boxed{
R_{\rm cert}
=\arg\min_{u\in\operatorname{span}(D_1)}\|R-u\|_2^2
=\frac{\langle R,D_1\rangle}{\|D_1\|_2^2}D_1.
}
\]

这正是实现中的 per-sample orthogonal projection。它没有拟合参数；而且

\[
R=R_{\rm cert}+R_{\rm unsupported},
\qquad
R_{\rm unsupported}\perp D_1
\]

给出了严格的因果消融。

这里必须区分两种“投影”。Hermite degree 的正交分解发生在 density/score
函数空间 $L^2(\phi)$；上式则是在每个采样状态的 latent tensor 上作欧氏投影。
后者不是对完整 Hermite 子空间的严格函数空间投影。理论精确提供的是局部证书轴
$D_1(z,t)$，实验检验的是 raw revision 沿这条轴的分量是否有用；不能把实现直接
等同为恢复了全部 probability-shape component。

## 七、为什么只在第一段使用谱证书

当前 IG 原协议已经分成

\[
[0,0.25),\quad[0.25,0.5),\quad[0.5,1].
\]

新方法没有从 FID 再选择一个 cutoff，而是在原有第一段使用 $R_{\rm cert}$，
第二段使用原始 $R$，最后一段无 revision。

这样做的理由有理论和观测两部分：

- 理论上，Hermite score 线性化只应在高噪声、接近 Gaussian fixed point 时可信；
- 实测 $D_1$ 与 raw revision 的 cosine 从 $t=.05$ 的 `0.969`，降到
  $t=.20$ 的 `0.733`、$t=.40$ 的 `0.443` 和 $t=.46875$ 的 `0.247`；
- 全区间强制使用 common 的 FID-1K 均值约为 `63.70`，不如只在第一段使用的
  `62.07`。

因此这里不是说低噪声的分量“没有用”，而是说 OU 线性概率谱已不足以认证它；
在没有更强理论前保留经验 raw revision 更诚实。

## 八、degree-1 不是事后随便挑出的投影方向

为了检验“任意相关 OU 向量投影都会改善”的反解释，使用完全相同的代码、查询数、
Heun-32、样本、标签和 FID reference，构造 degree-2 消去器

\[
D_2=r_t-\left(\frac{a_t}{a_\tau}\right)^2r_\tau.
\]

它消掉二次/covariance mode，却不消掉 mean mode。两组严格配对 FID-1K 为：

| 条件 | seed 20260904 | seed 20260905 | 均值 |
|---|---:|---:|---:|
| ordinary IG | 66.9619 | 65.9055 | 66.4337 |
| degree-1 common-first | **62.0006** | **62.1512** | **62.0759** |
| degree-1 unique-first | 65.7160 | 63.7827 | 64.7494 |
| degree-2 common-first | 66.2889 | 65.4531 | 65.8710 |
| degree-2 unique-first | 64.7667 | 63.9252 | 64.3460 |

四个条件在各自 seed 内的 noise SHA256 与 label SHA256 完全一致，调用数也完全
一致。优化前后的 degree-1 重复运行因 CUDA kernel 非 bitwise deterministic，FID
只漂移 `0.008--0.011`，远小于 degree-1/degree-2 的差距。degree-2
common 几乎退回 ordinary；degree-1 common 在两个银行都大幅更好。这个对照支持：

\[
\boxed{
\text{有用的不是任意 OU 投影，而是保留 shape、消去 location 的特定 vanishing moment。}
}
\]

它还没有证明所有数据集上 degree-1 必然最优，但已否掉当前最直接的任意投影解释。

由于最新最好方法会在选定方向上恢复 raw revision 的范数，还需要排除一个更严格的
反解释：上表中 degree-2 是否只是被普通投影额外缩小，若同样恢复范数也会与 degree-1
一样好？为此使用同一个 strong head 构造 $D_1/D_2$，并对二者都求完全相同的
norm-preserving polar 解。Heun-22、模型查询数、FFE、noise 和 label 均逐项相同：

| seed | ordinary IG, H32 | strong $D_1$-polar, H22 | strong $D_2$-polar, H22 |
|---:|---:|---:|---:|
| 20260904 | 66.9619 | **61.6037** | 65.5041 |
| 20260905 | 65.9055 | **61.5506** | 64.2561 |
| mean | 66.4337 | **61.5771** | 64.8801 |

$D_2$-polar 在两个 bank 中仍比 ordinary IG 好 `1.46/1.65`，因此它不是完全无效的
方向；但 $D_1$-polar 分别再优 `3.90/2.71`，两 seed 平均优势为 `3.3030 FID-1K`。
这排除了“任何 OU 消去器只要恢复到 raw 大范数都一样有效”：决定主要收益的是
degree-1/location null 所保留的 probability-shape 方向，而不是 polar normalization
本身。

## 九、100 类审计直接检验 location/shape 分解

为了不只依赖 FID 的间接解释，我们在相同标准 Gaussian OU-coordinate bank 上，
对 ImageNet-100 的所有类别逐类评估；每类 128 个 state，并用两个独立 bank 重复。
对任意 field $u(y,c)$ 作 sampled decomposition：

\[
u(y,c)=\bar u_c+\left(u(y,c)-\bar u_c\right),
\]

其中第一项是 class-constant/location-like energy，第二项是类内随 state 变化的
shape-like energy。报告使用有限样本无偏修正，避免把类内方差泄漏进类均值能量。

degree-1 与 degree-2 defect 的无偏 class-constant energy fraction 为：

| $t$ | $D_1$, seed 04 | $D_1$, seed 05 | $D_2$, seed 04 | $D_2$, seed 05 |
|---:|---:|---:|---:|---:|
| .02 | .153 | .136 | .743 | .753 |
| .05 | .203 | .209 | .455 | .475 |
| .10 | .199 | .209 | .259 | .268 |
| .20 | .070 | .071 | .157 | .161 |
| .30 | .026 | .026 | .111 | .115 |
| .40 | .022 | .022 | .068 | .070 |

因此在最关键的高噪声区，$D_2$ 大量携带 class-constant location signal，而
$D_1$ 显著更少；结果跨两个随机 bank 高度稳定。有限弱头并不是 population
OU score，所以 $D_1$ 的 class-constant fraction 没有严格为零，这恰好量化了
模型误差与理论对象之间的边界。

另一个直接结果是 $D_1$-common 占 raw revision 的能量比例会自然随时间衰减：
seed 04 在 $t=.02,.05,.10,.20,.30,.40$ 分别为
`.986, .945, .791, .319, .129, .010`，seed 05 几乎相同。这不是人为设定的
时间 schedule，而是理论证书本身在离开 Gaussian 端后逐渐失去解释力。

## 十、因果拆分支持“谱认证分量”，而不是只支持相关性

在两个配对 1K 银行中：

- early degree-1 common + raw late：`62.0006 / 62.1512`；
- early degree-1 unique + raw late：`65.7160 / 63.7827`；
- full projected PFR：`62.1853 / 61.8294`；
- time-only PFR：`62.5167 / 62.3615`；
- ordinary IG：`66.9619 / 65.9055`。

也就是说，把同一个 raw revision 精确拆成 common 与 orthogonal unique 后，主要
终点作用跟随理论 common；被剔除的 exact complement 在两个银行都明显更弱。

为了检验这个方向是否只是 depth-4 弱头的私有 approximation bias，又保持 raw
revision 不变，只把认证轴换成 strong v800 在固定 OU 坐标上的 $D_1$。额外完整
query 由 Heun-22 补偿到等计算。两个配对 1K 银行中：

- strong-$D_1$ common：`62.1207 / 62.5272`；
- strong-$D_1$ unique：`65.8297 / 64.2567`。

common 在两个银行都明确优于 exact unique complement。strong-common 与
weak-common 的两 seed 均值仅差约 `0.05 FID-1K`，说明被选择的更像 strong/weak
共同近似的概率形状轴，而不是某个弱头独有的误差方向。

这一点也能在不经过 FID 的局部量上看到：从 $t=.05$ 到 `.46875`，strong 与
weak 的 $D_1$ cosine 依次为 `.802, .827, .839, .813, .778, .746`。二者不是同一个
向量，但在模型、深度和近似误差不同的条件下仍维持稳定正对齐；结合两套
common/unique 因果排序，更符合“共享半群形状轴”而不是“某个 head 的偶然误差”。

FID 不是线性量，因此不能把 FID gain 的比例直接叫作“机制百分比”。这里能安全
声称的是条件排序及其跨两个配对银行的一致性。

## 十一、FID-5K 的变化与概率形状解释相容

weak certificate 相对原 projected PFR：

\[
37.5299\to37.2123.
\]

拆开 ADM FID：

- mean component：`5.8926 -> 5.9054`，几乎不变且略差；
- covariance component：`31.6373 -> 31.3069`，改善 `0.3304`；
- sFID：`68.9555 -> 67.5400`，改善 `1.4155`；
- IS：`40.8774 -> 40.9181`。

这与“保留已有语义/均值 transport，筛出分布形状修订”的解释相容。它仍然只是
一致证据，不等于 Hermite degree 与 Inception covariance 一一对应的定理。

更严格的 strong certificate 在相同 5K 银行上进一步得到：

\[
\boxed{
\operatorname{FID}=37.0840,
\quad
\operatorname{sFID}=67.5429,
\quad
\operatorname{IS}=41.1875.
}
\]

它相对 weak certificate 的 FID 再改善 `0.1283`；变化仍主要来自 covariance
component：`31.3069 -> 31.1562`，mean component 则
`5.9054 -> 5.9278`。这与 shape-certificate 解释方向一致，但单个 FID-5K bank
不足以把 `0.1283` 宣称为稳定方法增益；更可靠的是 strong common/unique 的跨 bank
因果排序。

## 十二、等计算比较是成立的

固定 Heun 中，每步两次完整 strong/depth-4 paired forward。额外弱头 prefix query
按实测成本计为完整 forward 的 `0.391`。

ordinary IG, Heun-32：

\[
64\ \text{full calls/batch}=64\ \text{FFE/batch}.
\]

OU common-first, Heun-25：

\[
50\ \text{full calls}+34\ \text{prefix calls},
\]

\[
50+0.391\times34=63.294\ \text{FFE/batch}.
\]

5K、batch size 4 时分别是 `80,000` 与 `79,117.5` FFE。新方法并不是用更多
模型计算换 FID。

strong certificate 的第一段每个 active evaluation 使用一次 weak prefix query
和一次 future strong full query。Heun-22 的预算为

\[
44+23\times0.391+11=63.993
\]

FFE/batch，即 5K 共 `79,991.25`，也没有超过 ordinary IG。实测 integration wall
time 为 `417.8s`，ordinary 为 `399.8s`，慢约 `4.5%`；因此这里只能声称
full-forward-equivalent/FLOPs 近似等价，不能声称 wall-clock 加速。

## 十三、无阈值的能量自适应版本

`common-first` 沿用了原 IG 已有的 $t=.25$ 分段。为了检验能否由理论量本身
决定退火，而不再依赖离散 cutoff，定义每个样本的 explained-energy ratio

\[
q(t)=\frac{\|R_{\rm cert}(t)\|_2^2}{\|R(t)\|_2^2}.
\]

因为 $R_{\rm cert}\perp R_{\rm unsupported}$，有 $q\in[0,1]$。由此得到

\[
\boxed{
R_{\rm adapt}
=R_{\rm cert}+(1-q)R_{\rm unsupported}.
}
\]

它没有新增阈值、指数或可学习参数：当 OU 证书几乎解释全部 revision 时，自动
退回 $R_{\rm cert}$；当证书失去解释力时，自动恢复 raw revision。两组配对
FID-1K 中，Heun-32 的均值为 `61.9525`；在等计算 Heun-23 下均值为
`62.4120`，与分段版 Heun-25 的 `62.3751` 基本相同。

正式 FID-5K 为：

\[
\boxed{
\operatorname{FID}=37.3211,
\qquad
\operatorname{sFID}=67.3602,
\qquad
\operatorname{FFE}=79{,}982.5.
}
\]

它相对 ordinary IG 改善 `8.78%`，但 FID 比分段版 `37.2123` 差 `0.1088`；
sFID 则略好。结论应当保守：理论能量足以无参地产生合理的连续退火，并保留绝大
部分收益，但当前简单 shrinkage 还没有严格支配分段结构。

## 十四、两尺度证书没有产生可辨认增益

为了检验单尺度 $D_1(t,t+h)$ 是否只是一个过窄的证书，我们构造第二个完全由
同一理论导出的 dyadic 尺度 $D_1(t,t+2h)$，再把 raw revision 逐样本正交投影到

\[
\operatorname{span}\{D_1(t,t+h),D_1(t,t+2h)\}.
\]

两个基向量都精确消去 OU-consistent degree-1/location path；系数由最小二乘唯一
决定，没有新增可调 gain、cutoff 或训练参数。为补偿第三次 prefix query，使用
Heun-23，其预算为 `63.595 FFE/batch`；单尺度 Heun-25 为
`63.294 FFE/batch`，二者只差 `0.48%`。

两个严格配对的 balanced FID-1K 银行得到：

| seed | 单尺度，H25 | 两尺度 span，H23 | 差值 |
| ---: | ---: | ---: | ---: |
| 20260904 | 62.5988 | 62.5224 | -0.0764 |
| 20260905 | 62.1515 | 62.1130 | -0.0385 |
| mean | 62.3751 | 62.3177 | -0.0574 |

方向虽在两次重复中一致，但幅度远低于 FID-1K 的可信分辨率，而且两尺度预算还略
高。因此按预先约定的停止规则不做 FID-5K，也不继续扫尺度。这一结果限制了理论：
“更丰富的 OU-null 子空间”不会自动转化成更好的生成质量；当前强证据仍来自
degree-1/location null 本身，而不是不断增加谱基的自由度。

## 十五、per-sample 投影并未靠逐样本系数制造结果

per-sample projection 比全局 field projection 更灵活，因此需要排除一个反解释：
方法是否只是利用每张样本一个自由系数，而不是依赖 $D_1$ 的结构？我们在相同
Gaussian OU-coordinate bank 上比较三种最小二乘系数：每个样本一个系数、每类一个
系数、整个 100 类 bank 共用一个系数。两组独立 bank 的 explained-energy fraction
如下：

| $t$ | pointwise | classwise | global | global coefficient |
|---:|---:|---:|---:|---:|
| .02 | .986/.987 | .985/.986 | .985/.986 | 1.059/1.062 |
| .05 | .944/.945 | .942/.942 | .941/.941 | .967/.966 |
| .10 | .788/.799 | .787/.798 | .786/.797 | .774/.775 |
| .20 | .317/.328 | .315/.325 | .313/.323 | .411/.415 |
| .30 | .130/.133 | .129/.132 | .127/.130 | .232/.233 |
| .40 | .011/.011 | .011/.010 | .010/.009 | .064/.063 |

斜线两侧为 seed `20260904/20260905`。关键 early interval 中，pointwise 与 global
只差约 `0.1%--0.3%` 的绝对 explained fraction；每类系数标准差也只有
`.018--.029`。因此当前 per-sample 形式主要是数值上稳健的实现，而不是靠样本级
自由度拟合。更准确的观测是：early $R$ 与 $D_1$ 近似满足一个由时间决定的全局
比例关系。

## 十六、跨尺度方向稳定，但不是纯 degree-2 模态

为了判断 $D_1(t,t+h)$ 是否给出稳定方向，我们比较 $h=1/32$ 与 $2h$。两组各
256 个样本中，weak-$D_1$ 的跨尺度 cosine 在 $t=.02$ 已为 `.935--.940`，
$t=.05$ 为 `.966--.969`，从 $t=.10$ 起为 `.991--.998`；长尺度相对短尺度的
正交能量从 `.12` 很快降到 `.005` 左右。这解释了上一节两尺度 span 为什么几乎
不增加有效维度，也与“局部概率尺度方向稳定”的解释相容。

但幅值给出一个重要反证。若高噪声端完全由线性化 degree-2 模态主导，两尺度
振幅比在 $t=.02/.05$ 应约为 `2.06/2.06`；weak 实测只有约
`1.59--1.60/1.71`，反推的 descriptive effective degree 约为
`1.13--1.20/1.07--1.11`，小于 degree-1 消去后 population Hermite 模式允许的
最小 degree 2。因此 finite network 不能被描述成一个纯 Hermite mode：若
$\widehat r=r+e$，则实测 defect 还包含

\[
\widehat D_1=D_1+e_t-\frac{a_t}{a_\tau}e_\tau.
\]

换言之，跨尺度**方向**高度稳定是真实现象；把它的全部振幅都解释成 covariance
或某个单一 Hermite degree 则不成立。后续理论必须允许模型的 semigroup
inconsistency，而不能把 population identity 原封不动贴到 neural field 上。

## 十七、方向/幅值交换证明收益不是简单 attenuation

令 early raw revision 为 $R$，certificate projection 为 $C=R_{\rm cert}$，并记
每样本 RMS 为 $n(\cdot)$。我们构造两个没有新参数的交叉控制：

\[
R_{\rm norm}=\frac{n(C)}{n(R)}R,
\qquad
R_{\rm dir}=\frac{n(R)}{n(C)}C.
\]

$R_{\rm norm}$ 保留 raw 方向、只复制 certificate 的幅值；$R_{\rm dir}$ 保留
certificate 方向、只复制 raw 的幅值。二者只在原有第一段使用，之后都精确退回
raw revision。正式 balanced FID-5K 为：

| early revision | FID | FID mean | FID covariance | sFID | IS | FFE |
|---|---:|---:|---:|---:|---:|---:|
| raw direction + raw norm | 37.7279 | 5.9001 | 31.8278 | 68.8409 | 40.9134 | 73,741.3 |
| raw direction + certificate norm | 37.6576 | 5.9687 | 31.6889 | 68.5044 | **40.9778** | 79,117.5 |
| certificate direction + raw norm | 37.2113 | **5.8356** | 31.3758 | 67.5978 | 40.5913 | 79,117.5 |
| certificate direction + certificate norm | **37.2082** | 5.9048 | **31.3034** | **67.5413** | 40.9185 | 79,117.5 |

四行具有完全相同的 noise/label SHA256；后三行的模型查询数与 FFE 完全相同。
完整投影相对 raw 的数值 FID 改善为 `0.5197`；只换范数改善 `0.0703`，只换方向
改善 `0.5166`，即数值上保留 `99.4%`。FID 并非线性 functional，因此这不能叫
“99.4% 的机制被解释”；但条件排序足以否定纯 shrinkage：有用信息几乎完整地
跟随 certificate direction，而不是 certificate norm。sFID 给出相同排序；IS
没有严格支配，因此不声称所有指标都由同一分量改善。

这一步把算法的最小直觉进一步收紧为：**先由一个与 FID 无关的 OU 不变量定义
方向，再让原 PFR 决定沿该方向走多远。** 它不是数值积分校正，也不是靠降低
guidance 强度取得收益。

### strong 证书揭示了正交投影中隐藏的幅值耦合

对任意 raw revision $R$ 和证书轴 $D$，普通投影

\[
C=\frac{\langle R,D\rangle}{\|D\|^2}D
\]

必然满足

\[
\boxed{\|C\|=\|R\|\,|\cos\theta|},
\]

其中 $\theta$ 是 $R$ 与 $D$ 的夹角。因此正交投影同时做了两件事：选择证书方向，
以及按夹角缩小修订。后者并不是 OU 不变量推出的要求。

在两组各 256 个样本的独立审计中，early raw revision 与 weak $D_1$ 在
$t=.02,.05,.10,.20$ 的平均 cosine 分别约为
`0.991/0.991`、`0.969/0.967`、`0.893/0.899`、`0.733/0.757`；与 strong $D_1$
则只有 `0.816/0.831`、`0.761/0.742`、`0.679/0.682`、`0.543/0.562`。
所以 strong 证书虽然提供了更好的方向，普通投影也会把 raw 半径无意缩到约
`54%--83%`。这正好解释为什么恢复 raw 范数对 strong 证书的收益远大于 weak
证书。

若把理论证书只解释为方向约束、把 raw PFR 的有限差分解释为样本自适应作用强度，
则自然得到另一个不含可调参数的变分问题：

\[
\boxed{
R_{\rm polar}
=\arg\min_u\|u-R\|^2
\quad\text{s.t.}\quad
u\in\operatorname{span}(D),\quad \|u\|=\|R\|.
}
\]

当 $\langle R,D\rangle\ne0$ 时，其唯一解为

\[
\boxed{
R_{\rm polar}
=\operatorname{sign}\langle R,D\rangle
\,\|R\|\frac{D}{\|D\|}
=\frac{\|R\|}{\|C\|}C.
}
\]

实现使用逐样本 RMS，因所有候选维数相同，它与上面的欧氏范数约束等价。可以把
这个解理解为一个极坐标分工：**OU 消去律给角度，raw PFR 给半径。** 它不是在
经验上新增一个 guidance gain，也不是为了更好 FID 再调一项系数；它是同时满足
“证书方向”和“保留既有修订能量”时，对 raw revision 改动最小的解。

seed-5 正式 5K 方向/幅值交换为：

| strong-certificate early revision | FID | FID mean | FID covariance | sFID | IS | FFE |
|---|---:|---:|---:|---:|---:|---:|
| raw direction + certificate norm | 37.7666 | 6.1709 | 31.5956 | 68.0792 | 40.6545 | 79,991.3 |
| certificate direction + certificate norm | 37.0827 | 5.9277 | 31.1550 | **67.5420** | 41.1940 | 79,991.3 |
| certificate direction + raw norm | **36.1902** | **5.4434** | **30.7467** | 67.9882 | **41.5645** | 79,991.3 |

这组结果不证明“任何不变量方向做 norm matching 都会提升”，但它排除了当前最直接
的替代解释：收益不是由投影缩小 correction 制造的。相反，在 strong 证书上，
保留证书方向并**解除**投影收缩才得到当前最好结果。独立 seed-6 复验中，普通
strong projection 为 `36.3598`，norm-preserving 解为 `35.7588`，相对 ordinary
IG 的 `39.6538` 改善 `9.82%`；因此该排序并非来自单个 5K noise bank。

这个结论没有原样迁移到 RAEv2。冻结官方 DINOv3-L K7、沿用同一 raw-PFR
`h=1/32,rho=.05` 而不重调参数时，两个官方 FID-1K bank 的均值为：ordinary IG
`38.6583`、raw PFR `38.1535`、OU-common `38.3600`、OU-polar `38.3290`。
恢复 raw norm 能追回一部分投影衰减，但 OU-polar 仍比 raw PFR 差 `0.1755`。
更重要的是，raw PFR 自己的正式 FID-5K 从 ordinary IG 的 `7.0345` 恶化到
`7.2242`。因此 OU 消去律提供的是一个可证伪的候选方向约束，不是跨 representation
保证；SiT 上的强结果必须与这个迁移反例同时报告。

## 十八、fixed point 给了研究品味，但不是本结论

NeurIPS 2025 Foresight Guidance 最值得借鉴的是：先独立定义具有生成语义的
golden object，再让 fixed-point iteration 成为求解方式；fixed point 本身不负责
证明对象是好的。

这里借鉴的是这个研究顺序，而不是固定点形式。一个 independently meaningful
对象也可以来自不变量、等变律、守恒律或变分最优性。本轮使用的是 OU 半群等变律：
OU-consistent degree-1/location score 在 interaction coordinate $e^s r_s$ 中必须
保持不变。因此额外 shape refinement 的证书应消去这条普适漂移。$D_1$ 是对应的
唯一归一化两尺度消去器，正交投影则是沿该局部证书轴、离原 revision 最近的
最小二乘修正。这个推导不需要构造 $x=F(x)$，也不依赖数值积分误差。

本轮曾据此测试一个更“合法”的候选：OU conditional-score expectation identity
把不同时间的 score 通过 posterior conditional expectation 联系起来。我们用当前
score 的 Tweedie posterior mean 作确定性 query；该代入对 affine/Gaussian score
路径是精确的。

但它与 PFR revision 的 cosine 很快变负：weak closure 从 $t=.05$ 的 `0.272`
下降到 $t=.20$ 的 `-0.360`，之后约为 `-0.29` 到 `-0.44`。因此被数据支持的
PFR 信号不是这个 posterior-mean semigroup closure。

这并没有否定精确的 conditional-expectation identity；它否定的是“用 posterior
mean 单点闭包即可解释/替代 PFR”。本轮不把现象硬塞进 fixed point。

相反，这里的 independently meaningful object 是一个**不变量要求**：纯
OU degree-1/location mode 不应触发 detail refinement。该要求唯一导出 $D_1$，
算法再由最小二乘投影自然得到。这保留了 Foresight Guidance 的理论品味，却不受
fixed-point 形式束缚；同样的研究顺序也可以由等变律、守恒律或变分原理完成。

## 十九、现在可以声称什么，不能声称什么

可以声称：

1. linear bridge 到 OU channel 的变换是精确的；
2. relative score 的跨尺度 conditional-expectation identity 与归一化后验
   martingale 是精确的；
3. $D_1$ 等于 OU score-FPE shape operator 的有限尺度加权积分，这不依赖小步长
   展开；
4. density ratio 的 Hermite degree 在 OU 半群下具有精确特征值 $a_t^k$；
5. $D_1$ 是唯一归一化、消去任意 OU-consistent degree-1/location path 的
   两时间线性滤波器；
6. 对 translated-Gaussian location family，$D_1\equiv0$ 是精确结论；
7. 高噪声一阶展开下，它是一个 probability-scale band-pass / one-moment wavelet；
8. 在当前 SiT/IG 上，raw PFR 的 early common 分量因果上携带主要收益；
9. degree-2 matched control 与 100 类 location/shape audit 支持 degree-1 特异性；
10. strong 与 weak $D_1$ 都得到跨 seed 的 common-over-unique 因果排序；
11. pointwise/classwise/global projection 在 early interval 几乎一致，结果不是靠
   每样本投影系数的额外自由度制造的；
12. 跨 horizon 的 $D_1$ 方向高度稳定，但其振幅不能由纯 degree-2 Hermite 模式
   解释；
13. 方向/幅值交换中，生成收益跟随 certificate direction，而非幅值衰减；
14. 在“证书方向 + raw 范数”两个约束下，norm-preserving polar revision 是离 raw
   revision 最近的唯一解，并且不增加可调参数；
15. 两个独立等计算 FID-5K bank 分别改善 11.54% 和 9.82%，同时 mean/covariance、
   sFID、IS 没有出现只靠单一 FID 分量制造的明显反号。
16. 在相同 strong certificate source、raw norm、Heun-22 和 FFE 下，$D_1$-polar
   跨两个 FID-1K bank 平均优于 $D_2$-polar `3.3030`，degree-1 特异性不依赖
   投影附带的范数衰减。

不能声称：

1. 这是像素 Fourier 高频；
2. finite neural score 在全时间严格按 Hermite degree 分解；
3. 消去 degree-1 对任意数据、架构和 guidance 都必然提升 FID；
4. posterior consistency 或 fixed point 已解释 PFR；
5. 当前 5K 内部结果等于 ImageNet 公共 benchmark SOTA；
6. (t=.25) 是由理论普遍推出的相变点；
7. 当前 pointwise latent projection 等价于完整 $L^2(\phi)$ Hermite 子空间投影；
8. $D_1$ 对任意带有非 Gaussian shape 的平移族都严格不变。
9. SiT 上得到的 certificate 方向会原样迁移到 RAEv2；现有两个 FID-1K bank 与
   一个 raw-PFR FID-5K 已明确否定这一说法。

## 二十、核心直觉的最短版本

> **扩散时间隐藏着一条概率 scale-space。先除掉 OU 对 degree-1/location score
> 的通用指数衰减，translated-Gaussian path 就会成为常数；剩下的跨尺度变化精确
> 积分了 score-FPE 的非平凡动力学。这个变化也等于“合法 cleaner 后验平均”与
> “同坐标反事实 cleaner 查询”之差，因此给出一个带 vanishing moment 的局部概率
> 小波方向。OU 证书负责决定角度，原始 PFR 负责决定半径；二者的 norm-preserving
> polar projection 是满足这两个约束时离原 revision 最近的唯一解，并在略低等效
> 计算下得到更好的质量。**

这一版本比“更精确地做数值积分”更符合现有反例，也比“模型在补图像高频”更严格：
它给出了明确坐标、明确谱、唯一消去器、正交因果对照和会失败的 matched control。

## 参考

- Wang et al., *Towards a Golden Classifier-Free Guidance Path via Foresight
  Fixed Point Iterations*, NeurIPS 2025.
- Duston and Bui-Thanh, *Variance-Reduced Diffusion Sampling via Target Score
  Identity*, arXiv:2601.01594. 本文使用的中间尺度 conditional-score 形式是同一
  semigroup/gradient-commutation 恒等式的直接特例，不把恒等式本身作为新贡献。
- Lai et al., *FP-Diffusion: Improving Score-based Diffusion Models by Enforcing
  the Underlying Score Fokker--Planck Equation*, ICML 2023. Score FPE 本身是已有
  结果；本文候选贡献是识别其 OU interaction-coordinate 有限积分与 $D_1$、以及
  将它作为 training-free IG revision certificate，而不是重新声称该 PDE。
- Littlewood--Paley theory for Ornstein--Uhlenbeck semigroups. 本文只借用
  “半群尺度差 + vanishing moment”的数学语言，不声称现有定理直接推出采样收益。
- Sadat et al., *Eliminating Oversaturation and Artifacts of High Guidance
  Scales in Diffusion Models* (APG), ICLR 2025；Jin et al., *Angle Domain
  Guidance*, 2025；以及 *Magnitude-Direction Decoupling for Fast Video
  Generation with Flow Matching Models*, 2026。它们已经说明 guidance/flow 中
  “方向”和“幅值”可以分开处理，因此 polar factorization 本身不是新颖点。本工作
  的候选区别是：方向并非由 conditional prediction、latent norm 或缓存经验残差
  定义，而由 Gaussian channel 的 OU 不变量和唯一 vanishing-moment 消去律导出。
- Dieleman, *Diffusion is spectral autoregression*, 2024. 该博客只提供空间频率
  直觉；本文的概率谱结论不依赖该博客。
