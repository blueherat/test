# PFR 的指数重定时理论与机制审计

日期：2026-09-03

状态：**精确代数成立；关键机制已有配对因果实验支持；尚不宣称对 FID 的无条件改进定理。**

对应实现与数据：

- 精确代数：`experiments/pfr_exponential_retiming.py`；
- strong/weak 时序审计：`experiments/audit_pfr_exponential_retiming.py`；
- strong/weak 因果拆分：`experiments/pfr_retiming_controls.py`；
- 固定 Heun 与 FID runner：`experiments/run_imagenet100_sit_pfr_equal_compute.py`；
- 256 样本几何数据：
  `/data/users/zhoushunyu/eqvae/imagenet_sit_flow/pfr_exponential_retiming_v1/n256_strong_weak_seed20260903/`；
- 配对 FID-1K：
  `/data/users/zhoushunyu/eqvae/imagenet_sit_flow/pfr_retiming_controls_v1/fid1k_seed0_balanced/`；
- 配对等计算 FID-5K：
  `/data/users/zhoushunyu/eqvae/imagenet_sit_flow/pfr_stage_reuse_v1/fid5k_seed5_balanced/`。

---

## 一、结论先行

现在最能同时解释公式与实验的版本是：

> **在线性 flow matching 中，跨时间复用 raw velocity 不是任意的“未来查询”。
> 它唯一地对应于把未来 score 沿 Gaussian prior 到未来分布之间的指数测地线拉回当前时间。
> PFR 的 time-only 修正正是当前 score 偏离这条指数直线的有限残差。**

因此，PFR 可以精确写成两部分：

\[
\boxed{
\text{PFR}
=
\text{same-time depth contrast}
+
\text{probability-path e-geodesic defect}.
}
\]

最新因果实验又把第二项进一步收紧：

> **真正有用的主要不是 weak head 独有的时间变化，而是 strong 与 weak 对概率路径弯曲的
> 共享判断。weak-specific 的正交剩余几乎不改善 FID。**

这使 internal weak head 获得一个比“坏模型”更准确的角色：它是一个便宜、受容量约束的
**概率路径弯曲探针**。完整 strong 模型也能测到相似弯曲，但需要额外完整 forward；weak
head 用较低成本估计其中的共享成分。

这一理论明确否掉两个旧故事：

1. `W(z,t+h)` 不普遍是一个更差或更准的 teacher；它的 teacher MSE 随时间可向两个方向变化。
2. PFR query 并不是方向特异的 pointwise hard negative。许多普通 forward/inward 方向都能提高
   局部 weak-score work，甚至比 PFR 更强。

---

## 二、从线性 flow 的基本恒等式开始

训练路径为

\[
Z_t=(1-t)E+tX,\qquad E\sim\mathcal N(0,I),\quad t\in(0,1).
\]

记边缘密度为 \(p_t\)，score 为

\[
s_t(z)=\nabla_z\log p_t(z),
\]

Bayes velocity 为

\[
v_t(z)=\mathbb E[X-E\mid Z_t=z].
\]

因为

\[
Z_t=E+t(X-E),
\]

所以

\[
\mathbb E[E\mid Z_t=z]=z-tv_t(z).
\]

另一方面，Gaussian corruption 的 Tweedie 恒等式给出

\[
\mathbb E[E\mid Z_t=z]=-(1-t)s_t(z).
\]

两者相等，得到

\[
\boxed{
v_t(z)=\frac{z+(1-t)s_t(z)}{t}
}
\]

以及

\[
\boxed{
s_t(z)=\frac{t v_t(z)-z}{1-t}.
}
\]

这不是近似，也不依赖数值积分器。

---

## 三、odds 坐标揭示了真正的不变量

定义时间 odds

\[
r(t)=\frac{t}{1-t},
\]

标准 Gaussian prior 的 score 为

\[
s_\phi(z)=-z.
\]

由上式直接得到

\[
s_t(z)-s_\phi(z)
=s_t(z)+z
=r(t)\,[v_t(z)-z].
\]

因此定义

\[
\boxed{
m_t(z)
:=
\frac{s_t(z)-s_\phi(z)}{r(t)}
=v_t(z)-z.
}
\]

这个量的直觉是：

> \(s_t-s_\phi\) 是当前分布相对 Gaussian prior 的 score evidence；
> 除以 odds 后，\(m_t\) 是每单位信息 odds 的 evidence direction。

如果存在一个与时间无关的保守场 \(m(z)=\nabla F(z)\)，使

\[
m_t(z)=m(z),
\]

那么

\[
s_t(z)=\nabla_z\bigl[\log\phi(z)+r(t)F(z)\bigr].
\]

空间积分后得到

\[
\boxed{
p_t(z)\propto \phi(z)\exp\{r(t)F(z)\}.
}
\]

也就是说，整条概率路径位于从 Gaussian prior 出发的一条一维指数族直线上。时间变化只改变
自然参数 \(r(t)\)，不改变 sufficient potential \(F\)。

这给出了一个真正可证伪的不变量：

\[
\boxed{
\text{路径 e-flat}
\Longrightarrow
v_t(z)-z\ \text{在固定 }z\text{ 上与时间无关}.
}
\]

---

## 四、跨时间 raw velocity 自动定义指数重定时

取 \(0<t<\tau<1\)。在相同空间点 \(z\) 处计算未来 velocity \(v_\tau(z)\)，但把这个 raw
velocity 当作当前 \(t\) 的 velocity。它在当前时间隐含的 score 是

\[
\widetilde s_{t\leftarrow\tau}(z)
=\frac{t v_\tau(z)-z}{1-t}.
\]

代入未来时间的 velocity-score 关系，得到

\[
\boxed{
\widetilde s_{t\leftarrow\tau}
=a(t,\tau)s_\tau
+[1-a(t,\tau)]s_\phi,
}
\]

其中

\[
\boxed{
a(t,\tau)
=\frac{r(t)}{r(\tau)}
=\frac{t(1-\tau)}{(1-t)\tau}
\in(0,1).
}
\]

如果 \(s_\tau\) 是密度 \(p_\tau\) 的 score，那么右侧正好是

\[
\boxed{
\bar p_{t\leftarrow\tau}(z)
\propto
p_\tau(z)^a\phi(z)^{1-a}
}
\]

的 score。这是 \(p_\tau\) 与 Gaussian prior 的 geometric mixture，也就是指数测地线上的点。
因为 \(0<a<1\)，Hölder 不等式保证这个乘积可积；归一化常数存在。

所以，raw velocity 的跨时间复用不是随意选的坐标技巧。velocity-score 参数化强制它成为一个
**canonical exponential retiming operator**：

\[
\mathcal R_{t\leftarrow\tau}(s)
=s_\phi+\frac{r(t)}{r(\tau)}(s-s_\phi).
\]

它还有严格半群性质：

\[
\boxed{
\mathcal R_{t\leftarrow\tau}
\circ
\mathcal R_{\tau\leftarrow u}
=
\mathcal R_{t\leftarrow u}.
}
\]

这是因为 odds ratio 相乘。

---

## 五、PFR time-only 正好加入有限 e-geodesic defect

定义一个模型在 \(t\) 与 \(\tau\) 之间的指数重定时 defect：

\[
\boxed{
\delta(t,\tau;z)
:=
s_t(z)-\mathcal R_{t\leftarrow\tau}(s_\tau)(z).
}
\]

利用第三节的 \(m_t\)，它还有两个完全等价的精确形式：

\[
\boxed{
\delta(t,\tau;z)
=r(t)[m_t(z)-m_\tau(z)]
=r(t)[v_t(z)-v_\tau(z)].
}
\]

所以它测量的是：

> 当前单位-odds evidence 与未来单位-odds evidence 是否一致。

若概率路径真在一条固定指数族射线上，\(m_t=m_\tau\)，defect 恒为零。非零 defect 因而是
路径偏离 e-flat reference 的有限 chord residual。这里称它为“概率路径弯曲”时，指的正是这个
可计算定义，不是样本 ODE trajectory 的欧氏曲率。

defect 还满足 cocycle：

\[
\boxed{
\delta(t,u)
=\delta(t,\tau)
+a(t,\tau)\delta(\tau,u).
}
\]

因此不同 horizon 的 defect 不是任意有限差分，而有严格的跨尺度组合律。

---

## 六、PFR 与 AutoGuidance 的精确关系

记当前 strong/weak velocity 为 \(S_t,W_t\)，\(\beta=1+\gamma\)。ordinary IG 是

\[
G_t=W_t+\beta(S_t-W_t).
\]

由于系数和为 1，它在 score space 中对应

\[
s_G=s_{W,t}+\beta(s_{S,t}-s_{W,t}).
\]

time-only PFR 是

\[
F_t=W_t+\beta(S_t-W_\tau).
\]

将未来 raw velocity 按第四节重解释为当前 score，得到

\[
\boxed{
s_F
=s_{W,t}
+\beta\left[s_{S,t}-\mathcal R_{t\leftarrow\tau}(s_{W,\tau})\right].
}
\]

加减 \(s_{W,t}\)：

\[
\boxed{
s_F
=s_G+\beta\delta_W(t,\tau).
}
\]

因此 PFR 并不是换了一个更准的 teacher。它在 ordinary depth contrast 之外，加入 weak path
相对指数重定时基准的弯曲残差。

如果各 score 都保守且对应的乘积可归一化，那么局部 implied density 为

\[
\boxed{
\pi_{\rm PFR,t}
\propto
p_{W,t}
\left(
\frac{p_{S,t}}
{\bar p_{W,t\leftarrow\tau}}
\right)^\beta.
}
\]

这与 AutoGuidance 的 density-ratio 形式同族，但 negative reference 不再只是“一个坏模型当前的
密度”，而是：

\[
\boxed{
\text{由未来 weak density 与 Gaussian prior 唯一确定的 e-geodesic counterfactual。}
}
\]

这个说法只严格覆盖 time-only PFR。完整 projected PFR 还改变 query state；有限神经 field 又不
保证全局保守，因此完整方法不能未经证明地写成一个全局 normalized density。

---

## 七、为什么两个很大的分项单独都失败

defect 可以精确拆成

\[
\delta
=
\underbrace{(s_t-s_\tau)}_{\text{score evolution}}
+
\underbrace{(1-a)(s_\tau-s_\phi)}_{\text{Gaussian retiming}}.
\]

真实 SiT 审计显示，这两项：

- RMS 都远大于最终 defect；
- cosine 从 `-0.9987` 到 `-0.9368`；
- 相消后只剩两项总能量的 `0.16%` 到 `6.47%`。

而已有因果采样中：

- 只保留 parameterization/Gaussian transport：FID-1K `170.624`；
- 只保留 score evolution：FID-1K `271.154`；
- 精确重组 raw velocity residual：FID-1K `61.969`。

这不再只是“完整公式碰巧最好”。理论预测了只有配对系数 \((1-a)\) 才构成归一的指数重定时
reference；拆开任一大项都会破坏该结构。实验的灾难性反例与这个预测一致。

---

## 八、strong/weak 共识实验

在 ordinary depth-4 IG rollout 的 256 个样本上，分别计算 strong 与 weak 的 defect。结果为：

| \(t\) | weak defect RMS | strong defect RMS | cosine |
|---:|---:|---:|---:|
| 0.05 | 0.00401 | 0.00391 | 0.792 |
| 0.10 | 0.00621 | 0.00724 | 0.770 |
| 0.20 | 0.01285 | 0.01681 | 0.669 |
| 0.30 | 0.02296 | 0.03214 | 0.544 |
| 0.40 | 0.03776 | 0.05645 | 0.525 |
| 0.46875 | 0.05211 | 0.07824 | 0.544 |

这说明它既不是“两个头完全相同”，也不是“weak 独有方向”。早期共享更强，中后期出现显著
差异。

随后对每个样本把 weak defect 投影到 strong defect，并真正 rollout。固定 Heun-32、1000 张、
100 类严格平衡、同 noise/label bank 的结果为：

| 条件 | FID-1K ↓ | 相对 ordinary |
|---|---:|---:|
| ordinary IG | 66.925 | 0.000 |
| weak time defect | 65.018 | -1.908 |
| strong time defect | 65.463 | -1.462 |
| strong defect，按样本 RMS 匹配 weak | 64.931 | -1.994 |
| weak defect 投到 strong defect 的共享分量 | **64.622** | **-2.303** |
| weak-specific 正交剩余 | 66.278 | -0.648 |
| 完整 projected PFR | 64.886 | -2.039 |

需要谨慎解释：FID 是非线性的，所以不能把 shared/unique 的改善相加，也不能从一个 FID-1K
训练 seed 宣称普适最优。但因果排序已经明确支持：

\[
\boxed{
\text{shared path curvature}
\gg
\text{weak-specific residual}
}
\]

作为当前机制解释。

一个自然的有限模型模型是

\[
\delta_S=\kappa+\eta_S,
\qquad
\delta_W=\kappa+\eta_W,
\]

其中 \(\kappa\) 是二者共同感知的概率路径弯曲，\(\eta_S,\eta_W\) 是各自估计误差。实验支持
“共享部分更有用”，但尚未证明噪声独立、无偏或正交，因此这只是解释性统计模型，不是定理。

---

## 九、hard-negative 与 off-channel 故事为何不成立

### 9.1 方向特异性失败

在同范数方向审计中，PFR projected query 的 weak-score line work 为正，但以下方向也几乎全部
为正：current/future strong、current/future weak、guided、radial inward、weak-score ascent。
其中 radial inward 与 score ascent 的平均 work 还大于 PFR。正交与 donor 对照约为零。

所以可以说：

\[
\text{PFR query 通常提高局部 weak implied density}
\]

但不能说：

\[
\text{只有 PFR 找到一个方向特异的 hard negative}.
\]

### 9.2 “故意让 weak 变差”也失败

在真实 teacher path 上，以真实 conserved velocity 测 weak query MSE：

- \(t=0.05\) 时，推进到真实未来 state 反而更准；
- \(t=0.1\) 基本持平；
- \(t\ge0.2\) 才逐渐更差。

与此同时，PFR candidate 的局部 MSE 在中后段随未来比例增加而改善。因此 `W(q)` 自身准确度
和 residual `W_t-W(q)` 的效用不是同一个问题，不能用“off-channel degradation”统一解释。

---

## 十、FID 与计算协议审计

### 10.1 FID 口径

当前 evaluator 与 OpenAI guided-diffusion 的 ADM evaluator 对齐：官方 Inception graph、
uint8 NHWC `[0,255]`、pool-3 FID/sFID/IS。reference NPZ 恰好是 ImageNet-100 validation，
每类 50 张，共 5000 张；cached stats 与从 activations 重算的 FID 误差约 `1e-3`。

但 FID-1K 有明显有限样本偏差。真实 validation 内部做两个分层 2500 子集比较，FID 仍约
`22.63`，而 unbiased KID 约为 0。因此：

- FID-1K 只用于同 noise/label bank 的快速排序；
- 不把 FID-1K 与 FID-5K 的绝对数值横比；
- 小于约 0.3--0.5 的 FID-1K 差异只视为筛查信号；
- 关键结论必须用 FID-5K 或多独立 bank 复核。

### 10.2 等计算

实测 depth-4 prefix/full latency ratio 为 `0.391`。以 full-forward equivalent (FFE) 计：

- ordinary IG Heun-32：`64.000 FFE/batch`；
- full PFR Heun-27：`64.557 FFE/batch`；
- stage-reuse PFR Heun-29：`63.474 FFE/batch`；
- time-only PFR Heun-27：`64.557 FFE/batch`。

配对 FID-1K 中，time-only Heun-27 为 `65.589`，ordinary Heun-32 为 `66.925`。因此纯时间
曲率项在匹配计算量后仍有 `1.34` FID 的筛查收益。完整等计算 FID-5K 已有：

| 条件 | FFE/batch | FID-5K ↓ |
|---|---:|---:|
| ordinary IG Heun-32 | 64.000 | 40.912 |
| full PFR Heun-27 | 64.557 | 37.530 |
| stage-reuse PFR Heun-29 | 63.474 | 37.626 |

因此 PFR 的已观察收益不能归因于多做网络 forward。

---

## 十一、这个理论解释了什么，又没有解释什么

### 已解释并被实验支持

1. 为什么 raw velocity 跨时间差比单独 score-evolution 或坐标项稳定：它们只有按 canonical
   系数组合才形成指数测地 reference。
2. 为什么 query horizon 与 correction scale 近似按乘积折叠：小 horizon 下 defect 是路径弯曲
   的有限差分，已有固定 \(h\times\text{scale}\) 扫描在 \(h\le1/16\) 内近似重合。
3. 为什么 time-only 能拿走主要收益：它已经包含完整的指数重定时 defect；空间 query 是额外的
   conditional increment。
4. 为什么 weak 独有能量未必有用：因果投影显示共享 strong/weak 的 curvature component 才是
   主体。

### 尚未解释，不能越界

1. 没有定理保证增加 \(+\beta\delta\) 必然降低真实分布距离或 FID。
2. 有限 neural velocity 未必对应某个全局 conservative score，因此 density 公式首先是
   implied-score 解释。
3. FID 改善可能同时包含 precision/recall tradeoff；尚需正式 precision/recall 验证。
4. full projected query 的空间部分不属于纯 time-retiming 定理。
5. 当前 strong/weak 共识因果表是 FID-1K、单模型训练 seed；它适合决定机制方向，尚不足以写成
   跨模型普适结论。

---

## 十二、与现有工作的边界

- [AutoGuidance](https://arxiv.org/abs/2406.02507) 给出 strong-minus-weak 的局部
  density-ratio 解释，并明确指出逐噪声级的 implied densities 未必组成合法的全局 diffusion
  path。这里新增的是：在线性 FM 中，未来 raw velocity 唯一诱导 Gaussian-to-future 的指数
  重定时 reference，以及对应 defect 的半群/cocycle 结构。
- [The Spacetime of Diffusion Models](https://arxiv.org/abs/2505.17517) 研究 denoising posterior
  family 的 Fisher-Rao/information geometry。它证明的是 posterior spacetime 的指数族结构；
  本文推导的是 marginal score 在 raw-velocity 跨时间复用下的 geometric-mixture identity，
  两者不能混写成同一定理。
- 几何混合在 annealed importance sampling、Schrödinger bridge 与 guidance 中并不新。当前潜在
  新意不在“两个 density 做乘积”，而在 raw FM velocity 所确定的 canonical retiming、它与
  Internal Guidance 的精确组合，以及 shared-curvature 的因果证据。

---

## 十三、当前最值得发展的设计原则

现有证据支持的设计原则不是“构造更坏的 weak output”，而是：

\[
\boxed{
\text{用便宜内部头估计跨时间 e-geodesic defect，
并保留跨模型或跨尺度稳定的 curvature consensus。}
}
\]

这给出两条直接、可证伪的设计：

1. **strong-consensus control**：用完整 strong future defect 投影 weak defect。它昂贵，但已作为
   机制 oracle 获得最佳 FID-1K；可用于判断“过滤 unique residual”是否值得蒸馏。
2. **multi-horizon semigroup consensus**：只查询 cheap weak head 的两个 future horizons，利用
   cocycle/方向一致性保留跨 horizon 稳定成分，作为无需额外 full-model query 的部署近似。

第二条只有在它能够预测 strong-consensus component，并在等计算 FID 中超过单 horizon PFR 时，
才应升级为方法；否则应保留为负结果，而不是继续包装。
