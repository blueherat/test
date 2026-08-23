# AdvFD 的 selective amplification 候选反例

记录日期：2026-08-23

状态：**候选改进方向，不属于 paper-only 首轮复现。** 必须先完成独立 AdvFD
复现并冻结结果，再把本文件中的改动加入同一骨架对照。

## 1. 核心问题

AdvFD 的 real-feature whitening 约束可学习表示在真实分布下满足

\[
\mathbb E_p f=0,\qquad \operatorname{Cov}_p(f)=I.
\]

这精确消除了同时作用于 real/fake 的公共可逆仿射变换，但它没有直接约束
\(p\) 低密度或零密度区域上的 feature amplitude。神经 critic 因而可能保持真实特征
不变，只放大 fake-specific 区域。这是 population objective 自身的单边性，不依赖
finite batch、EMA 或 detach。

## 2. 一维严格反例

令

\[
p=\tfrac12\delta_a+\tfrac12\delta_b,
\qquad
q_\varepsilon=(1-\varepsilon)p+\varepsilon\delta_c,
\]

并取

\[
f_M(a)=-1,\qquad f_M(b)=1,\qquad f_M(c)=M.
\]

在 \(p\) 下始终有均值 0、方差 1，所以 real whitening 不随 \(M\) 改变。生成
特征的矩为

\[
\mu_q=\varepsilon M,
\qquad
\sigma_q^2=1-\varepsilon+\varepsilon(1-\varepsilon)M^2.
\]

一维 Gaussian Fréchet 距离因此为

\[
D_M=\varepsilon^2M^2+
\left(1-\sqrt{1-\varepsilon+
\varepsilon(1-\varepsilon)M^2}\right)^2,
\]

且

\[
D_M\sim \varepsilon M^2\rightarrow\infty.
\]

分布和 artifact mass \(\varepsilon\) 完全没有变化；增加目标只需要持续放大同一
fake-only response。

## 3. 与 Pearson chi-square 的关系

只看标量 mean branch，并令 \(q\ll p\)。在

\[
\mathbb E_p f=0,\qquad\mathbb E_p f^2=1
\]

下，满函数类极限满足

\[
\sup_f(\mathbb E_qf-\mathbb E_pf)^2
=\chi^2_{\rm Pearson}(q\|p)
=\int\frac{(q-p)^2}{p}.
\]

因此 real-only calibration 天生强烈惩罚 fake-only/low-real-density mass；若
\(q\not\ll p\)，该极限可以为无穷。这个结果与上面的离散反例是同一件事的函数空间
版本。

## 4. pooled calibration 的理想性质

令 \(m=(p+q)/2\)，改用 mixture moments whitening：

\[
\tilde f=(f-\mu_m)\Sigma_m^{-1/2}.
\]

在精确、满秩、无正则的 population setting 下，令 whitening 后
\(\mu_p=\mu\)、\(\mu_q=-\mu\)。由

\[
I=\tfrac12(\Sigma_p+\Sigma_q)+\mu\mu^\top
\]

可得 \(\|\mu\|^2\le1\) 以及

\[
\operatorname{Tr}(\Sigma_p+\Sigma_q)
=2d-2\|\mu\|^2.
\]

故完整 Fréchet/Bures 目标满足更紧的上界

\[
\begin{aligned}
D_{\rm FD}^{\rm pooled}
&=4\|\mu\|^2+d_B^2(\Sigma_p,\Sigma_q)\\
&\le4\|\mu\|^2+
2d-2\|\mu\|^2\\
&\le\boxed{2d+2}.
\end{aligned}
\]

原始分析分别粗略上界 mean/covariance 后得到 \(2d+4\)，虽然正确但不紧。
\(2d+2\) 利用了二者共享的 mixture-covariance 恒等式。

标量 mean branch 则退化为 Fisher GAN 已有的 pooled Fisher IPM：

\[
\sup_f(\mathbb E_pf-\mathbb E_qf)^2
=2\int\frac{(p-q)^2}{p+q}\le4.
\]

因此不能把 pooled mean normalization 声称为新贡献。潜在的新问题只可能是：
**AdvFD 的 full-covariance Fréchet/Bures game 中，pooled calibration 是否能消除真实
selective amplification，并改善生成 post-training。**

## 5. 必须保留的反面风险

1. bounded 不等于 useful；critic 可能很快饱和到一个有界但对 generator 无帮助的解。
2. pooled moments 依赖当前 \(q_\theta\)。若在 G-step 对 whitening statistics detach，
   forward 值仍受约束，但梯度不是精确 pooled objective 的全导数。
3. 若使用 EMA，当前批的 selective amplification 不一定在同一步被分母抵消，只会被
   后续统计追上。
4. 当前有限维上界随 feature dimension 线性增长，不能单独保证 critic 泛化。
5. Fisher GAN、McGAN、MMD-GAN 已覆盖大部分“联合统计校准/可学习特征矩”邻域；最终
   novelty 必须来自 AdvFD 特有的真实失败和生成收益，而不是换一个 normalization 名字。

## 6. 生死实验顺序

在 paper-only AdvFD 首轮冻结以后：

1. **解析反例**：扫描 \(M\)，验证 real-whitened FD 发散、pooled-whitened FD 有界。
2. **固定生成器的 artifact critic**：只给少量 generated images 注入固定 artifact，
   比较 raw/real/pooled critic 的 train/held-out FD、real/fake feature RMS、输出谱和
   artifact response 是否持续放大。
3. **同骨架短训**：在 pMF-B 上从同一 checkpoint 比较 static FD、AdvFD 和 pooled
   AdvFD；先看稳定性、held-out feature FD、precision/recall 和对
   \(\lambda_{adv}\) 的敏感性，不直接追求 SOTA。

停止标准：若真实 critic 中没有 selective amplification，或 pooled calibration 虽然
有界却系统性削弱 generator improvement，则停止把它发展成方法。

## 7. 与当前 cleanroom 工作的边界

- paper-only 首轮严格使用 AdvFD 论文的 real whitening；
- pooled 只作为首轮结果冻结后的机制对照；
- 已有 matched-distribution critic toy 显示 real whitening 仍有明显 train/held-out gap，
  但这只支持“需要审计泛化”，尚未证明真实图像中的 selective amplification；
- 官方 AdvFD 源码继续保持封存，直到独立首轮结果完成。

主要一手来源：

- [AdvFD](https://arxiv.org/abs/2608.11205)
- [FD-Loss](https://arxiv.org/abs/2604.28190)
- [Fisher GAN](https://arxiv.org/abs/1705.09675)
- [McGAN](https://proceedings.mlr.press/v70/mroueh17a.html)
- [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401)
