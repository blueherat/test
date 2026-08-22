# 矩精确残差 Flow Matching：理论、实现与泄露审计

## 1. 结论先行

这条理论的闭式概率部分是正确的：在线性 Flow Matching 路径上，任意数据分布的
Bayes velocity 都可以在 `L2(p_t)` 意义下唯一分解成：

\[
\boxed{
v^*(z,t)=v_G(z,t)+r^*(z,t)
}
\]

其中 `v_G` 是所有仿射函数中对真实 conditional velocity 的最优投影。它只由数据
均值和协方差决定，精确负责路径的一阶、二阶矩演化；`r^*` 与全部仿射函数正交，
只负责低阶矩无法表达的非高斯结构。

以下内容需要严格区分：

1. 仿射投影公式、正交性、moment dynamics 和高斯精确性是严格命题。
2. 高噪声端的 cumulant 阶数结论是带正则条件的渐近结论。
3. “解析旁路一定改善有限神经网络”不是无条件定理。它依赖模型容量、架构和优化。
4. 当前 toy 是支持性证据，不是 ImageNet 方法结果，更不是 SOTA 证据。

## 2. 统一时间约定

本文件使用当前 ImageNet-100 SiT baseline 的约定：

\[
Z_t=(1-t)E+tX,\qquad t\in[0,1],
\]

其中：

- `X` 是 clean data latent；
- `E ~ N(0,I)` 是独立高斯 source；
- `t=0` 是纯噪声；
- `t=1` 是数据；
- conditional path velocity 为

\[
V=\frac{dZ_t}{dt}=X-E.
\]

训练普通 SiT velocity model 时使用：

\[
L_{\mathrm{FM}}(\theta)
=
\mathbb E\|v_\theta(Z_t,t)-V\|^2.
\]

仓库早期 prediction-target toy 使用相反约定：

\[
\widetilde Z_s=(1-s)X+sE,
\qquad
\widetilde V=E-X.
\]

两者通过 `s=1-t`、`widetilde V=-V` 完全对应。本文所有符号均以 SiT 约定为准，
避免把 toy 中的正负号直接搬入真实训练。

## 3. 目标是 conditional mean，而不是随机配对速度本身

对平方损失，population-optimal deterministic field 是：

\[
\boxed{
v^*(z,t)=\mathbb E[V\mid Z_t=z]
}
\]

随机变量 `V=X-E` 在给定 `Z_t` 后通常仍不确定。因此训练 target 中包含不可约的
conditional variance。任意预测器 `f` 都满足：

\[
\mathbb E\|V-f(Z_t,t)\|^2
=
\mathbb E\operatorname{Var}(V\mid Z_t,t)
+
\mathbb E\|v^*(Z_t,t)-f(Z_t,t)\|^2.
\]

解析分解能改变第二项的表示与优化难度，但减去一个由 `Z_t` 决定的函数不会消除
第一项。因此不能把本方法描述成降低了不可约 target noise。

## 4. 线性路径的一阶、二阶统计

记：

\[
\mu=\mathbb E[X],
\qquad
\Sigma=\operatorname{Cov}(X),
\qquad
a_t=1-t.
\]

因为 `E` 与 `X` 独立，且 `E[E]=0`、`Cov(E)=I`，有：

\[
m_t:=\mathbb E[Z_t]=t\mu,
\]

\[
\boxed{
C_t:=\operatorname{Cov}(Z_t)=a_t^2I+t^2\Sigma
}
\]

以及：

\[
\mathbb E[V]=\mu.
\]

中心化变量为：

\[
Z_t^c=Z_t-t\mu=a_tE+t(X-\mu),
\]

\[
V^c=V-\mu=(X-\mu)-E.
\]

它们的交叉协方差为：

\[
\begin{aligned}
B_t
&:=\operatorname{Cov}(V,Z_t)\\
&=\mathbb E[V^c(Z_t^c)^\top]\\
&=t\Sigma-a_tI.
\end{aligned}
\]

`C_t` 表示当前 path marginal 的协方差，`B_t` 表示当前状态的线性变化与真实速度
之间的关系。

## 5. 最优仿射 velocity 的严格推导

考虑全部仿射预测器：

\[
g_{b,A}(z)=b+A(z-m_t).
\]

我们求：

\[
(b_t^*,A_t^*)
=
\arg\min_{b,A}
\mathbb E\|V-b-A(Z_t-m_t)\|^2.
\]

对 `b` 的 normal equation 给出：

\[
b_t^*=\mathbb E[V]=\mu.
\]

对 `A` 的 normal equation 给出：

\[
A_t^*C_t=B_t.
\]

在 `0 <= t < 1` 时，`a_t^2I` 使 `C_t` 正定，因此：

\[
\boxed{
A_t=B_tC_t^{-1}
=
[t\Sigma-(1-t)I]
[(1-t)^2I+t^2\Sigma]^{-1}
}
\]

最终得到：

\[
\boxed{
v_G(z,t)
=
\mu+A_t(z-t\mu)
}
\]

当 `t=1` 且 `Sigma` 低秩时，使用 Moore-Penrose pseudoinverse；对属于数据仿射
支撑的 `z`，公式仍取正确极限。此时矩阵在支撑外不唯一，但预测函数在 `p_t`
几乎处处意义下仍唯一。

`v_G` 有两个等价含义：

1. 它是任意数据分布的 Bayes velocity 在仿射函数空间中的 `L2(p_t)` 正交投影。
2. 它是与数据具有相同 `mu,Sigma` 的高斯分布所对应的精确 Bayes velocity。

第二个含义解释了下标 `G`，但计算它不要求真实数据是高斯。

## 6. 残差的正交性

定义随机配对残差：

\[
R_t=V-v_G(Z_t,t).
\]

由最小二乘 normal equations：

\[
\boxed{\mathbb E[R_t]=0}
\]

和：

\[
\boxed{
\mathbb E[R_t(Z_t-m_t)^\top]=0.
}
\]

真正需要网络学习的是 conditional residual：

\[
r^*(z,t)=\mathbb E[R_t\mid Z_t=z].
\]

因为条件期望保持上述矩关系：

\[
\mathbb E[r^*(Z_t,t)]=0,
\]

\[
\mathbb E[r^*(Z_t,t)(Z_t-m_t)^\top]=0.
\]

于是：

\[
v^*(z,t)=v_G(z,t)+r^*(z,t).
\]

这里的“正交”只指 `L2(p_t)` 中对常数和线性函数的正交。它不等于 manifold
切向/法向正交，也不等于网络参数梯度正交。

## 7. 为什么称为 moment-exact

真实 path 的均值导数是：

\[
\dot m_t=\mu.
\]

而解析场满足：

\[
\mathbb E[v_G(Z_t,t)]=\mu.
\]

因此 `v_G` 单独就给出精确的一阶矩速度。

协方差导数为：

\[
\dot C_t
=
-2(1-t)I+2t\Sigma
=2B_t.
\]

仿射场诱导的协方差导数是：

\[
\begin{aligned}
\mathbb E[&(v_G-\mu)(Z_t-m_t)^\top]\\
&+\mathbb E[(Z_t-m_t)(v_G-\mu)^\top]\\
&=A_tC_t+C_tA_t^\top\\
&=B_t+B_t^\top\\
&=2B_t\\
&=\dot C_t.
\end{aligned}
\]

由于 `r^*` 的均值和交叉协方差均为零，它对规定路径的一阶、二阶矩导数没有贡献。
所以这个分解不是一句“高斯近似”，而是：

\[
\boxed{
\text{解析分支精确负责 mean/covariance transport，
神经分支负责其余分布形状。}
}
\]

这里的“精确”有明确作用域：上述等式是在规定的 training marginal `p_t` 下成立。
纯 `v_G` 从标准高斯出发时会精确生成具有 `mu,Sigma` 的 moment-matched Gaussian
路径；完整 population field `v_G+r^*` 会生成真实 `p_t`。但有限网络
`v_G+r_theta` 的实际 rollout marginal 是 `q_t`，若 `r_theta` 有误，`q_t` 不一定仍
保持目标 moments。本文不把 moment-exact 误写成“任意有限 rollout 自动矩精确”。

## 8. 高斯数据时为什么神经残差严格为零

如果：

\[
X\sim\mathcal N(\mu,\Sigma),
\]

那么 `(V,Z_t)` 联合高斯。联合高斯的条件均值必为仿射函数，因此：

\[
\mathbb E[V\mid Z_t=z]
=
\mu+B_tC_t^{-1}(z-t\mu)
=v_G(z,t).
\]

所以：

\[
\boxed{r^*(z,t)=0.}
\]

这提供了一个严格 sanity check：如果目标分布就是高斯，理论方法应退化成纯解析
flow，任何非零神经 residual 都只是有限训练误差。

## 9. 残差协方差与归一化

随机 residual 的无条件协方差为：

\[
Q_t
:=\operatorname{Cov}(R_t)
=\Sigma+I-B_tC_t^{-1}B_t^\top.
\]

`C_t`、`B_t` 都是 `Sigma` 的多项式，因此彼此可交换。化简可得：

\[
\boxed{
Q_t=\Sigma C_t^{-1}.
}
\]

在 `Sigma` 的特征方向 `i` 上，若特征值为 `lambda_i`，则：

\[
c_i(t)=(1-t)^2+t^2\lambda_i,
\]

\[
a_i(t)=\frac{t\lambda_i-(1-t)}{c_i(t)},
\]

\[
q_i(t)=\frac{\lambda_i}{c_i(t)}.
\]

这里：

- `c_i` 是当前 noisy state 在该方向的方差；
- `a_i` 是解析 velocity 对当前状态的线性系数；
- `q_i` 是去除最优仿射项后，随机 residual 的方差。

可以定义 support 上的 normalized residual target：

\[
Y_t=Q_t^{\dagger/2}[V-v_G(Z_t,t)].
\]

它满足：

\[
\mathbb E[Y_t]=0,
\qquad
\operatorname{Cov}(Y_t)=P_{\operatorname{supp}(\Sigma)},
\]

并且仍与 `Z_t-m_t` 不相关。网络预测：

\[
h_\theta(Z_t,t)\approx\mathbb E[Y_t\mid Z_t,t],
\]

最终恢复：

\[
\boxed{
v_\theta(z,t)=v_G(z,t)+Q_t^{1/2}h_\theta(z,t).
}
\]

正式公平训练仍使用原始 velocity-space loss：

\[
L(\theta)=\mathbb E\|V-v_\theta(Z_t,t)\|^2.
\]

这样不同 parameterization 比较的是同一个物理 velocity error。若改为对 normalized
target 直接做等权 MSE，就额外改变了各协方差方向的 loss weighting，必须作为另一
种方法单独消融。

## 10. 非高斯信息为什么是高阶部分

高斯 source 的三阶及以上 cumulant 为零。独立随机变量的 cumulant 可加，缩放后
第 `k` 阶 cumulant 乘缩放系数的 `k` 次方。因此对 `k >= 3`：

\[
\boxed{
\kappa_k(Z_t)=t^k\kappa_k(X).
}
\]

这严格说明高噪声端 `t -> 0` 时，三阶及以上的非高斯统计比均值、协方差更快消失。

若数据具有足够有限矩，且在高斯卷积后密度光滑，可以用 Edgeworth/Gram-Charlier
展开进一步得到：若首个非零高阶 cumulant 是 `k` 阶，则 score 相对 moment-matched
Gaussian 的首项通常为 `O(t^k)`；由线性路径关系：

\[
v^*(z,t)=\frac{z}{t}+\frac{1-t}{t}\nabla_z\log p_t(z),
\]

velocity residual 通常从 `O(t^{k-1})` 出现。三阶首先非零时是 `O(t^2)`。

这一段是渐近分析，不是对任意奇异 manifold 分布和全部 `t` 的统一界。正式理论若
使用它，必须补充矩条件、余项界和维度依赖，不能直接把 `O(t^2)` 当作已证明全程规律。

## 11. 两种实现层级

最小实现是：

\[
v_\theta=v_G+r_\theta.
\]

完整预条件实现是：

\[
v_\theta=v_G+Q_t^{1/2}h_\theta.
\]

前者只移除解析仿射函数；后者还把随机 residual 的无条件协方差标准化。两者不能
混为一个消融。

为了防止有限网络重新把低阶仿射分量学回 residual，可增加 moment-neutral penalty：

\[
L_{\mathrm{mean}}=
\|\mathbb E_{\mathrm{batch}}[r_\theta]\|^2,
\]

\[
L_{\mathrm{cross}}=
\|\mathbb E_{\mathrm{batch}}[r_\theta(Z_t-m_t)^\top]\|_F^2.
\]

完整 `D x D` cross-covariance 不必显式构造。令随机 probe `xi ~ N(0,I)`，对：

\[
M=\mathbb E[r_\theta(Z_t-m_t)^\top]
\]

有：

\[
\mathbb E_\xi\|M\xi\|^2=\|M\|_F^2.
\]

因此少量 Hutchinson probes 就能给出无偏 Frobenius penalty 估计。这个 penalty 是
可选方法组件，不属于基础分解定理。

## 12. 条件生成

若类别标签为 `Y=y`，严格条件版本只需替换：

\[
\mu\rightarrow\mu_y,
\qquad
\Sigma\rightarrow\Sigma_y.
\]

此时 `v_G(z,t,y)` 是该类别条件下的最优仿射 velocity，conditional residual 对该类
的常数和线性函数正交。

真实 ImageNet 中每类样本数远小于 latent dimension，直接估计 full `Sigma_y` 很不稳。
可行顺序应是：

1. global `mu,Sigma`；
2. class mean `mu_y` + shared covariance `Sigma`；
3. train-only shrinkage class covariance；
4. 只有统计功效足够时才测试 per-class low-rank covariance。

第 2 项不再是严格的 class-conditional 最优仿射投影，只是受限 affine family，报告中
必须明确。

若训练包含 classifier-free label dropout，则 dropped/null 条件使用 global moments，
有标签条件使用对应条件 moments。不能让同一 null token偷偷访问原始类别统计。

## 13. SD-VAE latent 的统计必须怎样计算

当前 cache 存储每张图的 posterior mean 与 standard deviation。若实际训练 latent 为：

\[
X_i=\mu_i+\sigma_i\odot\xi,
\qquad
\xi\sim\mathcal N(0,I),
\]

那么不能只对 posterior mean 做 PCA。正确的 train-distribution moments 是：

\[
\widehat\mu=\frac1N\sum_i\mu_i,
\]

\[
\widehat{\mathbb E[XX^\top]}
=
\frac1N\sum_i
[\mu_i\mu_i^\top+\operatorname{Diag}(\sigma_i^2)],
\]

\[
\widehat\Sigma
=
\widehat{\mathbb E[XX^\top]}
-\widehat\mu\widehat\mu^\top.
\]

所有 VAE scaling、channel normalization 和 latent reshape 必须与 baseline 训练完全
相同。只使用 posterior mean 会系统性漏掉 encoder posterior variance，并使解析场与
真实训练分布不匹配。

## 14. 是否作弊或泄露

按以下协议实现，不构成作弊或数据泄露：

| 项目 | 是否允许 | 原因 |
|---|:---:|---|
| 用 ImageNet train latent 估计 `mu,Sigma` | 允许 | 普通 train-only 统计预处理 |
| 使用已知 Gaussian source moments | 允许 | source 是模型定义的一部分 |
| 使用当前 `z_t,t` 计算 `v_G` | 允许 | 推理时本来可见 |
| class-conditional 模型使用给定 class label | 允许 | 与 baseline 条件相同 |
| 报告统计估计和矩阵乘法 FLOPs | 必须 | 保证计算公平 |
| 用 validation 选择 shrinkage/rank | 可用于开发 | 必须与 final test 分开并登记搜索量 |
| 用 validation/test latent 估计 moments | 禁止 | 直接数据泄露 |
| 用 FID 最优结果反向选择 covariance basis | 禁止作为 confirmatory result | 指标选择泄露 |
| 用 `v800` 输出拟合解析分支再和从头 baseline 比 | 禁止或必须标为蒸馏 | 引入额外 teacher |
| 真实图像实验使用真实 manifold projector `P` | 禁止 | 不可获得的 oracle 信息 |
| 推理时访问对应 clean `X` 或配对 `E` | 禁止 | 目标泄露 |
| 用更多外部图像估计 moments 却不计入数据预算 | 不公平 | 额外训练数据 |

所以正确的真实 SiT 实验应只读取训练 cache 一次，冻结统计量，然后从头训练 baseline
和 residual model。验证集只用于评价和预先声明的超参数选择。

## 15. 现有 toy 是否泄露

现有 `run_prediction_target_spectral_preconditioning_toy.py`：

- 从 toy 的训练分布额外抽取 131,072 个样本估计均值和协方差；
- covariance seed 与训练、评价 seed 分离；
- 不读取 evaluation samples、生成结果或 metric；
- 所有条件在同一 seed 内共享初始化、训练 batch、noise 和 time；
- 所有条件统一最小化 recovered velocity MSE。

因此它没有 evaluation leakage，也没有使用逐样本 clean target 做推理。但它访问的是
一个可无限采样的 toy generator，相比有限数据训练拥有额外的 distribution samples。
这在机制 toy 中可以接受，却不能直接当作有限数据方法证据。正式 finite-data toy 应
固定同一个 training set，并只从该集合计算 moments。

另一个 `subspace_velocity`/operator hybrid 条件在线性 toy 中利用 covariance 的精确
active support，等价于获得真实线性子空间。它应继续标为 oracle upper bound，不能和
纯 LMMSE residual 合并成真实图像方法结果。

## 16. 现有三 seed 结果

统一配置为 `D=64`、output rank 16、5,000 steps。完整逐 seed CSV 位于：

`docs/data/prediction_target_spectral_preconditioning_toy_v1/`

三 seed 平均：

| curvature | condition | teacher velocity MSE | SWD-2D | ambient SWD | MMD-2D |
|---:|:---|---:|---:|---:|---:|
| 0.0 | native-x | 0.27738 | 0.06591 | 0.08289 | 0.000107 |
| 0.0 | LMMSE residual | 0.25975 | 0.03512 | 0.04390 | 0.000317 |
| 0.5 | native-x | 0.77186 | 0.04778 | 0.09759 | 0.000199 |
| 0.5 | LMMSE residual | 0.37355 | 0.04229 | 0.07316 | 0.000682 |

可以得出的结论：

1. 解析仿射 residualization 在 rank-limited toy 中稳定降低 teacher velocity MSE。
2. 线性 toy 的 ambient SWD 三 seed 均明显改善。
3. 曲面 toy 的平均 ambient SWD 改善，但单 seed 不完全稳定。
4. MMD 在两组中都没有同步改善，曲面条件尤其明显。
5. 输入 whitening 条件表现很差，说明“有闭式统计”不等于任意 whitening 都有效。

因此现有实验支持继续做 train-only phase-0 audit，但不支持“已经普遍改善生成质量”。

## 17. 有限容量论断的准确边界

精确正交分解本身不推出神经网络一定改善。若 hypothesis class `H` 对加减仿射函数
封闭，且优化能达到全局最优，则：

\[
\inf_{f\in H}\|v^*-f\|_{L2}^2
\]

与：

\[
\inf_{h\in H}\|r^*-h\|_{L2}^2
\]

可能只是重参数化，没有统计优势。

解析旁路真正改变的是有限模型的有效函数类：baseline 使用 `H`，新模型使用
`v_G+H`。在线性 rank-r 模型中，如果 affine modes 与 residual modes 的奇异子空间
正交，可以用 Eckart-Young 定理证明：baseline 的 rank 必须在两组 modes 间竞争，
而解析旁路免费保留全部 affine modes，并把 rank-r 全部留给 residual。此时旁路误差
不大于 direct rank-r error，并在 affine mode 挤出 residual mode 时严格更小。

如果两组奇异子空间不正交，或者 Transformer 已通过 residual path 几乎免费表示
`v_G`，上述严格排序不再自动成立。因此正式论文需要：

1. 在受控 rank 模型中证明带条件的 capacity theorem；
2. 在真实 SiT 中测量 affine branch 实际占用的输出/feature rank；
3. 做 scalar preconditioning 与 full moment branch 对照；
4. 检查收益是否随 width 增大而按理论缩小。

不能提前写成“解析旁路必然释放容量”。

## 18. 计算代价与可扩展性

flatten 后的 SD-VAE latent 维数约为数千。full covariance 的存储尚可，但每个 NFE
执行 dense `D x D` 乘法可能显著增加成本。理论 exact 版本与工程版本应分开：

1. full covariance：理论和小规模基准；
2. diagonal/channel covariance：EDM-like 强 control；
3. low-rank plus diagonal：主要可扩展候选；
4. Kronecker 或 stationary convolutional covariance：只有数据支持时再做。

low-rank approximation 不再精确匹配 full covariance，但仍可精确匹配所选受限 affine
family。任何 FID 改善必须同时报告额外 preprocessing、训练 FLOPs 和 sampling FLOPs。

## 19. 第一轮无训练验收

在启动真实训练前，只使用 ImageNet-100 train cache 完成：

1. 按 VAE posterior 公式估计 `mu,Sigma`；
2. 报告 spectrum、effective rank 和 shrinkage stability；
3. 计算 `v_G` 对随机 FM target 的 affine explained variance；
4. 将现有 `v800` 输出投影到 affine family，比较其投影与 `v_G`；
5. 测量 baseline 输出中 affine/non-affine 能量随 `t` 的变化；
6. 用完全独立 validation bank 只做评价，不回写统计量；
7. 估算 full、diagonal、low-rank 三种分支的实际 FLOPs。

若以下任一核心条件成立，应停止：

- train latent 基本各向同性，full branch 与 scalar control 几乎相同；
- `v800` 已以极小误差表示 `v_G`，没有可见容量负担；
- affine component 在有用时间区间解释的 predictable field energy 很小；
- low-rank 近似需要接近 full rank 才准确，计算优势消失。

通过后才做同初始化、同数据、同时间分布、同 recovered-velocity loss 的短训对照。

## 20. 最终可声明与不可声明内容

现在可以声明：

> 线性 FM path 的 Bayes velocity 存在唯一的 moment-exact affine projection；其残差
> 对常数和线性状态正交。现有 rank-limited toy 支持解析 residualization 改变有限模型
> 的 field risk 和部分生成指标。

现在不能声明：

- 它已经改善 ImageNet；
- 它一定优于 x/v/eps；
- 它降低了不可约 conditional target variance；
- 它在任意网络容量下都更优；
- 曲面 toy 的 MMD 已改善；
- 它已经具有充分新颖性或达到 SOTA。

理论主体没有使用未来 clean sample、测试集或 decoder 信息。只要真实实现严格遵守
train-only statistics 和 matched-compute protocol，它不是作弊，也不存在数据泄露。
真正尚未解决的是有限模型为什么、何时从这个精确分解中获益。这正是下一阶段需要
用容量定理和短训因果实验回答的问题。

## 21. ImageNet-100 SiT raw-only 终验与停止结论

按照同初始化协议，在 ImageNet-100 的 SD-VAE latent cache 上训练两份 SiT-S/2 到
`20,000` step：

1. `native`：直接回归 conditional velocity `X-E`；
2. `diagonal_lmmse`：用 **train split** 的逐坐标均值和方差构造解析 diagonal affine
   branch，网络只回归剩余项。

两者均使用 raw `model` 权重，不使用 EMA；采样均为无 guidance、相同 5000 个初始
噪声与类别、相同 Dopri5 配置、相同 VAE 和相同 ImageNet-100 validation reference。
训练统计文件的 SHA256 为：

```text
aa923e6e8fdca8d93a02c3863793c1c52f2591b6884d9f9676ae0b28633f6695
```

### 21.1 配对 validation velocity risk

在 5000 个固定 validation latent、固定 posterior noise、固定 FM source noise 和固定
时间上：

| 模型 | velocity MSE | residual - native |
|---|---:|---:|
| native raw | 0.8283545 | - |
| diagonal residual raw | 0.8286877 | +0.0003332 |

相对变化为 `+0.0402%`，paired 95% CI 为
`[+0.0001417, +0.0005248]`。时间分箱并非单调：残差模型在 `[0,0.3)` 与
`[0.9,1]` 略好，但在 `[0.3,0.9)` 整体更差。因此不能把总体差异解释成某个单一端点
数值问题。

### 21.2 闭环生成结果

| 模型 | FID-5K | sFID | IS |
|---|---:|---:|---:|
| native raw | **155.6983** | **84.2994** | **9.0112** |
| diagonal residual raw | 161.0571 | 89.0443 | 8.2944 |

残差模型相对 native 的 FID 恶化 `+5.3587`，sFID 恶化 `+4.7450`，IS 也下降。
两路 sampling manifest 的逐 rank noise/label SHA256 完全一致，排除了样本配对差异。

残差模型的 adaptive solver NFE 反而更少：rank 0 为 `3164`，native 为 `4166`；rank 1
也呈同样趋势。这说明解析旁路让当前场更易积分，但“更平滑/更低 NFE”没有转化为更好
的终端分布。

### 21.3 最终判断

这轮真实模型结果否定了当前方法主张，而不是否定第 5--7 节的概率恒等式：

- moment-exact affine projection 仍然数学正确；
- train-only 统计、checkpoint 绑定和配对评估未发现泄露或作弊；
- 但 diagonal residualization 在真实 SiT-S/2 上没有降低 raw validation risk，也没有
  改善生成质量；
- toy 中的 rank-limited 优势没有迁移到当前 Transformer regime。

因此本路线在当前仓库中定性为**已终止的负结果**。不再追加 EMA、长训、full
covariance、low-rank 分支或新的预条件变体。代码与小型结果保留，仅用于复核精确分解、
泄露边界和这个负迁移结论。
