# 谱加权、随机输运与生成质量：理论解释、因果复现和剩余 Gap

更新日期：2026-07-18

## 技术摘要

本轮研究经过两轮文献检索和六轮逐步收紧的因果实验，结论不是“频域加权能否
加速 diffusion”这么简单，而是：

> **固定输出谱权重改变的是神经网络看到的函数空间度量。其效果由权重基底相对
> 于模型 Jacobian/NTK、数据语义联合统计和 endpoint leverage 的方向共同决定；
> 在有限容量、随机训练和多步生成中，它不是保持原解的纯预条件器。**

现有理论已经能较好解释四件事：

1. 仅知道权重矩阵的特征值谱，不能预测优化效果；相对特征向量决定
   `J^T W J` 的有效谱。
2. 频带能量或功率谱不保留相位、跨频率依赖和语义，因此频带 proxy 可以改善而
   图像语义、mode coverage 和 FID 同时恶化。
3. teacher-state 的局部 L2 速度误差不单调控制 self-generated rollout 的 endpoint
   分布质量；误差会经过状态依赖的 tangent flow 有符号地传播。
4. 同一目标的 Monte Carlo 时间抽样会产生参数相关、各向异性的梯度噪声；非凸
   训练可以因此选择不同路径和解。

但现有理论仍不能解决一个已经严格复现的预测问题：

> **固定数据、初始化、minibatch、Gaussian bridge、优化器、全部 128,000 个时间值
> 及其边缘分布，只重新排列 1,000 个训练 step，为什么 endpoint feature-FID 能在
> `66.3--219.3` 之间变化，并使相同谱加权的相对效果在 `0.736--2.112` 之间翻转？**

非交换随机更新理论说明“顺序可以重要”，但目前没有理论或低成本统计量能够在
训练前或训练早期预测排列的符号、幅度及最终生成质量。这是本轮剩下的真实 gap。

## 一、需要区分的三个命题

### 命题 A：网络有 spectral bias

神经网络通常更快学习低频函数。Rahaman 等人的
[spectral bias 工作](https://proceedings.mlr.press/v97/rahaman19a.html)给出了理论和
实验依据；Basri 等人进一步说明
[输入密度会改变频率收敛率](https://proceedings.mlr.press/v119/basri20a.html)。

这只回答“哪些函数成分先被拟合”，不回答：

- 改变频带 loss 后是否保持相同有限模型解；
- 更快降低某个频带 MSE 是否改善生成分布；
- 多步 rollout 是否放大该方向的误差；
- 最终 FID 是否随 teacher MSE 单调变化。

### 命题 B：输出残差加权是 preconditioning

对残差 `r_theta(x,t)` 使用固定正定矩阵 `W(t)`：

```text
L_W(theta) = 1/2 E[r_theta^T W(t) r_theta]
```

其参数梯度和 Gauss-Newton 曲率近似为：

```text
grad L_W = E[J^T W r]
H_GN,W    = E[J^T W J].
```

因此它一般**不是**只改变收敛速度而保持有限模型解的参数空间预条件器。只有在
无限函数容量、每个 `(z,t)` 可独立拟合且 `W` 与样本无关时，加权和不加权平方损失
才共享条件均值 minimizer。

[Preconditioned Flow Matching](https://arxiv.org/abs/2603.02337)讨论的是对数据和
target 进行可逆、成对的路径预条件，从而改善整个 flow path 的条件数；这与只改
输出残差度量不是同一个操作。把两者都叫“预条件”会掩盖本项目最关键的解偏移。

### 命题 C：频域误差更低会带来更好生成

[Focal Frequency Loss](https://openaccess.thecvf.com/content/ICCV2021/html/Jiang_Focal_Frequency_Loss_for_Image_Reconstruction_and_Synthesis_ICCV_2021_paper.html)
说明自适应复频域误差在若干重构/生成任务中有用，但它没有证明任意 marginal
band weighting 都能改善 flow endpoint。尤其是：

- 复系数误差保留相位关系，band energy 不保留；
- image-to-image 重构目标与无条件 transport endpoint 不同；
- adaptive hard-frequency weighting 与固定 inverse-variance weighting 不同。

因此该文是相关先例，不是当前方法的理论保证。

## 二、解释框架

### 1. 权重谱必须和 Jacobian/NTK 联合看

线性化残差动力学可写成：

```text
dr/dtau ~= -K W r,
```

其中 `K = J J^T` 是当前模型和样本上的经验 NTK。真正控制收敛的是
`W^(1/2) K W^(1/2)` 的谱，而不是 `W` 单独的谱。
[NTK 理论](https://arxiv.org/abs/1806.07572)说明线性化训练沿 kernel eigendirections
进行；因此两个等谱 `W` 若和 `K` 的特征向量对齐不同，可产生完全不同的动态。

在参数空间同样有：

```text
H_eff = J^T W J.
```

若同时旋转数据 Jacobian、target 和 `W`，这是 gauge change；若只旋转 `W`，则是
真实改变目标几何。此前将“DCT 与 random 有相同 eigenvalues”理解为它们应等价，
正是忽略了这一区别。

### 2. endpoint 误差是有符号、状态依赖的路径积分

对 ODE `dx/dt = v(x,t)` 的小扰动 `delta v`，endpoint 一阶变化为：

```text
delta x(0) = integral Phi(0,t) delta v(x(t),t) dt,
```

`Phi` 是沿 rollout 的状态转移/tangent flow。于是：

- 相同大小的 velocity MSE 在不同时间、方向和状态上 leverage 不同；
- 各时间误差可相消或相长；
- teacher path 上的误差不等于 learner rollout state 上的误差；
- 一个无符号 L2 上界不能给出 FID 改善的符号。

[Flow Matching 误差界](https://arxiv.org/abs/2305.16860)需要 regularity/Lipschitz
条件才能把向量场误差传到分布误差，且仍不提供“更低 MSE 必然更低 FID”的
单调性。[DAgger](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)所刻画的
训练状态与 learner-induced states 的 covariate shift，也支持 teacher/rollout 必须
分开测量。

### 3. 功率谱是高度不充分的语义统计量

频带二阶矩：

```text
m_b = E[sum_{k in band b} |c_k|^2]
```

丢失了：

- coefficient sign/phase；
- 跨位置、尺度和方向的依赖；
- 类别比例和 mode coverage；
- feature-space 的语义几何。

Oppenheim 与 Lim 的经典
[phase 研究](https://dsp-group.mit.edu/wp-content/uploads/2024/11/ImportancePhaseSignals_1981.pdf)
说明相位对可辨识结构至关重要；Portilla 与 Simoncelli 的
[纹理统计模型](https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Portilla99)
则需要跨尺度、方向和位置的联合统计，而非只匹配 marginal power。

FID 在冻结语义特征空间比较高斯一二阶矩，见
[Heusel 等人的原始论文](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html)；
[MMD](https://www.jmlr.org/papers/v13/gretton12a.html)也明确依赖所选 kernel。
因此 raw DCT band energy 与 feature-FID 本来就测量不同对象。

### 4. 时间抽样是随机积分，也是优化路径输入

flow objective 含有对时间的积分：

```text
L(theta) = E_t E_x,eps [ell(theta; x, eps, t)].
```

训练用有限 `t` 样本近似该积分。不同时间的梯度方差和任务难度不同：

- [Improved DDPM](https://proceedings.mlr.press/v139/nichol21a.html)用
  `p_t proportional sqrt(E[L_t^2])` 降低 VLB gradient noise，同时也报告更低
  likelihood objective 并不保证更好 FID。
- [Min-SNR weighting](https://openaccess.thecvf.com/content/ICCV2023/html/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.html)
  将不同时间视作有梯度冲突的任务。
- [P2 weighting](https://openaccess.thecvf.com/content/CVPR2022/html/Choi_Perception_Prioritized_Training_of_Diffusion_Models_CVPR_2022_paper.html)
  说明不同 noise levels 对内容和细节的作用不同。
- [Adaptive timestep sampling](https://arxiv.org/abs/2411.09998)直接利用不同
  timesteps 的 gradient variance 差异。
- 2026 年预印本
  [CARV](https://arxiv.org/abs/2605.21489)用 importance sampling 和 stratified
  inverse CDF 降低 diffusion-teacher Monte Carlo 方差，但也观察到大幅降低
  gradient variance 后 downstream FID 未必改善。

这些工作解释了“为什么时间抽样方差值得控制”，但主要研究改变 `p(t)` 或 estimator
方差后的平均效果；它们没有预测固定 `p(t)`、固定 time multiset 下的训练顺序会让
谱加权相对 FID 翻转。

### 5. 顺序效应来自非交换更新，而非直方图

在局部二次近似中，一步更新为：

```text
delta_theta_(s+1) ~= (I - eta H_(t_s)) delta_theta_s - eta g_(t_s).
```

最终状态含有矩阵乘积：

```text
product_s (I - eta H_(t_s)).
```

若不同时间的曲率不对易：

```text
H_(t_i) H_(t_j) != H_(t_j) H_(t_i),
```

则相同 multiset 的不同排列也不等价。Lok 等人的
[random reshuffling 理论](https://proceedings.mlr.press/v272/lok25a.html)明确得到由
随机矩阵构成的 non-commutative polynomial，并指出 gradient-flow 近似会漏掉
离散顺序效应。

另外，SGD noise 不是球对称噪声：

- [Shape Matters](https://proceedings.mlr.press/v134/haochen21a.html)说明参数相关
  noise covariance 会改变 implicit bias；
- [anisotropic SGD noise](https://proceedings.mlr.press/v97/zhu19e.html)强调噪声
  covariance 与曲率的对齐；
- [SGD generalization bounds](https://proceedings.mlr.press/v134/neu21a.html)
  依赖沿实际训练路径的局部 gradient variance 和 smoothness。

这为顺序效应提供了定性理论。但当前使用 AdamW，二阶矩状态也依赖历史，因而实际
路径比上述 SGD 二次模型更复杂。

## 三、第一轮循环：现有理论能否解释原始反常现象

## 3.1 等谱 DCT/random 为何不同

### 复现实验

`experiments/isospectral_alignment_toy.py` 构造精确线性 least-squares：DCT-aligned
与 random `W` 拥有相同特征值，权重条件数都为 `10`。

| quantity | aligned | random |
|---|---:|---:|
| `cond(W)` | 10.0 | 10.0 |
| `cond(A^T W A)` | 10,000.0 | 2,164.79 |
| final relative parameter error | 0.24986 | 0.14063 |

两者 `W` eigenvalues 最大差仅 `2.33e-15`；联合旋转 `A,b,W` 后 Hessian 最大误差
`4.44e-16`。

### 判断

**已解释。** 同谱不等于同优化；决定量是相对 eigenvector alignment。该 toy 是
精确机制复现，不依赖非线性网络或生成指标。

结果：
`$HOME/data/eqvae/experiments/isospectral_alignment_toy/seed0_20260717_160145/`

## 3.2 band proxy 改善而语义崩坏是否矛盾

### 复现实验

`experiments/small_image_dct_sign_scramble.py` 对每张图的 DCT 系数随机翻转符号，
保留 DC。它逐 coefficient 精确保留功率，8-band power 的 fp32 最大相对误差小于
`4.1e-7`。

| dataset | reference accuracy | scrambled accuracy | feature FID |
|---|---:|---:|---:|
| FashionMNIST | 0.83545 | 0.16382 | 231.24 |
| MNIST | 0.95557 | 0.16235 | 575.27 |

把图像 clamp 到合法范围后 accuracy 仍只有 `0.17334/0.17236`，排除了纯 clipping
artifact。

### 判断

**已解释。** marginal power 不含 phase 和联合语义统计，因此它不是生成质量的
充分 proxy。频带能量变好与 mode coverage 变差可以同时成立。

结果：

- `$HOME/data/eqvae/experiments/small_image_dct_sign_scramble/fashion_mnist_seed0_20260717_180028/`
- `$HOME/data/eqvae/experiments/small_image_dct_sign_scramble/mnist_seed0_20260717_180746/`

## 3.3 teacher MSE 改善而 rollout 变差

已有 RAE 和 small-image 实验已经定位：

- teacher path 局部速度可以更准；
- middle rollout states 上同一 field difference 会反转或放大；
- endpoint 中高频 marginal 继续收缩；
- protected baseline path 能移除伤害，却没有稳定正收益。

详细证据见：

- [`TEACHER_ROLLOUT_MECHANISM_ZH.md`](TEACHER_ROLLOUT_MECHANISM_ZH.md)
- [`TRANSPORT_REVERSAL_MECHANISM_STUDY_ZH.md`](TRANSPORT_REVERSAL_MECHANISM_STUDY_ZH.md)
- [`MECHANISM_TO_QUALITY_STUDY_ZH.md`](MECHANISM_TO_QUALITY_STUDY_ZH.md)

### 判断

**机制层面基本解释。** 输出风险分配、gradient coupling、endpoint leverage 和是否
有 protected path 能解释伤害方向；teacher/rollout gap 由状态分布和 tangent flow
解释。但尚不能从训练前静态谱量预测最终 FID 数值。

## 四、第二轮循环：seed 翻转到底来自哪里

## 4.1 四因素 seed 析因

固定其余因素，在 MNIST 上交叉 data、init、training stream 的 seed 3/4：

| data | init | stream | feature-FID ratio |
|---:|---:|---:|---:|
| 3 | 3 | 3 | 1.7167 |
| 3 | 3 | 4 | 1.1219 |
| 3 | 4 | 3 | 1.2623 |
| 3 | 4 | 4 | 1.1596 |
| 4 | 3 | 3 | 1.2194 |
| 4 | 3 | 4 | 0.8612 |
| 4 | 4 | 3 | 1.4329 |
| 4 | 4 | 4 | 0.6791 |

log-ratio factorial effect 的绝对值：

| term | absolute log effect |
|---|---:|
| training stream | 0.4012 |
| data | 0.2537 |
| three-way interaction | 0.1849 |
| data:stream | 0.1461 |
| initialization | 0.0877 |

结论：初始化不是主要 trigger，training stream 最大，但有显著交互。

结果：
`$HOME/data/eqvae/experiments/small_image_seed_factorial/factorial_20260717_190808/`

## 4.2 training stream 分解

先把 stream 分成 minibatch indices 与 stochastic-interpolant bridge：

| batch seed | bridge seed | feature-FID ratio |
|---:|---:|---:|
| 3 | 3 | 1.43295 |
| 3 | 4 | 0.99723 |
| 4 | 3 | 2.24088 |
| 4 | 4 | 0.67908 |

bridge log effect 为 `-0.7782`，batch log effect 仅 `+0.0315`。普通数据顺序不是
主因。

再将 bridge 分成 Gaussian noise 与 time draws：

| noise seed | time seed | feature-FID ratio |
|---:|---:|---:|
| 3 | 3 | 2.24088 |
| 3 | 4 | 1.16915 |
| 4 | 3 | 1.64143 |
| 4 | 4 | 0.67908 |

time log effect 为 `-0.7666`，noise 为 `-0.4273`。时间序列是最大来源，Gaussian
bridge noise 次之。

结果：

- `$HOME/data/eqvae/experiments/small_image_stream_factorial/factorial_20260717_205045/`
- `$HOME/data/eqvae/experiments/small_image_bridge_factorial/factorial_20260717_214818/`

### 判断

**因果来源已定位，符号尚未解释。** 这一步把“随机 seed”收紧成时间抽样序列，
但还不能判断是时间覆盖还是顺序。

## 五、第三轮循环：时间分布是否解释翻转

## 5.1 精确时间序列审计

`experiments/small_image_time_sequence_audit.py` 严格按原训练循环的 RNG 调用顺序
重放 `indices -> Gaussian noise -> t`，不训练模型。

seed 3/4 各含 128,000 个时间值：

| statistic | seed 3 | seed 4 |
|---|---:|---:|
| mean | 0.498417 | 0.499622 |
| std | 0.289158 | 0.289401 |
| `P(t<0.1)` | 0.101930 | 0.101359 |
| `P(0.4<=t<0.6)` | 0.200297 | 0.197852 |
| `P(t>=0.9)` | 0.099094 | 0.100047 |

两流比较：

| metric | value |
|---|---:|
| Wasserstein-1 | 0.001309 |
| KS distance | 0.003937 |
| 20-bin total variation | 0.004898 |
| max bin frequency difference | 0.001195 |

全局 8-band 平均权重最大差 `0.000817`；任一训练十分位、任一 band 的最大平均
权重差 `0.00683`。

### 判断

**全局时间覆盖解释被否定。** 两条经验分布差异很小，无法直接解释 `1.641` 与
`0.679` 的 FID ratio 翻转。

结果：
`$HOME/data/eqvae/experiments/small_image_time_sequence_audit/audit_20260717_222416/`

## 5.2 单次顺序和分层干预

固定 data/init/batch/noise，只替换 time schedule：

| schedule | baseline FID | weighted FID | ratio |
|---|---:|---:|---:|
| IID seed 3 | 72.71 | 119.30 | 1.641 |
| IID seed 4 | 148.35 | 100.74 | 0.679 |
| seed-3 同一 multiset，step permutation | 80.28 | 98.94 | 1.232 |
| per-batch stratified | 73.80 | 87.68 | 1.188 |

这首次揭示 ratio 翻转的一部分来自 baseline 本身：seed 4 baseline 明显更差，
weighted 看起来“有益”主要是因为它避免了该次 baseline collapse。

结果：
`$HOME/data/eqvae/experiments/small_image_time_order_study/study_20260717_223049/`

## 六、第四轮循环：多 seed 与跨数据验证

本节所有数值都是 tiny classifier feature-FID，不是 ImageNet Inception FID。每个
training condition 用 3 个 rollout seeds；表中的跨条件标准差来自 12 个 training
time seeds。

## 6.1 IID time-seed sweep

| dataset | baseline mean/std | weighted mean/std | weighted/base variance | paired mean diff | harm rate |
|---|---:|---:|---:|---:|---:|
| MNIST | 104.57 / 28.30 | 104.61 / 11.57 | 0.167 | +0.04 | 7/12 |
| FashionMNIST | 96.24 / 20.77 | 139.62 / 35.28 | 2.885 | +43.38 | 11/12 |

MNIST 的 paired-difference 近似 95% normal CI 为 `[-19.82, 19.90]`；FashionMNIST
为 `[29.41, 57.34]`。MNIST 中 baseline FID 与 ratio 的相关为 `-0.932`：baseline
越差，weighted 越显得有益。weighted 在 MNIST 像方差收缩器，但该机制没有跨
数据复现；Fashion 上它既提高均值，也提高跨-seed 方差。

### 判断

**“谱加权普遍稳健化”被否定。** MNIST seed 3/4 的相反结论主要是 treatment effect
heterogeneity 加上 baseline path variance；Fashion 上的系统伤害仍由 coarse-risk
neglect、gradient weak coupling/conflict 和 endpoint leverage 解释。

结果：

- `$HOME/data/eqvae/experiments/small_image_time_seed_sweep/study_20260717_223751/`
- `$HOME/data/eqvae/experiments/small_image_time_seed_sweep/fashion_mnist/study_20260717_224815/`

## 6.2 12-seed stratified sampling

MNIST 每个 batch 在 128 个等宽 strata 各采一个时间点，保持 uniform target
distribution：

| sampling | baseline mean/std | weighted mean/std | paired mean diff | harm rate |
|---|---:|---:|---:|---:|
| IID | 104.57 / 28.30 | 104.61 / 11.57 | +0.04 | 7/12 |
| stratified | 89.67 / 19.81 | 100.24 / 20.10 | +10.57 | 8/12 |

stratification 降低 baseline 方差并改善其均值，但没有稳定 weighted 的相对效果；
weighted variance 反而回升，paired-difference 区间 `[-6.85, 27.98]` 跨 0。

### 判断

**“普通 Monte Carlo 方差降低足以解决翻转”被否定。** CARV/importance-sampling
类理论能解释 estimator variance，但 endpoint quality 仍依赖优化路径和 objective
geometry。

结果：
`$HOME/data/eqvae/experiments/small_image_time_seed_sweep/mnist/stratified/study_20260717_234152/`

## 七、最终因果复现：固定全部时间值，只改顺序

`experiments/small_image_time_permutation_sweep.py` 固定：

- MNIST data split、normalization 和 classifier train data；
- model initialization；
- minibatch indices；
- Gaussian bridge noise；
- AdamW、学习率、gradient clipping 和 1,000 updates；
- seed-3 的全部 128,000 个 float32 时间值；
- 三个 rollout seeds 和评价流程。

唯一变化是把 `[1000,128]` 时间 tensor 沿 step 维做 12 种 permutation。CUDA
0--3 上 source stream 的有序 SHA-256 都是：

```text
53c2da959901752455fe43870e4ae0ff104cf4c1cfa63571294b8642d4043910
```

排序后的 multiset SHA-256 都是：

```text
a567118e1c36d318ab3ad5e05f295303d87242de26d757b9fb941bb1ad18789d
```

代码测试也逐元素验证任意两个 permutation 的 sorted tensor 完全相等。

### 结果

| quantity | min | max | mean/std |
|---|---:|---:|---:|
| baseline feature-FID | 66.34 | 219.25 | 101.21 / 41.71 |
| weighted feature-FID | 87.39 | 161.34 | 109.52 / 23.75 |
| weighted/baseline ratio | 0.736 | 2.112 | median 1.170 |

12 个排列中 4 个 weighted 有益、8 个有害。paired mean difference 为 `+8.31`，
近似区间 `[-10.53, 27.15]`，不能支持稳定平均收益。

几个代表条件：

| permutation | baseline | weighted | ratio |
|---:|---:|---:|---:|
| 2 | 219.25 | 161.34 | 0.736 |
| 5 | 119.16 | 91.83 | 0.771 |
| 0 | 102.26 | 92.51 | 0.905 |
| 3 | 80.28 | 98.94 | 1.232 |
| 1 | 66.34 | 105.51 | 1.591 |
| 7 | 67.48 | 142.46 | 2.112 |

### 判断

这是本轮最特别、也最可靠的新发现：

> **时间样本的 marginal distribution、直方图甚至完整 multiset 都不足以决定训练
> 结果；step order 本身可以改变 baseline 质量并翻转谱加权的 paired effect。**

结果：
`$HOME/data/eqvae/experiments/small_image_time_permutation_sweep/study_20260718_000125/`

## 八、现象解释状态表

| phenomenon | status | existing theory explains | still missing |
|---|---|---|---|
| 同谱 DCT/random 不同 | 已解释 | `J^T W J` / NTK eigenvector alignment | 大模型上低成本估计联合谱 |
| teacher MSE 好、rollout 差 | 基本解释 | tangent-flow leverage、covariate shift、有限容量 risk allocation | 训练前预测 endpoint 符号 |
| band energy 好、语义差 | 已解释 | phase 与联合统计缺失、metric mismatch | 无 |
| local final-checkpoint derivative 预测错 | 已解释其局限 | 一步 hypergradient 是 truncated approximation；完整训练 Jacobian 非平稳 | 可计算的 full-path influence estimator |
| seed 3/4 结论相反 | 已定位 | baseline path variance + time/noise stochasticity + interaction | 为什么具体时间顺序选中具体 basin |
| stratification 不解决相对效果 | 可解释 | variance reduction 不等于 objective/endpoint alignment | 哪类 variance 分量真正影响 endpoint |
| 固定 time multiset，顺序翻转 FID | 仅定性解释 | 非交换 Hessian products、Adam history、anisotropic noise | **符号和幅度的前瞻预测理论** |
| MNIST 近似方差收缩、Fashion 系统伤害 | 部分解释 | basis/data/Jacobian alignment、coarse leverage | 可跨数据预测的 alignment statistic |

关于 one-step proxy，2026 年预印本
[When Losses Align](https://arxiv.org/abs/2605.07756)把一步下游变化写成训练梯度与
下游梯度的内积，同时明确它是 truncated one-step approximation；这与本项目中
final-baseline signed derivative 无法外推 1,000-step-from-init 的结果一致。

## 九、剩余的真实 Gap

## Gap 1：Pathwise spectral-order-to-endpoint theory

需要解释并预测：

```text
given:
  same data, initialization, optimizer, minibatches, Gaussian noise,
  same W(t), same complete time multiset, different time ordering

predict:
  sign and magnitude of Q(theta_weighted) - Q(theta_baseline),
```

其中 `Q` 是 rollout feature-FID/SWD，而不是 training loss。

现有理论的边界：

- NTK/线性谱理论通常固定 kernel 或研究期望动力学；这里 `J_theta` 随路径变化。
- SGD noise 理论说明 covariance shape 影响 implicit bias，但不提供这组排列的 FID
  排序。
- random reshuffling 理论说明非交换矩阵乘积存在，但主要分析 least-squares 的
  train/generalization error，不含 time-conditioned neural flow 和 endpoint rollout。
- timestep-sampling 工作研究 `p(t)`、difficulty 或 estimator variance，不研究固定
  multiset 的顺序因果效应。
- flow error bounds 给上界，不给 feature-FID 的 signed treatment effect。

因此不能把 gap 写成“我们发现 SGD 对 seed 敏感”；真正的新问题是：

> **输出谱度量 `W(t)` 如何改变 time-conditioned gradient/Hessian cocycle，使同一
> Monte Carlo quadrature multiset 的排列选择不同 transport field，并怎样把这种
> 训练路径差异映射为 endpoint semantic quality？**

## Gap 2：可迁移的 basis-data-model alignment 指标

现有实验能事后测量：

- coarse/detail gradient cosine；
- weighted coarse-descent ratio；
- per-band endpoint splice leverage；
- decoder/feature sensitivity。

它们成功解释 Fashion DCT/PCA 为什么伤害、random 为什么近中性，但需要已训练
checkpoint 或 rollout。尚缺一个只使用 initialization/早期训练即可预测以下现象的
统计量：

- MNIST IID 中 weighted 近似收缩跨-seed方差；
- Fashion 中 weighted 系统增加均值和方差；
- 同一 DCT weight spectrum 在不同数据上符号不同。

这不是“功率谱 condition number”的问题；至少需要 `W`、empirical NTK/Jacobian、
gradient covariance、数据语义子空间和 endpoint adjoint 的联合量。

## 十、下一步只值得做的低成本理论实验

当前 inverse-variance output weighting 方法线应继续关闭；以下实验用于研究 gap，
不是继续调 gamma。

### 1. Time-conditioned commutator atlas

在训练早期 checkpoint 和少量 `t_i,t_j` 上，用 Hessian-vector products 估计：

```text
C_ij(v) = H_W(t_i) H_W(t_j) v - H_W(t_j) H_W(t_i) v.
```

比较：

- baseline 与 weighted；
- MNIST 与 Fashion；
- IID permutation 中好/坏路径；
- SGD 与 AdamW。

若 commutator energy 与 permutation endpoint 排名无关，则“非交换曲率”只是一句
正确但无预测力的事后解释，应被否证为主机制指标。

### 2. Pathwise gradient-noise decomposition

按时间 bin 估计：

```text
Cov[g_W(t) - g_I(t)],
alignment(Cov, H),
alignment(g_W-g_I, endpoint adjoint).
```

重点不是总 variance，而是和曲率及 endpoint-sensitive directions 的对齐。
普通 stratification 已经证明只减时间覆盖方差不够。

### 3. Optimizer discriminant

用同一固定 time multiset 的 6--12 个 permutations 比较：

- plain SGD；
- SGD + momentum；
- AdamW；
- AdamW 但定期重置 second moment。

若顺序翻转只在 AdamW 强，gap 主要落到 adaptive-state history；若 SGD 也强，则
non-commuting model curvature 足以产生现象。

### 4. 前瞻 gate

任何候选 path statistic 必须在不看 endpoint 的情况下：

1. 用一半 permutations 拟合阈值或方向；
2. 在 held-out permutations 预测 baseline FID 排序和 weighted treatment sign；
3. 在 MNIST 与 Fashion 都显著优于随机；
4. 再考虑一个 tiny RAE screen。

若无法过这个 gate，它只能算解释性相关，不能成为 ICLR 级方法主张。

## 十一、研究决策

### 可以确定

1. FxLMS/频域 LMS 的“各频段不同收敛率”直觉在神经 flow 中不够，因为输出方向
   与参数方向并不解耦，且 endpoint leverage 不等于 teacher variance。
2. 当前 inverse-variance DCT output loss 没有稳定平均生成收益；Fashion 上有强负
   结果，MNIST 上平均接近零。
3. seed sign reversal 不能被当成方法有时成功的证据；它首先反映 baseline 和
   treatment 都具有很强的 time-order path dependence。
4. 最值得研究的不是继续找一个更好的静态 band weight，而是建立
   `time order -> optimization cocycle -> endpoint distribution` 的可预测理论。

### 不能声称

- 不能说谱加权普遍加速 flow matching。
- 不能说 gradient variance 降低会改善 FID。
- 不能说相同权重 eigenvalues 意味着基底无关。
- 不能用 band-energy proxy 代替生成质量。
- 不能把 MNIST 的方差收缩外推到 Fashion、ImageNet 或 RAE。
- 不能把“非交换更新可能导致顺序效应”写成已经解释了具体 FID 翻转。

## 十二、复现入口

新增代码：

- `experiments/isospectral_alignment_toy.py`
- `experiments/small_image_dct_sign_scramble.py`
- `experiments/small_image_seed_factorial.py`
- `experiments/small_image_stream_factorial.py`
- `experiments/small_image_bridge_factorial.py`
- `experiments/small_image_time_sequence_audit.py`
- `experiments/small_image_time_order_study.py`
- `experiments/small_image_time_seed_sweep.py`
- `experiments/small_image_time_permutation_sweep.py`

对应测试：

- `tests/test_isospectral_alignment_toy.py`
- `tests/test_small_image_dct_sign_scramble.py`
- `tests/test_small_image_seed_factorial.py`
- `tests/test_small_image_stream_factorial.py`
- `tests/test_small_image_bridge_factorial.py`
- `tests/test_small_image_time_sequence_audit.py`
- `tests/test_small_image_time_order_study.py`
- `tests/test_small_image_time_seed_sweep.py`

所有正式结果保存在 `$HOME/data/eqvae/experiments/`，不写入仓库内 `outputs/` 或
`artifacts/`。

## 最终一句话

> **目前已排除“只看权重谱”“只看频带功率”“只看 teacher MSE”“只看时间分布”
> 四种过度简化解释；剩余 gap 是固定时间 multiset 下，谱输出度量与非交换随机优化
> 路径如何共同决定 rollout endpoint 的符号和语义质量。**
