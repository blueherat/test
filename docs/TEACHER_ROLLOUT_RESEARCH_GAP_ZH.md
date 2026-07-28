# Teacher 指标与自生成输运断层：Research Gap 与低成本验证路线

## 结论先说

可以，而且现在应该暂停新的 RAE 训练，把主要机制研究移到小数据和小模型。

现有 MNIST 实验已经证明，这不是一个只能在大 RAE 上看到的现象：在严格配对的初始化、batch、噪声和时间下，频率加权模型的低中噪声 teacher velocity MSE 平均改善 `3.62%`，但 rollout feature FID 平均恶化 `16.3%`。增加 Euler 步数和校准频带能量都不能消除差距。

因此下一步不需要再用 ImageNet 和大 DiT 来回答“断层是否存在”。真正需要回答的是：

> 在什么最小条件下，teacher 分布上的局部回归改善会反转成模型自身轨迹上的边缘输运退化？

这是一个适合用二维分布、MNIST、FashionMNIST 和 CIFAR-10 逐级研究的问题。RAE 只应作为最后的外部有效性检查，而不是日常试错平台。

## 最反常、也最值得研究的现象

原先基于 FxLMS / inverse-variance weighting 的预期是：

1. 固定正定方向权重不改变无限函数类中的 Bayes velocity。
2. 它会让弱方向收敛更快。
3. teacher MSE 的改善至少应与 rollout 改善同向；即使不同向，增加训练或采样步数也应缩小差距。

实验却显示：

1. 多个方向的 held-out teacher MSE 改善，但最终生成分布变差。
2. `50 -> 200` 个 Euler 步不能消除差距，说明主要问题不是数值积分。
3. 匹配每个频带的边缘能量后差距仍在，说明不是简单的方差不足。
4. RAE 中造成损伤的主要部分并非 teacher MSE 最差的 band 0，而是 teacher MSE 已经改善的非零频带。
5. 最关键的是符号翻转：同一个 partial field 在中段 teacher states 上的边缘 drift 更好，放到 baseline rollout states 上却更差。

第五点把问题定位到 teacher risk 和 self-induced transport 之间，而不只是“loss 与视觉指标不一致”。令 `p_t` 为训练插值的 teacher marginal，`q_t^theta` 为模型 ODE 自己产生的 marginal，则训练优化的是

```text
R_teacher(theta) = E_{t, z~p_t} ||v_theta(z,t) - v_target||^2,
```

采样质量却由 `q_t^theta` 满足的 continuity equation 决定。对任一频带或子空间 `b`，二阶矩输运满足

```text
d E_q[||z_b||^2] / dt = 2 E_q[<z_b, v_theta,b(z,t)>].
```

因此点对点速度误差变小，并不保证状态与速度的交叉矩正确；一旦早期误差令 `q_t^theta != p_t`，网络还必须在训练未覆盖的状态上维持正确响应。现有 shared-state probe 支持一个两阶段解释：

- 高噪声阶段已有 on-path marginal-drift mismatch。
- 中段主要是早期偏移触发的 off-path generalization failure。
- 两者沿 ODE 组合，形成最终 endpoint degradation。

## 已有研究覆盖到哪里

截至 2026 年 7 月，下面这些宽泛表述都不能再作为主要新颖性：

| 宽泛命题 | 已有代表工作 | 对当前工作的边界 |
|---|---|---|
| 协方差病态会拖慢 Flow Matching | [Preconditioned Flow Matching](https://arxiv.org/abs/2603.02337) | 已覆盖 whitening / invertible preconditioning；普通 FxLMS 类故事不新颖 |
| 高噪声 conditional target 方差大 | [Stable Velocity](https://arxiv.org/abs/2602.05435) | 已覆盖无偏 target variance reduction；不能只把异常归因于高噪声方差 |
| 训练状态与生成状态不一致 | [Input Perturbation](https://arxiv.org/abs/2301.11706)、[Elucidating Exposure Bias](https://arxiv.org/abs/2308.15321) | 一般 exposure bias 已知；新意必须来自更具体、可证伪的断层机制 |
| 频域辅助损失能改善生成 | [Spectral Regularization for Diffusion Models](https://arxiv.org/abs/2603.02447) | “加频域 loss”本身不新；当前价值在于解释何时反而伤害输运 |
| 对齐生成轨迹与真实边缘分布 | [FlowConsist](https://arxiv.org/abs/2602.06346) | 泛化的 marginal alignment 不新；不能声称首次发现或解决轨迹错位 |
| 局部 field error 会传到 endpoint | [Flow Matching Error Analysis](https://proceedings.mlr.press/v267/zhou25l.html) | 已有整体误差界，但没有定位当前 time x direction 的反转机制 |
| 中间扰动的未来放大率很重要 | [Finite-Time Spectral Sensitivity](https://arxiv.org/abs/2607.12616) | 标量 flow-map sensitivity 已出现；我们的空间只能在方向化、因果干预和训练目标审计上 |
| 按下游敏感度加权 velocity residual | [Decision-Weighted Flow Matching](https://arxiv.org/abs/2606.16790) | adjoint weighting 也不是空白；直接做“敏感度加权 loss”新颖性风险很高 |
| 时间加权会改变有限模型解 | [Training Flow Matching: Weighting and Parameterization](https://arxiv.org/abs/2603.06454) | 该工作在低分辨率 U-Net 上发现 teacher denoising 与 FID 强相关；我们的反向结果更像一个尚未解释的适用边界 |

当前最有希望的 gap 不是“频谱预条件”，而是下面这个更窄的命题：

> **Objective-induced transport reversal:** 一个目标重加权可以改善未见 teacher states 上的条件回归，却系统性损害该模型自身诱导的边缘输运；这种反转可以通过 time x direction field intervention 和 shared-state drift 被因果定位。

在已检索的工作中，没有发现同时提供以下完整证据链的论文：

1. 严格配对的训练目标干预。
2. teacher metric 改善而 rollout metric 反向变化。
3. 时间与方向的 field splice 定位 endpoint 因果贡献。
4. 同一 vector field 在 teacher states 与 rollout states 上出现 drift 排名反转。
5. 用这些量预测何种目标修改会伤害生成。

这仍然只是“有希望的 research gap”，不是可以立即声称的首次发现。它需要跨数据、架构和方向基底复现，才能排除 MNIST/DCT/当前 RAE 配置特例。

## 低成本三级实验

### Level A：二维或低维可解析 toy

**数据：** 8-Gaussians、two moons 和一个各向异性 Gaussian mixture，维度 `2-16`。

**模型：** 共享瓶颈的 tiny MLP，而不是无限容量模型。至少比较一个能表达 exact velocity 的模型和一个故意欠参数化的模型。

**目标：** baseline MSE、方向 inverse-variance weighting、time-only weighting，以及只改变 optimizer 而不改变输出风险的 preconditioning control。

**测量：** teacher MSE、exact/Monte-Carlo marginal drift、rollout Wasserstein/SWD、shared-state sign reversal、连续小剂量 field splice。

这一层应在 CPU 或单卡数分钟级完成。它回答“有限函数类投影是否足以产生反转”，并允许直接画出 `p_t`、`q_t` 和向量场。若这里完全无法产生反转，也有价值：说明图像架构的局部性或高维数据流形是必要条件。

### Level B：MNIST + FashionMNIST 主实验

现有 MNIST 配置可直接复用：train `8192`、test `1024`、width `24`、`1000` updates、`5` seeds。每个 seed 独占一张 GPU；四卡同时跑四个 seed，第五个随后补跑。

不要只重复当前 DCT 实验。最小 factorial 是：

| 因素 | 取值 | 要排除的解释 |
|---|---|---|
| 数据 | MNIST、FashionMNIST | 反转是否只来自数字的极强低频结构 |
| 容量 | width 12、24、48 | 是否是有限容量共享参数的 trade-off |
| 方向基底 | DCT、PCA、随机正交 | 是频率特例、低方差方向问题，还是任意重加权都会发生 |
| 目标 | baseline、direction-weighted、time-weighted | 方向风险修改是否比普通时间加权更容易破坏输运 |
| 状态 | teacher、baseline rollout、weighted rollout、两者插值 | 反转从多远的 off-path 扰动开始 |

MNIST 负责高统计功效和快速 debug；FashionMNIST 负责更复杂轮廓、纹理和类内变化。评价以 SWD、独立小分类器 feature FID、batch drift 和短 rollout discrepancy 为主，不需要 Inception，也不需要大型 decoder。

### Level C：CIFAR-10 小桥接，只在前两层通过后运行

使用 `32x32` pixel flow 或一个小型 frozen ConvAE latent，不使用 RAE。模型只需 tiny U-Net 和 tiny patch Transformer 各一个，因为这里真正想验证的是：反转是否依赖 U-Net 局部归纳偏置，还是 Transformer 共享 token 参数也会出现。

这一层不做大网格搜索，只带入 Level B 中最有区分力的 `baseline + harmful weighting + benign control` 三个设置，跑 `3-5` seeds。它是从灰度 toy 到高维语义 RAE 之间的桥，不是新的算力坑。

## 什么时候才允许回到 RAE

只有同时满足以下 gate，才值得做一次 RAE paired screen：

1. teacher-to-rollout 符号反转在至少两个小数据集上出现，并在不少于 `4/5` seeds 上同向。
2. 反转不能由 solver steps、边缘能量或单一 DCT 基底解释。
3. shared-state drift 或短 rollout discrepancy 比 teacher MSE 更早、更稳定地预测 endpoint degradation。
4. 至少找到一个控制变量，能打开或关闭反转，例如模型容量、基底与协方差的对齐程度，或一到两步 off-path 训练。
5. 候选修复先在 MNIST/FashionMNIST 上同时恢复 rollout 且不伤害 baseline teacher metric。

通过后，RAE 阶段也不需要从头训练大模型。先用已有 checkpoints 和 `64-256` 个固定 validation latents 做 field splice/shared-state probe；只有候选修复表现稳定，才允许一次 tiny paired training。

## 最推荐的下一步

最有信息量、成本最低的下一轮不是继续调 `gamma`，而是实现一个统一的 `small_transport_gap` runner：

1. 先加入 FashionMNIST 数据适配，复用现有 MNIST velocity field 和评价器。
2. 加入 `DCT / PCA / random orthogonal` 三种基底，并匹配权重条件数，避免强度不公平。
3. 加入 teacher、baseline-rollout、weighted-rollout 和插值状态的 shared-state drift 曲线。
4. 用 `5` 个 paired seeds 运行 width `12/24/48`，先不加入更多模型。
5. 只有观察到明确的容量或基底 phase boundary，再添加 tiny patch Transformer。

这一轮能直接区分三种机制：

- **只在 DCT 出现：** 频率与图像局部结构的特殊耦合。
- **PCA 出现、随机基底不出现：** 低方差/协方差对齐方向的有限容量投影问题。
- **随机基底也出现：** 更一般的输出风险重加权与 self-induced state shift 问题。
- **只在小容量出现并随 width 消失：** 主要是有限容量资源分配，而不是新的 flow 几何现象。

这比再跑一个 RAE epoch 更能决定论文是否成立，也更接近当前真正的 research gap。

## 证据边界

当前 RAE 结论来自三个 paired seeds 和固定的 64 个 ImageNet validation latents；MNIST 结论来自三个 seeds。它们足以支持机制假设和下一步设计，但尚不足以支持跨架构普遍性或“首次发现”的论文级主张。

详细现有证据、排除项和复现入口见 `docs/TEACHER_ROLLOUT_MECHANISM_ZH.md`。
