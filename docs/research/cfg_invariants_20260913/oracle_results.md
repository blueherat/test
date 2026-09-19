# CFG 不变量的 CPU oracle 实验：已完成结果

2026-09-13。两种二维已知分布、两个独立评价 seed、每个 seed 2,048 个样本，完成 9 个积分分支和一个直接目标采样分支，共 40 行结果。主采样使用 FP64、128 步 Heun，到 `t=.97`；未使用 GPU、神经网络、图像数据或评价分类器。

**结果支持“跨噪声层对应同一个终点目标”这一严格要求，但不支持“保持零旋度便能普遍改善采样”。** 真正目标的 oracle 场在两种分布上都达到直接目标抽样的误差量级。证据势相容反馈只在相关网格上改善指定目标误差；环形分布没有相应收益。带非零旋度的 norm 反馈反而在环形分布上表现更好，必须保留这个反例。

这里的“误差”全部相对于事先指定的概率分布；没有产生真实图像质量或 CFG FID 收益结论。

## 1. 指定目标与可精确计算的过程

基础 clean 分布为有限二维支撑点 `x_i`，质量 `p_i`，并指定真实软类别概率 `L_i=P(C=1|X=x_i)`。两种分布为：

- `correlated_grid`：4 个横坐标 × 3 个纵坐标，12 点；纵向质量随横坐标相关；`L_i=sigmoid(1.2*x_i[0])`。每个等证据层含 3 点。
- `anisotropic_rings`：3 个半径 × 8 个角度，24 点，纵向缩放 `.8` 且角度质量不均匀；三个半径的 `L` 分别为 `.15,.5,.85`。每个等证据层含 8 点。

参数与完整概率质量均保存在各分布的 `*_definition.json`。主目标固定为

\[
Q_2(X=x_i)=\frac{p_i L_i^2}{\sum_jp_jL_j^2}.
\]

所有分支使用同一 Gaussian corruption `Z_t=tX+(1-t)epsilon`，真实评价目标是 `Q_{2,.97}`，即对 `Q_2` 使用同一 corruption 到 `.97`。有限支撑端点在 `t=1` 奇异，本轮没有把一个小噪声截断冒充精确的连续终点积分。

对每个状态，可以精确求和计算基础后验 `P(X=x_i|Z_t=z)`，因此同时得到：

\[
q(z,t)=\mathbb E[L\mid Z_t=z],\qquad
h(z,t)=\mathbb E[L^2\mid Z_t=z],
\]

\[
v_u=\frac{\mathbb E_P[X\mid z]-z}{1-t},\quad
v_c=\frac{\mathbb E_{P(\cdot\mid C=1)}[X\mid z]-z}{1-t},\quad
v_{Q_2}=\frac{\mathbb E_{Q_2}[X\mid z]-z}{1-t}.
\]

这使得真正终点倾斜对应的各噪声层场可以直接作为 oracle，无需用 posterior 预测器替代。

## 2. 实際跑过的分支

记 `g_v=v_c-v_u`，所有反馈写成 `v=v_c+a(t,z)g_v`。普通 CFG2 为 `a=1`；令 `T=.97`、`b(t)=sin²(pi*t/T)`、`lambda=.75`。

| 分支 | 完整定义 |
|---|---|
| conditional | `v_c`，无额外放大 |
| cfg_2 | `v_c+g_v` |
| time_feedback | `a=1+lambda*sin(2*pi*t/T)` |
| norm_feedback | `a=1+lambda*b*tanh(log(r_ref(t)/(||g_v||+epsilon)))` |
| evidence_potential | `a=1+lambda*b*(1-2*q)` |
| true_terminal_tilt | 上述精确 `v_Q2` |
| 三个 cfg_match 分支 | 分别给三种反馈匹配额外作用量的固定 CFG 系数 |
| direct_target | 从 `Q_{2,.97}` 独立直接抽样，用来估计有限样本误差底线 |

`r_ref(t)` 来自独立校准数据的条件加噪状态上 gap 范数中位数，并作时间插值。它没有访问评价 seed。所有反馈都在 `t=T` 回到 `a=1`，避免主比较中引入任意不同的末端幂指数。配平 CFG 是作用量对照，其固定系数可以不同于 CFG2；不声称它本身保证实现 `Q_2`。

证据分支的总 score 势为

\[
(2+\lambda b)\log q-2\lambda bq.
\]

因此每个固定时刻保留梯度身份，但它仍不等于 `log E[L²|Z_t]`。这个区别正是本轮检验的重点。

## 3. 指标与直接抽样底线

`target_atom_tv` 是输出按最近支撑点归类后的经验质量与 `Q_2` 的 TV；`within_evidence_tv` 则在每个相同 `L_i` 的组内，比较经验条件质量与原始 `P(X|L)`，再按目标组质量加权。没有样本的组记 TV=1，并单列缺失组的目标质量。

这里有两层需要区分：clean 支撑上的等证据组是精确定义的；将 `.97` 的连续输出赋给支撑点，是近似分类。两种分布的最小点间距分别为 **42.03 倍和 20.60 倍终端噪声标准差**，全部直接目标样本的最近点分类混淆率为 **0**。另外用相邻 `L` 的中点给真实 noisy posterior `q` 分箱，并保存它与原子组别的不一致率；这仍是有限噪声下的近似分箱，不是严格连续等值面的检验。

`target_sw2` 是对 32 个固定方向计算的投影平方 Wasserstein 距离，参考是另外一份同样本数、独立 seed 的直接目标样本。`evidence_mean` 使用当前真实 noisy posterior `q`。证据更大不是更接近指定目标的充分条件：过度集中也能提高它。

下表是两个评价 seed 的平均值；没有把两 seed 均值称作总体显著性。

| 分支 | 网格目标 TV↓ | 环形目标 TV↓ | 网格层内 TV↓ | 环形层内 TV↓ |
|---|---:|---:|---:|---:|
| 直接目标抽样 | 0.02179 | 0.03472 | 0.02034 | 0.03513 |
| conditional | 0.13766 | 0.16637 | 0.01339 | 0.04252 |
| CFG2 | 0.08309 | 0.06553 | 0.02040 | 0.03726 |
| 时间反馈 | 0.15748 | 0.06374 | 0.04127 | 0.04513 |
| norm 反馈 | 0.08842 | 0.05612 | 0.02379 | 0.03900 |
| 证据势反馈 | 0.05770 | 0.06514 | 0.02194 | 0.04009 |
| 真正终点 tilt 场 | 0.02049 | 0.03580 | 0.01035 | 0.03620 |
| 匹配时间反馈的 CFG | 0.16749 | 0.08231 | 0.03180 | 0.04046 |
| 匹配 norm 反馈的 CFG | 0.08163 | 0.06813 | 0.01943 | 0.03796 |
| 匹配证据反馈的 CFG | 0.07186 | 0.06429 | 0.02071 | 0.03742 |

真正 tilt 场的 TV 与直接抽样底线接近；某个 seed 上比直接抽样误差更低只是 Monte Carlo 波动。conditional 分支相对 `Q_2` 的误差较大是因为它精确瞄准 `P(X|C=1)`，不是因为原条件生成场有模型错误。

![指定目标误差与直接抽样范围](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/target_error_summary.png)

### 证据势反馈的核心配对比较

从保存的终点 NPZ 进行 1,000 次 bootstrap，没有追加 rollout。每个 seed 内按共享初始 Gaussian 的索引配对重采样，统计量为两个 seed 各自 TV 差的平均。差值定义为“证据势反馈 − 对应作用量配平 CFG”，负值有利。

| 分布 | 两 seed 实际差值 | 平均差 | 95% bootstrap 百分位区间 |
|---|---|---:|---|
| 相关网格 | −0.011885、−0.016433 | −0.014159 | [−0.018028, −0.000892] |
| 非均匀环形 | +0.000718、+0.000988 | +0.000853 | [−0.002948, +0.003828] |

这是针对两个固定 oracle、固定校准规则与评价 seed 的有限样本近似区间。TV 非光滑，只有两种分布，不能用这个 bootstrap 证明跨分布普遍效果或真实图像收益。它支持保留网格上的局部收益，同时明确环形未获益。

### 作用量确实作了近似配平

测量实际路径上的

\[
A=\mathbb E\int_0^T\|v_{\rm method}(Z_t,t)-v_c(Z_t,t)\|\,dt.
\]

先用独立 seed `902613`、384 条路径、80 步 Heun，对固定 CFG 的额外系数二分 10 次；校准作用量误差为 `0.052%–0.079%`。正式评价没有再次调参。下面是正式评价上 `(反馈作用量/配平CFG作用量)-1`，因此这些控制是**近似**而非逐样本严格配平。

| 分布 | 时间反馈：两个 seed | norm：两个 seed | 证据势：两个 seed |
|---|---|---|---|
| 网格 | +0.244%、+0.303% | +0.139%、+0.028% | −0.022%、−0.135% |
| 环形 | −1.018%、−0.394% | −0.606%、−0.496% | +0.140%、−0.170% |

尤其核心证据势比较的正式作用量误差都小于 `0.17%`；所有分支最大误差约 `1.02%`。作用量相同不表示方向、时间位置、能量或输出位移相同；积分平方能量也独立保存在结果中。

## 4. 旋度、证据收益与失败结果

在同一独立状态库、4 个噪声时刻测额外 score 的二维旋度。CFG、纯时间反馈、证据势反馈及真正 tilt 场只有有限差分量级的非零值；归一化旋度最高约 `6.1e-8`。norm 反馈在网格 `t=.35` 为 `0.0765`，环形 `t=.65` 为 `0.1155`，确实破坏了相应梯度身份。

但 **norm 反馈在环形上的目标 TV 为 0.05612，优于 CFG2 的 0.06553，也优于其作用量配平 CFG 的 0.06813**。这直接否定“有旋度必然导致更差终点”的强推论；同时，证据势反馈并未显示跨分布稳定优势。这里没有把 norm 反馈冒称为原论文 CFG-CTRL 的完整复现。

网格的 oracle 目标 clean 证据均值为 `0.75745`，真正 tilt 场得到 noisy 证据均值 `0.75632`；CFG2 提高到 `0.78998`，时间反馈更高达 `0.81600`，目标 TV 却更大。环形目标均值 `0.70958`，真正 tilt 场为 `0.70721`，CFG2 为 `0.73046`。这表明“证据越来越高”可以是目标外的过度选择。

层内条件 TV 大多接近直接抽样量级；**本轮没有显示证据势反馈具有稳定的层内保留收益**。网格上目标 TV 的改善主要不能简单归结为层内误差降低。当前有限样本与离散支撑也不足以细分更小的层内效应，不能据此宣称这些方法已经实现严格等值层保持。

## 5. 求解器与算子核对

对每种分布第一 seed 的前 512 条共享噪声，比较 CFG2、norm、证据势和真实 tilt 的 64/128/256 步结果：

- 八组检查中，64→128 及 128→256 都没有最近原子分类变化。
- 128→256 的最大坐标 RMS 差为 `3.52e-4`，出现在环形 CFG2；真正 tilt 场的最大值为 `5.21e-5`。
- 主结果的 TV 差异不是这批步数检查中的原子分配变化造成的。检查只覆盖前 512 条及四个代表分支，不能扩大成所有样本的严格误差界。
- 不同算子的共享噪声终点确有区别；相对 CFG2 的最大分支 RMS 差，网格为 `0.3800`、环形为 `0.2091`。没有把同一个算子换名重复运行。

## 6. 幂等、密度保持和 clean mean 的独立控制

同脚本还实际执行以下 CPU 控制：

| 控制 | 实测或解析值 | 含义 |
|---|---|---|
| 硬事件条件化重复两次 | TV=0 | 同一事件的条件化幂等 |
| 固定软似然倾斜重复两次 | TV=0.127552 | 重复软乘权通常不幂等 |
| 标准二维 Gaussian 旋转 `.6` 弧度 | 总体 KL=0，配对位移 MSE≈0.6971 | 分布完全保持也可以逐点变化 |
| 同一 Gaussian 径向乘 `1.2` | 总体 KL=0.075357 | 确定控制也可能改变分布 |
| 精确 Gaussian FM 的 remaining endpoint | 沿轨迹最大漂移=0 | 固定 ODE 的未来终点满足流契约 |
| 同轨迹的 posterior clean mean | 起止向量变化=1.34853 | 精确 denoiser 的 clean 预测也不必沿轨迹恒定 |

![解析控制](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/analytic_controls.png)

## 7. 可检查图、数据与复现

![相关网格散点](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/correlated_grid_scatter.png)

![环形散点](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/anisotropic_rings_scatter.png)

脚本：[oracle.py](/home/zhoushunyu/eqvae/experiments/cfg_invariants_20260913/oracle.py)。完整运行：

```bash
python experiments/cfg_invariants_20260913/oracle.py --samples 2048 --steps 128
```

仅从既有终点重算配对 bootstrap 和正式作用量审计：

```bash
python experiments/cfg_invariants_20260913/oracle.py --postprocess-only
```

输出目录：[oracle](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle)。主要文件：

- [逐 seed 指标 metrics.csv](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/metrics.csv)、[均值及两 seed 范围 summary.csv](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/summary.csv)。
- [逐证据层数据](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/fiber_metrics.csv)、[旋度数据](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/curl.csv)。
- [独立校准](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/calibration.csv)、[正式作用量配平误差](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/action_matching_evaluation.csv)、[核心配对 bootstrap](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/paired_bootstrap.json)。
- [64/128/256 步核对](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/solver_checks.csv)、[运行清单](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/manifest.json)、[后处理清单](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/oracle/postprocess_manifest.json)。
- 各分支保存全部终点、原子概率与共享初始噪声的 NPZ；第一 seed 另保存前 32 条完整轨迹 NPY。

最后一次全量采样及图表过程约 **28 秒 CPU 墙钟**；随后只追加既有终点的 bootstrap 和匹配审计，没有追加 rollout。全量运行清单与后处理清单分别记录相应脚本 SHA256，后者的代码变化仅增加后处理功能。未进行参数筛选、GPU 训练或真实图像效果声明。
