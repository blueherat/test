# Prediction target、输出秩与算子目标：受控 toy 实验

## 1. 实验问题

本轮只回答两个问题：

1. `x / v / eps` 的有限模型差异，是否能由输出可表达秩与目标谱解释？
2. 如果把不同子空间分配给不同预测方式，是否能优于整块选择单一 target？

统一路径为

\[
z_t=(1-t)x+t\epsilon,\qquad v=\epsilon-x,
\]

生成时从高噪声端积分到数据端。所有 native target 都转换到同一个 velocity-space MSE；同一 setting 内共享初始化、训练 batch、噪声与时间。v2 协议还在不同 rank 之间共享随机数。

## 2. 实现与验证

- `experiments/audit_prediction_target_rank_spectra.py`：用 v4 的高容量 clean oracle 估计条件 target 谱及最佳 rank-r 仿射残差。
- `experiments/run_prediction_target_rank_symmetry_toy.py`：固定 trunk 为 5 层、宽度 512，只改变最终 affine output rank。
- `experiments/evaluate_prediction_target_rank_symmetry_ambient.py`：补充完整环境空间 SWD，避免只看二维投影。
- `experiments/run_prediction_target_operator_hybrid_from_rank_baseline.py`：在线性数据子空间上训练结构化算子 target。
- `tests/test_prediction_target_rank_symmetry_toy.py`：检查实际输出秩、target 转换、解析 skip 等价性和算子公式。

测试结果：相关测试与 v4 回归测试共 `14 passed`。

## 3. 谱审计

对于 target `y`，固定 rank-r 输出头不可避免的均方残差由条件均值输出协方差的尾部谱决定：

\[
\mathcal E_r(y,t)=\frac{a_y(t)^2}{D}\sum_{i>r}\lambda_i,
\]

其中 velocity-space 换算增益为

\[
a_x(t)=1/t,\qquad a_v(t)=1,\qquad a_\epsilon(t)=1/(1-t).
\]

在 `D=512`、rank 64 下，对可用 oracle checkpoint 和五个时刻求平均：

| curvature | target | effective rank | rank-95 | rank-64 后剩余方差 | velocity tail MSE / dim |
|---:|:---:|---:|---:|---:|---:|
| 0.0 | x | 1.98 | 2.0 | 0.0% | 0.00000 |
| 0.0 | v | 8.71 | 437.0 | 43.0% | 0.82376 |
| 0.0 | eps | 480.85 | 470.2 | 82.2% | 19.41507 |
| 0.5 | x | 2.69 | 8.8 | 0.0% | 0.00002 |
| 0.5 | v | 9.32 | 429.6 | 39.1% | 0.83451 |
| 1.0 | x | 4.98 | 13.6 | 0.0% | 0.00004 |
| 1.0 | v | 11.55 | 406.1 | 29.5% | 0.83817 |

关键点是：`v` 的 participation-ratio rank 很低，但有很长的弱谱尾，rank 64 仍漏掉 30%--43% 的总方差。`eps` 同时具有近满秩输出和 `1/(1-t)` 条件数；curvature=0.5、t=0.9 时，其 rank-64 velocity tail MSE / dim 约为 `81.95`。

这里的 oracle 是 v4 中训练得到的高容量 clean oracle，不是解析 Bayes oracle；`seed20260807, curvature=0` checkpoint 缺失并已显式跳过。

## 4. 显式 rank 干预

曲面数据的配对 v2 teacher velocity MSE：

| D | rank | x | v | 更优 target |
|---:|---:|---:|---:|:---:|
| 64 | 4 | 4.339 | 1.350 | v |
| 64 | 16 | 0.805 | 1.090 | x |
| 64 | 64 | 0.354 | 0.487 | x |
| 256 | 4 | 4.513 | 1.266 | v |
| 256 | 16 | 1.013 | 1.119 | x |
| 256 | 64 | 0.184 | 0.951 | x |
| 512 | 4 | 4.551 | 1.233 | v |
| 512 | 16 | 1.240 | 1.096 | v |
| 512 | 64 | 0.130 | 1.021 | x |

`D=64, curvature=0.5` 的 rank 4 / 16 / 64 结论又在两个额外 seed 上复现：rank 4 均为 `v` teacher risk 更低，rank 16 和 64 均为 `x` 更低。

因此 target preference 确实会随模型输出秩翻转。但决定量不是局部 intrinsic dimension 单独一个数：曲面虽只有二维，其全局 affine span 由 Fourier 特征扩展；有限训练下还叠加了优化和 target 换算条件数。

闭环生成必须另行判断。全部 24 个 paired native setting 中，`x` 的完整环境空间 SWD 都优于 `v`；低 rank 的 `v` 虽偶尔有更低二维 SWD，却带有很大的法向误差。因此本轮只确认了 teacher field risk 的因果翻转，没有确认 native endpoint distribution 的翻转。

## 5. 解析 skip 对照

把网络输出解释成 clean residual，再解析构造 `x/v/eps` target，在共同 velocity loss 下与 native-x 完全等价：

- 每步预测 velocity 相同；
- 最终参数最大差值为 `0`；
- 单测严格通过。

所以“只把恒等项写成解析 skip”本身不是新方法；若损失和最终 velocity 不变，优化轨迹也不变。

## 6. 结构化算子 target

在线性 toy 中，令 `P` 为真实数据子空间投影。因为

\[
(I-P)x=0,\qquad (I-P)z_t=t(I-P)\epsilon,
\]

法向速度可直接由当前状态得到：

\[
\hat v=P h_\theta(z_t,t)+(I-P)z_t/t.
\]

网络只学习数据子空间中的 velocity，法向不再学习高维 identity/noise mapping。

三 seed、`D=64`、rank 4/16/64 共九个 setting：

- teacher MSE：9/9 优于 native-x，约改善 5%--7%；
- 完整空间 SWD：9/9 优于 native-x，平均相对改善约 40%--47%；
- 二维 SWD：9/9 优于 native-x；
- MMD：仅 1/9 更低，整体略差。

严格跨-rank配对的单 seed 复验中，rank 4/16/64 的完整空间 SWD 分别改善 `38.1% / 31.1% / 17.2%`，teacher MSE 改善 `6.2% / 6.5% / 5.7%`。hybrid 的法向一致性误差约 `6e-8`，而 native-v 为 `0.26--0.94`。

旧协议单 seed 的 D=64/256/512 共 12 个 setting 中，hybrid 的 teacher MSE 为 12/12 更低，完整空间 SWD 为 11/12 更低。该跨-rank趋势可能含旧版随机数混杂，因此正式因果比较以 paired v2 为准。

## 7. 当前结论边界

1. 已确认：target 难度由条件 target 谱、模型可表达输出空间和换算条件数共同决定，不能只用 `D/d` 或 target 名称解释。
2. 已确认：低 effective rank 不代表高维尾部可以忽略；`v` 正是“少数强方向 + 很长弱尾”的反例。
3. 已确认：在线性 oracle 几何下，按子空间分配预测责任可显著降低 field risk 和 SWD。
4. 尚未确认：该算子 target 是否改善所有分布指标；MMD 的反向结果必须保留。
5. 尚未解决：真实图像没有已知精确 `P`。当前结果是机制证明与 oracle upper bound，不是可直接宣称的图像生成方法。

## 8. 数据位置

- 谱审计：`/home/zhoushunyu/data/eqvae/experiments/prediction_target_rank_spectrum_audit_v2`
- paired D64：`.../prediction_target_rank_symmetry_v2_paired_d64`
- paired D256：`.../prediction_target_rank_symmetry_v2_paired_d256`
- paired D512：`.../prediction_target_rank_symmetry_v2_paired_d512`
- D64 额外曲面 seeds：`.../prediction_target_rank_symmetry_v2_paired_d64_curv05_seeds22_23`
- paired operator：`.../prediction_target_operator_hybrid_v2_paired_d64`
- operator 三 seed：`.../prediction_target_operator_hybrid_v1*`

## 9. LMMSE residual / moment-exact 对照

后续 `run_prediction_target_spectral_preconditioning_toy.py` 没有使用真实 tangent
projector，而是只从训练分布样本估计均值和协方差。它解析减去 velocity 对当前
state 的最优仿射预测，并让 rank-limited MLP 学习标准化 residual；所有条件仍使用
同一个 recovered-velocity MSE。

`D=64`、rank 16、5,000 steps 的三 seed 平均结果：

| curvature | condition | teacher velocity MSE | ambient SWD | MMD-2D |
|---:|:---|---:|---:|---:|
| 0.0 | native-x | 0.27738 | 0.08289 | 0.000107 |
| 0.0 | LMMSE residual | 0.25975 | 0.04390 | 0.000317 |
| 0.5 | native-x | 0.77186 | 0.09759 | 0.000199 |
| 0.5 | LMMSE residual | 0.37355 | 0.07316 | 0.000682 |

它稳定降低 teacher MSE，并改善平均 SWD；MMD 没有同步改善，曲面 seed 间也存在
波动。因此这是支持解析 residualization 的初步机制证据，不是普适生成改进。

完整推导、数据泄露边界和真实 SiT 验收条件见：

`docs/MOMENT_RESIDUAL_FLOW_THEORY_AND_LEAKAGE_AUDIT_ZH.md`
