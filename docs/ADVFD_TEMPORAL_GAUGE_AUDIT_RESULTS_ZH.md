# AdvFD 时序坐标一致性审计

## 结论

AdvFD 官方实现对 critic 特征的一阶矩和未中心化二阶矩做 EMA，但 critic 本身仍在训练。数值上，这些张量当然可以逐步平均；问题在于，不同时刻的特征坐标并不一定一致，因此这种平均不满足随当前特征重参数化共同变换的等变性。

本轮审计在官方 PMF-B 复现的 `step_0005000` 和 `step_0009999` checkpoint 上确认了三个事实：

1. 即使输入仍来自同一个静态 ImageNet 真实分布，checkpoint 中的历史 EMA 统计与当前 critic 坐标下重新估计的统计之间，也存在远大于独立采样噪声的差异。
2. 这项差异会改变传给当前生成图像的 AdvFD 梯度。`step_0009999` 的扩大样本实验中，历史 EMA 梯度与当前统计梯度的 cosine 为 `0.7963 +/- 0.0689`，而两个独立当前统计库之间为 `0.9526 +/- 0.0211`；10 次配对试验全部满足后者更高。
3. 差异主要来自协方差项，并广泛分布在许多谱方向上，不是少数 top eigenmodes 造成的。

因此，本轮实验支持以下严格表述：

> AdvFD 当前官方统计更新会产生可测量的 temporal-coordinate mismatch，并向在线 generator 提供不同于当前 critic 统计目标的梯度。

它尚未证明这种 mismatch 必然损害最终生成质量，也尚未证明 moment transport、降低 EMA 动量或额外 measurement critic 能改善结果。那需要单独的因果训练对照。

## 审计对象

- AdvFD 官方仓库 commit：`4e4cfed944e4fc38a75fae3ea7701ae9e5587060`
- 复现模型：PMF-B，官方式 AdvFD continuation
- checkpoint：`step_0005000`、`step_0009999`
- critic 特征维度：2048
- 官方统计动量：`beta=0.99`
- 官方统计语义：均值与未中心化二阶矩的 EMA，协方差由二者恢复
- generator 梯度输入：保持训练协议，生成张量从 `[-1, 1]` 映射到 `[0, 1]`，不额外 clip 或量化
- 数值精度：本文所有正式 moment、spectrum 和 gradient 结果均由 FP64 moments 重建；早期误存为 FP32 的 NPZ 不用于本文数字

原始大体积结果保存在：

```text
/data/users/zhoushunyu/eqvae/experiments/advfd_temporal_gauge_audit_v1
```

仓库只保留复现实验代码、测试、精简 CSV 和本文档，不提交 moment NPZ、checkpoint 或生成图像。

## 问题定义

令第 `k` 步 critic 的特征映射为

\[
\phi_k(x) \in \mathbb{R}^d,
\]

当前数据在该坐标中的均值和未中心化二阶矩为

\[
m_k=\mathbb{E}[\phi_k(x)], \qquad
S_k=\mathbb{E}[\phi_k(x)\phi_k(x)^\top].
\]

官方实现更新

\[
\bar m_k=\beta\bar m_{k-1}+(1-\beta)m_k,
\]

\[
\bar S_k=\beta\bar S_{k-1}+(1-\beta)S_k.
\]

若每一步只是使用同一个固定特征坐标，这就是普通的时间平滑。可是 critic 在训练，`phi_k` 会变化。考虑不改变任何样本几何的等价正交重参数化

\[
\phi'_k(x)=Q_k\phi_k(x).
\]

当前矩应变为

\[
m'_k=Q_km_k, \qquad S'_k=Q_kS_kQ_k^\top.
\]

但直接平均历史张量得到的 `bar m'_k, bar S'_k` 一般不等于把旧统计先运输到当前坐标后再平均的结果。只有当坐标基本不动，或显式知道相邻坐标的 transport 时，两者才一致。这就是本报告所说的 temporal-coordinate mismatch。

## 静态真实分布对照

这里每个 fresh bank 都来自完全相同的 ImageNet 真实训练分布。因为真实分布没有随 generator 演化，这一侧最适合隔离 critic 坐标漂移。

`FD/dim` 是在 anchor covariance 白化后计算、按 2048 维归一化的 regularized FD。`fresh A/B` 是两个独立当前特征库之间的有限样本基线。

| checkpoint | 每个 fresh bank | EMA -> fresh FD/dim | fresh A/B FD/dim | EMA -> fresh mean Mahalanobis^2 | fresh A/B | EMA -> fresh log-eigen RMS | fresh A/B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 5,000 | 1.2102 | 0.6178 | 112.063 | 1.585 | 4.8526 | 1.2008 |
| 9,999 | 5,000 | 1.4831 | 0.9153 | 78.311 | 1.958 | 4.3046 | 1.3764 |
| 9,999 | 10,000 | 1.4614 | 0.3034 | 78.067 | 0.611 | 4.2074 | 0.9264 |

扩大到两个 10K bank 后，`step_0009999` 的历史/当前差异相对 fresh split 基线为：

- whitened FD/dim：`4.82x`
- mean Mahalanobis squared：`127.77x`
- covariance log-eigen RMS：`4.54x`

增加样本后 fresh split 噪声明显下降，而 EMA/current 差异基本不消失。这排除了“只是 2048 维下 5K 样本太少”这一解释。

## 当前 fake 统计对照

在 `step_0009999` 直接从在线 generator 生成两个独立 10K fake bank，并在当前 critic 中提取特征：

| 比较 | whitened FD/dim |
|---|---:|
| 历史 fake EMA -> 当前 fresh fake | 1.4351 |
| 当前 fresh fake A/B | 0.3298 |

前者是后者的 `4.35x`。不过 fake 分布同时受 generator 历史变化影响，因此该结果只能作为补充；静态真实分布对照才是坐标漂移的干净证据。

## Generator 输出梯度

在同一批当前生成图像上分别用历史 EMA moments、当前 fresh moments A、当前 fresh moments B 计算 AdvFD image gradient。扩大实验使用每个 current bank 10K 样本、10 个配对 trial、每个 trial 4 张生成图像。

| checkpoint | 每个 current bank | trials | EMA/current cosine | current A/B cosine | current/EMA norm ratio |
|---:|---:|---:|---:|---:|---:|
| 5,000 | 5,000 | 5 | 0.8401 +/- 0.0325 | 0.9403 +/- 0.0271 | 0.3721 |
| 9,999 | 5,000 | 5 | 0.7654 +/- 0.0389 | 0.8742 +/- 0.0708 | 0.4218 |
| 9,999 | 10,000 | 10 | 0.7963 +/- 0.0689 | 0.9526 +/- 0.0211 | 0.8867 |

扩大实验的 10 个 trial 中，`current A/B cosine - EMA/current cosine` 全部为正，范围约为 `0.057` 到 `0.260`。因此历史/当前梯度旋转显著大于当前统计的有限样本波动。

这项 cosine 测的是 generator 输出图像空间中的梯度，不包含 generator 参数 Jacobian；不能直接解释为参数更新 cosine。

## 协方差项与谱分布

`step_0009999` 扩大实验中，协方差梯度范数相对均值梯度范数为：

- 历史 EMA：`171.94x`
- 当前 fresh moments：`145.16x`

当前 20K real/fake 的 FD 分解与协方差谱贡献为：

| 统计 | total FD | covariance fraction | top-1 | top-16 | top-128 | 50% 所需 modes | 90% 所需 modes | participation rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current full | 223.740 | 99.53% | 1.60% | 8.55% | 32.42% | 273 | 968 | 615.55 |
| checkpoint EMA | 257.142 | 99.77% | 0.94% | 9.37% | 37.47% | 223 | 889 | 554.30 |

当前 full 统计中，前 128 个模式只解释 `32.42%` 协方差贡献，达到 90% 需要 968 个模式。按 anchor eigenvalue 中位数划分的低/高谱贡献分别为 `48.37%` 和 `51.63%`。因此当前证据反对“误差只集中在少数 top modes”的假设。

绝对 FD 在 2048 维仍有明显有限样本偏差，因此这里主要使用配对对照、贡献比例和谱分布，不把 `223.740` 当作生成质量指标。

## 模拟刷新路径

从 checkpoint EMA moments 出发，反复用同一组当前 moments 做 `beta=0.99` 的 EMA 更新，比较每个中间统计产生的图像梯度与完全当前统计梯度：

| refresh 次数 | cosine to current | current/refresh norm ratio |
|---:|---:|---:|
| 0 | 0.8167 | 0.9176 |
| 1 | 0.8592 | 1.0938 |
| 2 | 0.8783 | 1.2153 |
| 5 | 0.9038 | 1.4621 |
| 10 | 0.9209 | 1.7189 |
| 20 | 0.9355 | 2.0080 |
| 50 | 0.9520 | 2.2422 |
| 100 | 0.9633 | 2.0226 |
| 200 | 0.9748 | 1.4209 |
| 500 | 0.9917 | 0.9343 |

方向随刷新几乎单调收敛，而范数明显非单调。这说明观测到的变化不是给同一梯度乘一个常数，而是统计坐标与尺度共同变化。

该实验只在固定 checkpoint 上插值 moments，不等价于继续联合训练 critic 和 generator。

## 精确旋转高斯对照

二维 toy 固定真实/生成高斯分布，只让共享特征坐标每一步旋转。因为真实几何不变，当前坐标中的正确 FD 恒为 `1.4201027`。

当 `beta=0.99`、每步旋转 `90` 度时：

- 直接 EMA 历史矩的最终 FD 为 `0.1008906`，只剩当前正确值的 `7.10%`
- 先把历史矩运输到当前坐标再 EMA，最终比值为 `1.0000000`

这个 toy 是有限旋转下的精确反例，不依赖小扰动或 Taylor 展开：同一个统计对象在移动坐标中直接平均，可以得到错误目标；正确 transport 恢复不变量。

## 证据边界

本轮已经确认：

- 官方 EMA 统计在训练中的 critic 坐标下并非时间一致对象
- mismatch 同时出现在一阶矩、协方差和 generator 输出梯度中
- gradient discrepancy 超过同规模 current-bank 抽样噪声
- 协方差项是主导项，且作用谱很宽
- 一个可控的 exact toy 能单独由坐标旋转复现该问题

本轮没有确认：

- temporal-coordinate mismatch 是 AdvFD 效果的主要瓶颈
- 任何特定修复会改善 FID、recall 或训练稳定性
- fake EMA/current 差异能全部归因于 critic，而不是 generator 历史变化
- image-gradient cosine 能直接替代 generator 参数梯度或最终生成指标

## 仓库产物

- `experiments/advfd_cleanroom/temporal_gauge.py`：FP64 moments、合并、EMA 插值、FD 分解工具
- `experiments/advfd_cleanroom/audit_advfd_temporal_gauge_stats.py`：真实分布统计审计
- `experiments/advfd_cleanroom/audit_advfd_temporal_gauge_current_fake_stats.py`：当前 fake 统计审计
- `experiments/advfd_cleanroom/audit_advfd_temporal_gauge_fake_stats.py`：早期 PNG-based fake 审计入口，仅用于复现历史产物，不提供本文正式数字
- `experiments/advfd_cleanroom/audit_advfd_temporal_gauge_gradients.py`：图像梯度与刷新路径审计
- `experiments/advfd_cleanroom/audit_advfd_covariance_spectrum.py`：协方差贡献谱
- `experiments/advfd_cleanroom/audit_advfd_rotating_gaussian.py`：精确旋转高斯对照
- `tests/test_advfd_temporal_gauge.py`：核心统计、transport 和梯度分解测试
- `docs/data/advfd_temporal_gauge_*.csv`：本文使用的轻量结果表
