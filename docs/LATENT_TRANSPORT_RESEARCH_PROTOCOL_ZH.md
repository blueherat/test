# Latent Transport Compatibility 研究协议

## 研究问题

本研究不假设等变性是 RAE final latent 的内禀结构。等变 adapter 只作为一种
可控、可逆的表示几何干预。核心问题是：

> 可逆 latent 重参数化保持信息与重构时，标准 Gaussian straight flow
> matching 为什么仍会改变，以及使用坐标一致的 probability path 能否恢复或
> 改善生成？

## 四个不可混淆的分支

以下统一采用 RAE 官方 transport 的时间约定：`t=0` 为数据，`t=1` 为噪声。

| 分支 | 条件路径 | source | 作用 |
|---|---|---|---|
| Base | `z_t=(1-t)z+t eps` | `eps ~ N(0,I)` | 原始 latent 基线 |
| Gaussian straight | `(1-t)f(z)+t eta` | `eta ~ N(0,I)` | 同时改变 target 坐标、source 配对和路径 |
| Matched chord | `(1-t)f(z)+t f(eps)` | `f(eps)` | 消除 source endpoint mismatch，只保留 chord mismatch |
| Pushforward | `f((1-t)z+t eps)` | `f(eps)` | 原始条件路径在新坐标下的严格 pushforward |

Pushforward target velocity 为：

```text
v_f(t) = J_f(z_t) (eps - z)
```

实现必须使用 JVP，不能显式构造完整 Jacobian。

## 理论边界

若连续映射 `f` 对所有 `z, eps, t` 满足：

```text
f((1-t)z+t eps) = (1-t)f(z)+t f(eps),
```

则由保持所有线段的 Jensen 仿射性质，`f` 必须是 affine map。若还要求
`f#N(0,I)=N(0,I)`，则平移必须为零、线性部分必须正交。因此：

> 同时保持标准高斯 source 和所有 straight conditional paths 的全局坐标变换，
> 本质上只能是正交线性变换。

非正交线性映射不产生 chord defect，但会产生 Gaussian source mismatch；非线性
映射通常同时产生 source mismatch 和 chord defect。即使某个非线性映射保持
Gaussian marginal，它仍不必保持逐样本 chord。

## 假设与可推翻条件

- `H1`：DINOv2 深层 direct error 下降包含真实 token correspondence，而不只是
  空间同质化。
- `H2`：非线性 adapter 会改变 source distribution 或 conditional path，即使
  reconstruction 不变。
- `H3`：bridge defect / velocity ambiguity 比等变误差、rFID 或简单 smoothness
  proxy 更能预测有限模型的生成困难。
- `H4`：Pushforward 能恢复因错误坐标路径造成的性能差距。
- `H5`：只有同时控制表示几何和 transport compatibility，adapter 才可能稳定
  改善 RAE 生成。

若 Pushforward 在严格配对的 toy 中不能恢复 Gaussian-straight 与 Base 差距的
至少 50%，停止 `H4/H5` 主线，不进入 ImageNet。若 bridge defect 与任何 velocity
或 endpoint 指标均无稳定关系，暂停大模型训练并重新推导机制。

阶段 2 不再要求把所有坐标变换压缩成单一的 `defect-VIV` 相关性。非正交线性
映射是严格反例：它的 chord/bridge defect 为零，但会改变 `f#N(0,I)` 与协方差谱。
因此无训练审计必须分开检验两条机制轴：

1. `source-prior mismatch`：由 `f(eps)` 相对标准 Gaussian 的分布偏移，以及
   Gaussian-straight 相对 matched-chord 的变化刻画；
2. `path-curvature mismatch`：由 transformed chord 相对严格 pushforward 的 bridge
   defect、velocity 差异和局部速度歧义刻画。

VIV 只在固定随机投影子空间和共享类内协方差假设下报告为 `projected VIV`；近邻
条件速度方差必须称为 `local velocity ambiguity proxy`，不得冒充论文定义的 VIV。

## 阶段门槛

| 阶段 | 核心产物 | 进入下一阶段的硬条件 |
|---|---|---|
| 0 实现审计 | 四分支实现、cycle/decoder/JVP 单测、旧实验差异表 | 所有数值验收通过 |
| 1 Layerwise | residual、correspondence、随机置换对照 | residual error 下降 >=15%，correspondence >=2x 随机 |
| 2 无训练路径审计 | prior、bridge、projected VIV、velocity ambiguity、Jacobian、语义保持 | identity/orthogonal 对照通过；linear 与 nonlinear 两条机制可分；nonlinear 强度下 defect 与 matched/pushforward 差异同向 |
| 3 Toy 因果 | 2D、小型图像 latent、5 seeds | Pushforward 在 >=4/5 seeds 恢复 >=50% 差距 |
| 4 CIFAR-10 | 3 seeds、50k FID/KID/P-R | 恢复 >=30%，至少 2/3 seeds 同向 |
| 5 RAE screen | 严格配对 5k/10k/20k | 5k KID 平均改善 >=5%，至少 2/3 seeds 达标 |
| 6 方法化 | Pushforward FM 或 path-regularized adapter | 50k FID 改善 >=0.3 或同 FID 步数减少 >=20% |

## 实验纪律

- 大 artifact 只写入 `~/data/eqvae/`，仓库不写 `outputs/`。
- Notebook 不提交 execution count 和 cell output。
- Toy 使用 seeds `0..4`；ImageNet screen 至少 3 个预注册 seed。
- 配对实验固定模型初始化、样本索引、augmentation、噪声、time、global batch、
  world size、优化器、EMA 和数值设置。
- 自然图像的主要几何干预只使用 `flip_h`；`flip_v/rot180` 是 stress test，严格
  `D4` 只在人工对称数据上研究。
- rFID、teacher MSE、VIV 和 proxy 只能作为机制指标，不能替代最终生成指标。

## 阶段 2 验收细则

- Identity：prior mismatch、bridge defect、matched/pushforward state 与 velocity 差异
  均 `<=1e-6`。
- Signed orthogonal channel map：bridge defect `<=1e-6`；projected prior mismatch
  不得显著超过有限样本 identity-vs-independent-Gaussian 控制的 `1.5x`。
- Anisotropic linear map：bridge defect `<=1e-6`，但 prior mismatch 与 projected VIV
  随 condition number 总体上升；这用于证明两条机制轴确实可分。
- Scaled nonlinear adapter：所有强度 cycle relative L2 max `<=1e-5`；bridge defect
  从 `alpha=0` 到 `alpha=1` 总体上升，Spearman `>=0.8`。
- 在 nonlinear adapter 强度轴上，bridge defect 与 matched-chord/pushforward 的
  state 或 velocity 差异至少一项 Spearman `>=0.6`，bootstrap 95% CI 下界 `>0`。
- 若上述控制不通过，先修实现；若控制通过但 velocity ambiguity 与两条机制轴均
  无稳定关系，则阶段 2 记为机制证据不足，仍可进入低成本 toy，但不得进入 RAE
  stage-2 大训练。

## 代码入口

- `experiments/latent_transport_paths.py`：四分支条件路径、bridge defect、JVP。
- `experiments/audit_latent_transport_protocol.py`：真实 RAE/adapter 阶段 0 审计。
- `tests/test_latent_transport_paths.py`：identity、linear、nonlinear、cycle 数学单测。
