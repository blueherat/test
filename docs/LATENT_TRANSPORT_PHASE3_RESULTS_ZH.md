# Latent Transport 阶段 3A：四路径 toy 结果与停止决定

## 最终判断

阶段 3A 是一个**有效且机制可解释的负结果**：

> 正确的 probability-path pushforward 在连续数学上保持坐标共轭，但把同一个普通
> MLP 放到 nonlinear 坐标中重新训练，并不会自动恢复 Base 生成质量。

预注册的数值、solver、Base 质量和 coordinate-gap 条件全部通过；失败的是方法核心：
Pushforward 在 `0/5` seeds 恢复一半差距，平均 SW1 是 Base 的 `2.60x`。因此按门控
停止 `H4/H5`，不进入 MNIST/FashionMNIST、CIFAR 或 RAE stage-2 大训练。

正式 artifact：

```text
~/data/eqvae/experiments/latent_transport_four_path_toy/
  preregistered_v1_20260718_131400/
```

预注册见 `docs/LATENT_TRANSPORT_PHASE3_PREREG_ZH.md`。

## 实验有效性

主实验使用 8 模态环形 Gaussian mixture、quadratic invertible coupling、width-64
三层 residual MLP、4,000 updates、5 seeds、8,192 endpoint samples 和 100-step
Heun。四个分支每 seed 共享初始化、data、epsilon、time 和优化超参数。

| 检查 | 结果 | 门槛 | 状态 |
|---|---:|---:|---|
| 初始参数最大差 | `0` | `0` | 通过 |
| transform cycle relative max | `1.67e-7` | `<=1e-6` | 通过 |
| JVP vs central difference max | `2.62e-5` | `<=1e-4` | 通过 |
| 100/200-step endpoint relative max | `1.83e-4` | `<0.02` | 通过 |
| Base 有效 seed | `5/5` | `>=4/5` | 通过 |
| Gaussian/Base gap seed | `4/5` | `>=4/5` | 通过 |
| Pushforward 恢复一半 | `0/5` | `>=4/5` | 失败 |
| Pushforward mean / Base mean | `2.605` | `<=1.10` | 失败 |

所有分支在所有 seed 都覆盖 8 个 mode。Base mean SW1 为 `0.0298`，独立
reference-vs-reference 的有限样本 floor 为 `0.0194`；所以 Base 已经学到有效生成，
不能用“模型整体没训练好”解释结果。

## 主强度结果

主强度固定为 `a=1`，没有根据结果换强度。

| seed | Base | Gaussian | Matched | Pushforward | Gaussian/Base | Push recovery |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.0189` | `0.0718` | `0.0628` | `0.0740` | `3.79x` | `-0.041` |
| 1 | `0.0243` | `0.0760` | `0.0596` | `0.0786` | `3.13x` | `-0.050` |
| 2 | `0.0383` | `0.0654` | `0.0558` | `0.0781` | `1.71x` | `-0.469` |
| 3 | `0.0248` | `0.0720` | `0.0537` | `0.0811` | `2.90x` | `-0.194` |
| 4 | `0.0424` | `0.0414` | `0.0535` | `0.0758` | `0.98x` | 不定义 |
| mean | `0.0298` | `0.0653` | `0.0571` | `0.0775` | `2.20x` | 负 |

Matched chord 在主强度上平均比 Gaussian 好，说明 source endpoint/pairing 确实解释
一部分退化；但它仍明显不如 Base。严格 Pushforward 又比 Gaussian 更差，说明消除
chord mismatch 并不等于降低有限模型学习难度。

辅助强度 `a=0.5` 也没有反转：Base/Gaussian/Matched/Pushforward mean SW1 分别为
`0.0298/0.0464/0.0569/0.0667`。因此负结果不是只在一个极强变换点出现。

Exact mixture NLL 与 SW1 同向：主强度均值为
`1.244/2.792/2.363/3.050`。可视化中的 transformed 分支都在上半圆形成 mode 间
拖尾，Pushforward 最明显；这不是单个指标的偶然排序。

## 为什么 Pushforward 会更差

变换为：

```text
y1 = x1
y2 = x2 + a*x1^2
```

Base field `v=(v1,v2)` 的严格 pushforward 是：

```text
v_y1 = v1(x,t)
v_y2 = v2(x,t) + 2*a*x1*v1(x,t)
x1 = y1
x2 = y2 - a*y1^2
```

所以即使 Base field 能由 width-64 MLP 表达，transformed field 还需要：

- 在输入端实现 quadratic inverse composition；
- 在输出端实现 state 与 velocity 的乘法；
- 在同一固定宽度中分配额外容量；
- 在 Euclidean y-space MSE 下承受 `J_f^T J_f` 的方向加权。

函数类与损失都不是 coordinate-invariant。`f` 可逆只保证信息不丢失，不保证
`MLP_y` 与 `J_f MLP_x` 是同一个有限假设类，也不保证两个 Euclidean regression
objective 等价。

Held-out microscopic MSE 也印证了复杂度增加。主强度 Base/Gaussian/Matched/Push
均值为 `1.865/2.318/2.900/4.307`，对应 target velocity RMS 为
`1.738/2.481/2.377/3.047`。Push target 的动态范围和条件结构都更难。

## Conjugacy control：排除实现错误

为了区分“Pushforward 公式错了”和“普通 MLP 学不好共轭 field”，没有重新训练或
选择 checkpoint，而是把已训练 Base field 直接构造成：

```text
v_conj(y,t) = J_f(f^{-1}(y)) v_base(f^{-1}(y),t)
```

从同一个 `f(eps)` 出发积分，再应用 `f^{-1}`。结果：

| strength | Heun steps | Base vs conjugated endpoint relative mean |
|---:|---:|---:|
| 0.5 | 100 / 200 / 400 | `4.4e-5 / 1.1e-5 / 3.0e-6` |
| 1.0 | 100 / 200 / 400 | `8.9e-5 / 2.2e-5 / 5.8e-6` |

- 400-step 所有任务最大 mean error：`7.56e-6`；
- 100-step 到 400-step 的误差下降倍数最小：`14.78x`，符合 Heun 二阶趋势；
- Base 与 conjugated SW1 在显示精度内相同；
- field 定义的 full-batch absolute max：`5.72e-6`。

因此 JVP、source、时间方向、ODE 与 inverse 计算图是正确的。数学上 Pushforward
完全恢复 Base；失败只发生在“重新用相同普通 MLP 与 y-space MSE 学它”这一步。

主强度下，独立训练的 Push field 与 exact conjugated Base field：

- teacher-state prediction relative L2：均值 `0.174`；
- 同噪声 endpoint relative L2：均值 `0.087`；
- microscopic MSE 仅从 conjugated 的 `4.288` 变到 learned 的 `4.306`。

最后一项尤其重要：不到 `0.5%` 的 microscopic MSE 差异仍对应明显 endpoint 分布
退化。它再次复现仓库中已经观察到的 teacher-forcing/rollout gap，说明 pointwise
velocity MSE 会被 irreducible pair noise 淹没，不能保证多步 transport 正确。

Conjugacy audit 文件：

```text
conjugacy_endpoint.csv
conjugacy_field.csv
conjugacy_audit.json
```

## 对原始 RAE idea 的含义

原始“encoder 后加可逆 adapter、decoder 前加 inverse”只保证 reconstruction closure。
即使再把 stage-2 path 改成严格 Pushforward，也不保证固定 DiT 容量更容易训练；
它可能同时增加输入 composition、Jacobian multiplication、目标动态范围和 rollout
敏感性。当前证据不支持在 RAE 上继续付出大训练成本。

可被保留的理论结论是：

> Latent reparameterization、probability path、vector-field parameterization 与 loss
> metric 必须共同设计。只改表示或只改 path 都不是 coordinate-invariant 方法。

若未来另立新目标，只有两条逻辑自洽路线：

1. 显式共轭参数化 `v_y=J_f v_x` 并在 base metric 下训练；它应恢复 Base，但本质上
   是坐标重写，必须再证明 adapter 提供了额外收益；
2. 联合学习接近 Gaussian-preserving、低 Jacobian spread、低 bridge defect 的 `f`，
   直接把 transport complexity 纳入正则；这是新方法假设，不能算本阶段延伸成功。

按当前预注册，本研究主线在此停止，不做小图像/CIFAR/RAE 后续。

## 代码

- `experiments/latent_transport_four_path_toy.py`
- `experiments/audit_four_path_toy_conjugacy.py`
- `tests/test_latent_transport_four_path_toy.py`
