# 双目标 Flow 的 Teacher 与闭环生成诊断

## 结论先行

这组 toy 没有复现一个最强版本的负结论：

> 在已知真实分布、可以计算精确 Bayes 条件速度时，逐点最优的 `x/epsilon` oracle 混合不仅降低 teacher 路径误差，也在 `D=2` 和 `D=512` 的三个种子上都改善了最终分布。

因此，`x/epsilon` 速度混合本身并不会必然破坏闭环生成。ImageNet 双头 SiT 的失败不能仅归因于“连续 flow 里两个 endpoint target 不能混合”。

但实验同时确认了一个更细、也更重要的现象：

> 逐点速度误差最小，不等于最终生成分布在所有指标上都最好。

在 `D=512` 中，把端点安全 gate 施加到和 oracle 完全相同的 D0 基础头上，最终 ambient SWD、intrinsic SWD、MMD 和模式覆盖优于逐点 Bayes oracle；但它的 NLL 和离真实二维子空间的误差更差。也就是说，两者分别偏向了不同的分布性质，单步 MSE 无法决定闭环生成的全部结果。

## 实验回答的问题

SiT 线性路径为：

```text
x_t = (1 - t) epsilon + t x,    t: 0 -> 1
v*  = x - epsilon
```

两个 endpoint head 给出：

```text
v_x       = (x_hat - x_t) / (1 - t)
v_epsilon = (x_t - epsilon_hat) / t
v_r       = r v_x + (1 - r) v_epsilon
```

实验主要检查四件事：

1. 如果给出精确 oracle gate，teacher 路径更准是否会使最终分布也更好。
2. 当前 scaled gate loss 与真实 velocity gate loss 的区别是否来自端点降权。
3. `r = sigmoid(log((1-t)/t) + h)` 的端点安全参数化是否有效。
4. shared trunk、多任务梯度和基础 head 质量是否混入 gate 的收益。

## 实验设置

| 项目 | 设置 |
| --- | --- |
| 数据分布 | 已知二维坐标的 32 分量等权 Gaussian spiral mixture |
| 分量标准差 | `0.035` |
| ambient 维数 | `D=2, 512` |
| 尺度 | 两个维数均保持 clean ambient 每维 RMS 约为 1 |
| 模型 | 4 层 MLP，hidden `128`，单模型约 `0.19M-0.25M` 参数 |
| 训练 | 15,000 step，batch 2,048，3 个 seed |
| 采样 | 固定 200-step Heun，相同初始噪声 |
| 评估 | 4,096 个生成样本，8,192 个 reference 样本 |
| seed | `20260821, 20260822, 20260823` |

真实 clean 分布严格位于一个已知二维线性子空间，因此可以同时测量：

- intrinsic SWD：只看真实二维坐标中的分布差异。
- ambient SWD：同时惩罚二维坐标误差和法向空间残留，主结论优先使用它。
- NLL：样本在真实 32 分量窄 Gaussian mixture 下的负对数似然。
- component TV / coverage：模式权重与模式覆盖。
- off-subspace RMS：生成样本离真实二维子空间的距离。

Bayes sampler 的 Heun 收敛已单独检查。`D=512` 时从 100、200 到 400 step，intrinsic SWD 基本稳定在 `0.0202-0.0206`，200 step 的法向 RMS 仅 `0.00249`，因此正式结果不是积分步数不足造成的。

## 对照模型

| 名称 | 含义 |
| --- | --- |
| `B0_v_ind` | 独立 velocity 模型 |
| `B1_x_ind` | 独立 clean `x` 模型 |
| `B2_eps_ind` | 独立 noise `epsilon` 模型 |
| `D0_xeps` | shared trunk 的 `x + epsilon` 双头，无 learned gate |
| `D1_scaled` | 当前 ImageNet 风格的 scaled-loss gate |
| `D2_velocity` | 直接最小化真实 velocity residual 的 gate |
| `D3_oracle_bayes_gate` | 用已知分布的精确 Bayes 条件速度求解析 gate |
| `D4_safe` | true velocity loss 加端点安全 gate 参数化 |
| `S0_xv` | shared `x + v`，无 consistency |
| `S1_xv` | shared `x + v`，加 consistency |

`D3_oracle_pair_teacher_only` 使用真实配对的 `(x, epsilon)`，只用于 teacher sanity check，绝不进入采样。正式 oracle sampler 使用 `E[v* | x_t]`，只依赖当前状态和已知 toy 分布，没有使用当前样本的 clean target，因此不存在配对泄露。

## 最终分布结果

表中是三个 seed 的 ambient SWD 均值和标准差，越低越好。

| 方法 | `D=2` | `D=512` |
| --- | ---: | ---: |
| Exact Bayes | `0.0282 +/- 0.0056` | `0.0194 +/- 0.0056` |
| Independent `v` | `0.0502 +/- 0.0161` | `0.3684 +/- 0.0139` |
| Independent `x` | `0.1011 +/- 0.0040` | `0.1081 +/- 0.0283` |
| Independent `epsilon` | `0.1815 +/- 0.0664` | `961.62 +/- 0.31` |
| D0 fixed `x/epsilon` switch | `0.0421 +/- 0.0097` | `0.3554 +/- 0.0099` |
| D0 `r=1-t` | `0.0454 +/- 0.0099` | `0.3757 +/- 0.0089` |
| Scaled-loss gate | `0.0420 +/- 0.0066` | `0.0967 +/- 0.0058` |
| True-velocity gate | `0.0461 +/- 0.0080` | `0.0951 +/- 0.0245` |
| Bayes oracle gate | `0.0424 +/- 0.0084` | `0.0803 +/- 0.0079` |
| Safe true-velocity gate | `0.0406 +/- 0.0076` | `0.0687 +/- 0.0217` |
| `x+v`, no consistency | `0.0725 +/- 0.0054` | `0.1375 +/- 0.0069` |
| `x+v`, consistency | `0.0661 +/- 0.0042` | `0.1380 +/- 0.0047` |

关键结果：

1. `D=2` 和 `D=512` 的全部三个 seed 中，Bayes oracle 都优于 D0 自己最好的单分支。
2. `D=512` 时，D0 的 `x` 分支为 `0.1236`，oracle 为 `0.0803`，改善约 `35%`。
3. `D=512` 时，`epsilon` 和直接 `v` 无法清除大量法向噪声。只看二维投影会误判，因此必须报告 ambient 指标。
4. `x+v` consistency 在 `D=2` 有约 `9%` 改善，在 `D=512` 没有改善。这里只测试了一个权重和固定切换点，不能据此否定 SC-Flow。

## 公共基础头对照

为了区分“gate 选择”与“gate 训练改变了 shared trunk”，三种 learned gate 又被施加到同一个 D0 `x/epsilon` 基础头上。

### `D=512` 公共头结果

| gate | Ambient SWD | Intrinsic SWD | NLL | Component TV | Off-subspace RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 pure `x` | `0.1236` | `0.1006` | `20.62` | `0.286` | `0.0032` |
| Scaled gate on D0 | `0.0969` | `0.0807` | `18.99` | `0.232` | `0.0031` |
| Velocity gate on D0 | `0.0818` | `0.0692` | `18.88` | `0.184` | `0.0216` |
| Bayes oracle on D0 | `0.0803` | `0.0675` | `18.98` | `0.172` | `0.0148` |
| Safe gate on D0 | `0.0762` | `0.0639` | `19.33` | `0.150` | `0.0608` |

这张表揭示了两件不同的事：

1. Scaled gate 的最终改善主要来自它学到的 shared trunk/head，而不是 gate 本身。它自己的 gate 结果 `0.0967` 与自己的纯 `x` 分支 `0.0967` 几乎相同。
2. Safe gate 在三个 seed 上的 ambient SWD 都略优于逐点 Bayes oracle，但 NLL 在三个 seed 上都更差，法向误差约为 oracle 的 4.1 倍。它改善的是全局形状、模式权重和 SWD，不是所有意义下都更接近真实分布。

## Teacher 与 Rollout 的分离

在 `D=512, t=0.99` 的公共 D0 头上：

| gate | Teacher Bayes-field MSE | Rollout Bayes-field MSE | Endpoint ambient SWD |
| --- | ---: | ---: | ---: |
| Pure `x` | `5.290` | `9.199` | `0.1236` |
| Scaled gate | `5.273` | `9.131` | `0.0969` |
| Velocity gate | `0.871` | `4.762` | `0.0818` |
| Bayes oracle | `0.712` | `3.200` | `0.0803` |
| Safe gate | `0.915` | `60.253` | `0.0762` |

Bayes oracle 在相同 teacher state 上按定义是逐样本最优的 convex scalar mix。它也明显降低了 rollout state 上的局部场误差。

但是 safe gate 在末端的 rollout 局部误差更大，最终 SWD 却略低。这不是代码矛盾，而是说明：

```text
局部场误差的时间位置、误差方向、状态访问分布
        ↓
共同决定最终 transport
        ↓
最终不同分布指标还会偏好不同性质
```

特别是 `t=0.99` 附近只占很短的积分区间。一个很大的瞬时 MSE 不一定比更早、更持久的小偏差对终点影响更大。反过来，低 SWD 也不能自动代表高 likelihood 或更小的法向误差。

## Scaled Loss 的端点问题得到支持

scaled residual 满足：

```text
L_scaled = t^2 (1-t)^2 L_velocity
```

它对所有时间共享的 gate 网络改变了总体优化权重，并不只是乘了一个不影响训练的常数。

在 `D=512, t=0.99`，scaled gate 的 teacher error `5.273` 几乎等于 pure `x` 的 `5.290`；真实 velocity gate 已降到 `0.871`。branch audit 也显示 scaled gate 此时仍给 `x` 约 `0.998` 的权重，而解析 oracle 约为 `0.31`。

因此下面这条机制得到直接支持：

> scaled objective 在两个端点把 gate 梯度严重降权，导致它没有学到连续 velocity conversion 所需的端点切换。

`r=1-t` 虽然代数上完全消除了两个分母，得到 `v = x_hat - epsilon_hat`，但 `D=512` 的 SWD 为 `0.3757`。这说明“避免数值奇异”是必要条件之一，却不是充分条件；高维 `epsilon` 预测误差仍会直接进入速度场。

## Shared Trunk 梯度结果

D0 的两个任务在 shared trunk 上的整体梯度 cosine 为：

| 维数 | `cos(g_x, g_epsilon)` | `||g_x||` | `||g_epsilon||` |
| --- | ---: | ---: | ---: |
| `D=2` | `0.082` | `0.088` | `0.719` |
| `D=512` | `0.034` | `0.129` | `0.053` |

这不是强负 cosine，因此当前结果不支持“两个任务普遍反向冲突”这一最强说法；更准确的是：它们整体近乎正交，并且梯度尺度随维数发生反转。shared 模型的基础分支通常弱于相应 independent 模型，说明多任务优化代价确实存在，但不能只用一个平均 cosine 解释全部差距。

还要注意，gate loss 虽然 detach 了 `x/epsilon` 输出 residual，仍会通过 gate head 回传到 shared trunk；全局 gradient clipping 也会让 gate 梯度影响整个模型的更新尺度。因此端到端 D1/D2/D4 之间的差异不能全部归因于 gate 公式。公共 D0 头对照正是为排除这一混杂而加入的。

## 当前可以确认什么

可以确认：

1. `x/epsilon` continuous velocity mixing 在 toy 上有可实现的闭环收益，不是形式上必然失败。
2. ImageNet 当前 scaled gate objective 的端点降权是一个真实问题。
3. 高维下必须同时看 intrinsic 和 ambient 指标；法向误差不能被二维图隐藏。
4. Teacher/rollout 局部场误差、终点 SWD、NLL 和法向距离不是同一个目标，它们可以出现稳定分歧。
5. learned safe gate 给出了一个真实的闭环分布偏置现象，但它不是“全面胜过 oracle”，而是全局覆盖与局部 likelihood/法向精度之间的取舍。

不能确认：

1. 不能据此证明 safe gate 在真实图像上会改善 FID。
2. 不能据此说 SC-Flow consistency 无效。
3. 不能把 D4 端到端结果全部解释成 gate 参数化，因为其 shared trunk 也被 gate objective 改变。
4. 只有三个训练 seed，`D4` 端到端方差仍较大，现阶段应称为机制证据而非最终方法结果。

## 下一步最有信息量的实验

1. 在 toy 上加入固定 NFE sweep，检查 safe gate 的优势是否依赖 200-step Heun。
2. 把 gate trunk 完全 detach 或使用独立小 gate 网络，严格隔离 gate 选择与 shared representation 改变。
3. 用按时间积分加权的 rollout defect 指标替代单点 MSE，检验它是否比 teacher MSE 更能预测终点分布。
4. 在小型图像 flow 上只比较 D0 公共头的 `scaled / velocity / safe / oracle-proxy`，不要立即训练更大的 ImageNet 双头模型。

## 复现位置

代码：

```text
experiments/run_dual_target_closed_loop_toy.py
experiments/summarize_dual_target_closed_loop_toy.py
tests/test_dual_target_closed_loop_toy.py
tests/test_summarize_dual_target_closed_loop_toy.py
```

本机正式结果：

```text
/home/zhoushunyu/data/eqvae/experiments/dual_target_closed_loop_toy_v1/
```

主要汇总：

```text
aggregate/endpoint_seed_summary.csv
aggregate/cross_gate_seed_summary.csv
aggregate/cross_gate_teacher_seed_summary.csv
aggregate/cross_gate_rollout_seed_summary.csv
aggregate/branch_seed_summary.csv
aggregate/gradient_all.csv
aggregate/endpoint_ambient_swd.png
aggregate/teacher_vs_rollout_bayes_error.png
```
