# RAEv2 Internal Guidance 分布 AUC 审计

## 1. 问题

本实验只回答：在冻结 RAEv2 后，官方 internal guidance 是否让实际采样状态
`q_t` 更接近训练路径的真实 latent 分布 `p_t`。

训练路径由官方代码直接给出：

```text
p_t = (1 - t) * E(x) + t * epsilon
```

其中 `t=1` 是高斯噪声，`t=0` 是干净 latent。实际采样按官方 shifted Euler
网格从 `t=1` 向 `t=0` 积分。实验没有训练或修改 RAEv2。

## 2. 协议

- checkpoint：官方 DINOv3-L K7 stage2 checkpoint 的 EMA 权重；
- sampler：与本地既有 FID 对照一致的 100-step Euler ODE；
- full：使用 internal-guidance 公式但令 scale 为 `1.0`，数学上精确等于 full
  head；
- IG：官方 scale `1.78`，区间 `[0.1, 1.0]`；
- 时刻：请求 `0.2/0.4/0.6/0.8/1.0`，实际 shifted grid 时刻为
  `0.198347/0.410256/0.603774/0.797583/1.0`；
- 每个 seed 使用 5000 个样本，覆盖 1000 个 ImageNet 类别且每类恰好 5 张真实
  train 图；每张真实图对应一个同类别生成样本；
- full、IG、`p_t` 使用相同类别和相同初始高斯噪声；
- 800 个类别共 4000 对样本拟合探针，200 个完全未见类别共 1000 对样本只用于
  held-out AUC；同一类别不会跨 train/test；
- 探针是原始 `1024 x 16 x 16` latent 上的全维 diagonal-LDA 线性分类器，
  所有均值和方差只由训练类别估计；
- 置信区间由 held-out 类别做配对 bootstrap，重复 2000 次。

## 3. 正确性检查

以下检查均通过：

1. `t=1` 时 `p_t=q_t^full=q_t^IG`，两条 AUC 精确为 `0.5`；
2. full 与 IG 的初始 latent 哈希逐 rank 相同；
3. 在非初始时刻，两条轨迹确实分开，不存在误把同一分支采样两次的问题；
4. 每条真实 ImageNet 记录的 label 都在运行时与目标类别逐条核对；
5. held-out ID 在 `p/full/IG` 间逐项一致；
6. sampler 记录的是每个 Euler step 调用模型前的状态，不是模型输出或人为重写
   的近似轨迹；
7. 源码中不存在 optimizer、backward、训练模式或 checkpoint 写回；
8. 两个 seed 的 AUC 均已从保存的原始 held-out probe scores 独立重算，与 CSV
   完全一致；
9. 7 个静态/toy 测试通过，`py_compile` 和 `git diff --check` 通过。

## 4. 结果

表中 `Delta AUC = AUC(IG) - AUC(full)`。正数表示 IG 后的状态更容易与真实
`p_t` 区分，即在线性探针下更远。

| actual t | seed 20260801 | seed 20260802 | 两 seed 平均 | 稳定判断 |
|---:|---:|---:|---:|---|
| 1.000000 | 0.0000 | 0.0000 | 0.0000 | 硬负对照通过 |
| 0.797583 | +0.0054 | +0.0059 | +0.0056 | 同向但不显著 |
| 0.603774 | +0.0129 | +0.0137 | +0.0133 | 同向但不显著 |
| 0.410256 | **+0.0474** | **+0.0482** | **+0.0478** | 两 seed 均显著更远 |
| 0.198347 | **+0.1169** | **+0.1108** | **+0.1138** | 两 seed 均显著更远 |

`t=0.198347` 的绝对 AUC：

| seed | full | IG |
|---:|---:|---:|
| 20260801 | 0.6582 | 0.7751 |
| 20260802 | 0.6796 | 0.7904 |

IG 与 full 轨迹的跨 seed 平均相对 RMS 差异随采样推进由约 `8.3%` 增长到
`52.7%`。
`t=0.198347` 也恰好位于 IG 关闭前的最后阶段：后续网格为 `0.140351`、
`0.074766`、`0`，而 IG 在 `t<0.1` 关闭。

此前每类 1 张图、共 1000 样本的 pilot 只在 `t=0.198347` 稳定复现正差值，
中间时刻受样本选择影响较大。扩大到每类 5 张图后，两颗 seed 的曲线几乎重合，
并在 `t=0.410256` 额外得到稳定显著的正差值，说明 pilot 的中间时刻确实存在
较大的有限样本波动。

## 5. 结论

### 已被证据支持

- 没有跨 seed 的稳定证据表明 IG 在整条轨迹上持续把 `q_t` 拉向 `p_t`；
- 在 `t=0.410256` 和靠近输出的 `t=0.198347`，IG 在两个独立 seed 中都使
  `q_t` **更容易**与训练路径的 `p_t` 区分；
- 因而“IG 是全程分布误差校正器”这一强假设被当前实验否定；
- 结合已有 IG 改善 FID 的结果，当前证据更符合：IG 至少在后段是一种质量
  倾斜或外推，而不是简单最小化到训练路径分布的偏差。

### 不能声称

- AUC 不是严格分布距离，只能说明给定全维线性探针的可分辨性；
- 两个 seed 不足以精确估计跨 seed 方差；
- 单次运行的 bootstrap 只覆盖 held-out 样本，没有覆盖探针训练集的不确定性；
  两个独立 seed 用于检查训练类别划分、图像选择和噪声变化后的稳定性；
- 本实验没有记录 `t=0`，不能直接声称最终 latent 分布一定更远；
- `t=0.603774/0.797583` 虽然两 seed 均为小幅正差值，但单 seed 置信区间包含
  0，不能声称这些时刻已经显著更远。

## 6. 决策

当前不应继续基于“让分布误差归零”的假设训练 density-ratio critic、PID、
MPC 或多层控制器。最小二元问题已经得到足够明确的否定证据：

```text
D(q_t^IG, p_t) < D(q_t^full, p_t)
```

不在整条轨迹上稳定成立，并在靠近输出的已测后段稳定反向。

本地结果：

- `/home/zhoushunyu/data/eqvae/experiments/raev2_distribution_auc/n5000_seed20260801_v1`
- `/home/zhoushunyu/data/eqvae/experiments/raev2_distribution_auc/n5000_seed20260802_v1`
- `/home/zhoushunyu/data/eqvae/experiments/raev2_distribution_auc/cross_seed_n5000_v1`
