# RAEv2 Internal Guidance 的 Decoder 反转与参数化放大审计

## 1. 核心问题

此前 latent 侧实验发现，官方 internal guidance（IG）并没有把实际采样状态
`q_t` 拉向训练路径分布 `p_t`，反而在低噪声端让两者更容易被线性分类器区分。

本实验继续回答两个问题：

1. 这种 latent 偏离经过冻结 RAE decoder 后，是被放大、保留，还是转化成更好
   的图像？
2. 低噪声端偏离迅速增大，究竟来自 full/base 双头的 raw 差异，还是来自
   `x` prediction 到 velocity 的 `1/t` 参数化？

实验全程冻结 RAEv2，不训练或修改任何模型参数。

## 2. Decoder 后实验协议

- 模型：官方 RAEv2 DINOv3-L K7，EMA，checkpoint step `100080`；
- sampler：官方 shifted 100-step Euler，IG scale `1.78`，启用区间
  `[0.1, 1.0]`；
- 每个 seed 使用 5000 个 ImageNet 样本，每类恰好 5 张；
- full、IG、`p_t` 使用相同类别和相同初始噪声；
- 800 类、4000 对样本拟合 Inception-feature diagonal LDA；
- 200 个完全未见类别、1000 对样本只用于 held-out AUC；
- 解码时刻：`t=1`、`0.410256`、`0.198347` 和真实终点 `t=0`；
- FID 使用 `/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz` 中的标准
  ImageNet-256 Inception 统计；
- 两个独立 seed：`20260801`、`20260802`。

其中：

- decoded AUC 比较 `D(p_t)` 与 `D(q_t)` 的 Inception 特征；
- real FID 比较解码图像与真实 ImageNet 参考统计；
- path FID 比较 `D(p_t)` 与 `D(q_t)`。

## 3. Decoder 后结果

下表均为 `IG - full` 的两 seed 平均。负数表示 IG 更接近或更好。

| actual t | decoded AUC 差值 | real FID 差值 | path FID 差值 |
|---:|---:|---:|---:|
| 1.000000 | 0.0000 | 0.0000 | 0.0000 |
| 0.410256 | **-0.0510** | **-11.5198** | **-1.7747** |
| 0.198347 | -0.0352 | **-1.8279** | **-0.8645** |
| 0.000000 | -0.0359 | **-0.5213** | **-0.5234** |

`t=0.410256` 的 decoded AUC 反转在两个 seed 中都显著；`t=0.198347` 和
`t=0` 在两个 seed 中方向一致，但只有第二个 seed 的配对置信区间排除 0。
三个非初始时刻的两类 FID 差值在两个 seed 中全部为负。

终点 5000 样本 FID：

| seed | full | IG | IG - full |
|---:|---:|---:|---:|
| 20260801 | 7.7493 | 7.0932 | -0.6560 |
| 20260802 | 7.5879 | 7.2013 | -0.3866 |

这些是同规模分支比较，不是官方 50k gFID，不能与论文 50k 数字直接比较。

## 4. 空间反转

结合此前 5000 样本 latent AUC：

| actual t | latent AUC: IG-full | decoded AUC: IG-full |
|---:|---:|---:|
| 0.410256 | **+0.0478** | **-0.0510** |
| 0.198347 | **+0.1138** | -0.0352 |

因此当前证据支持一个明确反转：

```text
IG 让 q_t 在线性 latent 统计上更偏离 p_t
                    ↓ frozen decoder
IG 让图像特征更难与 D(p_t) 区分，同时降低真实参考 FID
```

这说明 decoder 不是距离保持映射。它会压缩或丢弃一部分 latent 差异，并可能
把偏离训练路径的方向映射成更有利于图像质量的变化。换句话说，训练路径分布
`p_t` 不是每个时刻唯一的“正确目标”；至少对当前 decoder，存在偏离 `p_t` 但
图像侧更优的状态。

## 5. `1/t` 参数化拆分

RAEv2 使用 clean-latent prediction：

```text
v_t = (x_t - x0_hat) / max(t, 0.05)
```

同一状态上，IG 相对 full 的速度差为：

```text
Delta v_t = -Delta x0_hat / max(t, 0.05)
```

一次 Euler 更新真正施加的状态 impulse 为：

```text
Delta x_step = h(t) * Delta x0_hat / max(t, 0.05)
```

使用不改变官方采样器的顶层 forward hook，对 1000 样本、两个 seed 记录每一步
full/base 双头输出。下表为两个 seed 的平均，取 full 轨迹：

| t | raw head gap | velocity gap | Euler impulse |
|---:|---:|---:|---:|
| 1.000000 | 0.0910 | 0.0710 | 0.000089 |
| 0.797583 | 0.2135 | 0.2088 | 0.001558 |
| 0.603774 | 0.1941 | 0.2508 | 0.004616 |
| 0.410256 | 0.1664 | 0.3164 | 0.010888 |
| 0.198347 | 0.1254 | 0.4930 | 0.028590 |
| 0.140351 | 0.1269 | 0.7055 | 0.046270 |

关键结果：

- raw head gap 在后段没有爆炸；
- velocity gap 因 `1/t` 明显增长；
- 最后一个启用 step 的 Euler impulse 是起点的约 `516-518` 倍；
- full 轨迹中，`t<=0.4` 的最后 6 个启用 step 占全部 impulse 平方和约
  `86.3%`；
- `t<=0.2` 的最后 2 个启用 step 占约 `62.4%`；
- IG 轨迹在 `t=0.140351` 的 raw gap 为 `0.1432`，高于 full 轨迹的
  `0.1269`，说明轨迹分离后还有递归反馈放大。

因此低噪声端 AUC 快速增大，主要不能解释为“语义逐渐形成”。更准确的机制是：

```text
中等大小的 full/base clean-head 差异
    + x-prediction 的 1/t 转换
    + shifted Euler 网格的 h/t 集中
    + IG 轨迹上的递归反馈
    = 后段高度集中的状态 impulse
```

## 6. 当前结论

当前最可信的机制链条是：

1. full/base 双头提供中等、平滑的 clean-prediction 对比方向；
2. `1/t` 与 `h/t` 把该方向集中放大到最后几个 IG step；
3. 这使 IG latent 偏离训练路径，而不是回到训练路径；
4. frozen decoder 压缩这种 latent 偏离，并把其中一部分映射为更好的图像统计；
5. 因此 IG 更像 decoder-aware quality steering，而不是 latent distribution
   correction。

## 7. 不能过度声称

- decoded AUC 和 FID都使用 Inception 特征，不是完全独立的两种感知证据；
- AUC 只测线性可分性，不是严格分布距离；
- 中间状态本来不会在实际生成中直接解码，中间 FID只用于机制诊断；
- 只验证了 DINOv3-L K7 一个 RAEv2 模型；
- 5000 样本 FID有有限样本偏差，论文质量结论仍需 50k 最终采样；
- 当前证明了机制存在，但还没有证明对 `h/t` 做归一化一定能进一步改善生成。

## 8. 下一步最小因果实验

不训练模型，只改变 IG 的时间调度，并保持总 impulse 预算可比：

1. 官方 constant clean-space scale；
2. 限制 IG 只作用于 `t>0.4`、`0.2<t<=0.4`、`0.1<=t<=0.2`；
3. 对 `s(t)-1` 做 `t/h` 补偿，使单步 impulse 更均匀；
4. 做相反的后段集中调度，检查是否进一步改善 FID；
5. 报告 5k FID、precision/recall、latent AUC 和 decoded AUC。

该实验可以直接区分：质量改善究竟来自双头方向本身，还是主要来自官方参数化
无意形成的后段强 guidance schedule。

本地结果：

- `/home/zhoushunyu/data/eqvae/experiments/raev2_decoded_distribution_audit/cross_seed_n5000_v1`
- `/home/zhoushunyu/data/eqvae/experiments/raev2_ig_parameterization_audit/n1000_seed20260801_v1`
- `/home/zhoushunyu/data/eqvae/experiments/raev2_ig_parameterization_audit/n1000_seed20260802_v1`
