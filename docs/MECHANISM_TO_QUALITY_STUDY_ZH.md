# 从失败机制到生成质量：P22-P24 结论

## 技术摘要

本轮严格执行“先预测、后研究”，结果给出了一个清楚但不应美化的结论：

> 已发现的 risk-allocation / gradient-neglect / endpoint-leverage 机制可以指导我们
> 消除方向加权造成的大部分生成退化，但它本身没有提供一个稳定优于原始 baseline
> 的正向生成信号。

冻结 baseline、只在非保护子空间训练 residual adapter 后，FashionMNIST DCT 的
feature-FID ratio 从 raw weighted 的 `1.461` 降到 `1.009`，消除约 `98.1%` 的
额外损伤；PCA 从 `1.787` 降到 `1.208`，消除约 `73.6%`。但它们都没有通过
绝对质量 gate。进一步的独立 validation/test residual-scale 选择只有 `0.9978x`
FID，三个 seed 直接选择 scale `0`，不能称为生成质量提升。

因此当前答案是：

- **这个机制会带来什么：** teacher 指标和 rollout 质量断裂，高杠杆方向被有限
  共享网络牺牲，随后 off-path dynamics 放大。
- **需要怎么处理：** 保留原 MSE/baseline transport 的受保护路径；不要在没有
  oracle conditional field 时给 self-generated states 复用原 paired target；新目标
  必须直接测量 marginal/endpoint，并用绝对 baseline gate。
- **能否提高生成质量：** 当前三轮预注册实验只证明“避免伤害”，没有证明“稳定
  提升”。因此不进入 MNIST 复现或 tiny RAE 训练。

## 为什么机制不自动等于质量提升

机制识别的是一个负风险：

```text
degradation risk
  ~= risk-budget shift
     x gradient decoupling
     x endpoint leverage
     x no protected path.
```

消除其中一项可以阻断退化，但并不保证剩余 correction 具有正 endpoint leverage。
更完整的质量变化应写成：

```text
net quality gain
  ~= positive endpoint leverage of the learned correction
     - trajectory-distribution shift introduced by that correction.
```

P22 精确保护了 group 0，却仍只把 DCT 拉回 baseline、PCA 仍高于 baseline。这说明
groups 1-7 的 teacher-MSE 改善不等于正的感知/分布杠杆。原 baseline 已经处在当前
小模型、训练预算和 rollout 指标下较强的局部折中；减少某些条件速度误差只是换了
近似，并不自动改善生成分布。

## P22：受保护 residual 消除退化，但没有实现 Pareto

固定 raw baseline 全部参数，增加参数量为 baseline `25.3%` 的 width-12、零初始化
adapter。adapter 输出严格投影到 groups 1-7，同一状态上的 group-0 field delta
最大仅约 `6.5e-7`。

五 seed 均值：

| basis | variant | feature FID / baseline | feature SWD | latent SWD | group-0 MSE | detail gain retention |
|---|---|---:|---:|---:|---:|---:|
| DCT | raw weighted | 1.461 | 1.198 | 1.241 | 1.086 | 100% |
| DCT | teacher residual | 1.009 | 1.005 | 0.997 | 1.000 | 42.2% |
| DCT | rollout/drift residual | 1.147 | 1.076 | 1.028 | 1.000 | 27.2% |
| PCA | raw weighted | 1.787 | 1.330 | 1.234 | 1.098 | 100% |
| PCA | teacher residual | 1.208 | 1.107 | 1.072 | 1.000 | 64.9% |
| PCA | rollout/drift residual | 1.368 | 1.183 | 1.104 | 1.000 | 52.2% |

预测审计：

- 精确 coarse 保护：命中。
- teacher residual 同时 `FID<=1.02` 且保留至少 50% detail gain：两个基底都未
  同时满足。DCT 质量接近但 retention 只有 `42.2%`；PCA retention 达标但质量差。
- rollout/drift `FID<=0.98`：失败。
- rollout/drift 不比 teacher-only 差 2% 以上：失败，两个基底都稳定差约 13%。

## P23：off-path 反效果来自错误 paired target

P22 combined objective 中，最终 drift scalar 只有约 `3.5e-4-4.6e-4`，而新增
off-path paired MSE 约 `0.20-0.25`。P23 在结果前预测 offpath-only 会复现伤害，
drift-only 会接近 teacher-only。

FashionMNIST DCT 五 seed：

| variant | feature FID / baseline | feature SWD | latent SWD | detail gain retention |
|---|---:|---:|---:|---:|
| teacher-only（P22） | 1.009 | 1.005 | 0.997 | 42.2% |
| offpath-MSE only | 1.131 | 1.069 | 1.024 | 27.6% |
| drift only | 1.034 | 1.018 | 1.005 | 40.9% |
| combined（P22） | 1.147 | 1.076 | 1.028 | 27.2% |

四条预测全部命中：offpath-only 比 teacher-only 差 `0.122`，与 combined 仅差
`0.016`；drift-only 与 teacher-only 差 `0.025`。三者 teacher nonzero MSE 接近，
endpoint 却明显分离。

原因是原 paired target `noise - data` 在直线 interpolation state 上定义监督问题；
self-generated state 已离开该条件分布，同一个 pair target 不再是该状态下可靠的
conditional vector field。把它继续当标签，会训练出错误的 off-manifold extension。

## P24：trust region 没有发现稳定正 leverage

P24 不训练模型。原 P22 的 1,024 张测试图全部排除，使用新的 1,024 张 validation
从 `{0, .25, .5, .75, 1}` 选择 residual scale，再用另 1,024 张 final test 和新
噪声报告一次。

结果：

| seed | selected scale | final-test FID ratio | feature SWD | latent SWD |
|---:|---:|---:|---:|---:|
| 0 | 1.0 | 0.9946 | 1.0001 | 1.0016 |
| 1 | 1.0 | 0.9946 | 1.0007 | 0.9941 |
| 2 | 0.0 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 0.0 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 0.0 | 1.0000 | 1.0000 | 1.0000 |
| mean | - | 0.9978 | 1.0002 | 0.9991 |

没有 seed 选择中间 scale，违反最核心预测；mean FID 未达到 `0.98` gate。`0.9978`
主要来自三个 seed 回退 baseline、两个 seed 得到约 `0.5%` 的小改进。这种幅度低于
本研究预先规定的实用阈值，也可能包含 validation model-selection bias，不能作为
质量提升结果。

## 处理原则

### 当前可执行

1. 停止固定 inverse-variance DCT output weighting；原始 baseline 是当前质量选择。
2. 若研究方向预条件，只允许在原 MSE 主路径之外添加增量，并为经验高杠杆方向
   提供严格 no-regression/protected path。
3. self-generated states 上不能复用原 paired velocity label。没有 oracle 时，只能
   使用 baseline trust region、可识别的 marginal constraints，或直接 rollout-level
   distribution objective。
4. 所有方法同时报告绝对 baseline、teacher risk、短 rollout 和最终生成分布；
   `teacher MSE` 改善不能作为进入 RAE 的 gate。

### 若继续寻找正向方法

唯一仍与证据一致的低成本方向，是在独立 train split 上直接优化短 rollout 的
边缘分布，例如 fixed random projections 下的 SWD、cross-band covariance 或冻结
独立 feature network 的分布距离，同时保留原 MSE 和 protected path。它必须：

- 使用与最终评价器不同的 train-only proxy，避免直接优化测试 FID；
- 在未见 validation 选择后，再在完全独立 test/noise 上达到至少 `2%` 改善；
- 同时改善两个分布指标，而不是只改善单一 feature FID；
- 先通过 FashionMNIST 和 MNIST 多 seed，再进入一次 tiny RAE screen。

这是一条新的 endpoint-aware 方法假设，不是当前机制已经证明的收益。继续研究前
应重新预注册；不能把 P24 的 `0.2%` 选择收益当作进入 RAE 的理由。

## 数据与复现

代码：

- `experiments/small_image_residual_adapter.py`
- `experiments/small_image_residual_trust_region.py`

正式结果：

- P22：`$HOME/data/eqvae/experiments/small_image_residual_adapter/fashion_mnist_residual_adapter_preregistered_20260716_215425/`
- P23：`$HOME/data/eqvae/experiments/small_image_residual_adapter/fashion_mnist_residual_adapter_preregistered_20260716_220534/`
- P24：`$HOME/data/eqvae/experiments/small_image_residual_trust_region/residual_trust_region_preregistered_20260716_221738/`

P22/P23/P24 顶层表分别有 `40/20/25+15` 行，复合主键无重复。数值列中的空值
只来自 baseline/raw weighted 不定义 `detail_gain_retention`；其余数值全部有限。
P24 每 seed 的 source/validation/final-test indices 均为 `1,024` 个且三者完全
不相交，validation 与 final test 使用不同噪声和 projection seed。
