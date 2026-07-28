# Latent Transport 阶段 0 审计

## 结论

阶段 0 的路径公式与真实 checkpoint 数值验收已经通过，但旧 stage-2 结果不能用于
判断 adapter 的因果生成效果。旧 adapter 实验执行的是 `Gaussian straight`，同时
混合了 source-prior mismatch、conditional-path mismatch 和不配对训练随机性。

审计 artifact：

```text
~/data/eqvae/audits/latent_transport_phase0/protocol_audit.json
```

## 当前真实计算图

```text
x
 -> frozen DINOv2 encoder
 -> reshape
 -> ImageNet latent mean/variance normalization
 -> z
 -> frozen invertible adapter f
 -> y=f(z)

y
 -> f^{-1}(y)
 -> latent de-normalization
 -> frozen ViT-XL decoder
 -> x_hat
```

源码证据：

- `external/RAE/src/stage1/rae.py:109-126`：encoder、reshape、latent normalization。
- `external/RAE/src/stage1/adapted_rae.py:112-119`：normalization 后执行 adapter，
  decode 前先执行 inverse。
- `external/RAE/src/train.py:428-432`：stage-1 encode 位于 `torch.no_grad()`，优化
  目标只传入 stage-2 model。

因此 adapter 的确位于 normalized public latent 坐标中，inverse 也确实位于 RAE
denormalization 和 decoder 之前。

## 旧 stage-2 实际路径

`external/RAE/src/stage2/transport/transport.py:155-182` 始终用
`torch.randn_like(x1)` 产生 `x0`。Linear IC path 在
`external/RAE/src/stage2/transport/path.py:115-137` 中实现为：

```text
x_t = (1-t) x1 + t x0
u_t = x0 - x1
```

adapter 配置中的 `x1=f(z)`，所以旧训练实际是：

```text
x_t = (1-t) f(z) + t eta,  eta ~ N(0,I)
```

`external/RAE/src/sample_ddp.py:283-313` 同样从标准 Gaussian 开始，ODE 结束后
直接交给 `AdaptedRAE.decode`。旧实验没有使用 `f(eps)`，也没有使用
`f((1-t)z+t eps)`。

## 数值验收

数据使用 ImageNet validation 的固定索引 `6890, 3900`。它们既不属于 adapter
训练使用的 ImageNet-train `0..32767`，也不属于旧 adapter test 使用的 validation
`0..2047`。

| 检查 | 观测 | 门槛 | 结果 |
|---|---:|---:|---|
| cycle relative L2 max | `2.37e-7` | `1e-6` | 通过 |
| decoder identity pixel abs mean | `2.82e-7` | `1e-6` | 通过 |
| 相同 latent 重复 decode max | `0` | `1e-7` | 通过 |
| JVP vs central finite difference | `4.23e-5` | `1e-4` | 通过 |
| identity bridge defect max | `0` | `1e-6` | 通过 |
| orthogonal bridge defect max | `0` | `1e-6` | 通过 |

cycle 后 decoder 的单像素最大误差为 `1.53e-5`，但全图平均为 `2.82e-7`；相同
latent 重复 decode 完全一致。因此这是 fp32 cycle roundoff 经 ViT-XL 放大的极少数
单点误差，不是 dropout、非确定性或 decode 逻辑错误。验收采用预定“像素误差”
的全图平均，并额外要求重复 decode 的最大误差为零。

当前 adapter 在这两个样本、`t=0.1/0.9` 上的 bridge defect 平均为 `0.280`、最大
为 `0.502`。这只是阶段 2 的强信号，不构成统计结论；需要时间网格、样本规模、
adapter 强度和对照映射后才能判断。

## 旧 20k 比较为什么不配对

| 项目 | 原始 latent | adapter latent | 后果 |
|---|---|---|---|
| stage-1 target | `RAE` | `AdaptedRAE` | 预期干预 |
| global batch | 16 | 16 | 相同 |
| 配置 global seed | 42 | 42 | 表面相同 |
| 初始 world size | 2 | 1 | 实际 rank-0 seed 分别为 84 与 42 |
| resume world size | 2 | 4 | noise、flip、归约与 RNG stream 改变 |
| 模型初始化 | seed 84 | seed 42 | 不相同 |
| source/path | Gaussian straight | Gaussian straight | adapter 分支同时改变 prior/path |

训练源码使用 `seed = global_seed * world_size + rank`，且 seed 在模型实例化之前
设置，所以不同 world size 已经改变模型初始权重，不只是数据分片。旧
`92.46 -> 80.13` 只能视为信号，不能视为 adapter 的因果提升。

Parquet dataloader 对该格式设置 `shuffle=False`。审计了前 320,000 个样本：覆盖
全部 1000 类，每类 176-377 张，前 10,000 个标签有 9,993 次变化，所以没有发现
类别顺序聚集。相同 global batch 下，各 world size 的全局索引集合也基本一致；
主要不配对来源仍是初始化、augmentation/noise RNG 和归约路径。

## 数值设置问题

旧训练命令虽然使用 `precision=fp32`，但 `external/RAE/src/train.py:13-15` 在模块
加载时强制开启 TF32。因此旧结果是 fp32 tensor/autocast disabled，但矩阵乘法允许
TF32，不属于严格数值审计所说的 full fp32。后续严格配对实验必须显式统一并记录
TF32 开关；阶段 0 数值审计已关闭 TF32。

## 阶段 0 决策

阶段 0 通过，可以进入阶段 1 和阶段 2。当前没有需要终止主线的理论矛盾；相反，
旧实验未区分 prior 与 path 的事实正是四分支因果实验的必要性来源。
