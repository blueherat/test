# Latent Transport 阶段 3A：2D 四路径因果实验预注册

## 唯一问题

在信息与重构完全不变的 nonlinear coordinate map 下，生成性能差异究竟来自：

1. source prior 不匹配；
2. straight chord 不等于原路径 pushforward；
3. 还是 nonlinear 坐标本身增加了有限模型的函数复杂度？

阶段 3A 不再使用 VIV、等变误差或 kNN proxy 作成功指标。四个分支训练后全部回到
同一个原始 2D 分布，以 endpoint sliced Wasserstein-1 裁决。

## 固定问题与变换

数据为 8 个等权二维 Gaussian 模态，中心均匀分布在半径 2 的圆上，每个模态标准差
为 `0.15`。训练数据每步从解析分布在线采样，不存在有限 train/test 样本泄漏。

可逆坐标变换：

```text
f_a(x1, x2) = (x1, x2 + a * x1^2)
f_a^{-1}(y1, y2) = (y1, y2 - a * y1^2)
```

主强度固定为 `a=1.0`；`a=0.5` 只用于机制趋势。不得在看到结果后改用另一强度
宣称主实验通过。

## 四个严格配对分支

每个 seed 的四个模型共享完全相同的初始化、每步 data、epsilon、time、batch size、
优化器超参数和更新数：

| 分支 | target | source | path |
|---|---|---|---|
| Base | `z` | `eps` | `(1-t)z+t eps` |
| Gaussian straight | `f(z)` | `eps` | `(1-t)f(z)+t eps` |
| Matched chord | `f(z)` | `f(eps)` | `(1-t)f(z)+t f(eps)` |
| Pushforward | `f(z)` | `f(eps)` | `f((1-t)z+t eps)` |

Pushforward velocity 必须由 JVP 得到。采样统一从各自训练 source 开始，以 Heun 从
`t=1` 积分到 `t=0`；后三个分支最后用 `f^{-1}` 回到 `z` 坐标。

## 固定训练与评价

- seeds：`0,1,2,3,4`；
- 4 张 GPU 按 task 并行；
- MLP width `64`，depth `3`，无 dropout；
- batch `512`，updates `4000`，AdamW，初始 lr `1e-3`；
- endpoint/reference 各 `8192`，固定 256 个 SW1 directions；
- Heun `100` steps；另用相同 2,048 source 比较 100/200 steps；
- 辅助指标：coordinate W1、exact mixture NLL、mode coverage/TV、mean/covariance error、
  held-out microscopic velocity MSE；这些不能替换主 SW1。

## 验收顺序

先验收实验有效性，再看方法：

1. 数值：transform cycle max `<=1e-6`；JVP 与 quadratic central difference
   (`step=0.1`) relative max `<=1e-4`；`a=0` toy 单测中四路径完全相同；
2. Solver：100/200-step 同 source endpoint relative L2 每分支 `<0.02`；
3. Base：至少 4/5 seeds 覆盖全部 8 个 mode，且 SW1 `<0.20`；否则训练/容量不足，
   实验无效，不解释路径；
4. Gap：主强度 `a=1` 下，Gaussian SW1 至少在 4/5 seeds 达到 Base 的 `1.10x`；
   否则没有“坐标路径导致可恢复退化”的稳定现象，停止 `H4/H5`；
5. Recovery：对有 gap 的 seed 定义

```text
Recovery = (SW1_gaussian - SW1_pushforward)
           / (SW1_gaussian - SW1_base)
```

Pushforward 必须在至少 4/5 seeds 达到 `Recovery>=0.50`，且均值不劣于 Base 的
`1.10x`，才授权进入小图像 latent。

## 结果解释边界

- Gaussian 差、Matched 好：source prior 是主要因素；
- Matched 差、Pushforward 好：chord mismatch 是主要因素；
- Pushforward 仍差：正确 probability path 不能抵消 nonlinear 坐标增加的有限模型
  复杂度，停止“只改 path 即恢复”的主张；
- 四者都接近：该 toy 缺少 gap，不能算正结果，也不进入大实验；
- 只有 2D 通过后，才做 MNIST/FashionMNIST 或 CIFAR 小 latent；在此之前禁止 RAE
  stage-2 训练。
