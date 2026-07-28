# RAE Layerwise Representation Path 预注册协议

## 研究问题

在保持原始 RAE final latent、decoder 和 DiT 张量形状不变时，生成路径是否应在高噪声
阶段优先运输 final-layer semantic component，并在中低噪声阶段逐渐引入由 middle
layer 定位的 spatial-detail component？

## 同端点分解

使用 ImageNet-1k train split 拟合 middle spatial residual 到 final spatial residual 的
ridge regression。取可预测 final covariance 的 top-k channel eigenspace `Pi_d`：

```text
u_detail = Pi_d (z_final - token_mean(z_final))
u_sem    = z_final - u_detail
```

因此逐元素满足：

```text
u_sem + u_detail == z_final
```

全局 token mean 完整保留在 `u_sem` 中。该设计避免训练新 decoder，也排除了 endpoint
latent、rFID、通道数和 decoder closure 的混杂。

## 路径

仓库约定 `t=0` 为数据、`t=1` 为 Gaussian noise。令：

```text
s(t) = c_sem(t) u_sem + c_detail(t) u_detail
z_t  = (1-t) s(t) + t epsilon
v*   = epsilon - s(t) + (1-t) ds(t)/dt
```

比较三种共享端点的路径：

| condition | c_sem(t) | c_detail(t) |
|---|---:|---:|
| static | 1 | 1 |
| annealed | 1 | `(1-t)^p` |
| reverse | `(1-t)^p` | 1 |

另用同 rank Haar-random channel subspace 作为 control。由于随机 rank-16 子空间只捕获
约 `16/768` 的空间方差，而 middle-guided 子空间捕获更多高能量方向，随机 detail 使用
预注册比例

```text
scale_random = sqrt(explained_final_fraction / (rank / channels))
```

做等能量缩放，同时定义 `u_sem = z_final - u_detail`，因此最终端点仍严格等于原始
`z_final`。该 control 用来区分“延迟任意等能量方向”和“延迟 middle-predictable 方向”。

## 固定 latent stream

- 每个 seed 从 ImageNet-1k train 无放回选择 160k 张图，并固定水平翻转。
- 4 卡只编码一次，保存官方归一化后的 fp32 final RAE latent。
- parquet 物理读取按索引排序；`stream_order.npy` 恢复预注册随机逻辑顺序。
- static、annealed、reverse、random 四路共享同一缓存，因此 encoder 完全退出生成比较。
- 训练恢复按已消费样本偏移继续，不重复 checkpoint 前的数据。

## Gate 1：职责分离

- 训练与验证 split 严格分离；projector 只使用 train。
- `u_sem` 的 ImageNet label kNN accuracy 至少为 full latent 的 `95%`。
- `u_detail` 在 `flip_h` 和 `rot90` 的平均 correspondence 至少一项严格优于
  `u_sem`：direct error 相对下降 `>=10%` 或 diagonal cosine 提高 `>=0.05`。
- `D(u_sem + u_detail)` 相比 `D(u_sem)` 的 LPIPS 或 rFID 必须改善。
- random-subspace control 不能复制全部优势。

若没有任何 rank 同时满足以上条件，第一方向被否定，停止 layerwise path 训练。

## Gate 2：tiny ImageNet 生成

- static、annealed、reverse、energy-matched random 使用相同 final endpoint、decoder、DiT、
  初始化、训练 stream、优化器、fp32 数值和采样噪声。
- 3 seeds，每个 10k optimizer steps，5k paired KID/FID。
- annealed 相对 static 平均改善至少 `5%`，且 `3/3` seeds 同方向。
- reverse 必须差于 annealed，否则“semantic-first”机制不成立。
- 等能量 random control 不能达到 middle-guided subspace 的同等收益。

Gate 2 失败即停止第一方向；不以 teacher MSE、rFID 或单 seed 结果替代生成验收。

## Gate 3：扩大实验

- 5 seeds 至少 `4/5` 改善。
- 达到相同 FID 的 epoch 减少至少 `20%`，或标准 50k FID/KID 明确改善。
- 与 original final RAE、RAEv2 static last-k 和静态 multi-layer fusion 对照。
- 50k paired bootstrap 置信区间不能跨零。
