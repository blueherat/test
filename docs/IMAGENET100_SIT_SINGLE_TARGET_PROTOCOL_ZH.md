# ImageNet-100 SiT 单预测目标对照协议

## 目的

这组实验用于解释 `x + epsilon` 动态双头 SiT 的失败究竟来自哪里。它补充两个独立单头：

- `x`：网络直接预测 SD-VAE clean latent；
- `epsilon`：网络直接预测 source Gaussian noise。

二者与既有 `velocity` 单头以及动态双头使用相同的 SiT-S/2、ImageNet-100 latent cache 和线性路径：

```text
x_t = (1 - t) * epsilon + t * x
v   = x - epsilon
```

## 正式训练目标

正式长训使用各自的原生逐元素 MSE：

```text
L_v       = MSE(v_hat, x - epsilon)
L_x       = MSE(x_hat, x)
L_epsilon = MSE(epsilon_hat, epsilon)
```

这与失败的动态双头中两个 endpoint head 的监督一致，因此可以隔离 shared-trunk 多任务干扰。它不是 JiT 的 `x-prediction + v-loss` 参数化实验；后者属于另一项消融，不与本实验混称。

采样时统一换算成线性路径速度：

```text
v_x       = (x_hat - x_t) / max(1 - t, 1e-3)
v_epsilon = (x_t - epsilon_hat) / max(t, 1e-3)
```

## 公平性约束

除了 `prediction_target` 与对应 loss 之外，三条单头路径保持：

- 同一个 SiT-S/2 架构和 32,617,760 参数；
- 同一 official SiT 源提交 `cbde832a40b153ccc79603412409da9c9b0c568c`；
- 同一个 ImageNet-100 latent cache manifest；
- seed 0、四卡 DDP、global batch 256；
- AdamW、学习率 `1e-4`、EMA `0.9999`；
- BF16、TF32、`torch.compile`；
- 完全相同的数据、posterior noise、source noise、time 和 class-dropout 随机流；
- 同一 EMA、无 guidance、5K samples、seed 0、Dopri5、SD-VAE 和 ADM-FID 流程。

200-step 实际 smoke 中，`velocity/x/epsilon` 三个 checkpoint 的四个 rank RNG state 逐项一致；data manifest 与 official SiT hash 也一致。

## 已完成检查

四卡 benchmark，local batch 64、global batch 256：

| target | step/s | images/s | 单卡峰值显存 |
|---|---:|---:|---:|
| velocity | 19.18 | 4911 | 3.61 GiB |
| x | 19.45 | 4979 | 3.61 GiB |
| epsilon | 19.03 | 4871 | 3.61 GiB |

原生 loss 的 200-step smoke 稳定且有限：

| target | 前 10-step loss | step 200 train loss | step 200 raw validation loss |
|---|---:|---:|---:|
| x | 0.6700 | 0.2904 | 0.2860 |
| epsilon | 0.9810 | 0.3739 | 0.3728 |

两种 checkpoint 均通过 64-sample 四卡端到端采样 smoke。

作为边界检查，`x/epsilon + common v-loss` 配合 `1e-3` 分母下限虽然数值有限，但早期 loss 会达到数百到数千，明显被 endpoint conditioning 主导。因此它不进入本轮正式长训。JiT 风格实验应单独使用论文的时间分布和 `t_eps=0.05` 复现，不能与原生单头对照混在一起。

## 正式流程

入口：

```bash
bash experiments/run_imagenet100_sit_single_targets_overnight.sh
```

顺序执行：

```text
x/native 100K -> FID-5K -> 200K -> FID-5K
epsilon/native 100K -> FID-5K -> 200K -> FID-5K
```

每 100K 保存完整 model、EMA、optimizer 与四 rank RNG state。采样过程限制每张 GPU 始终低于 9 GiB，已有 smoke 的训练峰值约 3.6 GiB。

## 解释边界

- 若独立 `x` 或 `epsilon` 已明显差于 velocity，说明 target/parameterization 本身是主要因素。
- 若独立单头接近或优于 velocity，但动态双头仍差，shared-trunk 多任务干扰或 learned mixing 才是首要嫌疑。
- 原生 loss 数值不能跨 target 直接比较；最终判断以相同协议下的 FID 曲线为主。
- 5K FID 是内部筛查指标，不等同于官方 ImageNet-1K FID-50K。
