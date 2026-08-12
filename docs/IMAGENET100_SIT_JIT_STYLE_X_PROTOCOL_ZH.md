# ImageNet-100 SiT：JiT-style x Prediction 对照

## 实验问题

本实验单独回答：在既有 SiT-S/2 线性 flow 中，保持训练协议不变，只把网络输出从原生 velocity 改成 clean latent `x`，再换算到共同 velocity-space loss，是否优于原生 velocity prediction？

路径仍为：

```text
x_t = (1 - t) * epsilon + t * x
v   = x - epsilon
```

网络输出 `x_hat`，训练时使用：

```text
d      = max(1 - t, 0.05)
v_pred = (x_hat - x_t) / d
v_tgt  = (x     - x_t) / d
L      = MSE(v_pred, v_tgt)
       = MSE(x_hat, x) / d^2
```

prediction 与 target 必须使用同一个 clamped denominator。否则在 `1-t < 0.05` 时，即使 `x_hat=x` 也会产生伪 velocity error。实现已经用端点解析测试覆盖这一点。

## “JiT-style”边界

这里复现的是 JiT 的核心 `x output + velocity-space loss + t_eps=0.05` 参数化，但为了和既有 SiT baseline 做单因素比较，仍保留 SiT 的 `Uniform[0,1)` 时间采样。

因此它不是完整 JiT recipe。完整 JiT 还使用 logit-normal 时间分布；若同时改变时间分布，就无法把结果只归因于 prediction target。后续若复现完整 JiT，应给原生 velocity 也配同一个 logit-normal 时间分布作为 matched control。

## 公平性

与既有 velocity baseline 相同：

- SiT-S/2，32,617,760 参数；
- ImageNet-100 SD-VAE latent cache；
- global batch 256、seed 0、400K optimizer steps；
- AdamW `lr=1e-4`、EMA `0.9999`；
- BF16、TF32、`torch.compile`；
- 相同 class dropout、数据路径、无 guidance FID-5K 协议。

由于资源限制，本轮只使用物理 GPU 2、3：world size 2、local batch 128。global batch 与四卡 baseline 相同，DistributedSampler 每一步覆盖同一全局 permutation 的 256 个位置；但 rank-local noise stream 不可能与四卡 checkpoint 逐字节相同。因此这是严格的协议匹配，而不是 bitwise paired training。

## 启动前验证

- 29 项预测目标、旧 velocity 回归、训练和采样测试通过；
- 两卡 synthetic benchmark：`16.16 step/s`，单卡峰值 `6.40 GiB`；
- 300-step 真实 ImageNet-100 smoke：稳定约 `15.2 step/s`；
- common velocity loss 从前 25 步的 `21.35` 降至 step 300 的 `1.44`；
- 无 NaN/Inf，单卡训练峰值约 `6.28 GiB`。

预计 400K 纯训练约 7 小时 19 分；包含每 100K 一次 EMA FID-5K，预计总计约 7 小时 40 分。

## 运行

```bash
CUDA_VISIBLE_DEVICES=2,3 \
bash experiments/run_imagenet100_sit_jit_x_2gpu.sh
```

流程：

```text
x/velocity 100K -> FID-5K
           200K -> FID-5K
           300K -> FID-5K
           400K -> FID-5K
```
