# 生成时间瓶颈：Latent Prior 预注册

## 1. 唯一问题

小型条件模型已经证明 `high_noise` latent 能在生成前半程确定 MNIST 类别，
随后即使不再读取 latent，类别信息仍能通过像素状态传到终点。但这还不构成
方法优势，因为 `all_time` 的表现几乎相同。

本阶段只问：

> 在相同 latent 维数、prior 网络、训练 batch、step 和采样 NFE 下，
> `high_noise` latent 是否比 `all_time` latent 更容易由无条件 prior 生成？

## 2. 固定设置

- 使用已冻结的 seed 0/1 `all_time`、`high_noise` encoder 和像素 flow。
- 两者 latent 均为 8 维；不再更新 encoder 或像素 flow。
- prior 为同一 3-block residual MLP flow，hidden 128。
- 每个 prior 训练 6,000 steps，batch 256，AdamW，fp32。
- latent flow 使用 `z_t=(1-t)z+t epsilon` 和 velocity target。
- prior 与像素 flow 均使用 50-step Euler；每个模型生成 2,000 张图。
- 所有数据仍只来自对应 seed 的 MNIST train 子集；test 只用于分布评估。

## 3. 指标

- latent：SWD、Gaussian Fréchet、nearest-train distance、线性 probe 预测的
  类别熵与置信度。
- image：固定分类器 feature FID、类别熵、分类置信度。
- oracle gap：prior 生成码的图像 FID 相对真实 test code 条件 rollout 的退化。
- 计算：prior 参数量、训练 step 和总 NFE 必须相同。

## 4. 进入联合端到端的门槛

两个 seed 必须同时满足：

1. 生成码与图像的类别熵均至少为 `log(8)`，避免只生成少数数字。
2. 两阶段图像 feature FID 比同 seed 无条件像素 flow 至少改善 20%。
3. `high_noise` 的 image feature FID 不得比 `all_time` 差 10% 以上。
4. 若要声称时间门控有方法价值，`high_noise` 必须在两个 seed 上都至少满足
   一项：image feature FID 改善 5%，或 latent SWD 改善 10%。

前 3 项只证明两阶段模型成立；第 4 项才证明显式时间分工优于普通 diffusion
autoencoder。若第 4 项失败，则不进入联合端到端训练，因为它很可能只是把
decoder 原本会自动学到的条件使用曲线显式编码了一遍。
