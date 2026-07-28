# RAEv2 LPL 5K 小规模验证协议

## 目标

从 RAEv2 官方 DINOv3-L-K7 完整训练态 checkpoint 出发，比较：

1. 官方原始 EMA；
2. 只用官方 Flow loss 继续训练的 EMA；
3. 在相同 Flow loss 上加入 decoder-feature LPL 的 EMA。

先执行 Flow-only 2000 step 门控。只有训练连续性和采样链路都确认正常，才继续
Flow 5000 step 与 LPL 5000 step。

## 已核对的官方语义

- 官方 checkpoint 包含 `step/model/ema/optimizer/scheduler/epoch` 六项。
- checkpoint 位于 `step=100080, epoch=80`，可推出原 scheduler 的
  `1251 step/epoch`。
- Stage 2 使用 `transport.prediction=x`。模型主头直接输出干净 latent；
  不能套用旧 RAE 的 `z_hat=z_t-t*v`。
- `DiTwDDTHeadIG` 返回 `(full_output, base_output)`。官方 Flow loss 是主头
  loss 加 `base_model_coeff * base loss`。
- 优化器是 GMuon + AdamW 的复合优化器，续训必须恢复完整 optimizer state。
- 采样使用 EMA、100 个 Euler ODE step、IG scale `1.78`，生效区间
  `[0.10, 1.0]`。

## 公平性边界

- 官方 checkpoint 不含原始 per-rank RNG 与 dataloader 游标，因此不能声称
  恢复了第 80 epoch 结束后的原始下一批图像。
- Flow 与 LPL 从同一 checkpoint 开始，并在模型构造后用相同 per-rank seed
  重置 RNG。
- 两个分支使用同一 ImageNet train parquet、同一 DistributedSampler 序列。
- 裁剪固定为 ADM center crop；水平翻转由 `seed + source row index` 决定，
  因而独立启动的两个分支得到逐样本相同的输入。
- pilot 使用 4 卡、global batch `16`、每卡 micro batch `1`、累积 `4` 次。
  这不是官方 global batch `1024` 的等价延长，只是恢复优化器和最终学习率
  后的低成本连续性验证。
- 共享 GPU 上保留 GPU EMA 会使 GMuon 首次建状态时超出显存。pilot 将 fp32
  EMA 放在 CPU，仅由 rank 0 在每个 optimizer step 后按官方
  `ema = decay * ema + (1-decay) * model` 更新。训练模型、梯度和完整
  GMuon/AdamW state 仍在 GPU；采样仍读取 checkpoint 的 EMA。

## Flow 2000-step 门控

保存：

- branch update 1000；
- branch update 2000。

实际 checkpoint 全局 step 分别是 `101080` 和 `102080`。

训练侧必须满足：

- 模型、EMA、optimizer、scheduler 均严格载入；
- 首个学习率保持在官方最终值 `2e-5`，不得跳回 `2e-4`；
- loss、梯度和参数均为有限值；
- 每张卡在反向峰值后至少保留 `2.5 GiB`，避免占满共享 GPU；
- 1000/2000 checkpoint 能够重新加载并继续训练。

采样侧对原始 EMA 与 Flow-2000 EMA使用：

- 相同的 5000 个均衡 ImageNet 标签；
- 相同的每 rank 初始噪声与 RNG seed；
- 相同的 bf16、100-step ODE 和 IG 配置；
- 相同的 deterministic RAE decoder。

如果 Flow-2000 没有改善，先检查以下项目，不立即启动 LPL：

1. 采样是否使用 EMA，而不是 model；
2. scheduler 是否保持 `2e-5`；
3. checkpoint 中 GMuon/AdamW state 是否完整恢复；
4. 数据预处理与标签是否正确；
5. 原始和续训采样的噪声、标签、步数、IG 是否逐项相同；
6. reduced-batch continuation 是否因 batch 差异导致质量回退。

## LPL 定义

LPL 作用于主头直接预测的干净 latent。真实 latent 的 decoder features 在
`no_grad` 下计算；预测 latent 的 decoder features 保留到 latent 的梯度，但
decoder 参数冻结。

使用 decoder 深度约 `20/40/60/80/100%` 的五层特征，保留原 LPL 的：

- prediction-branch cross normalization；
- percentile/outlier morphology mask；
- 空间平方和、通道平均。

在正式 LPL 分支前先运行 calibration，以源 checkpoint 上的统计把加权 LPL
平均贡献设为 Flow loss 的约 `20%`，不直接复用旧 RAE 的权重。

## 结果位置

所有模型、checkpoint、样本和指标写入：

`/home/zhoushunyu/data/eqvae/experiments/raev2_lpl_pilot`

仓库内不写 `outputs/` 或模型产物，只保存代码、配置、测试和审计文档。
