# RAEv2 LPL 官方 Batch 严格续训协议

## 目标

从 RAEv2 官方 DINOv3-L-K7 完整训练态 checkpoint 出发，比较：

1. 官方原始 EMA；
2. 只用官方 Flow loss 继续训练的 EMA；
3. 在相同 Flow loss 上加入 decoder-feature LPL 的 EMA。

先执行 Flow-only 50 step 门控。只有训练连续性和采样链路都确认正常，才继续
同预算的 LPL 分支。这里减少 continuation step，而不改变官方每个 optimizer
step 的有效 batch。

## 已核对的官方语义

- 官方 checkpoint 包含 `step/model/ema/optimizer/scheduler/epoch` 六项。
- checkpoint 位于 `step=100080, epoch=80`，可推出原 scheduler 的
  `1251 step/epoch`。
- Stage 2 使用 `transport.prediction=x`。模型主头直接输出干净 latent；
  不能套用旧 RAE 的 `z_hat=z_t-t*v`。
- `DiTwDDTHeadIG` 返回 `(full_output, base_output)`。官方 Flow loss 是主头
  loss 加 `base_model_coeff * base loss`。
- 优化器是 GMuon + AdamW 的复合优化器，续训必须恢复完整 optimizer state。
- 当前 GMuon 版本的 `param_groups` 是内部属性包装。PyTorch 通用
  `load_state_dict()` 会恢复动量 state，但不会自动更新 GMuon 真正执行更新的
  内部 group；若不修复，Muon 会从 checkpoint 的 `2e-5` 错误跳回构造值
  `2e-4`，并把错误 LR 写入下一 checkpoint。续训入口现在显式重建内部
  group，并同时核对公开 LR、内部 LR、重新序列化 LR 和 optimizer state
  数量。
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
- pilot 使用 4 卡、官方 global batch `1024`、每卡 micro batch `1`、累积
  `256` 次。它保留官方每个 optimizer step 的样本数；与官方 8 卡运行相比，
  DDP world size 和浮点累加顺序仍不同，而且原始 dataloader 游标不可恢复。
- `50` 个严格 step 共处理 `51,200` 张图，已经多于 reduced-batch
  `2000 x 16 = 32,000` 张图，因此不能只按 step 数判断实验规模。
- 共享 GPU 上保留 GPU EMA 会使 GMuon 首次建状态时超出显存。pilot 将 fp32
  EMA 放在 CPU，仅由 rank 0 在每个 optimizer step 后按官方
  `ema = decay * ema + (1-decay) * model` 更新。训练模型、梯度和完整
  GMuon/AdamW state 仍在 GPU；采样仍读取 checkpoint 的 EMA。

## Flow 50-step 门控

保存：

- branch update 10、20、30、40、50。

实际 checkpoint 全局 step 分别是 `100090`、`100100`、`100110`、
`100120` 和 `100130`。

训练侧必须满足：

- 模型、EMA、optimizer、scheduler 均严格载入；
- 首个学习率保持在官方最终值 `2e-5`，不得跳回 `2e-4`；
- GMuon 和 AdamW 两组 LR 在加载、实际内部 group、重新序列化三个位置都必须
  为 `2e-5`；
- loss、梯度和参数均为有限值；
- 每张卡在反向峰值后至少保留 `2.5 GiB`，避免占满共享 GPU；
- 10/20/30/40/50 checkpoint 能够重新加载并继续训练。

采样侧对原始 EMA 与 Flow-50 EMA 使用：

- 相同的 5000 个均衡 ImageNet 标签；
- 相同的每 rank 初始噪声与 RNG seed；
- 相同的 bf16、100-step ODE 和 IG 配置；
- 相同的 deterministic RAE decoder。

如果 Flow-50 没有改善，先检查以下项目，不立即启动 LPL：

1. 采样是否使用 EMA，而不是 model；
2. scheduler 是否保持 `2e-5`；
3. checkpoint 中 GMuon/AdamW state 是否完整恢复；
4. 数据预处理与标签是否正确；
5. 原始和续训采样的噪声、标签、步数、IG 是否逐项相同；
6. 4 卡梯度累积与官方 8 卡 DDP 的浮点累加差异是否造成可见偏移。

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
