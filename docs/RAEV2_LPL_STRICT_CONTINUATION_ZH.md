# RAEv2 LPL 官方 Batch 严格续训协议

## 目标

从 RAEv2 官方 DINOv3-L-K7 完整训练态 checkpoint 出发，比较：

1. 官方原始 EMA；
2. 只用官方 Flow loss 继续训练的 EMA；
3. 在相同 Flow loss 上加入 decoder-feature LPL 的 EMA。

先执行 Flow-only 10 step 门控。只有训练连续性和采样链路都确认正常，才继续
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
- 两个分支使用同一 ImageNet train parquet、同一词典序索引映射和同一
  DistributedSampler 序列。
- 官方 RAEv2 Arrow 数据在训练时只做 `Resize(256)+ToTensor`，不做在线水平
  翻转。本地 raw parquet 因此使用 ADM center crop 并关闭翻转。
- 下载官方第一个 500 MB Arrow shard 进行像素审计：跨类别边界抽查的官方图像
  都能在本地 raw ImageNet 中找到逐像素完全相同的 center-crop 结果。官方
  Arrow 与本地文件名词典序存在少量位置偏移，因此不能声称恢复原始样本游标；
  该映射只固定 Flow/LPL 的新配对序列。
- 本地索引映射覆盖全部 `1,281,167` 张训练图且为无重复全排列；映射文件哈希
  写入每个实验 manifest。
- pilot 使用 4 卡、官方 global batch `1024`、每卡 micro batch `1`、累积
  `256` 次。它保留官方每个 optimizer step 的样本数；与官方 8 卡运行相比，
  DDP world size 和浮点累加顺序仍不同，而且原始 dataloader 游标不可恢复。
- `10` 个严格 step 共处理 `10,240` 张图。这个 pilot 的目标是先确认官方续训
  链路、Flow 趋势和 LPL 接入是否正常，不把它解释为充分收敛实验。若门控通过，
  再按相同 batch 逐段增加训练预算。
- 共享 GPU 上保留 GPU EMA 会使 GMuon 首次建状态时超出显存。pilot 将 fp32
  EMA 放在 CPU，仅由 rank 0 在每个 optimizer step 后按官方
  `ema = decay * ema + (1-decay) * model` 更新。训练模型、梯度和完整
  GMuon/AdamW state 仍在 GPU；采样仍读取 checkpoint 的 EMA。
- DDP 使用 `gradient_as_bucket_view=True`，让梯度直接复用通信 bucket，
  避免为同一梯度保留两份 GPU storage。它只改变显存布局，不改变 global
  batch、loss、梯度值、optimizer 或 scheduler。

## Flow 10-step 门控

保存：

- branch update 2、4、6、8、10。

实际 checkpoint 全局 step 分别是 `100082`、`100084`、`100086`、
`100088` 和 `100090`。

训练侧必须满足：

- 模型、EMA、optimizer、scheduler 均严格载入；
- 首个学习率保持在官方最终值 `2e-5`，不得跳回 `2e-4`；
- GMuon 和 AdamW 两组 LR 在加载、实际内部 group、重新序列化三个位置都必须
  为 `2e-5`；
- loss、梯度和参数均为有限值；
- 每张卡在反向峰值后至少保留 `2.5 GiB`，避免占满共享 GPU；
- 2/4/6/8/10 checkpoint 能够重新加载并继续训练。

采样侧对原始 EMA 与 Flow-10 EMA 使用：

- 相同的 5000 个均衡 ImageNet 标签；
- 相同的每 rank 初始噪声与 RNG seed；
- 相同的 bf16、100-step ODE 和 IG 配置；
- 相同的 deterministic RAE decoder。

如果 Flow-10 没有改善，先检查以下项目，不立即启动 LPL：

1. 采样是否使用 EMA，而不是 model；
2. scheduler 是否保持 `2e-5`；
3. checkpoint 中 GMuon/AdamW state 是否完整恢复；
4. 数据预处理与标签是否正确；
5. 原始和续训采样的噪声、标签、步数、IG 是否逐项相同；
6. 4 卡梯度累积与官方 8 卡 DDP 的浮点累加差异是否造成可见偏移。

## Flow 150-step loss 审计

第一轮 Flow 续训已经从 branch step 0 运行到 150。模型、EMA、GMuon、
AdamW 和 scheduler 在 0、10、30 三个启动/恢复边界都完整载入；两组内部
optimizer state 数量保持为 `222/346`，学习率始终为 `2e-5`，梯度范数没有
出现跳变或爆炸。

历史 `train_metrics.jsonl` 有一个只影响观察、不影响训练的统计问题：每个
optimizer step 实际累积 1024 张图，但日志曾只记录四个 rank 的最后一个
microbatch，等效为 4 张图的 noisy estimate。梯度仍由全部 1024 张图正确
累积，checkpoint 不受影响。入口现已改为在每个 optimizer step 内累计
Flow/LPL/total loss，再按 `world_size * grad_accum_steps` 求真实全局平均；
新 manifest 使用 `logged_loss_scope=global_accumulation_mean` 标记口径。

旧日志仅作健康检查时，前 30 step 的均值为 `0.9095`，后 30 step 为
`0.9190`；差异不显著（Welch `p=0.755`），150 step 线性趋势也不显著
（`p=0.489`）。更严格的固定输入审计使用相同 64 张图、latent、噪声、时间和
CFG mask，得到：

| checkpoint | state | Flow loss | 相对 official |
|---|---|---:|---:|
| official | model | 0.951255 | 0 |
| Flow-10 | model | 0.951625 | +0.039% |
| Flow-110 | model | 0.952423 | +0.123% |
| Flow-150 | model | 0.951215 | -0.004% |
| official | EMA | 0.950239 | 0 |
| Flow-110 | EMA | 0.950006 | -0.024% |
| Flow-150 | EMA | 0.950096 | -0.015% |

因此没有“恢复后初始 loss 很大”或“续训 150 step 后 loss 漂移”的证据。
相反，Flow-only 分支处在训练平台区。EMA decay 为 `0.9995`，150 step 后新
更新在 EMA 中的累计权重仅为 `1-0.9995^150=7.23%`，所以短续训样本接近官方
EMA 是预期现象，不能把它解释为 Flow 已明显改善。

## LPL 定义

LPL 作用于主头直接预测的干净 latent。真实 latent 的 decoder features 在
`no_grad` 下计算；预测 latent 的 decoder features 保留到 latent 的梯度，但
decoder 参数冻结。

使用 decoder 深度约 `20/40/60/80/100%` 的五层特征，保留原 LPL 的：

- prediction-branch cross normalization；
- percentile/outlier morphology mask；
- 空间平方和、通道平均。

在正式 LPL 分支前先运行 calibration，以源 checkpoint 上的统计把加权 LPL
全局平均贡献设为 Flow loss 的约 `20%`，不直接复用旧 RAE 的权重。未通过
高 SNR gate 的 microbatch 按 LPL 贡献为零计入分母；同时单独记录 gate rate
和 gate 内的 conditional LPL 均值。这个口径与旧严格 LPL 实验一致。

## Step 100 非 EMA 退化与机制审计

长任务已安全暂停在完整的 branch step 150 checkpoint；以下比较全部使用
branch step 100 的在线 `model` 权重，而不是变化很慢的 EMA。官方、Flow100
和 LPL100 使用相同的 5000 个标签、四卡分片、初始噪声、100-step Euler 和
确定性 decoder。四个 rank 的首噪声与标签哈希均逐项一致。

正常 `IG=1.78` 的结果为：

| branch | FID | KID | IS |
|---|---:|---:|---:|
| official model | 10.9783 | 0.000208 | 146.68 |
| Flow100 model | 11.0645 | 0.000177 | 150.02 |
| LPL100 model | 14.1670 | 0.002290 | 129.52 |

Flow100 只比官方差 `0.0862` FID，而 LPL100 比 Flow100 差 `3.1025`。因此
退化来自 LPL 分支，不是普通 continuation 或读取非 EMA 权重本身。

固定四张训练图、固定 latent、噪声、时刻和标签后，LPL100 相对 Flow100：

- full 头的 Flow loss 增加 `1.7%--3.4%`；
- full 头的 strict decoder-feature LPL 降低 `11.6%--18.9%`；
- 经过 `base + 1.78(full-base)` 后，guided Flow loss 增加
  `6.8%--10.5%`；
- `full-base` RMS gap 增加约 `3%--7%`，base 头本身基本不变。

这说明 LPL 确实降低了真实插值路径上终点预测的 strict feature loss，但只
直接约束 full 头。RAEv2 采样再对 `full-base` 差值做 `1.78` 倍外推，因此把
full 头的小幅 Flow 偏移放大成更明显的采样向量场偏移。

为验证该解释，保持 checkpoint、噪声、标签和 sampler 其余部分不变，只把
internal guidance 改为 `IG=1.0`，即直接使用 full 头：

| branch | FID | KID | IS |
|---|---:|---:|---:|
| official model | 11.5240 | 0.000563 | 133.00 |
| Flow100 model | 11.6942 | 0.000469 | 135.55 |
| LPL100 model | 12.4807 | 0.001087 | 130.65 |

此时 LPL100 相对 Flow100 的 FID 差从 `3.1025` 缩小到 `0.7865`，消除了
约 `74.6%` 的退化。`IG=1.0` 让 official/Flow 变差，却让 LPL100 从
`14.1670` 改善到 `12.4807`，因此 dual-head internal guidance 是已被因果
消融确认的主要放大器，但不是全部原因。

剩余退化至少有两个已观测风险：

- 当前 LPL 权重按标量 loss 比例校准。首个严格配对 update 的参数梯度范数从
  Flow 的 `0.431` 增至 `2.955`，放大 `6.85` 倍并触发 clip；标量贡献为
  `20%` 并不等于梯度贡献为 `20%`。
- 中高噪声下 strict normalized feature loss 明显下降，但 raw feature loss
  略有上升；prediction-variance 梯度与 Flow 梯度转为负相关。这表明
  cross-normalization 的分母可以通过改变预测特征方差降低，而不一定让原始
  decoder features 更接近目标。

因此当前 `0.0007801168` 权重的 LPL 长任务不应继续。下一轮应先做低权重、
gradient-calibrated 的短分支，并分别测试 full-only、full+base 或直接约束
实际 guided 输出；不能把旧 RAE 的单头结论原样移植到 RAEv2 双头模型。

## Guidance-aware 10-step 对照

后续实验全部从同一官方 step `100080` checkpoint 出发，使用相同的四卡
ImageNet 数据序列、增广、latent、噪声、时间和 CFG mask。所有首批摘要逐
rank 完全一致。采样读取在线 `model`，不是 EMA，以免 10 step 的差异被
`0.9995` EMA 稀释。

标量 loss 比例不能代表参数梯度比例，因此先用每卡 64 个 microbatch、全局
256 个严格配对样本分别反传 Flow-only 和 LPL-only。每个 LPL 权重都校准为
参数梯度范数约等于 Flow 的 `20%`：

| LPL target | probe grad (`weight=1e-4`) | calibrated weight |
|---|---:|---:|
| full | 0.390374 | 3.47865e-5 |
| full+base | 0.462146 | 2.93840e-5 |
| guided | 0.515044 | 2.63661e-5 |
| guided multiscale | 0.437674 | 3.10270e-5 |
| guided common-gradient | 0.226412 | 5.99778e-5 |

Flow probe 的参数梯度范数为 `0.678986`。校准后的五个 10-step 分支首步总
梯度都约为 `0.45`，后续约为 `0.37--0.40`，没有触发裁剪。相比之下，旧
标量校准分支首步为 `2.955`。因此本轮已经排除“LPL 只是权重过大”这一混杂。

目标定义如下：

- `full`：只把 full 预测送入冻结 decoder。
- `full+base`：按数据 index 奇偶确定性交替选择 full/base，是二者等权目标的
  无偏单 decoder-graph 估计。
- `guided`：直接约束官方
  `base + 1.78 * (full - base)`。
- `guided multiscale`：按数据 index 确定性使用 `1.0/1.39/1.78`。
- `guided common-gradient`：前向数值仍为官方 guided 输出，但用
  straight-through 令 LPL 对 full/base 的输出梯度都为 `0.5`，避免普通
  guided 目标固有的 `+1.78/-0.78` 反向拉扯。

### 1k 非 EMA 筛查

官方 `IG=1.78`、100-step ODE、同噪声 1000 张结果：

| branch | FID | KID | IS |
|---|---:|---:|---:|
| official model | 41.3955 | 0.000077 | 56.84 |
| Flow10 model | **40.7326** | **0.000039** | **57.87** |
| full10 | 41.6531 | 0.000466 | 56.25 |
| full+base10 | 40.9203 | 0.000081 | 57.32 |
| guided10 | 41.3377 | 0.000395 | 55.97 |
| guided multiscale10 | 41.4927 | 0.000451 | 55.76 |
| guided common-gradient10 | 40.8615 | 0.000148 | 57.30 |

把同一批 checkpoint 改为 `IG=1.0`，同时强制保留相同的 dual-output forward
路径，得到：

| branch | FID | KID |
|---|---:|---:|
| official model | 42.4837 | 0.000438 |
| Flow10 model | **41.9278** | **0.000220** |
| full10 | 42.0946 | 0.000369 |
| full+base10 | 41.8199 | 0.000260 |
| guided10 | 42.0240 | 0.000323 |
| guided multiscale10 | 42.1034 | 0.000341 |

full-only 相对 Flow 的 FID 差从 `IG=1.0` 的 `+0.1668` 放大到 `IG=1.78`
的 `+0.9204`；guided 从 `+0.0962` 放大到 `+0.6051`，multiscale 从
`+0.1756` 放大到 `+0.7601`。这再次确认 internal guidance 是退化放大器。

同噪声逐图分析还显示，各 LPL 分支相对 Flow10 的新增像素更新与纯 Flow 更新
近似正交，平均 cosine 约为 `0.049--0.054`。其扰动幅度相对
`official -> Flow10` 更新分别为：

| target | LPL/Flow pixel-update ratio |
|---|---:|
| full | 0.891 |
| full+base | 0.426 |
| guided | 0.706 |
| guided multiscale | 0.812 |
| guided common-gradient | 0.540 |

full+base 在 1k 下较稳定，主要因为它引入的额外扰动更小，不能据此认定 LPL
产生了生成收益。

### 5k 非 EMA 终验

复用协议、噪声、标签、采样器完全相同的官方 5k 非 EMA 样本，只补采 Flow10
及两个最稳定候选：

| branch | FID | KID | IS |
|---|---:|---:|---:|
| official model | 10.9783 | 0.000208 | 146.68 |
| Flow10 model | **10.8278** | **0.000167** | **153.03** |
| full+base10 | 10.9289 | 0.000213 | 152.77 |
| guided common-gradient10 | 11.0360 | 0.000280 | 152.41 |

full+base10 比官方好，但比相同计算量的 Flow10 差 `0.1010` FID；common-gradient
比 Flow10 差 `0.2081`，甚至略差于官方。两者 KID 方向一致。因此五种
gradient-calibrated 目标中，没有一种提供超过 Flow continuation 的可靠收益。

### 终点预测改善与生成指标断层

固定四张真实 ImageNet latent、固定噪声和时刻的审计确认，LPL 实现确实在优化
声明的目标。以 noise/signal=`1.0` 为例：

- full10 把 full strict decoder-feature loss 从 `728.0` 降到 `684.6`。
- guided10 把 guided strict loss 从 `759.3` 降到 `706.2`。
- full+base10 同时把 base/full strict loss 从 `951.0/728.0` 降到
  `892.2/699.2`。
- common-gradient10 同时把 base/full/guided strict loss降到
  `903.4/692.0/724.7`。

common-gradient 将 `full-base` RMS gap 相对 Flow 的增加控制在约 `0.33%`，
小于普通 guided 的约 `0.47%`，但 5k 生成仍退化。共享主干意味着即使输出
梯度都设为 `0.5`，两个头的函数响应也不会完全相同。

这里的 LPL 不是采样器“单步”损失，也不应称为 teacher-forced loss。每个训练
时刻的网络输出都表示最终干净 latent 的估计，LPL 比较的是该终点估计与真实
干净 latent 经冻结 decoder 后的特征。“真实插值路径”只描述网络输入由真实
latent 与噪声插值得到；“模型轨迹”描述输入由 ODE 的历史预测产生。

新增的递归终点审计从同一个真实插值状态出发，继续进行
`1/2/4/8/16` 次模型自身的 Euler 查询。结果否定了“LPL 只在真实插值路径上
有效、进入模型轨迹就停止改善自己目标”的解释：

- full+base10 的 strict feature error 相对 Flow 平均降低 `4.02%`；
- common-gradient10 平均降低 `4.48%`；
- 优势随递归查询次数增加，没有消失；
- 但 raw feature error 分别恶化 `12.57%/15.45%`，latent error 也轻微恶化
  `0.124%/0.198%`。

所以真正的断层不是“真实路径 vs 自生成路径”，而是：

> strict LPL 比值在两类状态上都能继续下降，但这种下降不等价于更准确的
> clean latent，也不等价于更好的生成分布。

未见 ImageNet validation 图像也得到同样结果。full+base10 的 strict feature
error 降低 `4.28%`，但使用固定 clean feature 方差归一化的误差恶化
`2.34%`，预测 feature 方差增加 `5.77%`。这排除了训练样本记忆或数据泄露。

### 归一化与 internal-guidance 排查

为了分开“减小分子”和“增大预测方差分母”，额外训练了 target-normalized 与
symmetric 两个 10-step 分支，参数梯度同样校准到 Flow 的 `20%`。

- target-normalized 确实降低固定 clean-stat feature error，但 1k FID 为
  `41.8816`，差于 Flow10 的 `40.7326`。
- symmetric 同时改善 strict、target-normalized、raw、feature variance、
  centered cosine 和 mean error，但 1k FID 仍为 `41.5457`。

因此“原 LPL 只是在放大分母”不是充分解释；即使所有配对 endpoint feature
代理都改善，也不能保证 RAEv2 的生成分布改善。

### Prediction-denominator detach

为直接隔离 prediction-variance 分母梯度，新增：

```text
prediction_detach = residual_squared / stop_gradient(prediction_variance)
```

它与原 strict LPL 的前向值完全相同，只切断分母反向传播。训练从相同官方
step `100080` 出发，恢复相同 model、EMA、GMuon、scheduler、学习率和
四卡全局 batch `1024`，并保持原 full+base LPL 权重不变。四个 rank 的首批
图像、标签、latent、噪声、时间和 CFG mask 哈希与原 full+base 分支完全一致。

未见 validation16 上，detach10 相对 Flow10：

- prediction/target variance 降低 `2.85%`，确认没有继续放大分母；
- target-normalized error 降低 `2.18%`；
- centered feature cosine 提高 `0.34%`；
- raw feature error 增加 `12.15%`；
- normalized feature mean error 增加 `11.72%`；
- latent error 只增加 `0.016%`。

这说明 detach 消除了分母捷径并改善部分相对对齐，但把绝对误差重新分配到
高方差通道和 feature mean，而不是全面降低 decoder-feature 偏差。

严格同噪声 5k non-EMA 结果：

| branch | FID | KID | IS |
|---|---:|---:|---:|
| Flow10 | **10.827822** | 0.00016715 | **153.03396** |
| full+base LPL10 | 10.928858 | 0.00021306 | 152.76570 |
| prediction-detach10 | 10.851805 | **0.00016054** | 152.12838 |

detach 消除了原 full+base LPL 相对 Flow 的约 `76%` FID 退化，但仍比 Flow
高 `0.02398` FID。KID 的微小优势小于其估计波动，不能视为可靠胜出。因此
直接分母梯度是原退化的重要来源，但不是缺少净生成收益的完整原因。

### 无方差归一化的 raw feature matching

为检验“只匹配 decoder feature 差值”是否更合适，进一步训练了 raw 分支：

```text
raw = sum(mask * (feature_prediction - feature_target)^2)
```

它保留与 strict LPL 完全相同的五层 decoder features、异常区域 mask、
noise gate、`full+base` 目标和数据流，唯一删除 prediction/target feature
variance 分母。因此这是“去掉方差归一化”的单变量控制，不是另换一套
perceptual loss。

早期旧 RAE raw 实验按标量 loss 匹配权重，梯度过大且约 `77.4%` update 被
clip，不能用于判断。本轮在相同的 256 个训练样本上重新做参数梯度校准：

- Flow 参数梯度范数：`0.678986`；
- raw 在 probe weight `1e-6` 下的参数梯度范数：`1.668345`；
- 令 raw 辅助梯度为 Flow 的 `20%`，得到正式权重
  `8.1396425e-8`。

10 个 update 的总梯度范数稳定在 `0.37--0.47`，无异常 clipping 或
NaN。四个 rank 的首批图像、标签、latent、噪声、时刻和 CFG mask 与
Flow/full/detach 分支逐项同哈希。

未见 validation16 上，raw10 相对 Flow10 在全部
`base/full/guided x 0.5/1.0/3.0` 条件平均：

- raw feature error `-8.647%`，说明分支确实优化了自己的目标；
- strict feature error `-1.770%`，normalized mean error `-11.068%`；
- target-normalized error `+0.825%`，prediction/target variance
  `+2.147%`；
- latent error 仅 `+0.009%`。

只看采样实际使用的 guided 输出，raw feature error 改善缩小为
`-0.996%`，target-normalized error 为 `+0.516%`，variance ratio 为
`+0.823%`。这说明无归一化目标更重视绝对尺度大的 feature/channel；它能
降低总平方差，却不保证低方差方向的相对误差也下降。

严格同噪声、non-EMA、官方 `IG=1.78`、100-step ODE 的最终结果：

| branch | 1k FID | 5k FID | 5k KID | 5k IS |
|---|---:|---:|---:|---:|
| Flow10 | **40.732635** | **10.827822** | **0.00016715** | 153.03396 |
| strict full+base10 | 40.920267 | 10.928858 | 0.00021306 | 152.76570 |
| prediction-detach10 | 40.878017 | 10.851805 | 0.00016054 | 152.12838 |
| raw full+base10 | 40.933338 | 10.841450 | 0.00017576 | **153.84479** |

raw 将 strict 相对 Flow 的 5k FID 退化从 `+0.10104` 缩小到
`+0.01363`，消除了约 `86.5%` 的差距，并略好于 detach。但它仍没有超过
Flow；KID 也不支持净改善。IS 虽然最高，但单次 5k 的小幅分歧不足以推翻
FID/KID 的共同判断。

因此本控制给出的结论是：

> 方差归一化及其梯度是 strict LPL 额外退化的主要来源；完全删除分母比只
> detach 分母略好，但“真实配对上的 raw decoder-feature MSE 更低”仍不能
> 转化为成熟 RAEv2 prior 的可靠生成收益。

full+base10 与 Flow10 的严格同噪声 IG 扫描如下：

| IG scale | Flow FID | full+base LPL FID | LPL - Flow |
|---:|---:|---:|---:|
| 1.00 | 41.9278 | 41.8199 | -0.1079 |
| 1.20 | 41.4765 | 41.5316 | +0.0552 |
| 1.40 | 41.0495 | 41.1263 | +0.0768 |
| 1.60 | 41.0421 | 40.9887 | -0.0534 |
| 1.78 | **40.7326** | 40.9203 | +0.1876 |

LPL 会轻微改变最佳 IG scale，但重新调 IG 后仍不能超过 Flow 在官方 `1.78`
下的最佳值。需要特别区分：

- CFG scale 为 `1.0`，本实验中 CFG 不活跃；
- `base` 是 REPA 中间头的较弱 clean-endpoint 估计，不是 CFG 的 unconditional
  分支；
- `base + 1.78(full-base)` 是 internal guidance，确实会放大偏差，但不是
  LPL 失败的根因。

### Flow-compatible 梯度因果干预

输出 latent 梯度审计显示，原 full+base LPL 梯度只有约 `4.65%` 落在 Flow
梯度的正向平行分量上，其余约 `95.35%` 是相对 clean-latent MSE 的切向分量。
据此实现 `flow_parallel`：

1. 计算 LPL 对预测 clean latent 的梯度；
2. 只保留其在 Flow 梯度方向上的正投影；
3. 用梯度替换保持原 LPL 前向值不变；
4. 再将参数梯度校准到 Flow 的 `20%`。

该投影的一阶性质是：附加更新不能在预测 latent 处增大 Flow 目标。由于完整
GMuon state 与 ViT-XL decoder 在当前共享 GPU 上不能同时保留计算图，训练使用
精确 decoder-gradient bridge：先无梯度得到同一个 clean-latent 预测并计算
decoder 输出梯度，再恢复 RNG、重算 Stage-2 前向并接回该梯度。它不是近似：

- toy 网络中 bridge 参数梯度与直接 decoder 反传一致；
- 256 个真实样本中，bridge 前后参数梯度范数为
  `0.02166023/0.02165996`；
- Flow/LPL loss、gate 率、数据哈希全部相同；
- 两次 Stage-2 输出要求逐元素完全一致，否则直接报错。

10-step 训练恢复官方 model、EMA、GMuon、scheduler、全局 batch `1024` 和
学习率 `2e-5`。首批四个 rank 的图像、标签、latent、噪声、时间和 CFG mask
哈希与原 full+base 分支逐项相同。

最终严格 5k 结果：

| branch | FID | KID | IS |
|---|---:|---:|---:|
| Flow10 | **10.827822** | **0.00016715** | **153.03396** |
| full+base LPL10 | 10.928858 | 0.00021306 | 152.76570 |
| flow-parallel LPL10 | 10.828043 | 0.00017019 | 153.00689 |

flow-parallel 与 Flow 几乎逐指标重合。它消除了原 LPL 的退化，但没有提供
超过 Flow 的生成收益。这是当前最强的机制证据：

> RAEv2 中原 LPL 的非冗余信息主要位于相对 Flow 的切向梯度，而这部分会破坏
> 成熟 prior 的生成分布；安全的平行分量只是重复 Flow，因而不能新增收益。

因此当前证据支持：

1. 旧 LPL 的大权重和 internal guidance 都会放大退化。
2. 修复梯度尺度、同时约束两个头、直接约束 guided 输出、覆盖多个 IG scale，
   或显式避免反向拉开两个头，都只能缓解，不能产生净收益。
3. strict LPL 在未见图像和模型递归状态上都能改善自己的目标，因此失败不是
   泄露、没有优化到目标、所谓“单步损失”，也不是简单的路径覆盖不足。
4. 只保留 Flow-compatible 分量会严格退化为普通 Flow 表现；继续扩大同类
   endpoint feature alignment 或 head-placement 实验没有价值。
5. 下一条有区分度的路线应保持成熟 prior 不动，让 decoder 适应 prior 的真实
   endpoint 误差，并用 clean reconstruction regularization 防止遗忘。只有该
   方向能检验“RAEv2 的问题是 decoder 对 prior 误差不鲁棒”，而不是再次用
   decoder 梯度扰动已经校准好的向量场。

## 结果位置

所有模型、checkpoint、样本和指标写入：

`/home/zhoushunyu/data/eqvae/experiments/raev2_lpl_pilot`

以及：

`/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_aware_10step`

以及：

`/home/zhoushunyu/data/eqvae/experiments/raev2_flow_parallel_10step`

以及：

`/home/zhoushunyu/data/eqvae/experiments/raev2_prediction_detach_10step`

以及：

`/home/zhoushunyu/data/eqvae/experiments/raev2_raw_10step`

仓库内不写 `outputs/` 或模型产物，只保存代码、配置、测试和审计文档。
