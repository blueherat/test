# RAE-adapted LPL 大规模真实性验证结果

## 总结

截至 2026-07-28，本轮 `DiTDH-S_ep20 + DINOv2-B RAE` 确认实验达到预注册的
Phase 1 强复现标准。

最可靠的结论是：

> 在冻结官方 DINOv2-B RAE encoder/decoder、只对官方 DiTDH-S ep20 prior
> 做 5,000 次严格配对 post-training 时，RAE-adapted LPL 相对等更新数的
> Flow continuation，在 3/3 个训练 seed 和 3/3 个采样 seed 上都稳定降低
> 本实验的 5k FID，并同时改善相对 Flow 的 KID 和 IS。

这不等于：

> LPL 已在所有指标上全面超过官方 checkpoint，或已经在官方 50k gFID
> 协议、DiTDH-XL、不同 tokenizer 上完成真实性确认。

相对零更新官方 checkpoint，LPL 的 5k FID 在 3/3 个同噪声采样 seed 上
更好，但 KID 和 IS 在 0/3 个 seed 上更好。这一指标权衡必须保留。

## 实验身份

当前方法称为 **RAE-adapted LPL**，不是官方代码复现：

- LPL 论文没有发布可直接核对的训练代码。
- clean estimate 使用 RAE linear Flow-OT 对应的
  `z_hat = z_t - t * v_theta(z_t, t)`。
- 仅在 `t / (1 - t) <= 3` 时加入 LPL。
- target 和 prediction 都使用 prediction feature 的空间均值、方差做
  cross-normalization。
- 主实验的 prediction variance 不 detach；detach 仅是机制消融。
- RAE ViT decoder 没有卷积 decoder 的多尺度 stage。本实现取 decoder
  相对深度 `0.2/0.4/0.6/0.8/1.0` 的五层；五层均为 `16 x 16`，因此论文
  的逆尺度权重在这里退化为共同常数。
- outlier quantile/morphology 按论文补充材料实现，但实测 mask keep
  fraction 约 `99.99997%`，在 RAE ViT decoder 上几乎是恒等操作。

固定 LPL 权重为：

```text
0.000732496037420993
```

它只用 ImageNet-1K train 的 1,024 个样本，在任何确认训练和生成评估之前
按 `E[weighted LPL] / E[Flow] = 0.25` 固定。校准使用 seed 4101，因此
seed 4101 不是完全独立于权重选择；seed 4102/4103 才是未用于调权重的确认
seed。

## 训练配置

| 项目 | 设置 |
|---|---|
| 官方起点 | `DiTDH-S_ep20/stage2_model.pt` |
| RAE | DINOv2-B with registers + frozen ViT-XL decoder |
| 数据 | ImageNet-1K train，1,281,167 张，294 个 train parquet shard |
| 训练 seed | `4101 / 4102 / 4103` |
| 每分支更新 | 5,000 optimizer steps |
| 每分支见样本数 | 80,000，约 `0.0624` epoch |
| 并行 | 4 GPU，micro batch 1/rank，accumulation 4 |
| global batch | 16 |
| optimizer | fresh shared AdamW，`lr=2e-5`，betas `(0.9, 0.95)` |
| EMA | `0.9995` |
| 数值 | fp32，TF32 off，确定性算法 |
| 采样 | 50-step Euler，CFG 1.0，每类严格 5 张 |
| 指标 | 5,000 samples 对 10,000-sample virtual ImageNet reference |

官方 checkpoint 只有模型权重，没有 optimizer/scheduler 状态。因此这里是
从相同官方模型权重出发、使用相同 fresh optimizer/scheduler 的配对
post-training，不是官方训练轨迹的无缝续训。官方训练 global batch 为
1,024，本实验为 16。

## 三训练种子结果

固定 sampling seed `20260715`。FID/KID 越低越好，IS 越高越好。

| train seed | Flow FID | LPL FID | FID 改善 | Flow KID | LPL KID | Flow IS | LPL IS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4101 | 19.9848 | 17.9296 | 2.0551 | 0.005659 | 0.004825 | 74.7719 | 76.3161 |
| 4102 | 20.0493 | 18.0659 | 1.9834 | 0.006032 | 0.005201 | 70.4101 | 73.6076 |
| 4103 | 21.0154 | 18.6538 | 2.3616 | 0.006739 | 0.005682 | 70.3543 | 70.7757 |
| mean | 20.3498 | 18.2165 | 2.1334 | 0.006143 | 0.005236 | 71.8455 | 73.5665 |

描述性统计：

- 平均 FID 相对 Flow 降低 `2.1334`，即 `10.48%`。
- 平均 KID 相对 Flow 降低 `0.0009073`，即 `14.77%`。
- 平均 IS 相对 Flow 增加 `1.7210`。
- 三个 FID 改善的标准差为 `0.2009`。
- LPL 分支的平均 Flow training loss 略高于 Flow 分支，而不是更低：
  `0.42831` 对 `0.42541`。因此生成改善不能解释为 LPL 分支碰巧把 Flow
  objective 优化得更低。
- LPL 加权项约占总训练 loss 的 `17.7%`，接近预先设定的五分之一。

训练 seed 数只有 3，不能把描述性区间当作大样本统计显著性证明；但三个
方向一致，且改善幅度相对 seed 间波动较大。

## 追加的 2k/5k checkpoint 轨迹诊断

原确认实验预注册的验收 endpoint 是 5,000 updates；下面的比较是完成主实验
后追加的 checkpoint 轨迹诊断，不追溯修改原验收条件。

固定 sampling seed `20260715`，使用同一 5k 严格配对采样协议。seed 4101
原有的 2k 样本在 `interleaved-v3-provenance` 下重跑后，Flow 和 LPL 的
最终 NPZ 均与旧样本逐字节 SHA256 相同。

### FID

| train seed | Flow-2k | Flow-5k | LPL-2k | LPL-5k | LPL 2k 相对 5k |
|---:|---:|---:|---:|---:|---:|
| 4101 | 19.5340 | 19.9848 | 17.7561 | 17.9296 | -0.1735 |
| 4102 | 19.6039 | 20.0493 | 17.7377 | 18.0659 | -0.3282 |
| 4103 | 19.9701 | 21.0154 | 18.1928 | 18.6538 | -0.4610 |
| mean | 19.7027 | 20.3498 | 17.8955 | 18.2165 | -0.3209 |

LPL-2k 在 `3/3` training seeds 上优于各自的 LPL-5k，并且 `3/3` 都优于
本地同协议的官方零更新 FID `18.9098`。平均 LPL FID 从官方降低
`1.0142`，从 5k endpoint 降低 `0.3209`。

### KID 与 IS

| endpoint | Flow KID | LPL KID | Flow IS | LPL IS |
|---:|---:|---:|---:|---:|
| 2k mean | 0.005491 | 0.004685 | 73.8878 | 76.0254 |
| 5k mean | 0.006143 | 0.005236 | 71.8455 | 73.5665 |
| 官方零更新 | 0.004696 | 0.004696 | 77.3452 | 77.3452 |

2k 的平均 KID 略优于官方，但幅度只有约 `0.000011`；只有 `2/3` 单 seed
优于官方 KID。2k 的平均 IS 明显优于 5k，但仍低于官方。因此准确结论仍是：
2k 在当前 5k-sample 协议下给出最佳 FID，不能称为所有指标全面超过官方。

这个现象不是“LPL 到 5k 失效”。LPL 相对同一步 Flow 的平均 FID 优势从
2k 的 `1.8071` 增大到 5k 的 `2.1334`；与此同时，普通 Flow continuation
从 2k 到 5k 平均恶化 `0.6472`。因此更符合数据的解释是：

> LPL 的相对纠正继续累积，但 fresh optimizer、小 global batch 的整体
> post-training 轨迹继续远离强官方起点，后者最终超过了新增纠正收益。

EMA 也可能参与该现象。EMA decay 为 `0.9995`，递推中初始官方 EMA 的直接
系数在 2k 为 `0.9995^2000 = 0.3678`，在 5k 仅为 `0.0820`。这不等于
EMA 中官方信息的全部比例，但说明 2k endpoint 明显更受官方起点锚定。

为避免从两个点事后挑 checkpoint，后续已改为每 500 updates 保存一次：
`500/1000/.../5000`。先在固定配对噪声下筛查完整曲线，再对候选区间做
独立 5k 复评；最终 50k 只使用预先由验证曲线选定的 endpoint。

密集曲线已完成。training seed `4102`、sampling seed `20260715` 的
1k-sample 配对筛查中，Flow FID 从 step 500 的 `47.6758` 整体升至
step 4500 的 `49.8840`，KID 从 `0.004337` 升至 `0.006068`。LPL 在
每个 checkpoint 的 FID/KID 都优于同一步 Flow，但自身最佳筛查 FID 出现在
step 2500 的 `46.7574`，随后也开始恶化。这条 1k 曲线噪声较大，不能用于
最终 FID 排名；它的主要价值是显示：绝对质量恶化并不是 LPL 独有，普通
Flow continuation 本身就持续离开官方起点。

机器可读曲线和图位于：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_lpl_checkpoint_curve_seed4102/
  seed4102_screen/checkpoint_curve_n1000_seed20260715.json
  seed4102_screen/checkpoint_curve_n1000_seed20260715.csv
  seed4102_screen/checkpoint_curve_n1000_seed20260715.png
```

## 官方 continuation checkpoint 审计

旧版官方 RAE 的公开资产不能用于精确 continuation：

1. Hugging Face `nyu-visionx/RAE-collections` 当前共有 27 个文件。
   Stage-2 只发布多个 `stage2_model.pt`，没有 optimizer、scheduler、
   global step、epoch 或 RNG 状态文件。
2. 本地 `DiTDH-S_ep20/stage2_model.pt` 的 SHA256 与官方 LFS 记录一致：
   `926bed703011...b43462`。实际反序列化结果是包含 190 个 fp32 tensor 的
   纯 `OrderedDict`，不存在 `model/ema/optimizer/scheduler/step/epoch`
   wrapper。
3. 官方训练器保存的可恢复 checkpoint 则明确包含
   `model + ema + optimizer + scheduler + step + epoch`，恢复时六项全部
   加载。公开文件与这一格式不同。
4. 作者在 issue #45 明确说明，公开的 epoch 20/80 权重来自同一条
   800-epoch 训练轨迹中的中间点。因此 epoch 20 不是为短程微调单独训练的
   终点。

纯 state dict 无法自证它是 online model 还是 EMA；结合官方默认评估 EMA
和公开文件面向 sampling 的用途，它很可能是抽取后的 EMA 推理权重，但现有
公开元数据不足以把“很可能”提升为确定事实。无论它是哪一种，缺少 AdamW
一、二阶矩、原 scheduler 相位和 global step 都已足以判定：当前用 fresh
AdamW 从它出发的实验是 post-training/restart，不是官方训练轨迹的连续
续训。

新版官方 [RAEv2](https://github.com/nanovisionx/RAEv2) 另行公开了一个
`stage2/imagenet/dinov3l-k7/checkpoint.pt`，大小约 `10.5 GB`。RAEv2 的
远程文件头已做字节级检查：PyTorch ZIP 中首个 `data.pkl` 的顶层字段确实
包括 `step=100080`、`epoch=80`、`model`、`ema`、`optimizer` 和
`scheduler`。它是目前找到且已确认的官方完整 continuation checkpoint。
但它使用 DINOv3-L 多层表征、新版 `DiTwDDTHeadIG`、`x` prediction 和
GMuon，不能直接替代当前 DINOv2-B + DiTDH-S + AdamW 实验。为避免无必要
占用空间，本次只读取并检查了远程文件开头，未保留 10.5 GB 完整副本。

因此当前最优先的外部依赖是向原作者索取与
`DiTDH-S_ep20/DiTDH-XL_ep20` 对应的原始完整 checkpoint。得到它之前，
现有结果只能回答“从官方推理权重重新启动优化时，LPL 相对 fresh-Flow
是否更好”，不能回答“沿官方原训练状态继续训练时，LPL 是否更好”。

## 三采样种子三方结果

固定 training seed 4102，只改变 sampling seed。官方、Flow 和 LPL 使用
相同 label、初始 noise、迭代数和 CUDA RNG 流。

### FID

| sampling seed | 官方零更新 | Flow-5000 | LPL-5000 | LPL - 官方 | LPL - Flow |
|---:|---:|---:|---:|---:|---:|
| 20260715 | 18.9098 | 20.0493 | 18.0659 | -0.8438 | -1.9834 |
| 20260716 | 19.0835 | 20.4475 | 18.3557 | -0.7278 | -2.0919 |
| 20260717 | 18.8480 | 20.2290 | 18.0246 | -0.8235 | -2.2044 |

LPL 相对 Flow 和官方的 FID 都是 `3/3` 改善。

### KID

| sampling seed | 官方零更新 | Flow-5000 | LPL-5000 | LPL - 官方 |
|---:|---:|---:|---:|---:|
| 20260715 | 0.004696 | 0.006032 | 0.005201 | +0.000505 |
| 20260716 | 0.004635 | 0.006095 | 0.005356 | +0.000721 |
| 20260717 | 0.004480 | 0.005914 | 0.004995 | +0.000515 |

LPL 相对 Flow 是 `3/3` 改善，相对官方是 `0/3` 改善。

### IS

| sampling seed | 官方零更新 | Flow-5000 | LPL-5000 | LPL - 官方 |
|---:|---:|---:|---:|---:|
| 20260715 | 77.3452 | 70.4101 | 73.6076 | -3.7376 |
| 20260716 | 77.8122 | 72.6281 | 74.7890 | -3.0231 |
| 20260717 | 80.2732 | 73.1766 | 75.4151 | -4.8581 |

LPL 相对 Flow 是 `3/3` 改善，相对官方是 `0/3` 改善。

因此当前最准确的表述是：

> 普通 Flow continuation 在当前小 batch、fresh optimizer 条件下明显破坏
> 官方 checkpoint。LPL 稳定减轻这种破坏，并在 5k FID 上越过官方起点；
> 但它没有恢复官方起点的 KID 和 IS。

## 无泄露与配对审计

下列检查在 3 个训练 seed 上全部通过：

1. trainer 只构造 `split="train"`，并拒绝文件名不是 `train-*` 的 shard。
2. 每个 manifest 均记录 1,281,167 个 train 样本、294 个 shard；
   `evaluation_reference_loaded_by_trainer=false`。
3. encoder/decoder 均为 `eval()`、`requires_grad=False`、fp32、
   `noise_tau=0`。
4. optimizer 参数集合精确等于 Stage-2 prior 的可训练参数；不包含
   encoder、decoder 或 EMA 参数。
5. 第一次 backward 后 frozen RAE/EMA 的梯度全部为 `None`；训练结束后
   RAE tensor version 逐项不变。
6. Flow/LPL 从相同官方 checkpoint SHA256
   `926bed703011...b43462` 和相同 fresh optimizer/scheduler 状态启动。
7. 每个 seed 的首批 dataset index、image、label、time、noise、
   noisy latent、target velocity、初始 prediction 哈希逐项一致。
8. 每个 rank 累计 20,000 个 microbatch；四个 rank 的完整数据流滚动
   SHA256 在 Flow/LPL 间逐 rank 一致。
9. 终点 CPU/CUDA RNG、scheduler 和 optimizer parameter-group 哈希一致。
   模型参数和 optimizer moments 因 loss 不同而允许不同。
10. `micro_batch=1`，所以 `lpl_max_samples_per_rank=1` 不会截断 eligible
    样本；约 20% 的门内样本全部精确计算 LPL。
11. 每个采样目录都由 provenance 文件绑定 endpoint、物化 EMA、配置、
    采样脚本、seed、并行协议和最终 NPZ SHA256。
12. 5,008 张临时 PNG 中只把前 5,000 张写入 NPZ，每类严格 5 张；Flow、
    LPL、官方三方的采样 RNG 审计一致。
13. 在汇总器之外重新读取 8 个评估文件、13 行结果，并实际重算 47 个
    endpoint、EMA、config、sampling script、NPZ 和 provenance 文件的
    SHA256，结果零不一致。
14. 对 3 个 sampling seed、每个 seed 的 4 个 rank，逐字段比较官方、
    Flow、LPL 的 initial RNG、first noise、first label、iteration count
    和 final RNG，共 12 组三方比较，结果全部一致。

训练 seed 只改变训练数据顺序、增强、time 和 noise，不改变方法权重。
sampling seed 只改变生成初始噪声和对应的平衡类序列。

## 代码与运行事故

在 seed 4102 训练和终点审计完成后，评估器曾因缺少 `RAE_ROOT` import 抛出
`NameError`。错误发生在启动采样之前，没有生成部分样本，也没有修改训练
checkpoint。修复后新增回归测试，所有最终结果都使用
`interleaved-v3-provenance` 重新采样并绑定哈希；旧的无 provenance 样本
没有进入最终汇总。

本轮最终聚焦测试：

```text
57 passed
```

## 可信度判断

**结论状态：可分享，但必须带限定条件。**

可以可靠支持：

- RAE-adapted LPL 在这个 DINOv2-B + DiTDH-S ep20 配对 post-training
  设置下，不是单一训练 seed 或单一采样 seed 的偶然。
- 收益不是数据流、noise、label、训练步数或样本数不一致造成的。
- 收益不是 encoder/decoder 被意外更新造成的。
- 相对等步数 Flow，FID/KID/IS 三项都稳定改善。

目前不能支持：

- LPL 在官方 50k gFID 协议上已经有效。
- LPL 在所有指标上超过零更新官方 checkpoint。
- LPL 在 DiTDH-XL、DINOv2-S/L prior、MAE 或 SigLIP2 上已经完成同等级
  的真实性确认。
- LPL 是官方 RAE 原训练 schedule 下普遍更好的从头训练目标。
- 该收益来自 outlier mask；mask 在当前 ViT decoder 上近乎恒等。

## 资源边界与下一步

官方公开的成熟 Stage-2 prior 使用 DINOv2-B latent。DINOv2-S/B/L 的
Stage-1 RAE 都已完成 shape、冻结和梯度桥接审计，但不能把随机初始化的
DINOv2-S/L prior 冒充官方生成模型。

下一步应验证：

1. DINOv2-B `DiTDH-XL_ep20` 和 `DiTDH-XL_ep80`，区分 prior 大小与成熟度。
2. MAE-B 和 SigLIP2-B 的官方 `DiT-XL-ep80`，每个 tokenizer 独立校准。
3. 至少一个 XL 起点通过后，再做预固定的 50k ADM FID 终验。

当前每张 4090 有约 24.6 GB，其他用户任务占约 3.48 GB。DiTDH-XL 有
838,675,796 个参数，按完整 fp32 model、gradient、EMA、Adam moments、
冻结 RAE 和运行开销估计，本实验约需 19.6 GB/卡；叠加现有任务后预计只剩
约 1.5 GB，即约 6%，低于要求的 10%-20% 余量。因此尚未启动 XL、MAE 或
SigLIP2，不停止别人的进程，也没有改成 head-only 或低精度实验。

## 结果文件

主要机器可读结果位于：

```text
/home/zhoushunyu/data/eqvae/experiments/rae_lpl_authenticity/
  confirmation_summary_strong_threeway.json
  confirmation_summary_strong_threeway.csv
  ditdh_s_ep20_seed4101_pair_audit_s5000.json
  ditdh_s_ep20_seed4102_pair_audit_s5000.json
  ditdh_s_ep20_seed4103_pair_audit_s5000.json
```

完整预注册协议见：

```text
docs/RAE_LPL_LARGE_SCALE_AUTHENTICITY_PROTOCOL_ZH.md
```

权威来源：

- [LPL, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/204fee94c982a19230c39045aa54f977-Paper-Conference.pdf)
- [LPL supplemental](https://proceedings.iclr.cc/paper_files/paper/2025/file/204fee94c982a19230c39045aa54f977-Supplemental-Conference.pdf)
- [Official RAE repository](https://github.com/bytetriper/RAE)
- [Official RAE model collection](https://huggingface.co/collections/nyu-visionx/rae)
- [Official clarification: ep20/ep80 are snapshots from the 800-epoch run](https://github.com/bytetriper/RAE/issues/45#issuecomment-3578820605)
- [Official RAEv2 repository](https://github.com/nanovisionx/RAEv2)
- [Official RAEv2 model files](https://huggingface.co/nyu-visionx/RAEv2-models/tree/main)
