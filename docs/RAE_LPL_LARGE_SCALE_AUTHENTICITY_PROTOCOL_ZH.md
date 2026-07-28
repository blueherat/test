# RAE-LPL 大规模真实性验证协议

## 目标

本轮只回答一个问题：

> 之前观察到的 LPL 生成收益，是否来自正确实现、无数据泄露的
> decoder-aware 训练信号，并且能否跨官方 prior 大小和成熟度复现？

本轮不以提出新方法为目标，也不根据中途 FID 改权重、训练长度或筛选 seed。

## 方法身份

当前实现应称为 **RAE-adapted LPL**，不是“官方 LPL 代码复现”：

- LPL 论文没有发布可直接核对的训练代码。
- `z_hat = z_t - t * v_theta(z_t,t)` 与 RAE 的 linear Flow-OT 一致。
- 只在 `t/(1-t) <= 3` 时启用 feature loss。
- target 与 prediction 都使用 prediction feature 的空间均值和方差归一化。
- outlier quantile、margin、closing 和 opening 与论文补充材料一致。
- 原论文使用卷积 decoder 的五个多尺度 stage；RAE 是同分辨率 ViT decoder。
  当前取相对深度 `0.2/0.4/0.6/0.8/1.0` 的五层并等权，是必要的架构适配。
  因五层分辨率完全相同，论文的逆上采样权重退化为所有层共享的常数；该
  常数由下述 mean-contribution 校准吸收。
- `full` 不 detach prediction variance；这与论文公式一致。`detach` 只作为机制消融。

运行时还必须报告 mask keep fraction。当前 DINOv2-B/ViT decoder 长跑中
该值约为 `99.99997%`，说明原论文为卷积 AE decoder 设计的 outlier patch
在这里近乎恒等。本轮不据此改变已经固定的目标，但结论应明确：若收益复现，
主要证据来自 decoder hidden feature matching 和 prediction-stat
cross-normalization，而不是异常值屏蔽。

此前实验按初始 `mean(weight * LPL) / mean(Flow) = 0.25` 标定。重新逐段
核对后，这个规则与主论文 Figure 9 的描述直接对应：最佳 `w_LPL≈3` 时，
加权 LPL 约占总 loss 的 `1/5`，也就是 LPL/Flow 约为 `0.25`。因此本轮
主结果继续使用：

\[
w_{\mathrm{mean}}
=
0.25\,
\frac{\mathbb E[L_{\mathrm{Flow}}]}
{\mathbb E[\mathbf 1_{\mathrm{gate}}L_{\mathrm{LPL}}]}.
\]

门外样本的 LPL 记为零，对应论文公式 (3) 中门控指示函数位于期望内部。
补充材料 A.4 在公平比较不同 perceptual losses 时，另用了
`Var(weight * LPL) / Var(Flow) = 0.1`。它不是主实验默认选权规则。本轮
仍报告这个 variance weight，并用 seed 4101 做一个预先固定的敏感性分支；
它不参与主验收，也不能在看到 FID 后替换 mean-contribution 主定义。

## 无泄露与无作弊审计

每次训练必须自动满足：

1. 数据集固定为完整 ImageNet-1k `train`，共 294 个 parquet shard；
   validation/test 和 ADM reference 不进入 trainer。
2. encoder 与 decoder 都是 `eval()`、`requires_grad=False`、fp32、`noise_tau=0`。
3. optimizer 参数集合必须精确等于 stage-2 prior 的可训练参数集合，并与
   RAE/EMA 参数完全不相交。
4. 第一次 backward 后，RAE 和 EMA 的全部参数梯度必须仍为 `None`。
5. 训练结束时，RAE 参数的 tensor version 必须逐项不变。
6. Flow/LPL 配对分支从同一官方权重、同一 fresh optimizer/scheduler、
   同一训练 seed 出发。
7. 每个 rank 对 dataset index、label、增强后图像、time 和 noise 写滚动
   SHA256；同 seed 的 Flow/LPL 必须逐 rank 完全一致。
8. 首批 image、label、time、noise、noisy latent、target velocity 和初始
   prediction 的 SHA256 必须完全一致。
9. manifest 记录 config、官方 prior、decoder、normalization statistics 的
   路径与 SHA256，不能用来源不明的包装 checkpoint。
10. 采样使用同一 noise/label 是 paired evaluation，不是训练泄露；终点还要
    更换采样 seed，排除单一噪声流偶然性。
11. 5k 采样使用 `interleaved-labels-v2`，输出目录协议标记为
    `interleaved-v3-provenance`：标签池按最终 PNG 全局编号交错分给
    各 rank，先 5,000 张严格保持 ImageNet 每类 5 张，额外 8 张随机尾部只为
    对齐 global batch，写 NPZ 时全部裁掉。每个 rank 还记录首批 noise/label
    与最终 CUDA RNG 状态哈希；配对分支任一哈希不同即拒绝比较。
12. 评估表记录训练 endpoint checkpoint、物化的 EMA/model state 和最终样本
    NPZ 的 SHA256。物化 state 另带 source checkpoint 哈希；来源变化、sidecar
    缺失或物化文件哈希不符时必须原子重建，禁止静默复用旧采样权重。
13. 每个完整采样目录必须包含 `sampling_run_provenance.json`，逐项绑定 endpoint、
    物化权重、采样配置、采样脚本、seed、并行协议和 NPZ 哈希。完整 NPZ 若
    缺少该清单或任一哈希不匹配，必须拒绝复用；不能根据当前配置事后补写。
14. 配对训练终点还要比较 CPU/CUDA RNG、scheduler 和 optimizer 参数组的
    规范化哈希。模型和 optimizer moment 应因目标不同而变化，但随机流与
    schedule 不得分叉。

任一硬检查失败，训练立即终止，该分支不得进入结果表。

## 官方模型边界

本轮官方 RAE 代码基线为 commit
`a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041`；模型资产逐文件对照
`nyu-visionx/RAE-collections` 的 LFS SHA256，而不是只依据文件名。外部
RAE 工作树并非完全干净，实际运行还包含以下已审计补丁：

- `stage1/rae.py`：为新版 Transformers 预先把 decoder 配置中的占位
  `patch_size` 替换为 16；官方构造函数加载配置后本来也会赋相同值。
- `utils/train_utils.py`：新增 ImageNet parquet reader；本轮 trainer 自己
  构造 `DistributedSampler(shuffle=True)`，官方 center crop 与 EMA 更新
  函数没有修改。
- `sample_ddp.py`：修正 `LOCAL_RANK` 绑定，加入严格类别交错和
  noise/label/RNG 哈希；ODE、Stage-2 forward 和 RAE decode 数学没有修改。
- adapter 导出、官方 `train.py` 和 resume helper 的本地补丁不进入本轮
  自定义 trainer。

因此本轮应描述为“固定官方 commit 与权重，加经过审计的兼容/I/O 补丁”，
不能描述为运行了完全干净的官方工作树。

RAE 官方发布了 DINOv2-S/B/L 的 encoder、decoder 和 normalization
statistics，但成熟 ImageNet stage-2 prior 只发布在 DINOv2-B latent 上：

- `DiTDH-S_ep14`
- `DiTDH-S_ep20`
- `DiTDH-XL_ep20`
- `DiTDH-XL_ep80`
- `DiTDH-XL` 最终模型

因此：

- DINOv2-S/B/L 用于无训练的 shape、公式、冻结和梯度桥接审计。
- 真正生成验证比较 DINOv2-B 上的 DiTDH-S 与 DiTDH-XL，以及不同 epoch。
- 不把随机训练的 DINOv2-S/L prior 称为官方模型或大规模生成验证。

## 实验阶段

### Phase 0：三种 RAE 尺寸的零训练审计

- `DINOv2-S`: `[384,16,16]`
- `DINOv2-B`: `[768,16,16]`
- `DINOv2-L`: `[1024,16,16]`

每种尺寸检查 clean estimate、decoder hidden shape、full LPL 有限、target
无梯度、prediction 能通过冻结 decoder 把梯度传回 latent、RAE 参数无梯度。

### Phase 1：DiTDH-S 的成熟度复现

主起点为官方 `DiTDH-S_ep20`，`ep14` 作为已有结果与成熟度对照。

- 分支：Flow、mean-contribution-calibrated full LPL。
- 训练 seeds：`4101/4102/4103`。
- 每分支 5,000 optimizer updates，global batch 16，共见 80,000 个 train
  样本，约 `0.062` ImageNet epoch。
- fp32、TF32 off、4 GPU、micro batch 1/rank、gradient accumulation 4。
- `lpl_max_samples_per_rank=1` 在该 micro batch 下不会截断 eligible 样本，
  因而本轮门控 LPL 是批内精确值，不是 eligible 子样本估计。
- checkpoints：500、2,000、5,000 updates。
- 先在固定 sampling seed `20260715` 上做 5k FID/KID/IS。
- 5,000 endpoint 再用 `20260715/20260716/20260717` 三个 sampling seed。

官方 Stage-2 配置的 global batch 是 1,024，本轮受四张 24 GiB GPU 限制，
global batch 为 16。学习率 `2e-5` 是官方 schedule 的末期学习率，EMA
decay `0.9995` 与官方一致，但 model-only checkpoint 不含原 optimizer 和
scheduler 状态。因此本轮是严格配对的低学习率 post-training，不是官方
optimizer 轨迹的无缝续训。这个差异尤其可能导致普通 Flow 分支受到
fresh-optimizer 或小 batch 的影响，必须同时报告零更新官方起点。

### Phase 2：DiTDH-XL 的 prior 尺度与 epoch 复现

先对 `DiTDH-XL_ep20` 做 1-step 完整参数训练 memory smoke。若 24 GiB
显存不能容纳，不允许偷偷改成只训练 head 后仍称为同一实验；应记录资源
边界，再选择 FSDP 或明确标注的 parameter-efficient 次级实验。

若完整训练可运行：

- 起点：`DiTDH-XL_ep20` 和 `DiTDH-XL_ep80`。
- 分支：Flow、mean-contribution-calibrated full LPL。
- seeds：`4101/4102/4103`。
- 首轮 500 updates；通过方向门槛后扩到 2,000 updates。
- 每个起点先单独校准权重，不能从 DiTDH-S 迁移权重。
- 评价与 Phase 1 使用相同 sample 数、sampling seeds 和 reference。

## 预注册验收标准

单个官方起点称为“强复现”需要同时满足：

1. 3/3 training seeds 的 LPL 5k FID 优于等 update Flow。
2. 平均 KID 不恶化，且至少 2/3 seeds 的 KID 改善。
3. LPL 平均 FID 优于零更新官方起点，而 Flow 是否退化单独报告。
4. 三个 sampling seeds 上，LPL-Flow 的 FID 差值同号。
5. Flow/LPL 的四个 rank 数据流 SHA256 全部配对一致。
6. 所有资产哈希、冻结边界和有限值检查通过。

方法形式和 mean-contribution 权重是在此前实验后确定的；本轮是确认性复现，
不是完全盲的首次发现。seed `4102/4103`、XL 起点及新增 sampling seeds 在
本轮开始前未用于重新调权重，这与训练/评价数据泄露是两回事，但报告时必须
保留该方法开发历史。

“弱复现”为 2/3 training seeds 的 FID 改善、平均 FID 改善、平均 KID
不恶化。否则该起点判为失败。

只有 Phase 1 强复现，并且 Phase 2 至少一个 XL 起点弱复现，才运行一个
预先固定组合的 50k ADM FID 终验。不能根据 5k 表现临时挑最好 seed；
固定使用 training seed `4102`、sampling seed `20260716`。

## 允许得出的结论

- 多起点、多 prior 尺度复现：支持 RAE-adapted LPL 的收益是真实的
  decoder-aware optimization 效应。
- 只在 DiTDH-S 复现：说明收益可能依赖 prior 架构或优化状态，不能泛化。
- 不同 epoch 方向相反：说明 LPL 更可能是 checkpoint-dependent
  post-training correction，而不是普适训练目标。
- mean calibration 复现而 variance sensitivity 不复现：仍支持主实验；
  说明收益对 loss scale 敏感，不能宣称两种权重规则等价。

## 权威来源

- [LPL, ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/204fee94c982a19230c39045aa54f977-Paper-Conference.pdf)
- [LPL supplemental](https://proceedings.iclr.cc/paper_files/paper/2025/file/204fee94c982a19230c39045aa54f977-Supplemental-Conference.pdf)
- [Official RAE repository](https://github.com/bytetriper/RAE)
- [Official RAE model collection](https://huggingface.co/collections/nyu-visionx/rae)

## 当前执行状态

Phase 1 已于 2026-07-28 完成并达到预注册强复现标准。完整结果、三方同噪声
指标权衡、无泄露审计和资源边界见：

```text
docs/RAE_LPL_LARGE_SCALE_AUTHENTICITY_RESULTS_ZH.md
```

Phase 2 尚未启动。当前其他用户任务占用约 3.48 GB/卡，完整 fp32
DiTDH-XL 预计无法保留要求的 10%-20% 显存余量；不停止他人任务，也不把
head-only 或低精度替代实验混入本协议。
