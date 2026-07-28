# RAE 子空间路径课程：2k→5k crossover 结果

## 一句话结论

本轮发现了一条比“固定 floor path”更有价值、也更贴近生成模型的现象：

> RAE latent 的 coarse/detail path 适合做**训练阶段课程**，不适合从头到尾固定使用。
> 前 2k 使用 floor path、后 3k 切回标准 static flow，既消除了后期低噪声梯度冲突，
> 又在单 seed、5k online screen 上显著优于全程 static。

这已经是一个可复现的机制候选，但还不是论文结论：目前只有一个训练 seed、小 DiT、5k
updates 和固定 5k 样本。

## 1. 实验完整性

四路 2x2 crossover：

| 路径 | 前 2k | 后 3k |
|---|---|---|
| floor→floor | floor 0.20, p=2 | floor |
| floor→static | floor | static |
| static→static | static | static |
| static→floor | static | floor |

恢复审计发现旧 static 是连续 0→10k，而旧 floor 是 0→2k 后恢复到 5k。RAE transport
在 CPU 上采样 `t`，重建 DataLoader 会额外推进全局 CPU RNG。修复 loader RNG 隔离后：

- floor→floor 与旧 floor-5k 的 model、EMA、optimizer、scheduler、CPU/CUDA RNG 全部
  逐 tensor 一致；
- static→static 与旧 static-5k 也全部逐 tensor 一致；
- 两个 replay 的所有重叠训练日志逐值一致。

因此 switch 分支可以解释；首次恢复错误的 v1 目录没有用于任何结论。

## 2. 后期路径因果控制梯度方向

独立 held-out 128 latent、`t=0.1`、last shared block：

| 路径 | semantic descent ratio | gradient cosine | basis/semantic norm |
|---|---:|---:|---:|
| floor→floor | **0.901** | -0.120 | 0.829 |
| floor→static | **1.021** | +0.186 | 0.115 |
| static→static | **1.028** | +0.209 | 0.132 |
| static→floor | **0.928** | -0.087 | 0.830 |

- floor→static：descent ratio `+0.121`，冲突恢复为协同；
- static→floor：descent ratio `-0.099`，协同变成冲突；
- `t=0.3` 方向相同。

双向 switch 说明：低噪声梯度翻转主要由**当前后期路径目标**控制，不只是前 2k 状态的
伴随相关性。

## 3. 5k online generation 门控通过

固定 seed `20260718`、50-step、ImageNet reference、5k 样本：

| 路径 | FID ↓ | KID ↓ | IS ↑ |
|---|---:|---:|---:|
| floor→floor | 150.44 | 0.12460 | 6.68 |
| **floor→static** | **129.37** | **0.11008** | **8.56** |
| static→static | 146.86 | 0.13700 | 6.91 |
| static→floor | 159.61 | 0.14415 | 6.14 |

预注册的三项确认判据全部通过：

- floor→static 相对 floor→floor：FID `-14.0%`，KID `-11.7%`；
- floor→static 相对 static→static：FID `-11.9%`，KID `-19.6%`；
- static→floor 相对 static→static：FID `+8.7%`，KID `+5.2%`。

这不是“static 后程把坏模型救回一点”，而是两阶段路径在本次 screen 中超过了全程 static。

## 4. Online decoder closure 同方向

同一 64 noise/label 的 paired closure：

| 路径 | cycle residual ↓ | decoder sensitivity ↓ |
|---|---:|---:|
| floor→floor | 1.117 | 1.649 |
| **floor→static** | **1.068** | **1.580** |
| static→static | 1.088 | **1.574** |
| static→floor | 1.112 | 1.615 |

floor→static 相对 floor→floor 的逐样本 bootstrap：

- cycle 中位差 `-0.042`，95% CI `[-0.058,-0.034]`，`60/64` 样本改善；
- sensitivity 中位差 `-0.069`，95% CI `[-0.084,-0.056]`，`64/64` 改善。

static→floor 相对 static→static 的 cycle/sensitivity 都显著变差。closure 不是 FID 的证明，
但它说明生成提升同时伴随 latent 更接近 decoder 可处理区域，而不只是 classifier feature
偶然变化。

## 5. EMA 是短训练实验的重大混杂因素

训练使用 `ema_decay=0.9995`。从 2k 到 5k 的 3000 次更新后，EMA 仍保留：

```text
0.9995^3000 = 22.3%
```

的旧状态权重。在相同 1k 样本下，online FID 为 `160--187`，EMA FID 却为 `230--267`。
特别是 floor→static：

| 权重 | FID | KID |
|---|---:|---:|
| EMA | 255.54 | 0.2260 |
| online | **160.11** | **0.1114** |

所以此前“切回 static 只恢复一点”的判断主要描述的是 EMA 历史滞后，不是 online 模型本身。
短程机制实验必须同时报告 online，或使用 phase-reset/post-hoc EMA。扩散模型中 EMA 长度与训练
时间、网络和 guidance 存在强交互，已有工作也主张 post-hoc 调整 EMA，而不是固定一次
运行只留一个 EMA（Karras et al., 2024: https://arxiv.org/abs/2312.02696）。

## 6. 当前最合理的机制

证据支持下面的两阶段解释：

```text
训练早期 floor path
  -> 高噪声状态中延迟低秩 detail，降低早期任务复杂度
  -> 更快建立可生成的主体/语义结构

训练后期继续 floor
  -> detail 与 semantic 在共享 block 中由协同翻为冲突
  -> rollout 和 decoder closure 退化

训练后期切回 static
  -> 梯度恢复协同，完整 detail endpoint 得到直接训练
  -> online generation 与 closure 同时改善
```

第一行“更快建立语义结构”仍是最合理假设，不是已单独证明的中间变量；后两段有双向
crossover 支持。

## 7. 与已有工作的关系

- Flow Matching 本身允许选择不同 conditional probability path：
  https://arxiv.org/abs/2210.02747
- ICLR 2025 的 denoising curriculum 按 timestep/noise 难度安排任务：
  https://arxiv.org/abs/2403.10348
- 2026 DeLTa workshop 的 Curriculum Sampling 先 middle-biased `p(t)`、后 uniform：
  https://openreview.net/forum?id=LpmuITZLAk

因此不能声称“首次两阶段 flow curriculum”。当前可能的新点是：

1. 针对高维 RAE latent，沿数据驱动 coarse/detail 子空间改变**概率路径**，而不是只改
   timestep sampling；
2. 发现固定 path 的梯度关系会随训练由协同翻为冲突；
3. 用 path crossover 说明“早期有益、晚期有害”，并据此构造 subspace path curriculum。

暂定名称可以是 **Subspace Path Curriculum (SPC)**。

## 8. 论文价值与下一门槛

现在值得继续，但离 ICLR accept 仍有明显距离。下一轮只比较 `static` 与
`floor→static`，不再保留已被机制淘汰的固定 floor 搜索。

### 多 seed 门槛

- 至少 5 个训练 seed，相同数据量、更新数和 online/EMA 评估；
- 至少 `4/5` seeds 的 FID/KID 同时改善；
- paired mean FID 相对改善至少 `5%`，95% CI 不跨 0；
- closure 不得系统变差；
- 同时加入 phase-reset EMA 或 post-hoc EMA，排除仅 online checkpoint 偶然更尖锐。

### 持久性门槛

- 在 10k/20k 检查优势是否保持，而不是只提前达到 static 最终水平；
- switch 比例只在独立 validation seed 选择一次，随后冻结；
- 最终用 50k gFID/IS，并与相同小 DiT、相同 compute 的 static baseline 比较。

若多 seed 或长程优势消失，正确结论是“SPC 加速早期收敛”，仍有工程价值，但不足以作为
主会生成方法。若优势持续，才进入更大 DiT 和正式论文实验。

## 产物

```text
~/data/eqvae/experiments/rae_path_crossover_train_v2/crossover_evaluation/
~/data/eqvae/experiments/rae_path_gradient_interference/crossover_v2_n128_seed20260725/
~/data/eqvae/experiments/rae_path_schedule_closure/crossover_v2_online_step5000_n64_seed20260726/
```
