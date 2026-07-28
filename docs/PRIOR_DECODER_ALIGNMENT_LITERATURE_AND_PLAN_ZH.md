# Prior-Decoder 对齐：文献定位、联合训练判断与实验路线

## 结论

两阶段模型确实存在目标错位，但当前不应直接把 encoder、latent prior 和
decoder 全部解冻。最稳妥的顺序是：

1. 先联合训练目标，保持表示和解码标尺固定。
2. 证明 decoder-aware prior loss 能缩小真实 latent 与生成 latent 的解码差距。
3. 再只开放一个小 decoder adapter，并用真实 latent 重构路径约束它。
4. 只有这两步都成功，才考虑让 encoder 参与有限的联合微调。

简写为：**先联合目标，后联合参数；先固定靶子，再允许靶子移动。**

## 当前证据指向什么

Imagenette-64 的 `16/64/256d x 5 seeds` 实验已经得到：

- empirical latent 的 Oracle FID 随容量增加而改善；
- 相同预算 prior 的 modeling gap 为 `2.79/9.92/22.42` FID；
- 256d 的 raw/condition 指标并未显示全面崩坏，但 decoded feature
  C2ST AUC 达到 `0.637`；
- class-mass reweighting只解释 `15.9%` 的 256d gap；
- 剩余 FID 差距主要来自协方差和多样性，而非均值偏移；
- 等角度局部扰动实验没有发现 prior error 落在特殊高敏感 decoder 方向。

因此目前最合理的解释不是“prior 在 latent 中离得特别远”，而是：

> prior 没有充分学习 decoder 真正读取的联合变化模式。普通逐点 latent
> 损失可以看不见这种分布级错误，而 decoder 会把它显现出来。

这支持 decoder-aware 训练，但暂不支持移动 encoder 或完整 decoder。

## 相关文献与边界

### 1. 匹配 latent prior 与编码分布并不新

Adversarial Autoencoder 等经典工作已经通过正则化 aggregated posterior
使编码分布匹配可采样 prior。因而“两个阶段应当对齐”只能作为问题背景，
不能单独构成新意。

### 2. LPL 是最直接的强基线

ICLR 2025 的 *Boosting Latent Diffusion with Perceptual Objectives* 明确把问题
称为 diffusion model 与 AE decoder 的 disconnect。它冻结 AE，用预测 clean
latent 和真实 clean latent 经过 decoder 后的中间特征差异监督 diffusion，
并报告 `6%-20%` 的 FID 改善。

这说明“梯度经过冻结 decoder”可行，也意味着简单的逐样本 decoder feature
matching 已经是必须复现的基线，而不是我们的新方法。

### 3. tokenizer 可以为 denoising 目标重新塑形

`l-DeTok` 用被插值噪声或 masking 破坏的 latent 仍可重构图像作为 tokenizer
目标。`PAE` 则强调空间结构、局部连续性和全局语义比重构误差更能解释下游
生成。这些工作支持“latent 应按 prior 的任务塑形”，但也意味着泛泛的
diffusion-friendly tokenizer 叙事已经拥挤。

### 4. decoder 可以在生成器训练后适配

*Image Tokenizer Needs Post-Training* 直接研究 reconstructed token 与 generated
token 的分布差异，并在已有生成模型之后调整 decoder。它与我们的 mismatch
现象非常接近，表明 decoder post-training 必须作为第二阶段对照，但它也提醒：
只改善 decoder 鲁棒性不等于 prior 学好了目标分布。

### 5. 端到端联合训练可行，但朴素做法不可靠

`REPA-E` 直接报告：把标准 diffusion loss 反传到 VAE 的朴素端到端训练无效，
甚至降低最终性能；加入 representation alignment 后才得到有效联合微调。
`AlignTok` 采用冻结 encoder、联合微调并保语义、最后单独精修 decoder 的三阶段
流程。2026 年的 `UNITE` 证明从头单阶段联合 tokenizer 与 denoiser 可行，但它
依赖共享 Generative Encoder，是架构级重设计，不是低成本 retrofit。

这些结果支持联合训练的方向，但不支持直接全解冻。

## 四种训练方式的判断

| 方式 | 能回答的问题 | 主要风险 | 当前建议 |
| --- | --- | --- | --- |
| 冻结 E/D，只训练 prior | prior 是否能学会固定 decoder 的需求 | 目标设计不足 | 首先做 |
| 冻结 E，联合 prior 与完整 D | 系统能否共同适配 | D 可吸收 prior 错误并遗忘真实 latent | 暂不做 |
| 冻结 E/D 主干，训练 prior 与小 D adapter | 少量解码适配是否足够 | adapter 仍可能绕过 prior | 第二阶段做 |
| 联合 E、prior、D | 能否自动形成全新 latent | moving target、latent collapse、语义漂移、责任串通 | 最后才考虑 |

全量联合时，三个模块可以通过以下方式降低训练损失而不提升生成：encoder
压缩或扭曲分布使 prior 更容易；decoder 减弱对 latent 的依赖；decoder 只适配
prior 的错误而损害真实 latent；RAE 的 VFM 语义被破坏。此时即使 FID 改善，
机制也难以归因。

## 第一阶段：固定 E/D 的 decoder-aware prior

设真实 latent 为 `z0 = stopgrad(E(x))`，prior 从 noisy latent `zt` 预测
`zhat0`。固定 decoder 的某层响应记为 `F_l(s_tau, tau, z)`，其中 pixel state
`s_tau`、pixel time `tau` 和随机噪声在配对比较中完全相同。

基础目标：

```text
L = L_flow + lambda_pair * L_pair + lambda_dist * L_dist
```

逐样本项是 LPL 强基线：

```text
L_pair = sum_l ||F_l(s_tau, tau, zhat0) - stopgrad(F_l(s_tau, tau, z0))||
```

我们真正值得检验的是分布项：

```text
L_dist = Dist({F_l(..., zhat0_i)}, {F_l(..., z0_i)})
```

`Dist` 首选随机投影 sliced Wasserstein；备选为白化低维特征上的 mean/covariance
Bures 距离。它针对现有实验发现的 covariance/diversity gap，而非只缩短每个
样本的欧氏距离。训练时无需完整 pixel rollout，只需对相同 pixel state 做一次
冻结 decoder 前向和反向。

## 最小消融

只用当前 256d、seed 0 的正式 prior checkpoint，继续相同步数：

| 分支 | 目标 | 角色 |
| --- | --- | --- |
| A0 | `L_flow` | continuation control |
| A1 | `L_flow + clean latent MSE` | raw latent baseline |
| A2 | `L_flow + condition embedding loss` | decoder input baseline |
| A3 | `L_flow + pairwise decoder feature` | LPL-style baseline |
| A4 | `L_flow + distribution decoder feature` | 核心假设 |
| A5 | `A3 + A4` | 互补性检查 |

所有分支固定 E/D，使用完全相同的数据、batch、latent noise、pixel noise、pixel
time、训练步数、采样 NFE 和评估样本。先继续 `2k-5k` steps，不从头重训。

## 验收和停止标准

单 seed 进入多 seed 的门槛：

1. 至少一个 decoder-aware 分支相对 A0 改善 `>= 2.0` FID。
2. A4/A5 至少比 A1/A2 改善 `>= 1.0` FID。
3. class TV、decoded covariance gap 同时下降。
4. raw SWD、covariance coverage、effective rank 不发生 collapse。
5. empirical-latent decoder FID 保持不变。

五 seed 方法门槛：

1. 256d modeling gap 平均减少 `>= 20%`，即约 `4.5` FID。
2. 至少 `4/5` seeds 同方向。
3. A4/A5 必须稳定优于 A3，才能支持“分布级 decoder-readable matching”新意。
4. 报告真实算力和 wall-clock，不只报告 steps。

若只有 A3 成功，这是 LPL 的复现和迁移，不足以形成当前方法贡献。若所有分支
失败，停止 decoder-aware prior 路线，不通过解冻模型来挽救假设。

## 第二阶段：受约束的部分联合训练

只有第一阶段通过后，才增加小 decoder adapter 或 LoRA：

- encoder 和 decoder 主干始终冻结；
- prior 更新时使用 `L_flow + L_pair + L_dist`；
- adapter 更新同时看到真实 `z0` 和生成 `zhat0`；
- 对真实 `z0` 保留 reconstruction/perceptual anchor；
- prior 和 adapter 交替更新，避免共同追逐移动目标；
- 明确监控 Oracle FID，任何持续退化都立即停止。

它回答的是“固定 latent 下，prior 学习与有限 decoder 鲁棒性是否互补”。只有
这个问题得到肯定答案，才有理由讨论带语义锚点的 encoder 联合微调。

## 最终判断

你的直觉“两个阶段应该共同考虑，才能保持对齐”是对的；但“共同考虑”不等于
“所有参数同时更新”。对当前项目，最有信息量、成本最低、因果最清楚的下一步
是冻结 encoder/decoder，让 prior 通过 decoder-aware 分布目标训练。真正的全量
端到端联合训练既不是首选验证手段，也很难成为独立新意。

## 主要文献

- Berrada et al., [Boosting Latent Diffusion with Perceptual Objectives](https://proceedings.iclr.cc/paper_files/paper/2025/hash/204fee94c982a19230c39045aa54f977-Abstract-Conference.html), ICLR 2025.
- Leng et al., [REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](https://arxiv.org/abs/2504.10483), 2025.
- Chen et al., [AlignTok](https://aligntok.github.io/), ICLR 2026.
- Duggal et al., [UNITE](https://arxiv.org/abs/2603.22283), 2026.
- Yang et al., [Latent Denoising Makes Good Tokenizers](https://arxiv.org/abs/2507.15856), 2025/2026.
- Yue et al., [Prior-Aligned Autoencoders](https://arxiv.org/abs/2605.07915), 2026.
- Qiu et al., [Image Tokenizer Needs Post-Training](https://arxiv.org/abs/2509.12474), 2025.
- Makhzani et al., [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644), 2015.
