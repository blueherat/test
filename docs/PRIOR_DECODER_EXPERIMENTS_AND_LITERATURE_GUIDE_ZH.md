# 从两阶段生成到 Prior-Decoder 断层：实验与文献指南

更新时间：2026-07-22

## 1. 最短结论

这条路线研究的不是“RAE 的 encoder 好不好”，而是一个更一般的两阶段生成问题：

```text
真实图像 x --encoder--> 真实 latent z_real
高斯噪声 e --latent prior--> 生成 latent z_gen
像素噪声 n + latent z --随机 decoder--> 图像 x_hat
```

我们已经确认：

1. 更大的 latent 确实能帮助 decoder，但这种收益只在输入真实 latent 时成立。
2. 用同预算 prior 生成 latent 后，16/64/256d 的 decoded modeling gap 约为
   `2.79/9.92/22.42`，容量越大反而越脆弱。
3. 这不能简单解释为“256d prior 没训练好”：256d 的 held-out flow MSE 最低，
   raw latent SWD 也不大。
4. decoder 确实读取样本级 latent；把 latent 打乱会在高、中噪声显著破坏预测。
5. 但是，固定时间、固定 decoder 层的边际 response distribution mismatch 不能预测
   最终 decoded gap。1024 样本复核后，所有 leave-one-dimension-out 相关仍为负。

因此，当前被否定的是一个具体机制和训练目标：

> 不能用若干固定时间、固定层的 batch-level response 分布距离，代表整条生成轨迹的
> 最终 prior-decoder 断层。

当前没有被否定的是更一般的事实：两阶段生成存在 tokenizer/prior/decoder 目标错位。
但这个大问题已经有大量论文研究，不能本身作为新颖性。

### 与更早 RAE 实验的关系

这条实验线是从更早的 RAE 等变性、adapter 和频域实验之后另行建立的受控系统。
两者不要混在一起：

- 更早的 RAE 实验研究 DINOv2/MAE/SigLIP 表征、decoder 条纹、等变 adapter 和
  RAE latent flow 的训练动力学。
- 当前实验研究一个小型、从头训练的两阶段生成器里，latent 容量、learned prior 与
  stochastic decoder 如何分配生成责任。
- 当前主实验不使用 DINOv2、EQ-VAE、SD-VAE 或原始 RAE decoder，也不使用 RAE 的
  ImageNet gFID 配置。

之所以换成 Imagenette-64 小系统，是为了用 5 个 seed 和严格配对控制低成本地确认
机制，再决定是否值得回到大型 RAE。结果现在表明：静态 response proxy 没有通过
回到大型模型所需的门槛。

## 2. 先把系统讲清楚

### 2.1 Encoder 做什么

小型卷积 encoder 接收 `64x64` Imagenette 图像，输出一个向量：

```text
z = E(x),  z in R^d,  d in {16, 64, 256}
```

它不是 DINOv2，也不是原始 RAE encoder。这个小系统的目的是低成本、受控地研究机制。
encoder 和 decoder 从头联合训练，每个容量运行 5 个 seed。

### 2.2 为什么 decoder 是“随机生成式 decoder”

这里的 decoder 不是一次前向的确定性网络。它是 latent-conditioned velocity U-Net，
从像素高斯噪声开始，沿 50 步 ODE 生成图像：

```text
s_1 ~ N(0, I)
ds_t / dt = v_D(s_t, t, z)
s_0 = generated image
```

“随机”指输入有 pixel noise，不是说 decoder 权重没有训练。decoder 与 encoder 已经在
Imagenette train split 上联合训练了 20,000 step。

训练时构造直线路径：

```text
s_t = (1-t) x + t n
target velocity = n - x
```

并最小化：

```text
L_decoder = ||v_D(s_t, t, E(x)) - (n-x)||^2
```

训练中有 `10%` condition dropout，所以同一个 decoder 能接收真实 latent 或零 latent。
这使“有条件与无条件的差”成为合法的模型内对照，而不是完全 OOD 的临时改线。

### 2.3 Latent prior 做什么

训练好 encoder/decoder 后冻结二者。latent prior 只学习：

```text
高斯噪声 -> encoder 在训练集上产生的 latent 分布
```

它也是 rectified flow：

```text
y_t = (1-t) z_real + t e
target velocity = e - z_real
```

采样时从 `t=1` 的高斯走到 `t=0`，得到 `z_gen`，再交给冻结 decoder。

### 2.4 为什么要区分两种噪声

系统里有两个完全不同的随机源：

- latent prior noise：决定生成哪个 latent，即较高层内容、实例和外观信息。
- pixel decoder noise：在给定 latent 后，生成 decoder 没被 latent 唯一规定的像素细节。

所以这不是“一个 diffusion 拆成两段”。它是概率分解：

```text
p(x, z) = p(z) p(x | z)
```

第一阶段学 `p(z)`，第二阶段学 `p(x|z)`。

## 3. 四种 rollout 分别在问什么

为了知道问题来自哪里，我们没有只看一种生成结果，而是固定 decoder 后做四路对照。

### 3.1 Oracle rollout

```text
x_val -> E -> z_val -> D -> x_hat
```

这是“给 decoder 与目标图像配对的真实 code”时的质量，表示 encoder/decoder 组合的
条件上限。

### 3.2 Empirical rollout

```text
从真实 train latent 集合随机抽 z_train -> D -> x_hat
```

它仍然使用真实 encoder latent，但不和当前 val 图像配对。它回答：如果 latent 分布
完全正确，但只做无条件采样，decoder 能产生什么分布。

### 3.3 Prior rollout

```text
e -> learned latent prior -> z_gen -> D -> x_hat
```

这是完整两阶段生成系统。

### 3.4 Gaussian control

```text
未经训练的 Gaussian latent -> D -> x_hat
```

它是下限控制，用来确认 learned prior 确实学到了东西。

### 3.5 三个核心差值

```text
Oracle FID = FID(oracle images, real val images)

Total prior gap = FID(prior images) - FID(oracle images)

Modeling gap = FID(prior images) - FID(empirical images)
```

`Modeling gap` 最重要，因为 Oracle 还包含“val code 与目标图像配对”的额外优势；
Empirical 和 Prior 都是无条件 latent 采样，更公平地隔离 learned prior 的代价。

这里的 FID 是 Imagenette-64 上的 ResNet18 feature FID，只用于本实验内部配对比较。
它不是 ImageNet Inception gFID，绝对数值 `100` 左右不能和 RAE 论文的 `1.x` 比较。

## 4. 第一轮：噪声责任曲线

### 4.1 问题

我们先问：decoder 在生成的哪个时段真正使用 latent？

对同一个像素状态 `s_t`，比较四种条件：

```text
real:         正确配对 latent
null:         零 latent
shuffle:      另一张图的 latent
within-class: 同类别另一张图的 latent
```

定义：

```text
Delta_shuffle(t)
= MSE(v_D(s_t,t,z_shuffle), target)
  - MSE(v_D(s_t,t,z_real), target)
```

若 `Delta_shuffle(t) > 0`，说明该时刻正确 latent 提供了样本级有用信息。

### 4.2 结果

- 责任曲线是真实的，5 个 seed 都稳定。
- 16d 的相对收益更集中在高噪声早期。
- 64/256d 在中低噪声仍提供更多实例和细节信息。
- 同类别打乱也变差，说明 latent 不只携带十分类别标签。

但责任曲线没有比普通 validation velocity MSE 更好地预测最终 feature FID。因此：

> “decoder 何时使用 latent”是真现象，但“使用得更久”不等于“更适合完整生成”。

这轮没有训练 latent prior，因为原预注册把预测力设为进入下一阶段的门槛。

## 5. 第二轮：真正闭合两阶段生成

责任曲线的预测假设失败后，我们把问题重新预注册为一个独立问题：

> latent 给 decoder 的收益，是否会和 learned prior 生成这种 latent 的系统代价形成权衡？

### 5.1 公平控制

- `16/64/256d x 5 seeds`，共 15 个冻结 encoder/decoder。
- 每个配置训练同一个 7.096M 参数 latent flow prior。
- 三种容量都嵌入统一 256d 正交接口。
- 低维 latent 的高斯源也限制在同一低维子空间，避免从满秩高斯压到奇异流形。
- 同 seed 的网络初始化、batch index、time 和基础噪声全部配对。
- prior 只读 Imagenette train latent；val 只评估，没有数据泄露。

### 5.2 主结果

| latent | Oracle | Empirical | End-to-end | Modeling gap |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 101.03 | 100.65 | 103.44 | 2.79 |
| 64 | 98.25 | 96.53 | 106.44 | 9.92 |
| 256 | 97.64 | 96.02 | 118.44 | 22.42 |

解释：

- 真实 code 下，256d 比 16d 好 `3.39`，说明更多 latent 信息确实帮助 decoder。
- learned prior 替代真实 code 后，256d 的额外代价比 16d 多 `19.63`。
- 方向在 `5/5` seed 一致。
- 原先预期的 64d 中间最优没有出现，完整生成是 `16d < 64d < 256d`。

### 5.3 最反常的结果

```text
held-out prior flow MSE: 16d .978, 64d .979, 256d .451
raw latent SWD:          16d .0299, 64d .0594, 256d .0298
```

256d 的普通 prior loss 最低，raw SWD 也不差，却有最大的 decoded gap。因此不能说：

```text
高维 -> prior 在普通意义上学不会 -> 图像变差
```

更准确的说法是：普通 latent 平均指标没有测到 decoder 最终在意的误差。

## 6. 第三轮：decoder 是否特别放大 prior 错误

### 6.1 等角局部敏感性

decoder 先把 latent 映射到 192d condition embedding。我们在这个球面上，从每个真实
condition 出发，构造三个同为 `0.15 rad` 的方向：

- 指向匹配的 prior condition。
- 指向另一个真实 condition。
- 随机切向方向。

若 prior error 恰好落在危险方向上，相同角度的 prior 方向应比真实方向引起更大输出变化。

结果并非如此：prior/empirical response 在 256d 约为 `.988`。所以“prior 偏到局部
高敏感方向”这一简单 Jacobian 解释被否定。

### 6.2 输入空间与输出空间的反序

用分类器区分 empirical 与 prior：

| latent | condition-space AUC | decoded-feature AUC |
| ---: | ---: | ---: |
| 16 | .527 | .498 |
| 64 | .819 | .602 |
| 256 | .574 | .637 |

64d 的输入分布差异最显眼，256d 的输出分布差异最大。这说明 decoder 没有凭空增加
理想统计散度，而是把普通有限维观察器不容易看到的差异，变成当前图像特征容易看到的
差异。

类别质量重加权只能收回 256d gap 的约 `15.9%`；剩余差异主要在 feature covariance、
类别内覆盖和多样性，不是简单的类别比例偏移。

## 7. 第四轮：为什么没有直接训练 decoder-aware loss

文献已经表明，直接把冻结 decoder 的中间特征用于 diffusion loss 是可行的。因此我们
原本设计了：

```text
L = L_flow + lambda_pair L_pair + lambda_dist L_dist
```

- `L_pair`：同一个训练样本的预测 clean latent 与真实 latent，经冻结 decoder 后做
  逐样本 feature matching。这接近 LPL 强基线。
- `L_dist`：匹配一批 predicted/real latent 在 decoder 中间 response 的分布。这是我们
  想检验的候选新增部分。

但训练前必须先验证：`L_dist` 测到的差异是否真的预测最终 modeling gap。否则它可能
优化一个与最终图像无关、甚至排序相反的 proxy。

## 8. 第五轮：Frozen Decoder Response Atlas

### 8.1 实验究竟做了什么

对 empirical、另一组独立 empirical control、prior 三种 latent，使用相同 pixel noise，
分别沿 decoder 自己的 50 步轨迹运行。在 `t=0.9/0.5/0.1` 记录：

```text
condition, down0, down1, down2, middle,
up2, up1, up0, velocity
```

对每个状态和层，同时运行零 latent，定义 latent 的条件贡献：

```text
R_l(s_t,t,z) = h_l(s_t,t,z) - h_l(s_t,t,0)
```

空间激活先固定池化，再使用预先固定的高斯投影压到最多 128d。随后比较：

- prior response 与 empirical response 的 normalized Frechet/Bures、SWD、均值和协方差。
- 两组独立 empirical response 的距离，作为有限样本统计下限。
- matched latent 与 shuffled latent 的 velocity MSE，确认 decoder 接线和样本依赖。

最后不是看单个相关，而是要求该指标同时能：

- leave-one-seed-out：外推到未见 seed。
- leave-one-dimension-out：外推到未见 latent 容量。

### 8.2 为什么需要 independent empirical floor

有限样本下，即使两批数据来自同一真实分布，Fréchet/SWD 也不为零。若
`prior-vs-real` 没有显著超过 `real-vs-real`，就不能把差异解释为 prior 机制。

### 8.3 结果

256 样本时没有任何相邻层同时通过两个 held-out protocol。扩到每组 1024 样本后，
所有 leave-one-dimension-out Spearman 仍为负：

```text
down0 -0.479, down1 -0.671, down2 -0.439, middle -0.468,
up2 -0.907, up1 -0.682, up0 -0.425, velocity -0.475
```

固定时刻 response mismatch 大致排序是 `64d > 256d > 16d`，最终 modeling gap 却是
`256d > 64d > 16d`。最关键的 proxy 顺序反了。

### 8.4 为什么这不是 decoder 忽略 latent

shuffled/matched velocity MSE 比值为：

```text
t=0.9: 6.867x, 15/15 run 更差
t=0.5: 3.489x, 15/15 run 更差
t=0.1: 1.020x, 15/15 run 更差
```

decoder 在高、中噪声明显使用正确 latent。低噪声时当前 pixel state 已携带大量信息，
latent 的边际作用自然减弱。

### 8.5 为什么静态 response 会失败

response atlas 测的是每个时刻的边际：

```text
Law(R_l(s_t,t,z))
```

最终图像依赖整条耦合轨迹：

```text
s_0 = s_1 + integral v_D(s_t,t,z) dt
```

即便每个时刻的均值和协方差接近，以下量仍可不同：

- response 与当前 state 的联合关系。
- 同一样本在不同时间的误差相关性。
- 小误差经过非线性状态反馈后的累计方向。
- 类别内部 mode、空间结构和纹理的高阶组合。

例如：

```text
Var(integral v_t dt) = double integral Cov(v_t, v_u) dt du
```

只看每个 `Var(v_t)`，没有测到 `t != u` 的跨时间协方差。因此当前数据最多支持：

> 若 prior-decoder 断层存在于 decoder 动力学中，它更可能是 state-conditioned、跨时间
> 的联合错配，而不是几个固定时刻的边际 response 错配。

这仍是解释，不是已被新实验确认的因果机制。

## 9. 哪些想法已有论文做过

### 9.1 两阶段“先生成 latent，再解码”已经很成熟

- [VQ-VAE](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html)
  明确学习离散 code 和其上的 prior。
- [VQ-VAE-2](https://proceedings.neurips.cc/paper/2019/hash/5f8e2fa1718d1bbcadf1cd9c7a54fb8c-Abstract.html)
  使用分层 latent 与强 autoregressive priors。
- [Latent Diffusion](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)
  在预训练 autoencoder latent 中训练 diffusion。

因此，“把生成拆为 `p(z)p(x|z)`”不是新意。

### 9.2 生成式或 diffusion decoder 已被充分研究

- [Diffusion Autoencoders](https://arxiv.org/abs/2111.15640)：semantic code 加 stochastic
  detail code，diffusion 负责未被语义 latent 表达的变化。
- [SODA](https://openaccess.thecvf.com/content/CVPR2024/papers/Hudson_SODA_Bottleneck_Diffusion_Models_for_Representation_Learning_CVPR_2024_paper.pdf)：
  紧 bottleneck 和条件 diffusion decoder 联合学习表示。
- [InfoDiffusion](https://proceedings.mlr.press/v202/wang23ah.html)：用互信息避免强 diffusion
  decoder 忽略 latent。
- [SWYCC](https://arxiv.org/abs/2409.02529)：连续 encoder 和 stochastic diffusion decoder
  联合训练，让 decoder 采样未压缩的细节。
- [DiTo](https://arxiv.org/abs/2501.18593) 与
  [FlowMo](https://openaccess.thecvf.com/content/ICCV2025/papers/Sargent_Flow_to_the_Mode_Mode-Seeking_Diffusion_Autoencoders_for_State-of-the-Art_Image_ICCV_2025_paper.pdf)：
  把 diffusion/flow decoder 发展为可扩展 tokenizer。
- [PiD](https://arxiv.org/abs/2605.23902)：将 VAE、DINOv2、SigLIP 等 latent 统一交给
  pixel diffusion decoder，并支持提前终止 latent diffusion。

所以，我们的小型随机 decoder 是受控实验平台，不是方法创新。

### 9.3 Prior 与编码分布不匹配是经典问题

- [Adversarial Autoencoders](https://openreview.net/forum?id=ByIrzPVc)：匹配 aggregated
  posterior 与可采样 prior。
- [Two-Stage VAE](https://openreview.net/forum?id=B1e0X3C9tQ)：再训练一个 VAE 修正
  第一阶段 latent 分布。
- [Sinkhorn Autoencoders](https://proceedings.mlr.press/v115/patrini20a.html)：把 latent
  Wasserstein matching、重构和 decoder/generator capacity 联系起来。

因此，“prior 应与 encoder 对齐”不是新意。

### 9.4 Reconstruction 与 generation 的容量权衡已经被直接研究

- [When Worse is Better / CRT](https://arxiv.org/abs/2412.16326) 系统研究第一阶段压缩与
  第二阶段建模容量的 trade-off，并用第二阶段归纳偏置正则 tokenizer。
- [VA-VAE](https://openaccess.thecvf.com/content/CVPR2025/html/Yao_Reconstruction_vs._Generation_Taming_Optimization_Dilemma_in_Latent_Diffusion_Models_CVPR_2025_paper.html)
  直接展示更高 channel 改善重构但恶化固定 DiT 的 generation，并用 VFM alignment
  改善高维 latent。
- [Improving the Diffusability of Autoencoders](https://proceedings.mlr.press/v267/skorokhodov25a.html)
  研究高 channel latent 的异常高频，并用 scale equivariance 改善下游 diffusion。
- [RAE](https://arxiv.org/abs/2510.11690) 与
  [RAEv2](https://arxiv.org/abs/2605.18324) 继续研究高维语义 latent 的可生成性、层聚合
  和训练效率。

我们的 `Oracle 改善但 end-to-end 恶化` 方向与这些工作一致，不是独立的新现象。
更特别的是：本实验里 256d 的普通 prior loss 反而最好，说明“高维平均 loss 更难”
不足以解释 decoded gap。但这目前只在小型 Imagenette 系统成立。

### 9.5 Decoder-aware、prior-aware 与后训练也已有强工作

- [LPL, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/204fee94c982a19230c39045aa54f977-Abstract-Conference.html)：
  冻结 AE，通过 decoder 中间特征给 latent diffusion 增加逐样本 perceptual objective。
- [l-DeTok, ICLR 2026](https://arxiv.org/abs/2507.15856)：让 tokenizer 能从被噪声或 mask
  破坏的 latent 重构，直接适配下游 denoising。
- [Image Tokenizer Needs Post-Training](https://arxiv.org/abs/2509.12474)：明确研究 reconstructed
  token 与 generated token 的分布差异，并在 generator 训练后调整 decoder。
- [LV-RAE](https://arxiv.org/abs/2602.08620)：研究高维信息型 latent 下 decoder 对 off-manifold
  扰动的过度响应，使用 decoder 鲁棒化和 latent smoothing。
- [Prior-Aligned Autoencoders](https://arxiv.org/abs/2605.07915)：把空间结构、局部连续性和
  全局语义直接作为 diffusion-friendly latent 目标。

所以，单纯提出“让 prior 看 decoder”或“让 decoder 适应生成 latent”已经不够新。

### 9.6 两阶段联合训练也已有明确先例

- [REPA-E, ICCV 2025](https://arxiv.org/abs/2504.10483) 发现朴素把 diffusion loss 反传给
  VAE 会让 latent 变得容易去噪却降低最终生成；用 representation alignment 才能稳定
  联合微调。
- [AlignTok, ICLR 2026](https://aligntok.github.io/) 使用冻结 encoder、联合微调并保持语义、
  最后精修 decoder 的三阶段方案。
- [UNITE](https://arxiv.org/abs/2603.22283) 用共享 Generative Encoder 从头单阶段联合
  tokenization 与 latent denoising。

因此，“encoder、prior、decoder 一起训练”不是空白，而且已有证据说明朴素全解冻会
产生 moving target 和目标投机。

### 9.7 轨迹误差与累计误差也不是空白

- [Elucidating Exposure Bias, ICLR 2024](https://arxiv.org/abs/2308.15321) 研究训练状态与采样
  状态不一致。
- [On Error Propagation, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b465dd58ac50e1b0b22894fd581f62f-Abstract-Conference.html)
  区分单步误差和累计误差，并将累计误差用于正则化。
- [FlowConsist](https://arxiv.org/abs/2602.06346) 直接研究 flow 的真实轨迹一致性和跨步误差
  累积。

所以，“固定时刻不够，应该看整条轨迹”也不能单独成为贡献。若继续，必须把问题收窄为
learned latent prior 与 stochastic decoder 之间特有的、可干预的 pathwise interface
mismatch，并证明它不同于普通 diffusion exposure bias。

## 10. 我们的实验与文献到底是什么关系

| 我们做的事 | 最接近文献 | 是否已被覆盖 | 当前价值 |
| --- | --- | --- | --- |
| 小 latent + stochastic decoder | DiffAE/SWYCC/DiTo | 架构思想已覆盖 | 受控平台 |
| 训练 learned latent prior | VQ-VAE/Two-Stage VAE/LDM | 已覆盖 | 闭合系统所必需 |
| 16/64/256 容量 trade-off | CRT/VA-VAE | 主问题已覆盖 | 独立小模型复现 |
| Oracle/Empirical/Prior/Gaussian 四路分解 | 相近工作常分开报告 rFID/gFID | 未找到完全相同的标准协议 | 诊断设计有用 |
| noise responsibility curve | CFG/DiffAE/SODA 相邻 | 未找到相同容量、shuffle、跨时间协议 | 有趣但无独立预测力 |
| 等角 decoder 敏感方向 | Latent Space Oddity/LV-RAE 相邻 | 未找到完全相同干预 | 否定简单 Jacobian 机制 |
| input/output C2ST 反序 | decoder distribution-shift 工作相邻 | 未找到相同反序实验 | 候选现象，外部效度有限 |
| layerwise response atlas 预测 endpoint gap | LPL 使用 decoder 层作训练目标 | 未找到先做 held-out predictive gate 的同样协议 | 可靠负结果 |
| batch-level response distribution loss | LPL 是逐样本 feature loss | 分布版候选未获证据 | 当前应停止 |
| pathwise joint mismatch | exposure bias/error propagation/FlowConsist 相邻 | 一般问题已覆盖 | 必须提出接口特有证据才有新意 |

“未找到完全相同”只表示在本轮检索的生成式 autoencoder、latent diffusion、tokenizer、
decoder robustness 和 flow trajectory 文献中未发现同样协议，不等于数学上证明从未有人做过。

### 文献成熟度

截至 2026-07-22，本文引用中已有正式同行评审版本的关键工作包括 VQ-VAE、VQ-VAE-2、
Latent Diffusion、Diffusion Autoencoders、InfoDiffusion、SODA、LPL、VA-VAE、
Improving the Diffusability of Autoencoders、FlowMo、REPA-E、CRT、RAE、l-DeTok 和
AlignTok。

SWYCC、DiTo、Image Tokenizer Needs Post-Training、LV-RAE、PAE、RAEv2、PiD、UNITE
和 FlowConsist 在本轮核查时应按公开预印本或技术报告看待。它们的性能数字需要保持
审慎，但其已公开的方法主张仍然会影响论文新颖性判断。

## 11. 当前证据应分成四档

### 已确认

- latent 容量增加改善 empirical/Oracle decoder 上限。
- 相同预算 prior 下，decoded modeling gap 随容量稳定增大。
- decoder 在高、中噪声读取样本级 latent。
- 普通 latent flow loss、SWD、effective rank 不能解释 256d 的最终 gap。

### 已否定

- 64d 是完整系统的中间最优。
- 256d 只是因为普通 prior loss 更差。
- prior error 特别对齐局部高敏感 decoder 方向。
- 固定时间、固定层的 response marginal discrepancy 能跨容量预测 endpoint gap。

### 只有间接支持

- decoder 把 latent 中不显眼的细粒度/多样性偏差显露到图像空间。
- 真正问题可能在 state-conditioned、cross-time joint mismatch。

### 还不知道

- pathwise mismatch 是否真是因果机制。
- LPL 式逐样本 loss 能否改善这个小系统。
- decoder post-training 或联合训练能否缩小 gap，而不损害 Oracle。
- 这些现象是否能外推到 ImageNet、SD-VAE、RAE 或大型 DiT。

## 12. 论文价值的客观判断

当前结果不能单独形成 ICLR 方法论文，原因不是实验不严谨，而是：

1. 架构思想、容量权衡、decoder-aware loss、decoder post-training 和联合训练都有强先例。
2. 最独特的 response atlas 得到的是负结果，没有产生被验证的新方法。
3. 核心证据来自 Imagenette-64 小系统，评价是内部 ResNet feature FID，外部效度有限。
4. “可能是 pathwise mismatch”与 exposure bias、error propagation 和 trajectory consistency
   文献高度相邻，目前只有推断，没有接口特有的因果干预。

它目前最有价值的产出是一个严格的研究边界：

> 两阶段生成的接口误差不能仅由 latent 平均距离或 decoder 若干静态 response 边际概括；
> 在提出 decoder-aware 分布损失前，必须证明该 proxy 能跨容量预测最终生成差距。

这是一条很好的负机制结论和实验方法论，但还不是一个完整论文贡献。

## 13. 当前停止决定为什么合理

按照实验前写好的规则，response atlas 的主预测门槛失败，1024 样本、替换 projection
seed、独立 Fréchet 实现、trace 位级一致性和 shuffled control 均未发现代码问题。因此：

- 不训练原计划的 batch-level response distribution loss。
- 不用解冻 decoder 或 encoder 事后“救”这个假设。
- 不把 LPL 已有方法的可能成功说成我们指标的成功。
- 不把 pathwise mismatch 从解释直接升级为结论。

这不是整个 prior-decoder 问题被证明不存在，而是当前这条具体训练目标已经有足够反证，
应当停止。

## 14. 外部模型验证

当前 Imagenette-64 结论不能直接外推到公开两阶段生成模型。公开 checkpoint 的候选筛选、
统一对照协议和停止标准见
[EXTERNAL_PRIOR_DECODER_VALIDATION_ZH.md](EXTERNAL_PRIOR_DECODER_VALIDATION_ZH.md)。

优先顺序是 DiffAE FFHQ-128、D2C FFHQ-256、DC-AE f32/f64。DC-AE 1.5 已公开报告
“重构改善但生成变差”的相邻现象，因此本项目的小模型结果应定位为受控复现；真正未解决
的是该现象能否被统一的 prior-decoder 接口指标解释和干预。
