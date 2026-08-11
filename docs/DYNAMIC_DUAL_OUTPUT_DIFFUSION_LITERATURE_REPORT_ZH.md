# Dynamic Dual-Output Diffusion Models 文献审计、SiT 迁移与正式结果

> 调研日期：2026-08-10
>
> 核心对象：Benny 与 Wolf，CVPR 2022，*Dynamic Dual-Output Diffusion Models*
>
> 调研范围：原论文、补充材料、arXiv 原始 TeX、论文引用的 38 项工作、Semantic Scholar 检索到的 37 条后续引用记录，以及与 prediction target、few-step sampling、flow matching 和 guidance 最接近的工作
>
> 当前实验底座：ImageNet-100、SD-VAE latent、SiT-S/2、linear flow matching、4 张 RTX 4090

> 正式实验更新：2026-08-11；双输出模型已训练到 450K，并完成 400K/450K EMA checkpoint 的无 guidance 5K FID 评估

## 1. 结论先行

这篇论文值得严格复现，但不能把“同一网络同时预测两个目标”当作我们的创新点。

最重要的结论有六条：

1. DDO 已经在 2022 年实现了同一 backbone 同时预测 `epsilon`、`x0` 和逐样本、逐空间位置的混合系数 `r`。
2. DDO 的核心不是简单平均两个 clean prediction，而是在同一个 reverse-step mean 空间中学习插值。
3. DDO 只研究两个分支之间的插值，没有系统研究越过 `x0` 分支的外推，也没有把分支差异解释为 guidance direction。
4. 2023 年的 simultaneous image/noise estimation 又做了联合 `x0 + epsilon` 预测；2026 年的 Self-Consistent Flow 更直接地在 rectified flow 中联合 `endpoint + velocity`，并加入严格一致性损失。因此“给 SiT 加一个 x-head”本身已经不新。
5. SC-Flow 与我们当前 ImageNet-100 SiT 底座最接近。它已经报告：endpoint 监督利于训练，velocity 输出利于数据端附近的稳定采样，二者通过 consistency loss 联合可显著加快收敛。
6. 仍未被这些工作直接回答的问题是：**两个有限模型 prediction target 的分歧，除了应被插值或消除，能否作为有用的闭环 guidance；在什么条件下可以越过较强分支外推，而不会被无关误差主导？**

因此推荐顺序不是直接宣称新方法，而是：

```text
完成当前 SiT-S/2 baseline
  -> 严格复现 v-only / x-only / naive dual / SC-Flow
  -> 确认双目标收益来自哪里
  -> 再测试 disagreement interpolation/extrapolation
  -> 最后才考虑 learned policy 或 rollout-aware objective
```

### 1.1 正式实验后的结论更新

当前 SiT 迁移实验**没有复现原论文中 dynamic path 优于两个单分支的结果**。
在 450K EMA checkpoint、同初始噪声、同标签和同评估 reference 下：

| 450K 路径 | FID | sFID | IS |
|---|---:|---:|---:|
| only epsilon | **70.7303** | **68.4541** | 24.3442 |
| dynamic | 71.7942 | 68.9646 | 24.7589 |
| only x | 72.8160 | 69.6784 | **24.8607** |

这与原论文 ImageNet 实验的分支排序不同。原文在无 classifier guidance、50 个
DDIM steps 下得到 `dual 25.3 < only x 27.4 << only epsilon 49.1`；当前 latent
flow 中则是 `epsilon < dynamic < x`。因此，原论文赖以获益的 target regime 没有
直接迁移过来。

与此同时，当前实现确实复现了原论文的定性 gate 规律。450K EMA validation 中，
高噪声端的平均 x-head 权重为 `0.9466`，低噪声数据端仅为 `0.0370`。所以这不是
“gate 完全没有学会切换”的简单失败，而是：

> **逐步误差意义下正确的切换规律，并没有转化成当前 flow rollout 的最佳 FID。**

原论文自己在局限性中承认，利用一步 posterior-mean loss 学到的 greedy gate 不保证
最终图像质量最优。当前结果是这个限制在 linear-flow adaptation 中的直接实例，但还
不能据此断言原论文方法本身错误。

## 2. 调研证据与边界

本报告优先使用以下一手材料：

- [CVPR 2022 正式论文页面](https://openaccess.thecvf.com/content/CVPR2022/html/Benny_Dynamic_Dual-Output_Diffusion_Models_CVPR_2022_paper.html)
- [CVPR 2022 论文 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Benny_Dynamic_Dual-Output_Diffusion_Models_CVPR_2022_paper.pdf)
- [arXiv 2203.04304](https://arxiv.org/abs/2203.04304)
- CVPR 页面提供的 supplementary material
- arXiv source 中的 `main.tex`、`main.bib` 与补充材料 TeX

后续引用检索使用 Semantic Scholar 与 OpenAlex。2026-08-10 的快照中：

- Semantic Scholar 返回 37 条记录，其中 1 条是 supplementary material 重复记录。
- OpenAlex 返回 22 条记录。
- 引用数据库的覆盖范围会变化，因此数量不是论文结论；本报告保留完整标题索引，并对真正技术相关的论文单独深读。

截至调研日期，没有找到可验证的 DDO 官方代码或 checkpoint。CVPR、arXiv 和作者论文页只提供论文与补充材料，CatalyzeX 仍显示 request code。因此后续复现必须把原文没有写清的实现选择登记为消融变量，不能假装它们已有唯一官方答案。

## 3. DDO 到底做了什么

### 3.1 两种标准预测方式

DDPM 的反向一步写成：

\[
p_\theta(x_{t-1}\mid x_t)
=\mathcal N\bigl(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2I\bigr).
\]

真实 posterior mean 为：

\[
\tilde\mu_t(x_t,x_0)
=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.
\]

网络可以直接预测 clean image `x0`：

\[
\mu_x(x_\theta)
=
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_\theta.
\]

也可以预测 noise `epsilon`：

\[
\mu_\epsilon(\epsilon_\theta)
=
\frac{x_t}{\sqrt{\alpha_t}}
-
\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}\sqrt{\alpha_t}}
\epsilon_\theta.
\]

原文把前者称为 additive path，把后者称为 subtractive path。

### 3.2 为什么两条路径各有优缺点

论文的解释是：

- 在高噪声阶段，直接 `x0` 预测更容易快速给出低偏差的整体图像；从 `epsilon` 恢复 `x0` 会放大噪声预测误差。
- 在低噪声阶段，`epsilon` 分支更像 residual correction，只需去掉剩余小噪声；直接 `x0` 分支每步都要重建完整图像。
- 两条路径并不只是数值重参数化后的同一条有限模型轨迹。有限训练下，它们会形成不同的中间状态与最终图像。

这与我们 prediction-target toy 的核心观察相容：population optimum 可以等价，有限网络的 target conditioning、输出维度、误差放大和递归 rollout 却不同。

### 3.3 三输出结构

DDO 一次 forward 输出：

\[
\epsilon_\theta,\quad x_\theta,\quad r_\theta
=f_\theta(x_t,t).
\]

然后在 reverse mean 空间混合：

\[
\mu_\theta
=r_\theta\mu_x(x_\theta)
+(1-r_\theta)\mu_\epsilon(\epsilon_\theta).
\]

对于 `H x W x C` 图像，最后一层从 `C` 个输出通道变成 `2C+1`：

- `C` 通道预测 `epsilon`；
- `C` 通道预测 `x0`；
- `1` 通道预测空间 gate `r`。

因此 `r` 不是每个时刻只有一个标量，而是每张图、每个空间位置都可以不同。论文画出的曲线只是对空间与样本求均值后的结果。

### 3.4 三项损失与 stop-gradient

原文训练目标为：

\[
L_t^\epsilon=\|\epsilon-\epsilon_\theta\|^2,
\]

\[
L_t^x=\|x_0-x_\theta\|^2,
\]

\[
L_t^\mu=
\left\|
\tilde\mu_t-
\left(
r_\theta[\mu_x(x_\theta)]_{\rm sg}
+(1-r_\theta)[\mu_\epsilon(\epsilon_\theta)]_{\rm sg}
\right)
\right\|^2,
\]

\[
L_t=L_t^\epsilon+L_t^x+L_t^\mu.
\]

三个权重在所有正式实验中都设为 1。

最容易漏掉的细节是：`L_mu` 中的两个分支都 stop-gradient。也就是说：

- `L_epsilon` 训练 noise head；
- `L_x` 训练 clean head；
- `L_mu` 主要训练 gate `r`；
- gate 不会通过 `L_mu` 把两个 endpoint predictor 拉向某个容易混合的解。

作者明确说，不 detach 时，`mu_x` 与 `mu_epsilon` 的 Jacobian 带来强烈的 timestep rescaling，训练不稳定。这一点与我们此前对分母、detach 和时间放大的审计高度相关。

### 3.5 原文没有写清的关键实现

原论文和 arXiv TeX 都没有明确写出 `r_theta` 使用 sigmoid、clamp 还是无约束线性输出。TeX 中也没有 `sigmoid`、`softmax` 或 `clamp` 的说明。

论文的语言把它称为 interpolation，图中的均值也位于 `[0,1]`，但这不能替代实现证据。严格复现必须至少报告：

| gate 版本 | 定义 | 作用 |
|---|---|---|
| bounded | `r=sigmoid(a)` | 严格限制在两个分支之间 |
| clipped | `r=clip(a,0,1)` | 有边界但存在饱和梯度 |
| unbounded | `r=a` | 自动允许 interpolation 与 extrapolation |

如果不做这个消融，任何 `r>1` 的结论都可能只是误猜原实现。

### 3.6 DDO 优化的是一步，不是整条轨迹

`L_mu` 让当前 `x_t` 上的混合 mean 接近训练配对的 `tilde_mu_t`。它没有直接优化最终 FID，也没有对完整 rollout 求梯度。

原论文在局限性中主动指出：这种逐步 greedy 选择不保证最终图像最优，未来可考虑 beam search。这个限制非常重要，因为我们已经在 RAEv2 和 toy 中反复看到：

```text
teacher-forced 单步误差最优
不等于 rollout latent 分布最优
也不等于 decode 后 FID 最优
```

## 4. 原论文实验审计

### 4.1 训练与评估协议

补充材料给出的主要设置：

| 数据集 | 架构与训练 | 评估 |
|---|---|---|
| CIFAR-10 linear | DDPM 风格 U-Net，base 128，channel multiplier `[1,2,2,2]`，1M iterations，global batch 512，Adam `2e-4`，EMA `0.9999` | 5/10/20/50/100 steps，50K samples |
| CIFAR-10 cosine | Improved DDPM 风格，cosine schedule | 同上，多种采样步数 |
| CelebA 64 | 更深一层 U-Net，500K iterations，Adam `1e-5` | 5/10/20/50/100 steps，50K samples |
| ImageNet 128 | ADM class-conditional backbone；只加载官方 encoder，其余 decoder/residual 训练 80K；global batch 128，Adam `1e-5` | 25/50 steps，50K samples |

训练使用 4 张 RTX 2080 Ti。FID 使用 `torch-fidelity`。ImageNet 的 precision/recall 使用 10K real 与 50K generated，FID 使用 50K validation 与 50K generated。

### 4.2 CIFAR-10 与 CelebA 结果

线性 schedule 的关键 FID：

| 数据集/方法 | 5 steps | 10 | 20 | 50 | 100 |
|---|---:|---:|---:|---:|---:|
| CIFAR DDIM | 49.70 | 18.57 | 10.87 | 7.03 | 5.57 |
| CIFAR DDO | **35.12** | **11.68** | **8.62** | **6.68** | 5.54 |
| CIFAR fixed `r_t` | 38.50 | 12.08 | 8.71 | 6.89 | 5.57 |
| CIFAR only epsilon | 41.99 | 12.30 | 8.74 | 7.11 | 6.01 |
| CIFAR only x | 45.53 | 24.27 | 16.93 | 12.47 | 7.39 |
| CelebA DDIM | 56.16 | 16.90 | 13.38 | 8.80 | 6.15 |
| CelebA DDO | **26.22** | **14.96** | **8.74** | **5.54** | **4.07** |
| CelebA only epsilon | 64.82 | 27.53 | 12.64 | 9.03 | 8.68 |
| CelebA only x | 29.79 | 16.03 | 9.18 | 6.57 | 4.23 |

这些数字支持三点：

- DDO 的收益在 few-step 区域最明显。
- 固定平均 gate 不如逐样本动态 gate。
- `x` 与 `epsilon` 谁更好依赖数据集、schedule 与步数，不能把一个 target 永久叫 strong head。

### 4.3 ImageNet 结果应怎样解读

ImageNet 128、无 classifier guidance、25/50 steps：

| 方法 | 训练步数 | FID 25 | FID 50 |
|---|---:|---:|---:|
| 官方 ADM | 4.36M | 11.7 | 7.6 |
| DDO | 80K，仅部分网络 | **27.7** | **25.3** |
| only epsilon | 80K | 51.3 | 49.1 |
| only x | 80K | 29.5 | 27.4 |

classifier scale 1.0 下，DDO 为 24.5/22.1，仍优于同一 80K 训练里的两条单分支路径。

但这不是对官方 ADM 的公平超越：作者明确说明只复用了 ADM encoder，并只训练剩余部分 80K，而 ADM baseline 训练 4.36M。论文把 ADM 数字作为参考，没有声称 DDO 在该设置超过完整 ADM。

### 4.4 原论文最可靠与最薄弱的证据

最可靠：

- 相同 backbone 与训练条件下，dynamic gate 稳定优于 fixed gate 和两个单分支。
- 多数据集、多 schedule、多采样步数均有对照。
- 50K FID，不是小样本筛选数字。
- stop-gradient、固定 gate、单分支都有消融。

最薄弱：

- 没有官方代码，`r` 的约束方式不明确。
- learned gate 由一步 posterior mean loss训练，不直接对应最终 generation metric。
- ImageNet 只是部分网络短训，不能回答大模型充分训练后收益是否保留。
- 论文没有研究 `r>1`，也没有 precision-recall 随外推系数的系统曲线。
- 训练成本虽然增加很少参数，但同时监督两个完整输出；原文没有按固定训练 FLOPs 与单头 baseline 对齐。

## 5. 它引用的 38 项工作

原论文的 38 条参考文献可以分为五组。完整索引如下，编号与原文一致。

### 5.1 扩散模型与采样主干

| 编号 | 工作 | 与 DDO 的关系 |
|---:|---|---|
| 6 | Diffusion Models Beat GANs on Image Synthesis | ADM 与 ImageNet baseline |
| 9 | Denoising Diffusion Probabilistic Models | epsilon prediction 与 DDPM 基础 |
| 18 | SRDiff | diffusion super-resolution 应用 |
| 20 | Knowledge Distillation in Iterative Generative Models | 少步采样/蒸馏背景 |
| 21 | Diffusion Probabilistic Models for 3D Point Cloud Generation | 扩散应用 |
| 22 | Denoising Diffusion Gamma Models | 非高斯噪声扩展 |
| 23 | Improved Denoising Diffusion Probabilistic Models | cosine schedule 与 IDDPM baseline |
| 28 | Autoregressive Denoising Diffusion Models for Multivariate Forecasting | 时间序列扩散 |
| 30 | Image Super-Resolution via Iterative Refinement | SR3 |
| 33 | UNIT-DDPM | 图像翻译应用 |
| 34 | Deep Unsupervised Learning using Nonequilibrium Thermodynamics | diffusion 起点 |
| 35 | Denoising Diffusion Implicit Models | DDIM 与 deterministic sampling |
| 36 | Generative Modeling by Estimating Gradients of the Data Distribution | score model |
| 37 | Improved Techniques for Training Score-Based Generative Models | score training |
| 38 | Learning to Efficiently Sample from Diffusion Probabilistic Models | learned sampler |

### 5.2 生成模型背景

| 编号 | 工作 | 类型 |
|---:|---|---|
| 1 | Deep Generative Stochastic Networks Trainable by Backprop | 早期迭代生成 |
| 2 | Large Scale GAN Training for High Fidelity Natural Image Synthesis | BigGAN |
| 3 | Language Models are Few-Shot Learners | 大规模生成背景 |
| 4 | WaveGrad | 波形生成 |
| 7 | Generative Adversarial Networks | GAN |
| 11 | Progressive Growing of GANs | GAN |
| 12 | A Style-Based Generator Architecture for GANs | StyleGAN |
| 13 | Analyzing and Improving the Image Quality of StyleGAN | StyleGAN2 |
| 14 | Glow | normalizing flow |
| 15 | Auto-Encoding Variational Bayes | VAE |
| 25 | WaveNet | 自回归生成 |
| 26 | Conditional Image Generation with PixelCNN Decoders | 自回归图像模型 |
| 29 | Variational Inference with Normalizing Flows | flow |
| 31 | Improved Techniques for Training GANs | IS 与 GAN 训练 |
| 32 | MCMC and Variational Inference: Bridging the Gap | Markov chain 背景 |

### 5.3 数据与评价

| 编号 | 工作 | 用途 |
|---:|---|---|
| 5 | ImageNet | 数据集 |
| 8 | GANs Trained by a Two Time-Scale Update Rule | FID |
| 16 | CIFAR-10 and CIFAR-100 | 数据集 |
| 17 | Improved Precision and Recall Metric | precision/recall |
| 19 | Deep Learning Face Attributes in the Wild | CelebA |
| 24 | High-Fidelity Performance Metrics in PyTorch | torch-fidelity |
| 27 | On Buggy Resizing Libraries and Surprising Subtleties in FID | FID 预处理审计 |

### 5.4 理论背景

| 编号 | 工作 | 用途 |
|---:|---|---|
| 10 | Estimation of Non-Normalized Statistical Models by Score Matching | score matching 基础 |

这组引用说明 DDO 的历史定位很清楚：它建立在 DDPM/DDIM/IDDPM 的 few-step 问题上，而不是 flow matching 或 guidance 论文。后来的 rectified flow、EDM、v-prediction、AutoGuidance 都不在其 2022 年参考范围内。

## 6. 引用它的 37 条记录

下面保留 Semantic Scholar 2026-08-10 快照的全部记录，并按技术相关性分级。`A` 表示直接推进双目标机制，`B` 表示与 target/trajectory/sampling 机制相关，`C` 表示综述或将其作为背景，`D` 表示应用性或很弱的引用。

| 年份 | 级别 | 标题 | 判断 |
|---:|:---:|---|---|
| 2026 | D | SILICA: Repurposing Diffusion Priors for Joint Glass Segmentation and Depth Estimation | 下游任务引用 |
| 2026 | A | Self-Consistent Flow: Unifying Velocity and Endpoint Prediction for Rectified Flow Models | 最直接后继，必须作为主基线 |
| 2026 | D | CCDM: Continuous-Time Conditional Diffusion Model for Blind CMRI Super-Resolution | 医学超分应用 |
| 2026 | D | TS-DiT: A Diffusion Transformer Model for Time Series Forecasting | 时间序列应用 |
| 2026 | D | Turning Black Box into White Box: Dataset Distillation Leaks | 非本问题主线 |
| 2025 | D | ScanDTM | 非生成 target 主线 |
| 2025 | D | Physics-Aware Parameter Diffusion for Spatio-Temporal Continuum Fields | 应用扩散 |
| 2025 | B | LEAF: Latent Diffusion with Efficient Encoder Distillation for Aligned Features | 明确讨论 x0/noise target，但任务是医学分割 |
| 2025 | D | Generative Map Priors for Collaborative BEV Semantic Segmentation | 下游应用 |
| 2024 | D | Versatile Stain Transfer in Histopathology | 下游应用 |
| 2024 | D | Conditional Diffusion Model with Spatial Attention and Latent Embedding for Medical Image Segmentation | 下游应用 |
| 2024 | B | Lotus: Diffusion-Based Visual Foundation Model for Dense Prediction | 使用 target 选择服务 dense prediction |
| 2024 | C | Diffusion Models and Representation Learning: A Survey | 综述 |
| 2024 | D | Application of DDPMs in the Minecraft Environment | 应用 |
| 2024 | D | Spatio-Temporal Fluid Dynamics Modeling via Physical-Awareness | 应用 |
| 2024 | D | DiffSal | 应用 |
| 2024 | D | StableIdentity | 应用 |
| 2024 | D | Diffusion Enhancement for Cloud Removal | 应用 |
| 2024 | C | The Rise of Diffusion Models in Time-Series Forecasting | 综述 |
| 2024 | B | Bring Metric Functions into Diffusion Models | 训练目标改造，但非双 target |
| 2023 | D | LC-SegDiff | 医学分割 |
| 2023 | B | Distilling ODE Solvers of Diffusion Models into Smaller Steps | few-step sampling/solver 蒸馏 |
| 2023 | B | Elucidating the Exposure Bias in Diffusion Models | 训练路径与 rollout 输入失配 |
| 2023 | D | Contrast-Augmented Diffusion for Markup-to-Image | 应用 |
| 2023 | D | Fuzzy-Conditioned Diffusion for Facial Image Correction | 应用 |
| 2023 | D | Non-Autoregressive Conditional Diffusion for Time Series | 应用 |
| 2023 | C | On the Design Fundamentals of Diffusion Models: A Survey | 综述 |
| 2023 | B | Image Generation with Shortest Path Diffusion | 研究 corruption path，不是 dual output |
| 2023 | D | Distribution Shift Inversion for OOD Prediction | 非生成 target 主线 |
| 2023 | B | ERA-Solver | 误差鲁棒快速采样 |
| 2022 | C | Diffusion Models in Vision: A Survey | 综述 |
| 2022 | B | On Analyzing Generative and Denoising Capabilities of Diffusion Models | 提出生成阶段到去噪阶段的功能转折 |
| 2024 | D | Testing Generated Distributions in GANs to Penalize Mode Collapse | 很弱的背景引用 |
| 2024 | D | MMGInpainting | 应用 |
| 2024 | D | Multi-Resolution Diffusion Models for Time Series | 应用 |
| 2023 | D | Text-Driven Motion Synthesis with Keyframe Collaboration | 应用 |
| - | - | Distilling ODE Solvers Supplementary Material | 与正文重复，不计独立论文 |

真正直接延续 DDO 双目标结构的后续工作很少。引用次数不低，但多数论文只是把 DDO 当作 diffusion target 或应用背景。最重要的直接后继是 SC-Flow。

## 7. 不一定引用 DDO、但更接近我们的工作

### 7.1 Simultaneous Estimation of Image and Noise，ACML 2023

[论文页面](https://proceedings.mlr.press/v222/zhang24b.html)

这篇工作同样让网络同时预测 `x0` 与 `epsilon`，但检索其正文没有发现对 Benny 与 Wolf 的引用。它还把前向过程写成 quarter-circle parameterization：

\[
x_t=\cos(\eta_t)x_0+\sin(\eta_t)\epsilon,
\]

并利用两个输出构造更稳定的 ODE gradient。它报告联合估计在低采样步数下优于单目标。

对我们的意义：即使不考虑 DDO，“联合预测 x 与 noise”也已经有第二条独立先例。新颖性必须落在 disagreement 的用途、flow 场景或闭环策略，而不是输出头数量。

### 7.2 EDM，NeurIPS 2022

[论文页面](https://papers.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)

EDM 通过 `c_skip(sigma)`、`c_out(sigma)`、`c_in(sigma)` 和 loss weighting 对网络做连续预条件。它不是两个独立预测头，也不学习逐样本 gate，但其有效输出随噪声水平在 residual preservation 与 full denoising 之间连续变化。

因此，若我们的 gate 只学习出一个几乎完全由 `t` 决定的平均曲线，它可能只是重学 EDM 式 schedule，而不是发现新的样本级机制。必须比较：

- 仅时间 gate；
- 时间加样本 gate；
- fixed analytical preconditioning。

### 7.3 JiT / Back to Basics，2025

[arXiv 2511.13720](https://arxiv.org/abs/2511.13720)

JiT 系统比较 `x`、`v` 与 `epsilon` target，强调高维环境空间中 clean target 位于低维数据结构，而 velocity/noise target 需要表达更多环境方向。它解释了为什么有限容量下 x-prediction 可能更容易。

这正是 v3/v4/v10 toy 的理论起点。但 JiT 证明 `x` target 更容易，不等于 `x-v` gap 可以继续外推。我们现有 toy 已经表明：gap 可能主要是弱 target 的无关法向误差。

### 7.4 AutoGuidance，NeurIPS 2024

[arXiv 2406.02507](https://arxiv.org/abs/2406.02507)

AutoGuidance 用同任务的 bad/weak model 与 good model 做外推：

\[
f_{\rm guided}=f_{\rm good}+\gamma(f_{\rm good}-f_{\rm bad}).
\]

它与 prediction-target extrapolation 的数学形式最近，但 weak model 的构造不同：AutoGuidance 通常使用较小、较少训练或退化版本；我们希望使用同一模型中由不同 prediction target 产生的结构化偏差。

因此必须证明 target gap 不只是任意 bad-model error。至少要比较：

- 同 backbone 双 target gap；
- early-checkpoint AutoGuidance；
- smaller-model AutoGuidance；
- 参数量和推理调用数匹配的随机弱头。

### 7.5 Prediction Target 与 Training-Free Guidance，2026

[arXiv 2607.00647](https://arxiv.org/abs/2607.00647)

该工作研究不同 prediction target 在 training-free guidance 后是否保持在数据流形附近，并报告 x-prediction 对 guidance 的误差放大更稳。它回答“哪个 target 适合作为 guidance backbone”，而不是“能否把两个 target 的 disagreement 本身用作 guidance”。

与我们相邻的边界是：如果 beyond-x 外推把样本推离流形，这篇工作的理论会直接解释失败；如果外推改善 precision 但降低 recall，则必须如实报告质量-覆盖权衡，不能只报 FID。

### 7.6 Self-Consistent Flow，TMLR 2026

[arXiv 2607.12171](https://arxiv.org/abs/2607.12171)

这是当前最重要的近邻，也是我们必须优先复现的基线。

线性 rectified flow 为：

\[
x_t=(1-t)x_0+tx_1,
\qquad
u=x_1-x_0.
\]

SC-Flow 用同一网络、binary mode embedding `b` 预测两种目标：

\[
v_\theta=\operatorname{nn}_\theta(x_t,t,b=0),
\]

\[
\tilde m_\theta=\operatorname{nn}_\theta(x_t,t,b=1).
\]

两者满足解析关系：

\[
m_\theta=x_t+(1-t)v_\theta,
\qquad
\tilde v_\theta=\frac{\tilde m_\theta-x_t}{1-t}.
\]

训练目标为：

\[
L_v=\|v_\theta-u\|^2,
\quad
L_x=\|\tilde m_\theta-x_1\|^2,
\]

\[
L_c=\|\tilde m_\theta-[x_t+(1-t)v_\theta]\|^2,
\]

\[
L=w_vL_v+w_xL_x+w_cL_c.
\]

其核心理论是：

\[
\operatorname{Var}(x_1\mid x_t,t)
=(1-t)^2\operatorname{Var}(u\mid x_t,t),
\]

所以 endpoint target 在训练中方差更低；但从 endpoint 恢复 velocity 的误差会按 `1/(1-t)^2` 放大，在接近数据端时不稳定。

它因此在采样早期使用 X-Flow，在后期切到 V-Flow，默认 `tau=0.5`。ImageNet-256 报告：

| 模型 | baseline FID | SC-Flow FID | 设置 |
|---|---:|---:|---|
| L/4 | SiT 11.53 | Mix 9.85 | 400K，CFG 4.0 |
| XL/2 | SiT 6.22 | Mix 4.19 | 800K，CFG 1.5 |
| XL/2 长训 | SiT 2.06/7M | Mix 1.86/4M | 论文配置 |

一致性权重消融中，`w_c=0` 的 FID 为 11.14，`w_c=0.1/0.2` 为 9.85/9.86。这说明：

- 单纯多任务共享有小收益；
- 大部分额外收益来自一致性约束；
- x-only 与 v-only baseline 在充分训练时几乎相同，不能只凭 target 名称预测最终 FID；
- SC-Flow 的方法目标是**消除 disagreement**，而我们的候选目标是判断 disagreement 是否还含有可利用方向。

截至调研日期，未找到可验证的 SC-Flow 官方代码仓库；论文附录提供了较详细伪代码。后续实现必须以公式和伪代码为准，并把任何工程差异写入协议。

## 8. 方法边界对照

| 方法 | 双目标 | 共享方式 | 两目标关系 | 推理使用 | 是否研究外推 |
|---|---|---|---|---|---|
| DDO 2022 | epsilon + x0 | 同 trunk，三个输出头 | 独立监督，mix loss 对两头 detach | learned spatial gate 混合 reverse mean | 否，原文只按 interpolation 叙述 |
| Simultaneous Estimation 2023 | epsilon + x0 | 联合输出 | 用于构造 ODE gradient | 联合更新 | 否 |
| EDM 2022 | 单网络预条件 | 单输出 | 解析 noise-level scaling | 连续 skip/output mix | 否 |
| AutoGuidance 2024 | good + bad model | 两模型 | 不要求一致 | good-away-from-bad 外推 | 是，但不是 prediction-target gap |
| JiT 2025 | x/v/epsilon 对照 | 独立模型/目标 | population 等价，有限模型不同 | 选更易训练 target | 否 |
| SC-Flow 2026 | velocity + endpoint | 同网络 binary mode | 显式 algebraic consistency | early X、late V | 否，主要消除 gap |
| 我们的候选问题 | velocity + endpoint | 先做共享与独立对照 | 测量而非预设 gap 的含义 | interpolation/extrapolation/window | 待验证 |

## 9. 对当前 SiT-S/2 实验的直接意义

当前 ImageNet-100 模型使用：

\[
x_t=(1-t)\epsilon+tz,
\qquad
u=z-\epsilon,
\]

网络预测 velocity `u`。这里 `z` 是 SD-VAE latent endpoint。

因此 SC-Flow 的关系可直接写成：

\[
\hat z_v=x_t+(1-t)\hat v,
\]

\[
\hat v_x=\frac{\hat z_x-x_t}{1-t}.
\]

如果在 endpoint 空间做 prediction-target extrapolation：

\[
\hat z_\gamma
=\hat z_x+\gamma(\hat z_x-\hat z_v),
\]

则等价的 velocity 为：

\[
\hat v_\gamma
=\hat v_x+\gamma(\hat v_x-\hat v).
\]

这表明我们的想法可以嵌入同一个 ODE sampler，不需要改变数据路径。但也暴露出两个风险：

- `t -> 1` 时，`v_x` 有 `1/(1-t)` 放大，必须切换、clamp 或限制窗口。
- 如果 consistency loss 把 `z_x` 与 `z_v` 完全对齐，guidance gap 会接近零；如果不加 consistency，gap 又可能主要是无关学习误差。

真正的研究问题因此不是“加不加 x-head”，而是：

> **双目标 disagreement 中是否存在一个介于“应被消除的一致性误差”和“可用于改善 rollout 分布的系统偏差”之间的可识别部分？**

## 10. 推荐复现路线

### 10.1 Stage 0：完成并冻结 baseline

先完成当前 ImageNet-100 SiT-S/2 到 800K 的训练曲线。

必须固定：

- 数据 cache 与 100 类索引；
- SD-VAE posterior sampling；
- SiT-S/2 官方 backbone source hash；
- global batch 256、learning rate `1e-4`、EMA `0.9999`；
- unguided 5K screening protocol；
- 最终候选的 50K evaluation protocol。

如果 baseline 自身还在快速变化，不能拿不同训练进度比较双目标方法。

### 10.2 Stage 1：严格 target baseline

在同一个训练入口中比较：

| 编号 | 配置 | 目的 |
|---|---|---|
| B0 | v-only，当前实现 | 主 baseline |
| B1 | x-only endpoint | 检查 JiT/SC-Flow target 差异 |
| B2 | shared dual，无 consistency | 分离多任务共享收益 |
| B3 | SC-Flow，`w_c=0.1` | 最新强基线 |
| B4 | two-network SC-Flow | 分离共享 backbone 收益 |

训练 step、数据、随机种子、optimizer、EMA 和采样预算保持一致。训练 FLOPs 不能隐藏：SC-Flow 训练每 batch 需要两个 mode forward，应同时报告固定 step 与近似固定 FLOPs 两种比较。

Stage 1 的最低验收标准：

- `v-only` 复现当前 baseline 的 checkpoint FID 曲线；
- `x-only` 与 `v-only` 的公式通过 toy identity test；
- `L_c=0` 与 `L_c>0` 的实现能在人工数据上恢复解析一致性；
- 5K FID 改善至少跨两个训练 checkpoint 同号；
- 若准备声称方法收益，至少补 3 seeds 或 50K 正式评估。

### 10.3 Stage 2：复现 DDO 的 learned mixing 思路

DDO 与 flow matching 不完全相同，不能直接把其公式原样搬来。可在 current flow 中定义：

\[
v_r=r\,v_x+(1-r)v_v.
\]

需同时比较：

- `r(t)` 仅依赖时间；
- `r(x_t,t)` 逐样本标量；
- `r(x_t,t,h,w)` 逐 token gate；
- bounded 与 unbounded gate；
- gate loss 对两个 predictor detach 与不 detach。

但这里首先是 DDO-inspired baseline，不应冒充原论文严格复现。真正严格的 DDO 复现应另在 CIFAR-10 DDPM/DDIM 设置完成。

### 10.4 Stage 3：prediction-target disagreement

只在 B0-B3 都跑通后测试：

\[
v_\gamma=v_x+\gamma(v_x-v_v).
\]

预注册扫描：

```text
gamma = -0.5, -0.2, -0.1, 0, 0.05, 0.1, 0.2, 0.5
window = full, early, middle, late
```

其中：

- `gamma < 0` 是向 velocity head 插值；
- `gamma = 0` 是 endpoint-derived velocity；
- `gamma > 0` 才是 beyond-endpoint extrapolation。

必须加入三类 control：

- AutoGuidance 的 early-checkpoint 或 smaller-model weak predictor；
- 具有相同 gap RMS 的随机方向；
- 只保留 gap 在主成分/低频/高频子空间的版本。

如果 prediction-target gap 不能优于这些 control，就不能称其具有特殊结构。

### 10.5 Stage 4：只有出现稳定收益才学 policy

learned `gamma(x_t,t)` 或 rollout-aware gate 属于最后一步，而不是起点。它需要：

- held-out trajectory 训练；
- 固定 base generator；
- 不使用 evaluation labels 或 FID feature 反传；
- 严格 train/val/test seed 分离；
- 与时间-only schedule 比较。

否则 learned gate 很容易把 validation metric sweep 伪装成方法。

## 11. 指标与验收标准

### 11.1 机制指标

- `L_v`、`L_x`、`L_c` 的 timestep curve；
- `||v_x-v_v||` 的 timestep curve；
- `v_x-v_v` 与 paired velocity residual 的 cosine，仅作解释；
- endpoint-derived velocity 在 `t -> 1` 的误差放大；
- trajectory straightness、NFE、ODE solver rejection；
- gate 的均值、方差、空间熵与样本间变异。

### 11.2 生成指标

筛选阶段：

- 固定 5K seeds 的 FID、sFID、IS；
- 相同 reference 与 ADM TensorFlow evaluator；
- decoded precision/recall 或至少 KID bootstrap；
- 同 seed 图像网格只作辅助。

正式阶段：

- 50K FID/sFID/IS；
- precision 与 recall；
- 至少 3 个训练 seed，或明确报告单 seed 局限；
- 相同 NFE、相同 CFG、相同采样器与相同 VAE decoder。

### 11.3 继续条件

Stage 3 只有同时满足以下条件才值得成为主线：

1. `gamma > 0` 相对较强单分支和 SC-Flow baseline 的 FID 改善至少 5%，且跨两个 checkpoint/多数 seed 同号。
2. 改善不能仅由 precision 上升、recall 严重下降解释。
3. 5K screening 的候选在 50K evaluation 中方向一致。
4. prediction-target gap 优于 matched-RMS random direction，并不弱于简单 AutoGuidance baseline。
5. 最优 gamma/window 在 held-out seed 上可迁移，不是逐 checkpoint 扫参。

若结果只是 SC-Flow 改善而外推无效，结论仍然清楚：应把双目标当训练正则与数值切换，不应把 disagreement 强行解释成 guidance。

## 12. 当前最稳妥的研究目标

短期目标：

> 在当前可控的 ImageNet-100 SiT-S/2 上，严格区分 endpoint supervision、shared multi-task、algebraic consistency、inference switching 与 disagreement extrapolation 各自的贡献。

中期可能形成的方法问题：

> 当两个理论等价、有限模型不等价的 prediction targets 产生分歧时，应该消除、切换、插值还是外推该分歧？能否根据闭环 rollout 风险而不是 teacher-forced 一步误差作选择？

这比“我们也做双头”更有研究价值，也与现有 toy 中发现的 three-regime 现象直接相连：

- weak head 太好，gap 接近零；
- weak head 适度退化，gap 可能包含有用 bias；
- weak head 灾难性退化，gap 被无关 ambient error 主导。

但这仍是待验证假设。SC-Flow 已经显著压缩了 novelty 空间，因此第一篇真实模型实验必须把它作为主 baseline，而不能只与 v-only SiT 比较。

## 13. ImageNet-100 SiT 双输出实现与 smoke 审计

### 13.1 当前实现的边界

当前代码是在已复现的 SiT-S/2 linear flow 上实现 **DDO-inspired flow
adaptation**，不是把原论文的 DDPM reverse mean 公式伪装成原样复现。保持不变的部分包括：

- ImageNet-100 类别、训练/验证划分与 latent cache；
- `stabilityai/sd-vae-ft-mse` posterior sampling 与 `0.18215` scaling；
- 官方 SiT-S/2 backbone source revision；
- global batch 256、AdamW `1e-4`、EMA `0.9999`、BF16、TF32；
- seed 0、每个 rank 的数据顺序、posterior noise、source noise 与时间采样。

只修改 SiT 最后一个投影，使共享 backbone 输出：

\[
(\hat\epsilon,\hat x,r)\in\mathbb R^{C+C+1}.
\]

官方 SiT-S/2 末层本来计算 `2C=8` 个通道，再在 `forward` 中丢弃后一半；
双输出版本计算 `2C+1=9` 个通道。因此参数量只从 `32,617,760` 增至
`32,619,300`，增加 `1,540` 个参数，backbone 宽度和深度完全不变。

### 13.2 Flow 版本公式

当前线性路径为：

\[
x_t=(1-t)\epsilon+tx,\qquad v=x-\epsilon,
\]

其中 `t=0` 是高噪声起点，`t=1` 是低噪声数据端。两个原生输出对应的速度为：

\[
v_x=\frac{\hat x-x_t}{1-t},\qquad
v_\epsilon=\frac{x_t-\hat\epsilon}{t}.
\]

动态速度为：

\[
v_r=r v_x+(1-r)v_\epsilon.
\]

两个主头分别使用 clean MSE 与 epsilon MSE。gate 使用下面的有限 residual：

\[
L_r=\left\|rt(\hat x-x)+(1-r)(1-t)(\epsilon-\hat\epsilon)\right\|^2.
\]

它等于混合速度误差乘上共同标量 `t(1-t)`，不改变固定样本、固定时间下的
最优 gate，同时避免 `t=0/1` 的除零。和原论文一样，`L_r` 中两个预测分支都
stop-gradient；当前主版本使用 sigmoid gate。后者是登记过的实现选择，因为原文
没有公开 gate activation 的代码。

### 13.3 正确性检查

当前检查已通过：

- 人工构造完美 `x/epsilon` 预测时，两个速度及动态速度均精确恢复 `x-epsilon`；
- `t=0/1` 端点的动态速度有限且正确；
- 单独反传 gate loss 时，梯度只进入 gate 通道，不进入两个预测头；
- gate 会选择人工构造的精确分支；
- 换末层不推进全局 RNG，保证与 v-only baseline 的随机数据流可配对；
- 2 step 保存后恢复到 step 3，与连续训练 3 step 的 model、EMA、optimizer state
  逐张量 bitwise identical。

### 13.4 四卡 smoke 结果

配置：ImageNet-100、SiT-S/2、seed 0、global batch 256、4 张 RTX 4090、BF16、
`torch.compile`、200 steps。

| 区间 | step/s | image/s | 每卡峰值 allocated memory |
|---|---:|---:|---:|
| 0-50，含首次编译 | 2.43 | 623 | 3.59 GiB |
| 50-100 | 17.24 | 4,414 | 3.47 GiB |
| 100-150 | 17.68 | 4,525 | 3.47 GiB |
| 150-200 | 17.66 | 4,521 | 3.47 GiB |

200-step raw validation 已出现正确的定性门控方向：高噪声区间 `t in [0,0.2)`
的平均 x-head 权重约 `0.712`，低噪声区间 `t in [0.8,1]` 约 `0.060`，即高噪声
优先 clean prediction、低噪声优先 residual-like epsilon prediction。这个结果只证明
实现方向和优化信号合理，不能把 200-step smoke 当作正式论文复现。

按稳定速度估算，800K step 约需 12.6 小时，另加 checkpoint validation 与采样评估
时间。该段是训练启动前的 smoke 记录；本轮最终训练到 450K，并已对同 checkpoint 的
`x-only / epsilon-only / dynamic` 三条采样路径完成配对 5K 对照，正式结果见下一节。
由于 5K 结果尚未显示 dynamic 优势，目前没有继续投入 50K 评估。

## 14. ImageNet-100 SiT 正式实验结果

### 14.1 实验问题、指标与比较基础

本轮实验只回答一个问题：在同一个双输出 SiT checkpoint 中，learned dynamic path
是否比 `only x` 和 `only epsilon` 更好；以及在相同训练 step 下，它是否比原始
单头 velocity SiT 更好。

正式协议固定为：

- ImageNet-100，SD-VAE latent，输出分辨率 256x256；
- SiT-S/2，seed 0，使用 EMA 权重；
- 无 classifier-free guidance；
- 官方 SiT Dopri5 自适应 ODE sampler；
- 每个条件生成 5000 张图，类别为 100 类，标签与初始 RNG 顺序配对；
- 所有条件使用同一个 ImageNet-100 validation 5K reference；
- ADM TensorFlow evaluator，FID/sFID 越低越好，IS 越高越好。

这里的 `5K FID` 是受控筛选指标，不等于论文使用的 50K FID。不同数据集、分辨率、
样本数和 evaluator 下的 FID 数字不能横向比较绝对值；本报告只比较当前协议内部的
相对排序。

### 14.2 生成指标没有复现论文的 dynamic 优势

| 模型/路径 | step | FID | sFID | IS | 比较角色 |
|---|---:|---:|---:|---:|---|
| 原始单头 velocity SiT | 400K | **68.6537** | **68.8580** | **26.3078** | 同 step baseline |
| 双输出 dynamic | 400K | 73.8274 | 69.0210 | 24.2918 | 同 step 主对照 |
| 双输出 only epsilon | 450K | **70.7303** | **68.4541** | 24.3442 | 同 checkpoint 分支 |
| 双输出 dynamic | 450K | 71.7942 | 68.9646 | 24.7589 | 同 checkpoint 主路径 |
| 双输出 only x | 450K | 72.8160 | 69.6784 | **24.8607** | 同 checkpoint 分支 |

可以直接确认四件事：

1. 400K 到 450K 期间，双输出 dynamic FID 从 `73.8274` 改善到 `71.7942`，下降
   `2.0332`，因此模型在这段区间仍有训练收益。
2. 在完全相同的 450K checkpoint 内，dynamic 比 only x 好 `1.0218` FID，但比
   only epsilon 差 `1.0638` FID；它不是三个路径中最好的。
3. 同为 400K 时，双输出 dynamic 比原始单头 velocity baseline 差 `5.1737` FID。
4. 双输出 400K 的 FID 只接近原始单头 300K 的 `73.9829`，表现出约一个 checkpoint
   间隔的收敛滞后。这里只能称为现象，尚未证明原因是多任务竞争或模型容量不足。

因此，当前结果不支持“在 linear flow 中直接加入 DDO gate 即可改善生成质量”。

### 14.3 Gate 学会了论文描述的切换，但单步指标没有预测 FID

EMA validation 的全维 velocity MSE 为：

| step | dynamic MSE | epsilon MSE | x MSE | 高噪声 x 权重 | 低噪声 x 权重 |
|---:|---:|---:|---:|---:|---:|
| 400K | **0.8040** | 1.0125 | 1.1127 | 0.9476 | 0.0377 |
| 450K | **0.8034** | 1.0024 | 1.1055 | 0.9466 | 0.0370 |

当前时间定义中 `t=0` 是噪声端、`t=1` 是数据端。因此 gate 的行为和原论文一致：

- 生成早期、高噪声时主要使用 clean endpoint head；
- 生成后期、低噪声时主要使用 residual-like epsilon head。

而且 dynamic 在 teacher-forced velocity MSE 上明显优于两个分支。但 450K 的 FID
仍由 only epsilon 最优。400K 到 450K 的 dynamic validation MSE 只变化约
`0.0006`，FID 却改善约 `2.03`。这再次说明：

```text
训练配对上的一步速度误差
不等于实际 ODE rollout 的分布误差
也不等于 SD-VAE decode 后的 FID
```

原论文在 Discussion 中也明确承认，用下一步 posterior-mean loss 训练 gate 是 greedy
选择，并不保证最终图像质量最优。当前实验使这个限制变成了可观测结果，而不再只是
理论警告。

### 14.4 与原论文 ImageNet 结果的差异

原论文 ImageNet-1K 128x128、无 classifier guidance 的 50K FID 为：

| 原论文路径 | 25 DDIM steps | 50 DDIM steps |
|---|---:|---:|
| dual | **27.7** | **25.3** |
| only x | 29.5 | 27.4 |
| only epsilon | 51.3 | 49.1 |

原论文的关键前提是 `x` 分支远强于 `epsilon`，dynamic 再对两者取长补短。当前实验
却得到 `epsilon < dynamic < x` 的 FID 排序。也就是说，两条分支在当前 latent flow
中的有限模型偏差与原论文不同，dynamic 没有处在相同的工作区间。

另外，两套实验还有以下不能忽略的差异：

| 原论文 | 当前实验 |
|---|---|
| 离散 VP-DDPM/DDIM | continuous linear flow matching |
| 混合 reverse-step mean | 混合 endpoint-derived velocity |
| 像素空间 ImageNet-1K 128x128 | SD-VAE latent、ImageNet-100、256x256 |
| ADM U-Net | SiT-S/2 Transformer |
| 固定 25/50 DDIM steps | 自适应 Dopri5 ODE |
| ImageNet 使用预训练 ADM encoder，只训练部分网络 80K | 双输出 SiT 从头训练 450K |
| 50K samples | 5K samples |

因此，当前代码应继续称为 `DDO-inspired flow adaptation`，不能称为原论文严格复现。
它保留了 `2C+1` 输出、三项等权损失、空间 gate 和 stop-gradient，但把 reverse mean
mixing 改成了 velocity mixing。这一变化在代数上合理，却没有被原论文实验验证。

### 14.5 结果完整性检查

本轮不是由样本重复、标签错位或显存中断造成的假负结果：

- 三条 450K 路径均生成 `(5000,256,256,3)` 的 `uint8` 图像；
- 三条路径的 5000 个类别标签逐项完全相同；
- 400K dynamic 与 450K dynamic 的标签也逐项相同；
- 三个 450K 样本 NPZ 的 SHA256 不同，排除了三路重复采样；
- sampling 与 FID 子进程返回码均为 0，没有 NaN、缺样本或显存越线；
- 四卡采样峰值约 `2.78 GiB/GPU`，ADM-FID 峰值约 `7.68 GiB`；
- 400K/450K 与原始单头 400K 使用同一个 reference 和同一套 ADM evaluator。

本地可审计结果位于：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/
  runs/sit-s-2_dual-output_seed0/
  fid5k_dual-output_step400000_seed0/
  fid5k_dual-output_step450000_seed0/
  fid5k/sit-s-2_step400000_seed0/
```

仍需保留的统计限制是：当前只有一个训练 seed、每个条件只有 5000 张图。它足以作为
机制筛选的负信号，但不足以支撑论文级的普遍结论。

### 14.6 当前能确认、不能确认与最小下一步

**已经确认：**

- gate 的梯度隔离、端点公式、checkpoint 恢复和采样配对均通过测试；
- gate 学会了原论文描述的“高噪声用 x、低噪声用 epsilon”；
- 当前 dynamic 的一步 MSE 最好，但最终 FID 不是最好；
- 当前双输出模型在 400K 明显落后于同 step 单头 baseline。

**尚不能确认：**

- 不能仅凭当前结果判断原 DDO 方法失败，因为没有在其 DDPM/DDIM 设置中严格复现；
- 不能确定差距来自 velocity mixing、共享多任务优化、从头训练，还是自适应 sampler；
- 不能把 `5.17` FID 差距直接解释为多头容量竞争，当前没有固定 FLOPs/独立双网络对照；
- 不能根据 teacher-forced MSE 为 dynamic 最低，就推断更多训练最终会自动反超。

最小而有判别力的后续顺序应为：

1. 用当前 checkpoint 做固定 25/50 NFE 的 paired sampling，检查论文的 few-step 优势是否
   被自适应 Dopri5 掩盖。
2. 在 CIFAR-10 的标准 VP-DDPM/DDIM 上做低成本严格复现，确认原论文现象本身可由
   我们的实现恢复。
3. 若严格复现成功，再比较“从头训练双头”和“从充分训练单头 checkpoint 微调新头”，
   后者更接近原论文 ImageNet 的部分网络微调方式。
4. 只有某个 flow 版本在同 checkpoint 内稳定超过两个分支，并在相同训练预算下不弱于
   单头 baseline，才值得补 50K FID 和多 seed。

如果前两项仍显示 dynamic 不优于最强单分支，应停止把 DDO 直接迁移到 linear flow 的
路线，转而比较 SC-Flow 的一致性约束和解析切换，而不是继续用训练步数掩盖机制问题。

### 14.7 后续端点与 NFE 审计

现有 450K checkpoint 的逐 batch NFE、细时间网格 gate、端点放大系数、velocity
error 和分支误差 cosine 已补齐。结果表明 dynamic 比 x/epsilon 两条派生速度场更容易
积分；端点泄漏存在但较窄，而中段两个分支误差高度同向、缺少可供 gate 利用的互补性。

完整数据、方法边界与结论见
[`SIT_DUAL_OUTPUT_ENDPOINT_MECHANISM_AUDIT_ZH.md`](SIT_DUAL_OUTPUT_ENDPOINT_MECHANISM_AUDIT_ZH.md)。

## 15. 一手来源

- [Dynamic Dual-Output Diffusion Models，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Benny_Dynamic_Dual-Output_Diffusion_Models_CVPR_2022_paper.html)
- [DDO arXiv](https://arxiv.org/abs/2203.04304)
- [Improving Denoising Diffusion Models via Simultaneous Estimation of Image and Noise，ACML](https://proceedings.mlr.press/v222/zhang24b.html)
- [Elucidating the Exposure Bias in Diffusion Models](https://arxiv.org/abs/2308.15321)
- [Image Generation with Shortest Path Diffusion](https://arxiv.org/abs/2306.00501)
- [On Analyzing Generative and Denoising Capabilities of Diffusion Models](https://arxiv.org/abs/2206.00070)
- [EDM，NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)
- [AutoGuidance](https://arxiv.org/abs/2406.02507)
- [JiT / Back to Basics](https://arxiv.org/abs/2511.13720)
- [Prediction Target and Training-Free Guidance](https://arxiv.org/abs/2607.00647)
- [Self-Consistent Flow](https://arxiv.org/abs/2607.12171)
