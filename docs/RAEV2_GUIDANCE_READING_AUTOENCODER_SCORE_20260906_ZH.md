# RAEv2 基础阅读：去噪重建为何能给 score，何时不能

结论：两篇论文给出的是**特定训练目标下的后验均值／密度导数关系**，不是任意 autoencoder 回路的性质。当前 `E(D(z))−z` 没有因此获得 latent score 或 retraction 的解释。最有价值的启发是：从明确的观测噪声与 Bayes 估计问题导出向量场及尺度，并检查其局部响应是否相容；不能先有 cycle 残差再补一个密度解释。

本次只读两篇原文及已有源码／机制文档，没有加载权重、查询模型、编码图像、训练、采样或计算新 FID。归档：`R/reading_autoencoder_score_v1`（R 指 restart_20260906 数据根）；完整 URL、版本、文件字节与 SHA 在 `manifest.json`。

## 1. 原文读到哪里，真正证明什么

- Pascal Vincent，*A Connection Between Score Matching and Denoising Autoencoders*，Neural Computation 23(7):1661–1674，2011，DOI `10.1162/NECO_a_00142`。读取作者站正式排版 PDF 全部正文和唯一附录，重点 §3.2、§4.1–4.3、§5、附录 A.1–A.3。论文是理论连接，没有新增 ImageNet/FID 或 guidance 同成本实证。[正式 PDF](https://www-labs.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf)
- Guillaume Alain、Yoshua Bengio，*What Regularized Auto-Encoders Learn from the Data-Generating Distribution*，JMLR 15(110):3743–3773，2014。精读 §2–3 的 Theorem 1/2、Proposition 1、§3.2/3.5/3.6，以及附录 A–C；§4 的采样实证未逐实验复核，附录 D–F 未完整精读。独立 agent 另核了同一组关键证明；不是增加一篇论文。[正式页面](https://www.jmlr.org/papers/v15/alain14a.html)、[正式 PDF](https://jmlr.org/papers/volume15/alain14a/alain14a.pdf)

Vincent 的核心等价并非“任意 AE 天然有势”。对数据与 corruption 的联合分布，条件 kernel score 的后验均值等于平滑边缘 score：

\[
\mathbb E[\nabla_y\log q(y\mid X)\mid Y=y]=\nabla_y\log q_Y(y).
\]

把平方差展开，两种 score matching 目标的交叉项相同，只差与模型参数无关的常数。附录通过微分移入边缘积分证明；不需要在这一步借用 integration by parts。与 implicit score matching 的另一个等价才使用 §3.2 的可微性、有限二阶 score 矩和尾部通量消失条件。

其具体 DAE 对应 `tied weights + sigmoid encoder + linear decoder + Gaussian corruption + squared loss`；从所构造能量微分，才得到 `(r(y)−y)/σ²`。目标是 Gaussian Parzen 平滑密度。论文明确讨论有限 σ 的偏差／方差选择，**没有证明训练噪声 σ 本身免选**；也没有保证有限容量模型准确学到该 score。

Alain/Bengio 则在总体、非参数最优重建函数层面消除特定网络参数化限制。其 RCAE 惩罚是整个重建映射的 `||J_r||²_F`，不是只有 encoder 的 Jacobian，更不是普通权重衰减。变分必要条件为

\[
r-x=\sigma^2[J_r\nabla\log p+\Delta r].
\]

围绕 identity 的最优解族做小 σ 展开，得到 `r−x≈σ²∇log p`、`J_r−I≈σ²∇²log p`。Proposition 1 的 DAE→RCAE 近似另假设 `r_σ=x+o(1)`；一般固定 r 的展开还有 `σ²(r−x)·Δr`，不能任意删掉。

证明边界需如实保留：Theorem 2 字面只写正密度 `p∈C¹`、r 二阶可微，但附录 C 对 Hessian 渐近的展开实际使用更高阶导数和余项求导，没有完整列出统一极限控制。不能宣传为仅 C¹ 即有该 Hessian 保证。§3.6 也明确允许有限参数化残差不保守；不能把近似 score 与精确势函数混同。

## 2. 一个更干净的有限噪声机制

以下是从 Gaussian 条件核直接推导的关系；它也澄清两篇论文的适用对象。令同一空间中的 `X` 有有限二阶矩，`Y=X+ε`，`ε∼N(0,Σ)` 独立，固定 `Σ≻0`。总体平方损失最优值满足

\[
r_*(y)=\mathbb E[X\mid Y=y],\qquad
\boxed{r_*(y)-y=\Sigma\nabla\log p_Y(y)},\qquad
\boxed{J_{r_*}(y)=\operatorname{Cov}(X\mid Y=y)\Sigma^{-1}}.
\]

第一式是条件平方损失的正交分解；第二式微分 Gaussian 卷积；第三式微分后验期望并收集协方差。各向同性时 `J_r=Cov(X|Y)/σ²`，因此对称且半正定，但特征值可以大于 1。固定正噪声下即使 X 是奇异分布，其卷积密度也可使用这个关系；对未平滑原密度取 σ→0 的 ambient score 则需要额外光滑／正密度条件。

这里的自然系数来自**已知训练噪声协方差**。若实际重建误差为 `e=r_θ−r_*`，score 误差是 `Σ⁻¹e`；各向同性小噪声极限要保住一致性，需要相应范数的 `e=o(σ²)`，不是仅凭重建 loss 小。

这也不是 retraction 定理。最简单的 `X∼N(0,τ²I)` 给出 `r_*(y)=τ²/(τ²+σ²)y`，一般 `r_*∘r_*≠r_*`。有限噪声后验均值还可能位于数据支撑集之外。Retraction 至少需要在目标集合上恒等、输出落回该集合；最近点投影还需更强的几何条件。`||r−identity||²` 也不是负 log-density：score 在密度极大、极小与鞍点都可能为零。

## 3. 非线性 E 不能穿过条件期望：完整反例

记图像为 I、真实 latent 为 `A=E(I)`。即便把实际复杂图像损失换成理想 image L2，并假设 decoder 达到 `D_*(y)=E[I|Y=y]`，仍然只有

\[
E(D_*(y))=E\bigl(\mathbb E[I\mid Y=y]\bigr)
\quad\text{一般不等于}\quad
\mathbb E[E(I)\mid Y=y]=\mathbb E[A\mid Y=y].
\]

独立反例：`U∼Uniform[1,2]`，S 为独立等概率 ±1，令 `I=S√U`、`E(i)=i²`、`Y=U+σε`，固定 `σ>0`。符号对称性给出完美 image-L2 decoder `D_*(y)=0`，故 cycle 残差为 `−y`。但 `p_Y` 关于 1.5 对称，在 `y=1.5` 的 score 严格为零，cycle 残差却为 `−1.5`。这个反例具有真实 Gaussian latent corruption、总体最优 decoder、光滑非线性 E；失败无需归因于训练不足。

E 仿射是交换成立的一个充分条件；一般非线性 E 不成立。即使从 image score 变换到 latent score，也不能只套 encoder Jacobian：可逆非线性坐标变化还含 Jacobian 行列式项，降维映射更涉及纤维上的概率积分。

## 4. 与当前 RAE 实现逐项对照

本地 `external/RAEv2` commit：`8a0d238f8dc3b261aba98b217f6c79c0182e8e94`；以下三个文件工作区无修改，源码副本与 SHA 已归档。**这核实当前源码和配置，未独立核验所用 checkpoint 的完整训练 ledger。**

| 源位置 | 已核事实 | 与 score 定理的关系 |
|---|---|---|
| `src/stage1/rae.py:60`、`:74` | 每样本 `σ=.8*U(0,1)`；raw encoder token 加 `σ*randn`；随后才 stats 标准化；只在 training 启用 | 训练确有噪声；采样 `noise_tau=0` 不能解释为从未训练去噪。噪声尺度是混合且 decoder 不显式接收 σ |
| `configs/stage1/training/dinov3l-k7-imagenet.yaml` | `noise_tau=.8`，LPIPS 权重 1，`disc_start=8`，GAN 权重 .75 | 不是固定已知 σ 的同空间 latent L2 |
| `src/stage1/engine.py:106`–`:119` | pixel L1 + LPIPS；第 8 epoch 阈值后加入自适应加权 GAN generator loss | 不能先认定 `D=E[I|Y]`，更不能认定 `E∘D=E[A|Y]` |

此外，令 stats 标准差矩阵为 S，则 normalized latent 中条件噪声协方差是 `σ²S⁻²`，不是统一 `σ²I`。即使额外拥有合法 normalized latent posterior mean，也需要正确的噪声度量；此处只是条件推导，不给实际 cycle 添一个预条件器。

对于 hidden σ 混合，在允许微分移入积分、所需条件矩存在的位置，真正的边缘 score 是

\[
\nabla_y\log p_Y(y)
=\mathbb E[(A-y)/\sigma^2\mid Y=y],
\]

而同空间 L2 denoiser 的残差为 `E[A−y|Y=y]`。二者一般不能用固定 `Eσ²` 或 `E[σ²|Y]` 因子化。当前 σ 的分布可任意接近 0，不能额外默认全局逆方差矩有限。理想 encoder 的共同仿射约束使 raw normal 分量可携带 σ 的 χ² 信息（256 个 normal 维度）；**这只说明信息存在，不证明 decoder 使用它或任何投影候选有效。**

已有 [IG decoder pushforward 机制文档](RAEV2_IG_DECODER_PUSHFORWARD_MECHANISM_ZH.md) 的两个历史 5K seed bank 记录：IG 增量经 cycle 后 norm 比 1.3489/1.3538，cosine .5204/.5214，raw probe survival .1180/.1461。这里只引用历史描述量，没有重新算 payload，也不把两 bank 当严格独立确认；历史参考臂应读作 scale1。它们表明明显方向变换，不验证 score、收缩、重构偏差抵消或最近点投影；总增量放大本身也不反证“某个合法后验均值”，因为其 Jacobian 本来可以大于 1。

## 5. 可用的机制启发与当前裁决

可以继承的结构是 **观测模型 → Bayes 最优重建 → 带尺度的 score／条件协方差**。噪声和空间确定后，残差尺度、Jacobian 的对称性与半正定约束均有来源；无需把手选增益包装为理论。这些是对同一目标的结构一致性要求，并非单独通过就证明密度正确。

在理想 RAE bridge `Z_t=(1−t)X+tε` 中，若某 clean head 真是 `M_t=E[X|Z_t]`，同样直接有

\[
\nabla\log p_t(z)=\frac{(1-t)M_t(z)-z}{t^2},\qquad
J_{M_t}(z)=\frac{1-t}{t^2}\operatorname{Cov}(X\mid Z_t=z),\quad 0<t\le1.
\]

这是已有生成模型的后验均值结构，不是本轮发现的新 sampler；实际 Full/Base 及外推 IG 是否满足它，不能由名称或现有 MSE 推定。它说明未来机制应锚定**哪个条件分布、哪种噪声下的估计误差**，而不是把 `E(D(z))−z` 当现成额外 score。

本轮不据两篇论文启动 cycle guidance、重训、投影或新增性能臂。5% 公平成本 FID 目标仍未被这些基础定理保证；当前收获是一个可证明的结构入口与一个排除误用的反例。
