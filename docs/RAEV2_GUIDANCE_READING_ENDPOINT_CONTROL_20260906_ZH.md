# RAEv2 guidance 原文阅读：把终点质量变成采样中的控制目标

日期：2026-09-06。范围：三篇新增一手论文；仅阅读与 CPU 文本/源码检查，没有训练、GPU 实验或采样代码修改。本文不改变现有实验结论、冻结候选或最终 FID 目标。

**最值得保留的机制是：先规定终点需要改善的可观测量，再沿实际生成动力学学习它的剩余代价或反向敏感度。** 这样 guidance 的方向与时间变化来自终点边界问题。标量网络只是承载这个问题的参数化，不能靠“是一个势函数”自动获得质量目标。三篇文章分别提供了：随机控制与终点重加权的准确条件、可摊销的小型 h-transform、确定性 flow 的价值梯度控制。没有一篇已经证明在当前 RAEv2、相同总成本下实现真实参考 FID 降低 5%。

全文与阅读范围：

| 论文 | 实际读取的一手材料 | 本次最有用的部分 |
|---|---|---|
| **Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control**，ICLR 2025 | [arXiv 2409.08861v5 全文](https://arxiv.org/pdf/2409.08861v5)，§2–5、§7/Table 2，附录 C.4、D、E.3、G；[会议记录](https://openreview.net/forum?id=xQBRrtQM8u) | 何时终点指数重加权成立；终点梯度如何反向传播到中间时刻 |
| **DEFT: Efficient Fine-Tuning of Diffusion Models by Learning the Generalised h-transform**，NeurIPS 2024 | [官方完整 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/22d258dfbdf840ccbf266bbc545dd95f-Paper-Conference.pdf)，§2–4，附录 D、E.1、F.1、G.1–G.4 | 小网络摊销整个条件化修正；明确新目标与已有条件的区别 |
| **Value Gradient Guidance for Flow Matching Alignment**，NeurIPS 2025 | [官方完整 PDF](https://papers.nips.cc/paper_files/paper/2025/file/10b7e27c8eb9571fbbd2ae6a9f8c3855-Paper-Conference.pdf)，§3–6，附录 A、B、C；[作者实现](https://github.com/lzzcd001/vggflow/blob/main/lib/vggflow/algorithm.py) | 在不假设精确 score 的确定性动力学上定义终点控制 |

以下统一使用生成方向的时间 (s\in[0,1])：0 是噪声，1 是终点。DEFT 原文时间方向相反。不同论文的 \(\lambda\) 含义不同，数值不可横向比较。

## 1. Adjoint Matching：终点梯度是有传播方向的信号

**理论保证对象。** 论文考虑选定的基准 SDE，而非两个网络之间的形式差值：

\[
 dZ_s=[b(Z_s,s)+\sigma_s u(Z_s,s)]ds+\sigma_s dW_s,\qquad
 J(u)=\mathbb E\left[\int_0^1\tfrac12\|u\|^2ds+\ell(Z_1)\right].
\]

省略中间状态代价后，理想值函数满足
\[
 V(z,s)=-\log\mathbb E_b[e^{-\ell(Z_1)}\mid Z_s=z],\qquad
 u^*=-\sigma_s^T\nabla V.
\]

这给出明确的采样内设计：终点奖励取 \(r=-\ell\)，中间修正由未来终点奖励的条件期望决定，实际 drift 修正为 \(-\sigma_s\sigma_s^T\nabla V\)。这里的时间系数属于指定随机过程的控制几何。Girsanov 将控制能量等同于**给定相同起点的路径 KL**，并不自动得到无条件的终点密度 \(p_{\rm base}(z_1)e^{r(z_1)}/Z\)。固定随机初始分布时，最优路径还带有 \(e^{V(Z_0,0)}\) 因子；除非初值归一化不依赖 \(Z_0\)，终点倾斜有偏。原文 §4.2 正面处理了这个问题。[AM §4.1–4.2](https://arxiv.org/pdf/2409.08861v5)

**memoryless 条件解决的是上述初值问题，不是通用的 score 误差校正。** 对精确仿射桥 \(\bar Z_s=\beta_s\epsilon+\alpha_s Z_1\)，记 \(\eta_s=\beta_s(\dot\alpha_s\beta_s/\alpha_s-\dot\beta_s)\)。原文 Proposition 1 的独立性条件允许满足端点积分条件的一族噪声；不能只据“memoryless”就声称噪声系数唯一。Theorem 1 在要求训练后还能转换为任意噪声采样、包括 ODE 的更强条件下，确定规范选择 \(\sigma_s^2=2\eta_s\)。线性桥给出 \(\sigma_s^2=2(1-s)/s\)、\(b=2v_{\rm base}-z/s\)。这些转换使用精确桥边缘速度/score 假设。[AM §4.3、附录 D](https://arxiv.org/pdf/2409.08861v5)

**实际算法的价值。** AM 直接学向量控制，并没有训练一个标量 V。其 lean adjoint 从 \(\tilde a_1=\nabla\ell(Z_1)\) 出发，沿已生成轨迹向后求解
\[
 \dot{\tilde a}_s=-[D_z b(Z_s,s)]^T\tilde a_s,
\]
再回归 \(u_\theta\) 到 \(-\sigma_s^T\tilde a_s\)，轨迹与目标停止梯度。**终点误差被后续动力学的转置 Jacobian 传播回每一个时刻**，因此某个时刻的最佳修正可以和当地去噪残差不同，甚至反向。附录 E.3 Proposition 7 给出适当无限维控制空间中唯一临界点为最优控制的结果；不等于有限神经网络、离散步长和裁剪 SGD 的全局收敛。lean adjoint 也不能在任意非最优控制处被直接当成完整 cost-to-go 梯度。[AM §5、附录 E.3](https://arxiv.org/pdf/2409.08861v5)

**实验支持与成本。** Table 2 的 ImageReward 微调后 ODE 采样，AM 的 reward 系数 2500 时 HPS v2 从基准 17.89 提升到 24.12；DreamSim diversity 为 39.88，优于 DRaFT-1 的 27.39，但仍低于基准 56.53。支持的是奖励与多样性之间更好的折中，未验证 ImageNet FID。附录的 40 步、batch 40、两张 A100 80GB，1000 次 AM 更新约 156k 秒，即约 43.3 小时墙钟/86.7 GPU 小时。训练需要轨迹 rollout、基准网络的反向敏感度、终点奖励/解码反传以及控制回归，不能用“小控制器参数量”替代总成本核算。[AM Table 2–4、附录 G](https://arxiv.org/pdf/2409.08861v5)

**不能称为现成的无手调方法。** 附录 G 的实际噪声为 \(\sigma_s^2=2(1-s+h)/(s+h)\)、h=0.025：分母偏移处理起点奇异，分子偏移则是作者实测采用的终点加强；这与连续定理的精确系数有区别。训练始终取最后 10/40 个时间步，另随机取前 30 步中的 10 步，不能概括成“只训练最后 25%”；使用 \(L_{CT}=1.6\lambda^2\) 的 loss clipping，并明言常数需要调。它们不推翻控制目标，但实证成功尚未摆脱手设时间强调和稳定化常数。

**RAE 最致命缺口与正向用途。** 把实际 IG 速度代入 \(b=2v-z/s\) 不会自动产生与官方 ODE 同边缘、memoryless 的 SDE；Full/Base 也没有精确 score 保证。可迁移的是“终点敏感度沿真实 suffix 反向传播”的原则。若保持 RAE 的确定性官方轨迹，就应从该轨迹自己的离散更新推导控制，不把 AM 的 SDE 倾斜保证一起搬过来。可证伪预言：一个真正学到终点 cost-to-go 的修正，在未训练轨迹上应预测并改善终点目标的实际变化，而不仅降低某个局部匹配 loss。

## 2. DEFT：小网络有用，但必须先有新的终点条件

**理论与设计。** 指定终点似然 \(L_y(z)=p(y\mid z)\)，沿基准过程定义
\[
 h_s(z)=\mathbb E_b[L_y(Z_1)\mid Z_s=z],\qquad
 \Delta b_s=\sigma_s^2\nabla\log h_s.
\]

这使整条轨迹针对一个终点事件或软条件变化。Proposition 2.2 的后验采样保证同时要求正确的条件初始分布；原文明确该分布与无条件初始分布不同。附录 G.2 只在相应 VP/OU 条件下给出大噪声时间的初值误差衰减，不能据此忽略 RAE 有限时间 flow 的起点问题。Theorem 3.1 联结了后验路径测度、条件 denoising score matching、随机控制和正确双端点的 Schrödinger bridge。[DEFT §2–3、附录 D/G](https://proceedings.neurips.cc/paper_files/paper/2024/file/22d258dfbdf840ccbf266bbc545dd95f-Paper-Conference.pdf)

**真正实用的工程贡献是摊销。** 主方法直接学习向量 \(h_\phi\approx\nabla\log h\)，冻结大模型，以配对数据训练“目标条件 score 减预训练 score”的修正。训练不需反传大模型，也不需逐次完整模拟 SDE。网络输入包含 noisy state、预训练去噪估计、便宜的观测似然梯度；一个小标量网络学习该似然特征的时间系数。这里确有学得的时间依赖，不能简化为手调 CFG scale；但整体向量输出未被约束为某个标量的精确梯度。[DEFT §3.1–3.3](https://proceedings.neurips.cc/paper_files/paper/2024/file/22d258dfbdf840ccbf266bbc545dd95f-Paper-Conference.pdf)

**证据与真实成本。** ImageNet 550M 基准加 23M 修正网络，1000 张训练图、200 epoch、batch 16。inpainting Table 1：DEFT KID 0.29，RED-diff 0.86、ΠGDM 4.50、DPS 15.28；DEFT LPIPS 0.09。DEFT 用 100 DDIM 步，DPS/RED-diff 1000 步，DDRM 20 步。1000 张评估的含训练总时间 DEFT 5.2 小时（约 3.9 训练+1.2 采样），ΠGDM 2.83、DDRM 0.33、RED-diff 7.86、DPS 30.72 小时；单图 DEFT 4.36 秒。它相对昂贵对照节省总成本，但不是所有对照中成本最低，也不是等预算 ImageNet 无条件 FID 结果。附录 F.1 的 CT 消融中，将似然梯度输入和其残差分支换成观测反投影 \(A^*y\)，仍保留去噪估计，PSNR 从 35.81 降至 34.04；在这个配置上再移除去噪估计降至 26.62。反之，保留似然结构而仅移除显式去噪估计仍为 35.74。这支持有任务含义的条件特征与摊销结构的重要性，不能将 26.62 误读为没有观测信息的网络。[DEFT §4、附录 E.1/F.1](https://proceedings.neurips.cc/paper_files/paper/2024/file/22d258dfbdf840ccbf266bbc545dd95f-Paper-Conference.pdf)

**RAE 最致命缺口。** 现有 Full/Base 已经同 class；若把同一个 class 再称作新条件，并继续用原图加噪配对监督，理想回归目标仍是“该真实数据桥的条件去噪器减实际 G”。非精确预训练网络还会让修正吸收它本身的误差。因此在当前 sameclass、没有新终点目标的情形，DEFT 的 simulation-free 训练退回已有 bridge residual（再考虑参数化与梯度空间投影），不是新的终点质量机制。论文另有不依赖配对数据的 SOC/VarGrad 版本，但高维主实验未用它；展示主要是 3M 基准、70K 修正的 MNIST、小规模 RTX 3090 约 1 小时实验，不能把 ImageNet 主表成绩直接归给该版本。[DEFT §3.2、附录 G.3–G.4](https://proceedings.neurips.cc/paper_files/paper/2024/file/22d258dfbdf840ccbf266bbc545dd95f-Paper-Conference.pdf)

**可迁移且可证伪的洞见。** 把新信息放进有明确定义的终点观测中，可能比增加任意中间 head 特征更有效；随后学习其全部时间依赖，而非手设窗口。如果拿掉这项终点信息后，目标、无限容量最优解和 held-out 终点效果都不变，那么所谓新机制实际上仍是旧 residual projection。

## 3. VGG-Flow：确定性价值控制适合实际 RAE drift，但它不是精确密度倾斜

**最接近当前设定的理论起点。** 固定任意足够光滑的基准速度 \(b\)，不要求它是 score 或保守场，考虑
\[
 \dot Z_s=b(Z_s,s)+u(Z_s,s),\qquad
 \min_u\mathbb E\left[\frac\lambda2\int_0^1\|u\|^2ds+\ell(Z_1)\right].
\]

HJB 与最优控制为
\[
 \partial_sV+b\cdot\nabla V-\frac{1}{2\lambda}\|\nabla V\|^2=0,
 \quad V(z,1)=\ell(z),\qquad u^*=-\nabla V/\lambda.
\]

这准确说明为什么标量 guidance 可以有原理：**它是指定终点任务、指定实际动力学和指定控制能量下的值函数梯度**。时间行为由 PDE 与终点条件共同决定，不需要另加手工 gain 窗口。全局 \(\lambda\) 仍代表控制能量与奖励的折中，理论没有免除这个决策。[VGG-Flow §3–4、附录 A](https://papers.nips.cc/paper_files/paper/2025/file/10b7e27c8eb9571fbbd2ae6a9f8c3855-Paper-Conference.pdf)

**保证的边界很明确。** 论文将此称为 relaxed objective。附录 B.1 在基准 L-Lipschitz 假设下证明
\[
 W_2(q_1^u,q_1^b)^2\le e^{2L+1}\int_0^1\mathbb E_{q_s^u}\|u\|^2ds.
\]

这个上界控制相对基准的移动，不保证更接近真实数据。附录 B.2 的 KL 恒等式/上界还包括基准 score 与修正 divergence 项，不能仅由控制能量控制；原文也承认需要经验性地依赖隐式正则。因此确定性 \(L^2\) 控制不等于随机路径 KL，不能宣称最优终点恰为 \(q_1^b e^{-\ell}/Z\)。这一点不影响 HJB 对它自己所定义控制问题的意义。[VGG-Flow 附录 B](https://papers.nips.cc/paper_files/paper/2025/file/10b7e27c8eb9571fbbd2ae6a9f8c3855-Paper-Conference.pdf)

**论文实际学的仍然是向量，不是标量 V。** 其 \(g_\phi\approx\nabla V\) 由 one-step endpoint estimate 的 reward gradient 加向量残差构成，再以价值梯度 PDE 和终点边界 loss 训练，最后拟合 flow 速度差。SD3 主干用 rank-8 LoRA；value-gradient 网络来自缩小的 SD1.5 U-Net。默认 reward-gradient 权重 \(\eta_s=s^2\)，还比较了 \(s\)；线性与二次的最终折中相近但收敛速度不同。奖励相关 \(\beta=1/\lambda\) 分别设为 5e4、3e7、5e5，边界权重 10000，reward gradient 按第 80 百分位裁剪。它提供了可学习未来影响的结构，却不是当前“无需手调时间 gain”约束的直接完成品。[VGG-Flow §4、§5、附录 C](https://papers.nips.cc/paper_files/paper/2025/file/10b7e27c8eb9571fbbd2ae6a9f8c3855-Paper-Conference.pdf)

**一个应修复而非放大成全盘否定的导数问题。** 真正的价值梯度 PDE 含 \([D b]^Tg\)。附录 C.1 Eq.45 用沿 g 的速度有限差分近似它，但该差分实际趋于 \(D b\,g\)。作者公开 `algorithm.py` 的 `_compute_velocity_derivative` 确实做 \([b(z+\epsilon g)-b(z)]/\epsilon\)。以通常 \((Db)_{ij}=\partial b_i/\partial z_j\) 记号，二者只有在相应对称条件下才相同；RAE 实际 IG 场未保证这一条件。修复需要真正的 VJP/离散伴随，不能把去 curl 当成此次任务的替代目标。此外，标量 V 可保证其 Hessian 对称，向量 g 本身没有这个保证。附录 Eq.43 的“partial time derivative”还写成沿轨迹差分；公开代码固定 z 做时间差分，所以这一项更像可修复的表述/版本问题，不能据此声称代码重复计算了输运项。[VGG-Flow 附录 C.1](https://papers.nips.cc/paper_files/paper/2025/file/10b7e27c8eb9571fbbd2ae6a9f8c3855-Paper-Conference.pdf)，[作者代码](https://github.com/lzzcd001/vggflow/blob/main/lib/vggflow/algorithm.py)

**实验实际证明什么。** Table 1 的 Aesthetic：base reward 5.99、DreamSim 0.2312；VGG-Flow 8.24、0.2212；DRaFT 9.54、0.0778，支持保留多样性的奖励优化。文中 FID 是逐 prompt 比较“微调生成集与 base 生成集”，再平均，测的是 prior preservation；不是与 ImageNet 真图 stats 的质量 FID。相应 Aesthetic FID 的 base/VGG/DRaFT 为 212/375/1518。400 次更新、20 步 rollout、有效 batch 32、3 seeds 不等于相同总训练成本；附录有 AM 对照需 4 GPU/FP32 的信息，未给足可统一换算的所有方法端到端 GPU 小时与采样耗时。不能用更新步数或主干 LoRA 大小推导公平成本胜出。[VGG-Flow §5/Table 1、附录 C](https://papers.nips.cc/paper_files/paper/2025/file/10b7e27c8eb9571fbbd2ae6a9f8c3855-Paper-Conference.pdf)

**RAE 的正向设计启发。** 保持官方 G 作为 b，只训练一个小标量 V，并用明确的终点质量代价约束它的边界，是比假设 Full/Base 为精确 score 更直接的路径。可证伪点不是 V 的 MSE 是否变小，而是其梯度能否预测真实 suffix 后终点观测量的变化。连续 PDE 的保证必须落到当前真实 100 步 shifted grid 的离散更新上；用错误方向的 Jacobian 或 one-step endpoint 代替完整 suffix，会改变这个检验对象。

## 4. 与当前 pure bridge projection、旧 semigroup value 的区别

| 对象 | 训练状态与目标来源 | 理想目标 | 缺少什么就不能称为 endpoint-directed |
|---|---|---|---|
| 当前 pure bridge 势函数 | 真实 encoder latent 与独立噪声构成桥；拟合 X−G，按原 loss 权重限制为梯度修正 | 在指定桥测度下投影局部条件去噪/连续性误差 | 没有真实生成 suffix 的敏感度，也没有显式终点质量观测 |
| 旧 semigroup value 尝试 | 从 Full/Base 的形式差值出发，试图构造强弱端点密度比及其后验传播 | 若所有假设成立，修正指数倾斜的 semigroup 缺项 | 同一合法 semigroup 与 Full/Base 合法密度比并未建立 |
| 本文的 endpoint control | 实际选定基准动力学上的 rollout；明确终点 \(\ell(D(Z_1))\) 或观测矩约束 | 减少终点代价，同时限制对轨迹的干预 | 若终点标签仍只有 X−G，或只换 value 名字，机制没有改变 |

本地对照来源：[已有综合阅读](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md)、[探索归档中的 semigroup value 记录](RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md)。以上区别不撤销旧失败结果，也不把新术语当作重新启动旧训练的依据。

**一个具体而有边界的后续设计原则（本文综合，不是三篇已证明的 RAE 方法）：** 冻结真实 IG 动力学与正式采样网格，先把终点需要改善的 feature 均值/协方差等量写成观测约束，或写成可审计的终点损失。沿完整离散 suffix 将这些终点量反传回中间状态，在明确控制能量下得到方向；若再摊销到标量网络，应让它学这个 cost-to-go。约束形式中的乘子可由约束求解来决定，惩罚形式则必须诚实保留全局权衡常数，二者都不需要凭经验另画时间窗口。这里真正的新信息是“该状态的微小干预经过后续生成后，会怎样影响终点质量量”，而非“该状态离训练桥的条件均值有多远”。

折中也很具体：终点矩目标只识别相应可观测误差；局部线性响应、有限样本矩估计、scalar 蒸馏误差和离轨分布变化都可能让理论局部收益失效。若用真实参考 FID 的 feature 统计定义目标，必须独立估计/冻结训练统计并保留确认样本；不能把 reward 提升、base-prior FID 或同训练统计上的 loss 当作用户要求的最终 FID 改善。

**成本应分成两部分。** 小标量在采样时可能只需一次小网络 forward 和 input-gradient，具有接近现有势函数的摊销潜力，但耗时需要实测。训练/制标签仍可能包含完整 100 步 rollout、suffix VJP 和终点 decoder/feature extractor；这些是实在的新增总成本。若用冻结策略的 Monte Carlo return 回归 value，可以省掉标签生成时对大模型的整条反传，却不会省掉 rollout，且低 value MSE 不保证输入梯度准确。三篇提供的是把目标、传播机制和成本分开核验的方法，而非免成本的终点保证。

## 5. 来源锁定与可复核性

本次三篇 PDF、提取文本和已核对的 VGG-Flow 源码已从临时阅读目录持久归档到实验库 `reading_endpoint_control_v1/`。完整路径为 `/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_endpoint_control_v1/`；`source_manifest.json` 记录各文件来源、字节数、页数和内容 SHA，并逐文件核对复制前后相同。Manifest SHA-256 为 `69bd71b302190f22cccf5f5056741f353d86d70dd53ff8a2da6052077a938756`。实际 PDF 页数为 AM 55 页、DEFT 47 页、VGG-Flow 33 页。PDF SHA-256 如下：

| 文件 | SHA-256 |
|---|---|
| AM arXiv v5 | `e59e529f1d26d8ff9260837e61d53d1c29b5aef69d080a599ba3a33ed9889f8c` |
| DEFT NeurIPS 2024 | `237ac74a88e35e8009f4760a53eaf44ce67cfa95e4c3ea13f50ccc64d8790226` |
| VGG-Flow NeurIPS 2025 | `57e63c1b7e30467ded5f4be163008825fc62e607213f1e97e47d5282291ab803` |

VGG-Flow 公开 main 分支代码于本日只读获取；所检查 `lib/vggflow/algorithm.py` SHA-256 为 `6bea0aa035271e2828d46e8997ab76b40308de5321619755deb74e41deaaf4c6`。GitHub API 未取得 commit id，因此代码证据锁定到该内容 hash，不能声称精确复现论文当时提交。本文没有把无法取得全文的后续 Value Matching 当作已深读文献，也未展开其他代理正在阅读的 MOG/Adaptive SOC。

独立复核另读了 AM §4–5、附录 G 的时间选点/裁剪/成本，DEFT Proposition 2.2、Theorem 3.1、§3.1 与 Table 6，以及 VGG-Flow §4、附录 B/C 和当前公开源码。上述版本与数值核验支持本文主要结论；新增澄清为 AM 的前 30 步也有训练、DEFT 消融仍保留观测反投影。VGG-Flow 的转置问题由原文和源码共同确认，例如二维线性速度 \(b(z)=Az\)、\(A=\begin{psmallmatrix}0&1\\0&0\end{psmallmatrix}\)、\(g=(1,0)^T\) 时，原文所需 \(A^Tg=(0,1)^T\)，而源码方向差分为 \(Ag=0\)。这只是可直接复核的导数反例，不是关于 RAEv2 FID 的实验。没有新增第四篇文献、模型调用或 GPU 工作。

## 6. 本地独立离散推导：终点约束与整条轨迹的最小能量

以下是阅读后独立推导及双人核对，不是上述论文已经给出的 RAE 方法。对固定实际 Euler 轨迹 `z_{k+1}=T_k(z_k)`，在更新后加入 `h_k u_k`。记 `A_k=DT_k(z_k)`，终点观测为 Ψ，目标均值缺口为 δ。一阶扰动满足

\[
\xi_{k+1}=A_k\xi_k+h_ku_k,\quad
L u=\sum_kh_k\mathbb E[B_ku_k],\quad
B_k=J\Psi(z_K)A_{K-1}\cdots A_{k+1}.
\]

这里不包含 A_k，因为控制加在 T_k **之后**。在任意平方可积控制场中，若 δ 属于 G 的值域，则

\[
G=\sum_kh_k\mathbb E[B_kB_k^T],\qquad
u_k^*=B_k^TG^+\delta,\qquad
\min\sum_kh_k\mathbb E\|u_k\|^2=\delta^TG^+\delta.
\]

这是约束最小范数解；没有额外手设的时间 gain。时间行为来自实际 suffix 响应。令 λ=G⁺δ，每个时刻的终点对偶贡献均为 `h_k E||B_kᵀλ||²≥0`，因而不会沿这个共同终点 witness 相互抵消；不同坐标仍可产生必要的协调抵消。有限非线性修正只满足一阶约束，不能据此认领完整分布或 FID 保证。

确定性 Markov 轨迹的未来由当前状态确定，因此 B_k 可函数化；这不意味着无需计算未来。若未来有未知随机性，适应性约束要求先取 `E[B_k | 当前信息]`，再构造 Gram。直接使用未来实现的 B_k 会得到能预知未来的乐观控制器。

**标量势部署的位置也有区别。** `B_kᵀλ` 是剩余终点函数相对于后继状态 `y=T_k(z_k)` 的梯度；作为 z_k 的函数，它未必等于当前状态某个标量势的梯度。若使用“先做原 Euler，再在后继状态评价未来值函数梯度”的结构，需在这个位置重新训练/验证，不能直接搬用旧势网络。在离散目标中混淆两种梯度，会漏掉或多乘一个 A_kᵀ。

CPU 固定二维例子用 `h=(1/2,1/2)`、`A₀=diag(2,1/2)`、`A₁=[[1,1],[0,1]]`、`Ψ(z)=z`、`δ=(1,0)`。解为 `u₀=(.8,.4)`、`u₁=(.8,−.4)`，精确终点位移为 δ，能量 **0.8**。两条完整终点零空间基方向均与最优控制能量正交，验证最优性。所有矩阵、断言、源码 SHA 和 CPU 成本保存在实验库 `endpoint_adjoint_algebra_v1/{audit.py,summary.json}`；无 GPU 或图像评价。

这个结构给出下一次机制检验的明确对象：先用独立真实数据确定可复现的终点缺口，再检查 suffix-adjoint 是否预测完整有限扰动的实际效果。已有状态/F/B 缓存不能恢复 Jacobian，仍须重放模型并求输入 VJP；若随后摊销为小网络，训练准备与误差也必须进入成本和验证。仅完成上述代数，不足以启动质量结论。
