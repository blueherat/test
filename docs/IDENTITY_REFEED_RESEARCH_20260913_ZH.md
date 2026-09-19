# 恒等编辑与递归再输入：操作定义、反例和验证协议

对一张图片反复执行“保持当前图片不变”，研究对象首先是一个图像输入—图像输出的**保留任务**。最贴切的名称是“重复零编辑”（iterated no-op editing）；其中“递归再输入”描述输出进入下一次调用的连接方式，“恒等”描述任务要求。仅有循环连接，并不意味着里面的操作应当保持图片。

核心区别是：**模型每一轮还得到哪些原图信息，以及任务是否指定了原图本身作为正确答案。** 类别相同、描述相同、随机种子相同、边缘分布相同，都不足以指定同一张图片。完整源图作为额外条件时，内部可以使用随机噪声而仍以原图为目标；反之，源图只通过有损加噪状态进入模型时，精确生成器也未必能恢复产生该状态的那个实例。

目前最有依据的路线，是先在原生图像条件编辑接口上建立零编辑实验，再拆分编解码、生成、随机性和历史的影响。当前类别条件 SiT 的再加噪实验应解释为“迭代有损再采样”；同模型反演—生成应解释为“往返重建”；二者都不能直接充当通用生成模型质量或 CFG/IG 优劣的判据。这里否定的是既有解释所缺的条件，不是宣称所有未来构造都不可能。

## 1. 给“保持一样”一个可以检验的定义

令 \(A_{M,a}(x,u;\omega)\) 表示完整编辑系统：输入图片 \(x\)、指令 \(u\)、随机工作变量 \(\omega\)，设置 \(a\) 包括版本、预处理、分辨率、图像条件强度、采样器和解码流程。令 \(u_0\) 明确表示“不修改当前输入”。定义

\[
X_{r+1}=A_{M,a}(X_r,u_0;\omega_r).
\]

如果要求严格逐像素不变，接口的理想条件分布是

\[
K^*_{M,0}(dy\mid x)=\delta_x(dy).
\]

如果允许特定视觉误差，则预先声明距离与阈值：

\[
\Pr[d(A_{M,a}(x,u_0;\omega),x)\leq\epsilon]\geq1-\eta.
\]

\(\epsilon\) 和 \(\eta\) 是任务规范，不能根据实验结果事后放宽。文件字节相同、解码后的像素相同、布局相同、主体身份相同、语义相近，是不同层级。例如两张同种鸟的图片可以语义接近，但位置、枝干、视角和具体个体都不同。

“完整源图条件”在这里描述**接口接收了未被外部破坏的图片**，不表示内部编码器一定无损。若系统只使用非单射表示 \(\psi(x)\)，存在 \(x_1\ne x_2\) 却 \(\psi(x_1)=\psi(x_2)\)，仅依赖这个表示与独立随机数的解码器无法同时精确返回两张不同图片。因此，\(\delta_x\) 是外部任务目标；特定架构是否能达到它，需要另行分析。编码基线有助于解释误差，但不能把一次编码重建误差一概当作所有系统都无法突破的下界。

原生指令编辑接口提供了这种任务变量。InstructPix2Pix 将源图条件与生成中的噪声状态分开输入，并分别处理图像条件和文字条件的 guidance；它与“图片加噪后，只按原类别继续生成”的信息结构不同。[^1]

## 2. 历史是否保留，会改变整个实验

有历史时应写成

\[
X_{r+1}=A_{M,a}(X_r,u_0,H_r;\omega_r),\qquad
H_{r+1}=U(H_r,X_r,X_{r+1},u_0).
\]

只观察 \(X_r\) 通常不能把它当作时间齐次 Markov 链。历史可能仍含最初图片、先前图片及文字描述；当前图片已经丢失的细节，可以从历史中重新取得。历史还可能使模型回到最初图片，而最新要求实际上是保留当前图片。

| 协议 | 每轮实际可见的信息 | 所测能力 |
|---|---|---|
| 只传上一图 | 当前图片、固定指令；新请求不携带历史 | 图片自身经过重复零编辑后的保留 |
| 保留对话 | 当前图片、全部既有对话与图片 | 会话系统的持续保留和记忆利用 |
| 显式原图锚点 | 当前图片、最初图片及清楚的参考角色 | 有外部记忆的约束编辑 |

三者都可以研究，但必须分开命名与报告。原始观察的历史设置未确定，不能替它补设条件。主协议宜采用“只传上一图”，使问题可解释；另设带历史协议检验产品行为。

## 3. 五种经常被叫作“回灌”的操作

| 操作 | 数学形式 | 是否应保持当前图片 | 合理用途 |
|---|---|---|---|
| 原生零编辑 | \(A(x,u_0;\omega)\) | 是任务目标，达到程度受实现限制 | 对应“要求完全不变”的观察 |
| 编码—解码 | \(C(x)=D(E(x))\) | 理想无损编码可恒等；有损编码需分别看首次失真与后续稳定 | 编码瓶颈、重复压缩损伤 |
| 反演—同条件生成 | \(R_g(x)=F_g(I_g(x))\) | 在相同可逆映射下可成立 | 反演、积分、精度、codec 诊断 |
| 固定参考反演—另一模型生成 | \(R_{g\mid q}(x)=F_g(I_q(x))\) | 没有一般保证 | 特定潜空间对应关系的相容性 |
| 再加噪—生成 | \(K_g(x;\epsilon)=F_g^{t\to1}(tx+(1-t)\epsilon)\) | 一般不应要求逐图恒等 | 有损再采样的分布与实例演化 |

SDEdit 明确通过加噪与去噪在输入保真和生成真实性之间作权衡；它并不把该过程定义成原图精确复制。[^2] 因此在普通 img2img 接口里加一句“不要变化”，也不能自动改变模型的信息结构或赋予它指令编辑能力。

把同一个 caption 再用一次，只指定同类内容；把同一类别再用一次，只指定同类对象。它们都没有唯一指定原来的像素、构图和实例。

## 4. 完美的 SiT 式 flow 也会在旧操作下忘记原图

以下是独立解析推导，用来检验“理想模型应该不变”这个主张，不是对真实图像数据的数值预测。

设真实数据和初始噪声都为标准高斯，且独立：

\[
X,E\sim\mathcal N(0,1),\quad Z_t=tX+(1-t)E,\quad
s(t)=\sqrt{t^2+(1-t)^2}.
\]

流匹配的精确最优边缘速度为

\[
v^*(z,t)=\mathbb E[X-E\mid Z_t=z]
=\frac{2t-1}{t^2+(1-t)^2}z.
\]

由 \(v^*=(s'/s)z\)，从 \(t\) 到 \(1\) 的精确 ODE 映射为 \(F^*_{t\to1}(z)=z/s(t)\)。这种“边缘向量场实现概率路径”的关系来自 flow matching 的基本构造；具体高斯例子由此直接算得。[^3]

对当前 \(t=.25\) 的外部操作：

\[
X_{r+1}=\frac{.25X_r+.75E_r}{\sqrt{.25^2+.75^2}}
=\frac{1}{\sqrt{10}}X_r+\frac{3}{\sqrt{10}}E_r.
\]

每轮分布都精确为 \(\mathcal N(0,1)\)：生成模型没有学错，ODE 没有数值误差，分布也没有退化。但是

\[
\operatorname{Corr}(X_r,X_0)=10^{-r/2},\qquad
\mathbb E[(X_r-X_0)^2]=2(1-10^{-r/2}).
\]

| 再采样轮数 | 与原样本的相关系数 | 每维原图 MSE | 边缘分布 |
|---:|---:|---:|---|
| 0 | 1 | 0 | 标准高斯 |
| 1 | 0.316228 | 1.367544 | 标准高斯 |
| 2 | 0.100000 | 1.800000 | 标准高斯 |
| 3 | 0.031623 | 1.936754 | 标准高斯 |
| 4 | 0.010000 | 1.980000 | 标准高斯 |
| 5 | 0.003162 | 1.993675 | 标准高斯 |

**五轮后与原样本几乎不相关，但分布始终完全正确。** 人口层面的分布距离为零；有限样本 FID 仍会有估计误差。这足以否定“只要模型足够好，当前再加噪流程就应该保持原图”的一般推断。

固定噪声也不能修复定义。令每轮都使用同一个 \(e\)，则递推收敛到

\[
x_\infty=\frac{3}{\sqrt{10}-1}e\approx1.387426e,
\]

它仍然不是原来的 \(x_0\)。最终稳定可以意味着已经忘掉原图，而不是更忠实。

若把实际噪声 \(e\) 作为额外侧信息传给解码器，可以直接计算 \(x=(z-(1-t)e)/t\)。这是一项带侧信息的可逆编码任务，生成模型无需参与；它也不是当前网络看到的输入。

## 5. 随机性不是根本判据，信息集才是

独立高斯噪声提供新的随机自由度，但不携带关于原图的独立语义证据：\(I(X;E)=0\)。问题在于仅观察 \(Z=tX+(1-t)E\) 时，原实例往往已经不可唯一识别。只依据 \(Z\) 和独立随机数的解码器不能补回特定丢失的信息。对于平方可积随机变量，几乎必然精确恢复需要 \(\operatorname{Var}(X\mid Z)=0\)；高斯加噪例子的条件方差不为零。

但若同时提供完整源图 \(x\)，信息集成为 \((Z,x)\)，这个瓶颈就不同了。例如一个以 \(x\) 为条件的理想 flow 可以沿 \(z_t=tx+(1-t)\omega\) 在终点极限到达 \(x\)，无论内部使用哪份 \(\omega\)。精确点质量可能需要这样的终点极限、投影或复制通道；这不表示任意有限时间、全局 Lipschitz 的可逆 flow 都能把连续噪声压成一个点。因此应拒绝“任何新随机种子都使零编辑要求不合理”的过强说法。

还需区分 posterior mean、posterior sampling 与 ODE transport。对高斯加噪观测 \(Y=X+N\)，最小 MSE 重建器是 \(m(y)=ay\)，其中 \(0<a<1\)。它在自身任务上完全正确，重复直接作用于图像仍会收缩。概率重建从 \(p(X\mid Y)\) 采样则可以保持正确边缘分布，同时改变具体实例。去噪自编码器的 score 关系、生成性定理和感知—失真研究支持这些区别，不能互相替代使用。[^4][^5][^6]

尤其不能将不适定恢复的感知—失真权衡，直接当作完整原图复制任务必须损失信息的理由。

## 6. 恒等、幂等、平稳和可逆之间没有那条自动推理链

| 性质 | 条件 | 一个使质量推断失败的例子 |
|---|---|---|
| 恒等 | \(T(x)=x\) | 直接复制同时保留好图和伪影；不代表会生成或会编辑 |
| 幂等 | \(T(T(x))=T(x)\) | 所有输入变成同一常量，第一次严重损坏，以后完全稳定 |
| 随机核幂等 | \(K^2=K\) | 每次独立从真实分布重画，条件分布一步与两步相同，图片却不同 |
| 边缘平稳 | \(pK=p\) | 完美再采样核保持分布，但遗忘输入实例 |
| 往返可逆 | \(F\circ F^{-1}=I\) | 错误生成分布也可精确可逆 |

以 \(F(z)=100z\) 为例，它把标准高斯生成成错误尺度的分布，却有 \(F(F^{-1}(x))=x\)。同模型精确反演再生成首先测的是映射可逆性。DDIM 的重建实验、EDICT 的代数可逆构造和 RF-Solver 的高阶积分工作说明：实际往返误差还会受求解器与离散反演影响。[^7][^8][^9]

当前 guidance 若有探针缓存、动量或接受步历史，仅把时间倒着走还不一定构成同一个系统的逆。要谈精确可逆，必须连同相关状态及更新规则定义映射。

跨模型反演也不是自动的质量标签。若 \(F_2=F_1\circ Q\)，其中 \(Q\) 保持基础高斯分布，两个模型可具有相同生成分布，却给同一个 latent 对应不同图片。\(F_2(F_1^{-1}(x))\) 的偏移测到一种配对关系的差异，而不是已经被证明的错误。固定参考能让实验可定义，但参考本身会天然自反占优。

FlowEdit 给出另一个边界：在完整源目标场相同、共享噪声、没有末端额外单路采样等条件下，速度差可以逐步代数归零；此时恒等由算法保证，而不是由模型质量保证。仅仅使用相同文本还不满足这些条件。[^10]

## 7. 已有文献：哪些是真正同题，哪些只是结构相近

### 7.1 直接同题的两条路线

**Why Do DiT Editors Drift?（2026 预印本）**直接定义 no-op 编辑的固定点违约，运行十轮保留指令，并比较完整图片循环、DiT-only 和 VAE-only。它是此次检索中操作最贴近“要求不变、输出递归输入”的工作；论文报告特定编辑器中低频漂移更多来自 DiT。固定种子的说明不能代替对会话历史的控制，当前可见材料也不足以替其保证所有调用都排除了历史访问。[^11]

**后续作者代码核查补充：主编辑实验与组件消融不能混为同一个算子。** 在公开 commit `d577aabc1026398d4fc50fa8d2f51315d10b6fc5` 中，SD3-UE 主 runner 使用 `StableDiffusion3InstructPix2PixPipeline` 和图像 guidance；但 `FLUX.2/src/eval/vae_ablation_sd3.py` 的 SD3.5 分支选择普通 `StableDiffusion3Img2ImgPipeline`，设置 `strength=0.8`，把上一轮 latent 作为 `image` 再输入。该公开组件代码路径是加噪再采样；去掉轮间 VAE 不会让它自动成为原生零编辑任务。尚未核实图 2 对应的完整运行清单，不能断言论文图表必定由这一配置生成，但也不能把其 SD3 组件结论直接当成我们的恒等前提。[主实验代码](https://github.com/ZephinueCode/VAE-LFA/blob/d577aabc1026398d4fc50fa8d2f51315d10b6fc5/SD3-UE/run_ultraedit_metrics.py#L640)、[组件管线选择](https://github.com/ZephinueCode/VAE-LFA/blob/d577aabc1026398d4fc50fa8d2f51315d10b6fc5/FLUX.2/src/eval/vae_ablation_sd3.py#L394)、[组件加噪与再输入](https://github.com/ZephinueCode/VAE-LFA/blob/d577aabc1026398d4fc50fa8d2f51315d10b6fc5/FLUX.2/src/eval/vae_ablation_sd3.py#L477)。

**REED-VAE（2025）**研究重复编解码，并明确测试 NTI 反演后使用同一源提示词重建、再次把结果反演的无编辑循环。这是另一种合理操作，包含反演和专属编码预算；它对 VAE 的改进不能被解释为所有基础生成模型都提高了复制能力。[^12]

两者对主要误差组件的观察不同，不能用一句“VAE 才是根因”或“DiT 才是根因”统摄。不同接口、VAE、反演方法和频段可能给出不同归因；组件对照必须在自己的系统上做。

### 7.2 标题相近的论文仍需接受同样质疑

**The Drift Kernel（CVPR 2026）**测量 no-op/copy prompt 下的漂移及其与 strength 的关系，并以 decoder 的局部敏感性解释。它很相关，但普通 SD img2img 与原生图像指令编辑器的信息条件不相同；其名字不能消除这个差异。[^13]

对其推导还应作独立检查：设 decoder 前的实际残差是 \(r\)，均值 \(b\)、协方差 \(\Sigma\)，那么线性化后的误差项一般含

\[
\|J_D b\|^2+\operatorname{tr}(J_D\Sigma J_D^\top),
\]

并可能与已有 codec 偏差交叉。不能未经验证就把生成后的 \(\Sigma\) 换成 API strength 的平方乘单位阵，也不能仅根据聚合拟合推断每个模型都有相同规律。论文可提出待验证的敏感性假设，但不证明“扩散模型原则上无法服从原图保留”。这是对适用条件的分析，不是否认论文测得的输出差异。

### 7.3 跨领域真正可迁移的部分

| 工作 | 可迁移的思想 | 不应迁移的结论 |
|---|---|---|
| Idempotent Generative Network | 把真实数据固定点、生成幂等和固定点集合约束共同设计 | 仅惩罚任意预训练去噪器的重建残差就能改善质量 |
| Encoder–Decoder Manifold Alignment | 检查编码与解码表示是否闭合 | 任意幂等映射都是正确或最近的数据流形投影 |
| Idempotent Learned Image Compression | 首次失真与后续重复编码稳定性分别计量 | 重复不再变化意味着第一次没有损失 |
| Bayesian iterated learning | 传输链的长期形态可以反映学习者先验 | 链最终稳定意味着忠实保存起始信息 |
| CycleGAN steganography | 循环恢复可以由隐藏信息满足 | 中间图像语义正确可由循环误差单独保证 |

相应原始来源分别见脚注。[^14][^15][^16][^17][^18] 例如有效码上 \(E\circ D=I\) 足以使 \((D\circ E)^2=D\circ E\)，但这个等式不决定首次重建 \(D(E(x))\) 丢掉多少细节。粗量化也能第一次失真很大、之后完全稳定。

原始现象已经存在直接相关研究。下一步的贡献不能只写成“发现多次生成会漂移”；更需要明确新的操作控制、误差归因，或者证明某个缺陷信号对 guidance 有独立预测与干预价值。

## 8. 三种立场交叉反驳后的判断

下表列出理论、反演和跨领域分析之间真正需要修正的主张。它们不是投票式的“多数同意”，每一条都由反例或缺失条件限定。

| 初始主张 | 最强反驳 | 修正后可保留的结论 |
|---|---|---|
| 新随机噪声意味着不能要求不变 | 源图若仍独立可见，零编辑输出可以对随机变量不敏感 | 关键是源信息是否丢失；随机性本身不是充分否决条件 |
| 固定种子后就对应原样输出 | 完美高斯 flow 的固定噪声迭代仍收敛到别的点 | 固定种子只控制随机性，不创造逆映射或保留目标 |
| 改成反演重建就能比较模型强弱 | 任意可逆错误生成器也能精确往返 | 用作数值与编码诊断，不独立排名生成质量 |
| 原生图像条件一定能逐像素复制 | 内部编码可能非单射 | 精确复制是任务规范；架构可达性另测 |
| 必须禁止直接复制，否则实验作弊 | 零编辑任务本来就要求保留输入 | 产品任务允许复制；另设真实编辑能力和机制检查 |
| 更稳定就更值得用来指导生成 | 常量映射、错误流形投影、自反参考都可能得高分 | 必须有外部目标与独立质量验证，不能自证 |
| 编解码总是主要漂移源 | 不同模型的组件实验给出不同频段结论 | 在当前系统上干预组件再归因 |

最重要的区分是**能力评价与机制解释**。零编辑分数确实可以评价一个编辑系统是否忠实完成零编辑任务；但仅凭它，不能判断内部是否重新生成、是否调用了复制路径，更不能推出无条件生成先验的强弱。

## 9. 一个预算小、能分清问题的主实验

这一阶段先回答“定义是否对应现象、误差在哪里”，不再先搜索 guidance 参数。

### 9.1 固定输入和接口

选择原生支持“参考图片＋编辑指令”的接口。主实验每轮新请求，只传上一轮图片，指令固定为明确的保留要求，不额外描述图片内容，也不主动给它换图、增强或修复的任务。记录完整请求字段；若服务端内部历史、预处理或随机性无法访问，注明该限制。

R0 是实验起始图片，不必是当前模型生成的图片。初始样本同时包含真实图、模型自产图和跨模型图，可帮助识别“对自己的输出更熟悉”与一般保留能力；小规模阶段至少有细小文字、规则几何、重复纹理、人物/物体细节等容易暴露误差的内容。

固定解析尺寸、色彩处理和文件格式。字节校验与解码像素校验分开；只有格式元数据不同不能叫内容漂移。若接口必须缩放，以实际进入接口的规范化图作为明确基准，并单独保存规范化前原件。不要用事后几何配准掩盖真实构图移动。

### 9.2 三组主测量

| 测量 | 操作 | 排除什么歧义 |
|---|---|---|
| 五轮递归零编辑 | R1 以 R0 为输入，R2 以 R1 为输入，直到 R5 | 真正的输出递归链 |
| 原图分支重复 | 每次都从同一 R0 独立调用 | 单次随机性与递归累积的差别 |
| 小幅真实编辑 | 每图一条可核验的局部修改要求 | 所有指令都复制输入的退化方案 |

建议首批 16 张图。每张运行五轮，产生 80 个输出；再对原图补两个独立单次调用，产生 32 个输出；增加 16 个小幅真实编辑，总计每模型 **128 个输出**。这只是用于诊断和估计效应量的起点，不足以宣布普遍模型排名。跨轮数据共享祖先，统计时应以起始图片为聚类单位，不能把所有轮次当独立样本。

若 seed 可控，分别记录种子固定的链与独立随机链，后者可留给第二阶段扩展。跨模型用相同数字 seed 只是一项复现设置，不保证随机变量具有相同语义作用。若 API 不提供 seed，就用重复请求估计输出波动，不虚称确定性。

历史实验单独运行：明确让系统复制当前图，或明确以原图为锚点。两种目标也不能混写。

### 9.3 每轮测什么

必须同时记录 \(d(X_r,X_{r-1})\) 与 \(d(X_r,X_0)\)。前者小、后者大，可能意味着早期已经偏离，之后卡在另一个固定点；只看相邻误差会误判稳定。

像素误差、感知/结构距离和任务细节检查分别展示。文字可检查内容与排版，几何可检查边界与相对位置，人物可检查身份与姿态，但任何单一特征模型都不能替代完整图片保留。视觉质量另测；FID 不提供一一对应的原图保真证据。

对同一输入多次调用，以输出均值与输入之差估计偏置，以分支间波动估计随机性。在欧氏空间有

\[
\mathbb E\|Y-x\|^2=\|\mathbb E[Y]-x\|^2+
\operatorname{tr}\operatorname{Cov}(Y).
\]

它能区分系统性变化与随机变化，但不会自动指认是哪一层造成偏置。少量分支只能粗估，不能给出精确的高维方差模型。

### 9.4 白盒组件对照

保留三条独立路径：完整像素编辑链、只做 VAE 编码解码、避免中间 VAE 的编辑 latent 链。VAE 的 posterior 取均值/众数还是重新采样，必须写清楚；否则 codec 控制又引入另一种随机性。

可另外用同模型反演—重建检查求解误差，并比较两个步数或精度，判断变化是否随数值误差缩小。无需先跑完整 FID 网格。

组件误差不是可直接相减的标量预算。若总误差可在同一表示中写成 \(e=e_1+e_2\)，则 MSE 仍含 \(2\langle e_1,e_2\rangle\)；移除一层还可能改变下一层的输入分布。因此“完整 MSE 减 VAE MSE”等于“生成器 MSE”的解释一般不成立。

## 10. 何时才有资格连接 CFG 或 internal guidance

### 10.1 首先连接编辑保真，而不是新图生成质量

在具有源图条件的编辑系统上，可以研究 guidance 是否增加了无请求变化，以及减轻它时正常编辑是否仍完成。这是一个明确的编辑任务。文字 guidance 与图像 guidance 可能是两个不同控制量，不应直接套用类别 SiT 的同一标量解释。[^1]

若只有类别条件 SiT，没有额外图像条件、外部逆映射或数据一致性约束，就缺少定义“请保留这一个实例”所需的信息。目前没有找到仅凭这个接口、无需额外假设而同时具备“应逐图恒等”与“可识别生成质量”的非平凡算子。反演可补出一个任务，但也带入了配对与参考假设。

### 10.2 对 IG，一个比“弱模型误差大”更重要的可测条件

假设两套系统面对同一明确的零编辑目标 \(x\)，其输出为

\[
Y_s=x+e_s,\qquad Y_w=x+e_w.
\]

先只考虑**输出空间线性外推**，而不是实际采样过程中的 velocity guidance：

\[
Y_\alpha=Y_s+\alpha(Y_s-Y_w).
\]

它的误差满足

\[
\mathcal L(\alpha)=\mathbb E\|e_s\|^2+2\alpha B+\alpha^2 A,
\]

\[
A=\mathbb E\|e_s-e_w\|^2,\qquad
B=\mathbb E[e_s^\top(e_s-e_w)].
\]

当 \(A>0\) 且限制 \(\alpha\geq0\) 时，最优值是

\[
\alpha^*=\max(0,-B/A).
\]

所以正向外推可能有益，需要

\[
\mathbb E[e_s^\top e_w]>\mathbb E\|e_s\|^2.
\]

**弱模型 MSE 更大，不足以满足这个条件。** 若 \(e_w=k e_s\)、\(k>1\)，强弱误差同向且弱误差更大，取 \(\alpha=1/(k-1)\) 可以消掉误差。若强弱误差独立且弱误差均值为零，正向外推反而增加 MSE。\(A=0\) 时两个输出相同，外推没有效果。

这个初等推导提供了一个可检验的问题：在有效的零编辑任务上，强弱误差到底是否具备可利用的共同方向？它需要固定联合采样协议、同一坐标空间、二阶矩存在，并排除额外 clip 或非线性后处理的影响。不能把同一个数值 seed 当作天然唯一的联合分布。

进一步转成 velocity IG 还需要轨迹扰动的传播、局部 Jacobian 和非线性分析；上式不证明现有 IG 公式会改善。若只拿 no-op 数据调到最优，仍需独立的小幅编辑与外部质量任务验证，防止只学会不动。这比直接把任意回灌残差当作“弱模型错误方向”更有可识别依据，但还不是已验证方法。

## 11. 对当前仓库实验的解释修正

`recursive_guidance_20260913` 和 `recursive_guidance_focus_20260913` 的外部循环都使用 `.25*previous_latent + .75*fresh_noise`，随后运行既有采样后缀；生成器只接收状态和类别。这里保留了类别，却没有原生的干净图像条件与零编辑指令。实现位置见 [外部循环](../experiments/recursive_guidance_20260913/pipeline.py) 与 [采样器](../experiments/recursive_guidance_20260913/core.py)。

这些输出仍可用于研究迭代再采样后的 FID、样本变化和 guidance 效应；它们不是无意义的数据。但它们不能支持“这个模型更能原样复制”“漂移是生成错误”“更稳定的方向就应当用于 CFG/IG”这些结论。小幅 FID 改善也不能倒过来补上缺失的操作定义。

研究顺序应为：**先定义并复现原生零编辑操作；再归因误差；随后检验误差是否能预测、并帮助改善独立任务；最后才决定 guidance 公式。** 现阶段最有价值的实验，是上述小规模操作诊断，而不是恢复大网格继续挑赢家。

## 来源与定位

以下以原始论文和会议来源为主。2026 年预印本按所读版本标明，相关发现不视为跨模型定理。本文高斯反例、可识别性反例及输出外推误差展开是独立分析，不作新颖性声明。

[^1]: Tim Brooks, Aleksander Holynski, Alexei A. Efros. *InstructPix2Pix: Learning to Follow Image Editing Instructions*. CVPR 2023；arXiv v2，2023-01-18。§3.2–3.2.1：源图条件与文字条件、双条件 guidance。[原文](https://arxiv.org/html/2211.09800v2)
[^2]: Chenlin Meng et al. *SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations*. ICLR 2022；预印本 2021。§3 与 Fig.3：加噪去噪及 realism–faithfulness 权衡。[原文](https://arxiv.org/abs/2108.01073)
[^3]: Yaron Lipman et al. *Flow Matching for Generative Modeling*. ICLR 2023。§3–4、Theorems 1–2：条件路径边缘化与对应向量场。[原文](https://arxiv.org/html/2210.02747v2)
[^4]: Guillaume Alain, Yoshua Bengio. *What Regularized Auto-Encoders Learn from the Data-Generating Distribution*. JMLR 15，2014。Theorem 1、§3.4：去噪重建与 score。[原文](https://jmlr.org/papers/volume15/alain14a/alain14a.pdf)
[^5]: Yoshua Bengio, Li Yao, Guillaume Alain, Pascal Vincent. *Generalized Denoising Auto-Encoders as Generative Models*. NeurIPS 2013。§2.3、Theorem 1：条件重建与渐近平稳分布。[原文](https://papers.nips.cc/paper_files/paper/2013/file/559cb990c9dffd8675f6bc2186971dc2-Paper.pdf)
[^6]: Yochai Blau, Tomer Michaeli. *The Perception-Distortion Tradeoff*. CVPR 2018。§3–4、Appendix F：配对失真与边缘分布感知质量。[会议原文入口](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)
[^7]: Jiaming Song, Chenlin Meng, Stefano Ermon. *Denoising Diffusion Implicit Models*. ICLR 2021。§5.4、Table 2：反演重建及步数影响。[原文](https://arxiv.org/abs/2010.02502)
[^8]: Bram Wallace, Akash Gokul, Nikhil Naik. *EDICT: Exact Diffusion Inversion via Coupled Transformations*. CVPR 2023。§3.2、§4.1–4.2、Table 1：代数可逆、参数效用和 VAE 重建。[原文](https://arxiv.org/abs/2211.12446)
[^9]: Jiangshan Wang et al. *Taming Rectified Flow for Inversion and Editing*. ICML 2025；本文定位基于 2024-11-07 发布的 arXiv v1。§3.2：RF-Solver 的反演与数值积分误差。[原文](https://arxiv.org/html/2411.04746v1#S3.SS2)
[^10]: Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, Tomer Michaeli. *FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models*. ICCV 2025。§5、Alg.1、Appendix F：共享噪声、速度差和末端采样设置。[原文](https://arxiv.org/html/2412.08629v2)
[^11]: Xiaoce Wang et al. *Why Do DiT Editors Drift? Plug-and-Play Low Frequency Alignment in VAE Latent Space*. arXiv:2605.08250v1，2026-05-07。§3、§5、Appendix F：十轮 no-op、组件对照与 API 设置。[原文](https://arxiv.org/html/2605.08250v1)
[^12]: Gal Almog, Ariel Shamir, Ohad Fried. *REED-VAE: RE-Encode Decode Training for Iterative Image Editing with Diffusion Models*. Computer Graphics Forum 44(2)，2025。§5.5、迭代 NTI 图及补充材料。[原文](https://arxiv.org/html/2504.18989v1)；[期刊](https://doi.org/10.1111/cgf.70020)
[^13]: Gokul Srinath Seetha Ram, Rashmi Elavazhagan. *The Drift Kernel: Why Diffusion Models Change Even When Told Not To*. CVPR 2026。模型与 strength 设置、局部推导及拟合指标。[官方原文](https://openaccess.thecvf.com/content/CVPR2026/papers/Ram_The_Drift_Kernel_Why_Diffusion_Models_Change_Even_When_Told_CVPR_2026_paper.pdf)
[^14]: Assaf Shocher et al. *Idempotent Generative Network*. arXiv:2311.01462v1，2023。§2、§6：重建、幂等、tightness 及局限。[原文](https://arxiv.org/html/2311.01462v1)
[^15]: Dareen Alharthi, Abdul Waheed, Bhiksha Raj. *Encoder-Decoder Manifold Alignment for Idempotent Generation*. arXiv:2606.22304v1，2026-06-21。编码解码闭合约束及 Appendix B 的条件性命题。[原文](https://arxiv.org/html/2606.22304v1)
[^16]: Yanghao Li et al. *Idempotent Learned Image Compression with Right-Inverse*. NeurIPS 2023。重复压缩稳定性及 right-inverse 构造。[会议原文入口](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2a25d9d873e9ae6d242c62e36f89ee3a-Abstract-Conference.html)
[^17]: Thomas L. Griffiths, Michael L. Kalish. *Language Evolution by Iterated Learning With Bayesian Agents*. Cognitive Science 31(3)，2007。指定采样假设下的迭代学习链与先验。[原文入口](https://doi.org/10.1080/15326900701326576)
[^18]: Casey Chu, Andrey Zhmoginov, Mark Sandler. *CycleGAN, a Master of Steganography*. 2017。通过隐藏高频信息满足循环恢复的经验反例。[原文](https://arxiv.org/abs/1712.02950)
