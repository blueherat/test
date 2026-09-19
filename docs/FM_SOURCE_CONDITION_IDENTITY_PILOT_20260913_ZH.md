# 流匹配的原图条件、恒等构造与小规模实测

**流匹配可以表达“输入保持不变”。需要区分两种获得方式：学习一个始终看到源图的条件速度场；或在现有速度场外构造满足恒等的编辑算法。前者已有 FLUX.1 Kontext，后者已在当前 SiT 上完成 4 图、5 轮验证。** 现有类别条件 SiT 缺少经过训练的源图条件接口，不代表流匹配原理缺少这种能力。

本次实测是操作存在性与实现检查，未训练新模型，未恢复旧参数搜索，没有给出 FID 或 CFG/IG 收益结论。

## 1. 原图可以作为独立条件，而且仍然从噪声生成

普通类别条件流匹配学习

\[
v_\theta(z_t,t,c),\qquad z_t=(1-t)\epsilon+ty.
\]

这里类别 \(c\) 不能指定某张具体图片。图像编辑条件模型可以改为

\[
v_\theta(z_t,t,c,x,a),
\]

其中 \(x\) 是源图、\(a\) 是编辑指令、\(y\) 是目标图。始终保留未被外部加噪破坏的 \(x\)，加噪的只是目标侧状态 \(z_t\)。训练目标仍然是标准线性路径的速度回归：

\[
\mathcal L=\mathbb E\left\|v_\theta((1-t)\epsilon+ty,t,c,x,a)-(y-\epsilon)\right\|^2.
\]

零编辑时规定 \(a=\mathrm{id}\)、\(y=x\)。在理想条件场中，源图已知，因此目标速度可写成

\[
v^*_{\mathrm{id}}(z,t;x)=\frac{x-z}{1-t},\qquad t<1.
\]

解为 \(z_t=(1-t)\epsilon+tx\)，在 \(t\uparrow1\) 时趋于 \(x\)，与随机初始化无关。这是一个直接的存在性证明。它的末端是奇异极限，不能误说成全程光滑可逆的有限时间流把连续分布压成了点。

这里还有一个研究上的区别：**把这个公式硬编码进去，本身就能复制原图；训练一个通用编辑模型去近似这些不同条件场，才会留下可测的模型能力差异。** 严格复制也不需要生成先验，不能单靠复制成功评价先验质量。

FLUX.1 Kontext 已实现上述信息结构：源图经 VAE 得到固定参考 tokens，与目标噪声 tokens 一起输入网络；采样只更新目标 tokens。其能力来自专门训练，而非给任意类别条件模型新增一个推理参数。[^1] OminiControl 也提供了以条件 tokens 和适配训练给流模型增加图像控制的先例。[^2]

若源图通过有损编码 \(E\) 才进入模型，上述理想终点是在 latent 空间的 \(E(x)\)，最后仍要经过 \(D\)。保证 latent 不变不能自动保证原始 RGB 不变。

## 2. 无需训练、已在 SiT 上运行的操作

使用原图 latent \(x\) 构造一个参考状态

\[
q_t=tx+(1-t)\epsilon,
\]

但不再让单路生成器从 \(q_t\) 重新生成一张图。另设一个编辑位移 \(d_t\)，令

\[
d_{t_0}=0,\qquad
\dot d_t=v_{\mathrm{target}}(q_t+d_t,t)-v_{\mathrm{reference}}(q_t,t),
\qquad T(x)=x+d_1.
\]

源图 \(x\) 在这次操作中始终保留。两个模型分支共用同一参考噪声。若两边的**完整有效速度场**相同，包括模型、类别、guidance、随机网络状态，那么 \(d=0\) 处导数恒为零，最后严格返回原 latent。

这里噪声只是成对评估速度的工作变量：每轮可以换噪声，两路相同作用仍会抵消。它与过去只把混合状态交给单路生成器的信息结构不同。

这个构造来自 FlowEdit 的速度差思想。[^3] 本次实现是便于检查的变体：一轮内部固定一份噪声，从 \(t=.25\) 积分到 \(1\)，使用 48 步 Heun，全程差分，没有末端单路生成。它不是官方 FlowEdit 全算法复现，也不声称精确实现某个目标分布。

同场分支实际分别调用了两次网络，没有用 `if same: return x`。以独立状态变量保存 \(d\)，也避免通过浮点的 \(x+q-x\) 运算人为产生误差。

### 这个操作为什么值得保留，又不能拿它做什么

它解决了“能否在现有 FM 外部定义一个具有明确零编辑边界的操作”这个问题，也提供了可以验证的实现控制。

但它对坏模型同样恒等。把同一个坏速度场放在两边也会抵消，因此不能拿其零编辑误差给生成模型排序。

只改变目标分支的 CFG 或 IG 时，完整有效场已发生变化；输出的位移是**相对参考场的引导作用**，不能称作同一个零编辑任务下的失败。

## 3. 已完成的实测

使用现有 SiT-S/2、ImageNet-100 的 800K checkpoint 及已有 depth-4 内部速度头；取先前实验 R0 缓存的前 4 个样本，所有实验臂共享相同初图与各轮噪声。样本是便利选取的生成图，不代表真实图、跨模型图或 100 类总体。每张图连续执行 5 轮，每轮单独计算并保存。

除 codec 对照外，主实验在轮间传递 latent，每轮解码用于观察；它是白盒操作验证，不能称作已经复现了上传 RGB 的商业编辑接口。R0 像素在本次统一数值设置下重新解码，实际与旧缓存像素不完全一致；所有误差都相对本次 R0 计算，不跨数值设置比较。

共 5 个实验臂、100 个 R1–R5 输出，端到端约 33.35 秒。每轮记录相对前一轮和相对 R0 的 latent/像素 MSE、PSNR、模型调用计数；图册显示 R0–R5 全部轮次。

| 实验臂 | 操作 | R5 平均 latent MSE，相对 R0 | R5 平均逐图 PSNR，相对 R0 |
|---|---|---:|---:|
| 同场、共享噪声 | target = reference = full | **0** | **∞** |
| 同场、独立噪声对照 | 两路 probe 使用不同噪声 | 0.56951723 | 14.32 dB |
| CFG 改变量 | target = full + 0.25(full − null)，reference = full | 0.00801315 | 29.61 dB |
| IG 改变量 | target = full + 0.25(full − weak)，reference = full | 0.03358035 | 23.82 dB |
| 单独 VAE 编解码 | RGB → posterior mode → RGB，无 FM | 0.10812245 | 24.70 dB |

CFG 在 \(t<.75\) 生效，IG 在 \(t<.5\) 生效，跨同一接受步的 Heun 两个评估使用相同引导系数；两个截止点都位于积分网格上。两者时间窗口和差向量范数不同，0.25 不是相同有效干预量。因此这个表**不能比较 CFG 和 IG 的优劣**。PSNR 表项是先逐图算 dB 再平均，不是先平均 MSE 后取对数。

| 实验臂 | R1 latent MSE | R2 | R3 | R4 | R5 |
|---|---:|---:|---:|---:|---:|
| 同场、共享噪声 | 0 | 0 | 0 | 0 | 0 |
| 同场、独立噪声 | 0.14835151 | 0.27476152 | 0.38013736 | 0.47657801 | 0.56951723 |
| CFG 改变量 | 0.00234458 | 0.00346795 | 0.00457885 | 0.00621063 | 0.00801315 |
| IG 改变量 | 0.00277565 | 0.00739382 | 0.01407558 | 0.02295225 | 0.03358035 |
| 单独 VAE 编解码 | 0.01045529 | 0.03002711 | 0.05433649 | 0.08062013 | 0.10812245 |

同场共享噪声的全部 20 个输出与各自 R0 的 latent 和解码像素均严格相同。非零引导分支确实改变结果，未绕过更新。独立噪声对照说明路径抵消依赖两路耦合，不能据此说模型差。

VAE 对照进一步说明：如果每轮按真实上传流程重新编码 RGB，即使中间 latent 编辑严格恒等，整轮仍退化为 \(D\circ E\)，可能积累编解码误差。当前 codec 对照含 8-bit 像素量化和 posterior mode 编码；不是所有误差都可以单独归于某个网络模块。各实验臂误差不可直接相减作组件因果分解。

### 图片与原始记录

- [同场恒等：4 图 × R0–R5](/home/zhoushunyu/data/eqvae/experiments/fm_identity_pilot_20260913/shared_identity/rounds.png)
- [CFG 改变量：4 图 × R0–R5](/home/zhoushunyu/data/eqvae/experiments/fm_identity_pilot_20260913/cfg_delta_025/rounds.png)
- [IG 改变量：4 图 × R0–R5](/home/zhoushunyu/data/eqvae/experiments/fm_identity_pilot_20260913/ig_delta_025/rounds.png)
- [独立噪声对照](/home/zhoushunyu/data/eqvae/experiments/fm_identity_pilot_20260913/independent_noise_control/rounds.png)
- [VAE 对照](/home/zhoushunyu/data/eqvae/experiments/fm_identity_pilot_20260913/codec_only/rounds.png)
- [全部逐图逐轮数值](/home/zhoushunyu/data/eqvae/experiments/fm_identity_pilot_20260913/results.json)、[请求与配置](/home/zhoushunyu/data/eqvae/experiments/fm_identity_pilot_20260913/request.json)、[完成记录](/home/zhoushunyu/data/eqvae/experiments/fm_identity_pilot_20260913/summary.json)
- [独立实验脚本](../experiments/fm_identity_pilot_20260913/run.py)

每个实验臂目录另有每轮完整分辨率 PNG 与保存 latent、像素、标签的 NPZ。脚本拒绝覆盖已有输出，可用 `--out` 指定新目录复现。旧两组搜索队列保持暂停；本次独立 tmux 任务 `fm_identity_pilot_0913` 已正常结束。

## 4. 真正向“把原图作为条件”推进的最小方案

在当前 SiT 上，需要增加并训练源图条件通道。模型目前的 `forward(x,t,y)` 只接收状态、时间和类别。最小可行改造是保留现有目标 patch 投影和输出头，另加一个源图投影，以及少量编辑动作 embedding；训练源投影、动作 embedding 和必要的 LoRA。源投影可以零初始化，控制初始对原模型的影响。

不能只把输入通道从 4 改成 8：官方 SiT 的输入通道和输出通道设置存在绑定，必须把条件输入与目标速度输出明确分开。另一种更贴近 Kontext 的设计是拼接源/目标 tokens、区分位置角色、只输出目标速度，但注意力算量会增加。[^4]

第一版任务应包含：

1. identity：目标就是原图；
2. 有精确目标的小编辑，如预先指定的局部颜色修改或几何变换；
3. 丢弃源条件的训练样本，明确对应哪个目标分布。

只训练 identity 容易退化为学习解析复制器。混合非零编辑才能检查模型既利用源图、又服从动作；这些合成任务也仍是机制验证，不等同于通用语义编辑能力。

此时可以定义与源图直接相关的 image-CFG：

\[
v_{\mathrm{imageCFG}}=v_{\varnothing}+s(v_x-v_{\varnothing}),
\]

两边保持相同的类别、动作、时刻、当前目标状态，只改变源图条件是否存在。必须在训练中包含这种条件丢弃，不能默认旧类别模型等于新任务的无源条件分支。InstructPix2Pix 对图像条件与文字条件分别引导的设计提供了条件拆分先例。[^5]

若研究 IG，可用同一源条件训练任务的早晚 checkpoint，或经过验证的弱分支，在同一状态上比较速度；先看 held-out 条件速度误差和已知目标编辑误差。此处才有明确的“哪个误差应当被纠正”，而不是用任意循环漂移替代真值。

另一条路线是直接用现成 Kontext 做五轮零编辑。它的接口最接近最初观察，但本次没有下载或运行该大模型。其 `guidance_scale` 是蒸馏 guidance 条件，`true_cfg_scale` 才涉及额外两次前向；官方负文本分支仍保留源图，不能把它直接解释成 image-CFG。[^6]

## 5. 不训练时，差场操作对 CFG/IG 仍有何用

设目标场是参考场加一项引导：\(v_{\mathrm{target}}=v+\alpha g\)。在 \(\alpha\) 很小、沿途场足够光滑时，展开得到

\[
\dot d=J_v(q_t,t)d+\alpha g(q_t,t)+O(\|d\|^2+|\alpha|\|d\|).
\]

令 \(\Psi(1,s)\) 为参考场 Jacobian 诱导的线性传播算子，则

\[
d_1=\alpha\int_{t_0}^1\Psi(1,s)g(q_s,s)\,ds+O(\alpha^2).
\]

这个独立推导给出了更窄、但可检验的研究问题：**CFG/IG 的差向量经过后续动力学后，哪些时间和方向被放大、哪些相互抵消？** 它不把位移小等同于质量好；高质量改进也可能需要明确改变输入。若要研究幅度以外的方向作用，还需匹配实际干预量、设置外部质量目标和独立样本。

此外，若把源路径换成参考模型真实 ODE 轨迹，且末端为原图，上述差场会等价于跨场反演—生成；并没有避开之前讨论的参考模型依赖。线性源桥与真实模型轨迹必须明确区分。

不能用强终点锚定掩盖问题。例如

\[
\dot z=\frac{x-z}{1-t}+r_\theta(z,t)
\]

若 \(r_\theta\) 沿轨迹有界，则误差 \(e=z-x\) 满足

\[
e(t)=(1-t)\left[e(0)+\int_0^t\frac{r_\theta(z_s,s)}{1-s}\,ds\right]\to0.
\]

任意有界模型残差都可被该锚点强行抹掉，甚至本来想要的编辑也会被抹掉。用这类操作获得零误差，不能证明引导被改进。

## 6. 当前判断

已找到并运行一个具有严格零编辑边界的流匹配外部操作；已确认源图条件 FM 在数学上和现成模型中都存在。还没有得到一个无需训练、仅凭原图不变就能评价无源生成模型强弱的通用判据。

若下一阶段重点是原始多模态编辑现象，原生源图条件模型最合适；若重点是当前 SiT 与 CFG/IG 的机制，源图条件适配器加已知小编辑目标更有研究价值。差场原型则可保留作明确的恒等控制与引导响应探针。

进一步推导见[构造与反例审查](research/identity_operator_20260913/fm_construct_debate.md)，条件接口细节见[原始来源与实施方案](research/identity_operator_20260913/fm_conditioning_sources.md)。此前不同“回灌”操作的区分见[前次完整研究](IDENTITY_REFEED_RESEARCH_20260913_ZH.md)。

## 来源

[^1]: Black Forest Labs et al. *FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space*. 2025，§3。[原文](https://arxiv.org/html/2506.15742v2#S3)。
[^2]: *OminiControl: Minimal and Universal Control for Diffusion Transformer*. ICCV 2025，§3.2。[原文](https://arxiv.org/html/2411.15098v6#S3.SS2)、[作者实现](https://github.com/Yuanshi9815/OminiControl)。这里只引用条件 tokens 与适配训练的可行性，不外推其参数效率或性能到 SiT。
[^3]: Vladimir Kulikov et al. *FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models*. ICCV 2025，§5 与完整算法。[原文](https://arxiv.org/html/2412.08629v2#S5)、[作者实现](https://github.com/fallenshock/FlowEdit/blob/main/FlowEdit_utils.py)。恒等条件为本次对公式的代数分析。
[^4]: SiT 作者模型代码，`SiT.__init__`、`forward`。[官方代码](https://github.com/willisma/SiT/blob/main/models.py)。实际运行所用本地实现及哈希记录于实验 request.json。
[^5]: Tim Brooks et al. *InstructPix2Pix: Learning to Follow Image Editing Instructions*. CVPR 2023，§3.2。[原文](https://arxiv.org/html/2211.09800v2#S3.SS2)。原论文是扩散编辑器，本文的 SiT 条件 FM 训练方案尚未实施。
[^6]: Hugging Face Diffusers. *FluxKontextPipeline*，2026-09-13 查询的官方管线实现。[源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_kontext.py)。运行时需固定实际库版本，本次未运行此管线。
