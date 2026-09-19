# 流匹配中的原图条件与恒等操作：已核实构造和三个实施方案

日期：2026-09-13。本文是对“流匹配里是否存在要求输入保持不变的操作”的可行性核查。只查阅原始论文、作者实现与本地源代码；没有启动 GPU、下载权重或恢复旧实验。适用目录及祖先目录未发现 AGENTS.md。

**答案是存在，而且无需放弃从高斯噪声开始的流匹配。关键是把源图作为始终可访问的独立条件，而不是只把源图混入正在积分的状态。现成实例是 FLUX.1 Kontext；当前类别条件 SiT 尚未学习这种接口。另可不训练地构造恒等算子，但这类算子的恒等性通常是代数保证，不能据此给生成模型排名。**

## 1. 一个直接的存在性证明

以下是本次推导，统一采用本仓库的时间方向：噪声端为 0、图像端为 1。令需要保持的源 latent 为 \(x\)，随机初始化为 \(\epsilon\)，**源 \(x\) 在整个过程中作为额外条件保留**。取

\[
z_t=(1-t)\epsilon+tx,
\qquad
v_{\rm id}(z,t;x)=\frac{x-z}{1-t},\quad t<1.
\]

ODE \(\dot z_t=v_{\rm id}(z_t,t;x)\) 的解就是上面的直线路径，因此

\[
\lim_{t\uparrow1}z_t=x
\]

对任意初始化 \(\epsilon\) 成立。它是条件终点分布 \(p(z_1\mid x,\mathrm{id})=\delta_x\) 的显式流。这里不能误说为“有限时间、全程光滑可逆的流把连续噪声压成一个点”：速度表达式在末端奇异，严格命题是端点极限。理想场的直线路径可由只在左端点取值的 Euler 积分精确推进到终点，但学习场、舍入误差和不同求解器不自动继承这个性质。

这证明**流匹配没有禁止恒等任务**。它也揭示研究上的限制：若直接把这个已知速度写进程序，根本不需要生成先验；恒等成功是任务被显式实现的结果。若只访问 VAE 条件 \(E(I)\)，这里保证的是 latent 目标 \(E(I)\)，像素输出为 \(D(E(I))\)，不能自动得到原始 RGB 的逐像素恒等。

## 2. 方案一：原生图像条件流匹配编辑器，首选用于验证真实现象

### 已核实来源

FLUX.1 Kontext 的条件分布是 \(p_\theta(x\mid y,c)\)，其中 \(y\) 为源图，\(c\) 为编辑指令。论文 §3 将源图 VAE tokens 拼在目标 noisy tokens 后面，使用不同位置标识；训练从文生图 checkpoint 出发，在图像关系数据上微调。其 dev 版经 guidance distillation，主要针对图像编辑训练。这些内容不是推理时给任意 FLUX 权重多传一个参数就会自动获得的能力。[Kontext 原文 §3](https://arxiv.org/html/2506.15742v2#S3)

官方 Diffusers 的 `FluxKontextPipeline` 把源 `image_latents` 与目标随机 `latents` 分开创建；每次前向拼接两者，只截取目标部分的速度并更新目标 latent，源条件不被积分。当前实现中，`guidance_scale` 传入模型作为蒸馏 guidance 条件；`true_cfg_scale` 在提供负提示词且大于 1 时额外做两次前向，负文本分支仍保留相同源图 tokens。因此它不是“去掉图像条件”的 image-CFG。[官方管线实现：prepare_latents 与 denoising loop](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_kontext.py)，[接口文档](https://huggingface.co/docs/diffusers/api/pipelines/flux#diffusers.FluxKontextPipeline)

### 具体实验

使用已经训练好的 `black-forest-labs/FLUX.1-Kontext-dev` 时，用户侧不需再次训练。调用形态为：

```python
result = pipe(
    image=current_image,
    prompt="Return the input image unchanged. Preserve all content and details.",
    num_inference_steps=28,
    guidance_scale=2.5,
    true_cfg_scale=1.0,
    generator=generator_for_this_round,
).images[0]
```

此处数值是首轮协议选择，**不是论文证明最优的参数**。每轮将可见输出重新作为唯一源图，重复五轮；保留输入尺寸、图像预处理、完整像素输出与每轮独立随机种子。另跑 VAE 往返控制以及一个确实要求改变局部的编辑指令。若尺寸自动调整，要把调整后的规范源图定义为被比较对象，并保存原始图与规范图两者；否则几何预处理会冒充生成漂移。

恒等在这里是明确的**任务目标**，不是架构硬保证。也不能据现成论文断言零编辑指令已作为严格逐像素监督加入训练。这个方案最接近用户的最初观察，但尚不能仅凭 no-op 排出“生成先验强弱”。

### CFG / IG 研究接口

先区分文本 true-CFG、蒸馏 guidance 输入和图像条件强度，不能都命名为同一个 \(w\)。在同一编辑器上改变明确记录的一种引导，观察零编辑误差和小编辑成功率。若要做 weak–strong IG，应使用同一 latent 坐标、相同源图输入的模型或适配器对，在同一当前状态上评估两路速度。人为跳层只是一种待验证的弱化干预，不应预先称为“更差但同分布”的模型。该接口的收益需要真实编辑和生成评价独立确认。

## 3. 方案二：给现有小 SiT 增加源图条件，作为便宜的机制实验

### 本地检查

本仓库 [训练入口](../../../experiments/train_imagenet100_sit_flow.py) 的 `load_official_sit_module` 加载 `/home/zhoushunyu/data/research_repos/SiT/models.py`。已读实际代码：`forward(x,t,y)` 仅接收积分状态、时刻和类别，条件为 time embedding 加 label embedding。原图独立 tokens、图像适配器、语言指令均不存在。[官方 SiT 对应模型源代码](https://github.com/willisma/SiT/blob/main/models.py)

因此无需训练的“直接把原图作为新条件”不能被称为 SiT 已具备的能力。可以修改算法把源图注入，但修改后的输入语义与效果需要检验；加噪初始化、同类别、同种子、把图像随意拼到 batch 或 sequence 都不会自动建立复制契约。

### 可实施的最小改造（本次设计，不声称已实验）

取源图 latent \(x\)、编辑动作 \(a\)、目标 latent \(y\)。先使用有精确监督的合成小任务：identity、已知局部像素编辑后重新编码、一个预先声明的几何变换。动作可用少量离散 embedding 表示，不需要先接大型语言编码器。

目标侧正常做 FM：

\[
z_t=(1-t)\epsilon+ty,\qquad
\mathcal L=\mathbb E\|v_\theta(z_t,t,\ell,x,a)-(y-\epsilon)\|^2.
\]

源 \(x\) 始终未加噪。最小工程版本是在 patch 输入层引入独立源投影 \(P_x(x)\)，与原目标投影 \(P_z(z_t)\) 相加，源投影初始为零；保留原输出通道数，训练源投影、动作 embedding 和必要的 LoRA。**不能仅把 SiT 的 `in_channels` 从 4 改成 8**：官方代码把 `out_channels` 和 `in_channels` 绑定，这会同时改变输出头。应明确分离条件输入通道与目标输出通道。

更接近 Kontext 的版本是把目标与源 tokens 拼接，加源/目标类型位置标记，并仅输出目标部分；它改变序列长度，算量更高，需要适配注意力。OminiControl 已在 FLUX 上验证复用 VAE、拼接条件 tokens 和 LoRA 适配这条路线，因此“少量可训练参数增加图像条件”有先例；其具体参数比例和成功率不能直接外推到这里的 SiT。[OminiControl，ICCV 2025，§3.2](https://arxiv.org/html/2411.15098v6#S3.SS2)，[作者代码](https://github.com/Yuanshi9815/OminiControl)

为得到含义明确的 image-CFG，训练时要包含源条件丢弃的分支，并保持类别 \(\ell\) 一致：

\[
v_s=v_{\varnothing}+s\,[v_x-v_{\varnothing}],
\quad v_x=v_\theta(z,t,\ell,x,a),\quad
v_{\varnothing}=v_\theta(z,t,\ell,\varnothing,a).
\]

若同时研究类别 CFG 或动作 CFG，应另外定义对应的条件丢弃，不能混入同一个差分。只训练 identity 数据时，丢弃源图后仍是类别数据分布；混入编辑任务后，无源分支的目标分布可能变化，需真正训练该分支，不能默认冻结的旧 SiT 就等于新的无源条件模型。

InstructPix2Pix 已为图像和文字分别做条件丢弃并使用两个 guidance 系数，提供了可参考的条件拆分方式；它自身是 diffusion 模型，此处将线性引导组合应用于同参数化 FM 速度是本次方案，不能说原论文验证了这里的 SiT 版本。[InstructPix2Pix，CVPR 2023，§3.2–3.2.1](https://arxiv.org/html/2211.09800v2#S3.SS2)

强弱分支可以选同一训练流程的早晚 checkpoint，固定编码器、训练任务和源输入。测量 held-out 图上的条件速度误差、五轮 latent/像素误差，以及非零动作的正确率。只做 identity 训练会使任务接近学一个解析复制器；这是原图条件能力的最小验证，**不会单独产生可发表的“生成模型更强”结论**。训练成本与可达误差目前尚未估计，不应直接恢复 20 组调参。

## 4. 方案三：FlowEdit 式速度差，作为无需训练的恒等控制

FlowEdit 用源和目标分支的速度差直接演化编辑状态。作者实现明确区分差分演化段与可选的末端单分支生成段，后者由 `n_min` 控制；`n_min=0` 才全程保持差分形式。[FlowEdit，ICCV 2025 原文](https://arxiv.org/html/2412.08629v2#S5)，[作者实现 `FlowEdit_utils.py`](https://github.com/fallenshock/FlowEdit/blob/main/FlowEdit_utils.py)

用原论文的时间方向（数据端 0、噪声端 1）写单个噪声 probe：

\[
q_t=(1-t)x+t\epsilon,\qquad e_{\rm init}=x,
\]
\[
\frac{de}{dt}=v_{\rm tar}(q_t+e-x,t)-v_{\rm src}(q_t,t).
\]

以下是从公式和代码得到的代数推论：若两分支为**完全相同的有效速度场**，包括模型、提示词或类别、guidance、网络随机状态均相同，且共享同一 \(\epsilon\)，则 \(e=x\) 时右端严格为零；逐步保持原 latent。它允许每次取新的噪声，因为两分支中噪声的作用成对抵消。需关闭末端单分支采样，否则又进入不同操作。像素输出仍需另算 VAE 往返。

这一构造可以迁移到类别条件 SiT：把源/目标文字条件换成类别，并按仓库时间方向一致地改写。但它不让 SiT 学会语言“不变”指令，也不是原生源图条件训练；源图信息由外部差分算法持续使用。

最适合作为原型代码的正控制：任意强弱模型各自左右两边相同，都应接近零误差。这样确认实现确实具有 identity，而不是重复旧的单路加噪生成。**它不能把强弱模型拉开，因为恒等正是对任意场都成立。** 若把 strong 放一边、weak 放另一边，得到的是跨场编辑/运输差，不再有真值等于原图的理由；若只改目标 CFG，同样是在测 guidance 对参考场的响应。

## 5. 选择建议与可否定命题

| 方案 | 用户侧是否训练 | 恒等来自哪里 | 最适合回答什么 | 能否单靠恒等误差证明更强生成先验 |
|---|---|---|---|---|
| 原生 Kontext 图像条件 | 用现成权重不需训练；能力本身来自专门训练 | 任务要求、模型近似 | 真正的五轮零编辑是否漂移，哪些组件和引导导致漂移 | 不能 |
| 小 SiT 源图条件适配 | 需要 | 有监督条件 FM 目标、模型近似 | 原图条件如何消除来源歧义，image-CFG/IG 如何影响已知目标误差 | 不能；可形成机制证据 |
| FlowEdit 对称速度差 | 不需要 | 成对代数抵消 | 无需训练也可定义恒等；代码正控制 | 不能，任意场均可通过 |

最小且有意义的推进顺序是：先用解析条件桥和对称速度差作确定的算子控制；若已有原生编辑器权重，则直接做极小五轮试验；若重点必须保持现有 SiT 体系，则训练最小源条件适配器。零编辑成功证明的是“此系统能使用源条件保持目标”，必须额外检验非零编辑或生成任务才能连接回 CFG/IG 的更广泛收益。

**应被否定的旧前提：**“纯类条件 FM 只要模型足够好，就应该在随机加噪后回到同一原图。”

**现在可检验的新前提：**“在明确保留源图条件和任务目标后，近似条件速度场的误差与引导组合是否产生可预测、可减少的身份漂移？”这有严格目标、可分离的误差来源，也不要求把再采样误认作复制。
