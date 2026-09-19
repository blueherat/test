# 反演与未来端点：有界文献核查及一个候选

2026-09-13。只读直接相关论文、作者源码和已有仓库记录；未启动 GPU、采样或新队列。本次不扩展 guidance 分解、方法融合、时间调度等支线。

**最有用的文献事实：原图条件可以在采样器层通过解析边界约束实现，不必存在模型内部的 image-condition 接口；但反演更准本身不会把已经生成的图像变得更好。** 下文只保留一个尚未验证的正则化反演候选，不把它当成已有质量结果或独创方法。

## 1. 强前弱逆是不是 Euler 取负？

应分成三个层次，不能笼统回答“是”。

- Z-Sampling、W2SD 的已核作者 SDXL 实现：强 scheduler 前进；在**推进后的状态**重新运行弱预测，再调用 inverse scheduler；并非缓存原强速度后直接减回去。但弱预测仍用前一步模型时间 `t`，inverse scheduler 接收 `timesteps[i+1]`，是相邻时间近似，没有解隐式离散逆方程。SDXL 的 DDIM/Euler scheduler 系数应按原参数化描述，不能直接把全部版本写成 FM 的 `x-h*v`。[Z 固定提交](https://github.com/xie-lab-ml/Zigzag-Diffusion-Sampling/blob/eef8bb265deb8f33efd47c53e6ee5506606de360/utils/pipeline_stable_diffusion_xl.py#L1444)、[W2SD 固定提交](https://github.com/xie-lab-ml/Weak-to-Strong-Diffusion-with-Reflection/blob/c1c160d634837e92dd98851a74b52aa5415f5da5/utils/pipeline_stable_diffusion_xl.py#L1099)。FSG 的已核单步前瞻及 scheduler 时钟问题沿用[旧核查](fsg_w2sd_prior_art.md)，这里不重新扩成综述。
- 正确的 FM 逆向 Euler 是 `y-h*v_weak(y,t+h)`：已到新状态、新物理时间后重评估。它是逆 ODE 的一阶近似，仍不是精确撤销之前的显式 Euler。
- 精确撤销弱场 Euler 映射 `F_C(x)=x+h*v_c(x,t)` 要解 `x=y-h*v_c(x,t)`；未知前像上的网络调用不能省略。精细逆 ODE 则从 y 出发沿 `s→t` 多步积分，两者目标不同。

统一 `t=0` 噪声、`t=1` 图像。定义 `H=Φ_high(s←t)`、`C=Φ_cond(s←t)`。精确 `C⁻¹H` 返回同一时间层，通常改变状态；若下一腿也是同区间 C，则 `C C⁻¹ H=H`，只能复现已经计算的强场未来。再走 H 才是 `H C⁻¹ H` 的非平凡 reflection；基本构造已有 Z/W2SD 前例。

## 2. 三篇 RF 文献分别实际提供什么

|来源|实际操作与原图使用|可借用到当前 SiT 的部分|不能继承的结论|
|---|---|---|---|
|RF-Inversion，arXiv:2410.10792，2024-10-14 / ICLR 2025|逆腿把模型逆流与朝指定 Gaussian 端点的解析场混合；生成腿把模型场与朝**固定原图 latent**的解析场混合。原图是采样器的持续边界数据。|解析端点条件，无须新增图像 encoder；可将固定原图替换为自生成的未来状态，但这已改变任务。|外部图像编辑中的真实性/忠实度经验收益，不保证自生成 guide 能改善 CFG 分布；原方法的随机 noise 端点不能悄悄带入无 fresh-noise 任务。|
|RF-Solver / RF-Edit，arXiv:2411.04746，2024-11-07 / ICML 2025|二阶 midpoint 正/反向积分；RF-Edit 另存并注入 inverse 轨迹的 attention V。|正确物理时间、中点重评估、精细 inverse；当前 SiT 可直接用普通 midpoint/Heun。|数值 solver 本身没有独立原图条件；V 注入是另一机制，不能把编辑收益都归给高精度反演。|
|FireFlow，arXiv:2412.07517，2024-12-10 / ICML 2025|首步 midpoint，后续借上个中点速度预测当前中点，当前中点仍重评估；编辑另可注入 V。|在较直的场中节省逆积分调用的缓存策略，精度需本模型验证。|不是精确离散逆；FLUX 低步数结果不自动适用于当前 SiT；不能继承“仅 Lipschitz 就保证逆流收缩”的表述。|

原始来源：[RF-Inversion §3.3、§3.5、§4](https://arxiv.org/html/2410.10792v1)、[RF-Solver §3.2、Eq.(10–12)](https://arxiv.org/html/2411.04746v1)、[FireFlow 正式出版](https://proceedings.mlr.press/v267/deng25c.html)。算法源码：[RF-Solver](https://raw.githubusercontent.com/wangjiangshan0725/RF-Solver-Edit/main/FLUX_Image_Edit/src/flux/sampling.py)、[FireFlow 的 denoise_fireflow](https://raw.githubusercontent.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/main/src/flux/sampling.py)。后两链接为当日 main 读取，未把它们当固定提交复现实验。

RF-Inversion 的关键生成式，在本文物理时间下为

\[
\dot x_t=(1-\eta_t)v(x_t,t)+\eta_t\frac{y_{\rm source}-x_t}{1-t}.
\]

这不是 `v(x,t,c,image)` 的训练条件接口。作者 README 明确同时传递 `inverted_latents` 与 `image_latents`；现 diffusers community 实现将后者保存为 `y_0`，用于解析约束。逆腿还执行 `y_1=torch.randn_like(Y_t)`，所以原操作不能冒充“完全没有引入新 noise”的反演。[作者接口](https://github.com/LituRout/RF-Inversion)、[community 实现的 invert / __call__](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/pipeline_flux_rf_inversion.py)。

两个实现细节值得避免照抄：

1. RF-Inversion Appendix B.1 的有限终端惩罚解实际是 `(y−x)/(1−t+1/λ)`；正文的 `(y−x)/(1−t)` 取了 `λ→∞` 极限。硬终点条件和有限惩罚应区分。向接近端点的有界场混入奇异吸引项，也可能机械强迫重建，不能用此证明模型更强。
2. RF-Solver v1 Algorithm 1 的差分顺序与 Eq.(11) 相反；当前作者源码使用 `(pred_mid−pred)/(h/2)`，得到普通显式 midpoint。移植应按公式与代码核对后的有符号 h 实现，而非复制那个算法框的反号。FireFlow 换场或开始一条新 inverse 时应重置缓存，不能继承另一路场的速度。

RF-Inversion 相关论文概述只用于以上端点机制；不从其 SDE 等价性推导当前自生成锚点下的真实分布保证。更详细的逆向稳定性反例已在[之前的文献核查](inversion_literature.md)记录。

## 3. 唯一保留候选：带原状态约束的未来端点反演

这是一个**正则化逆问题**，不是要求所有反演都恒等，也不是增加一个 guidance schedule。它借用“以固定图像/未来为边界数据”的思想，但不照搬 RF-Inversion 的随机噪声端点或奇异解析场。

给定当前状态 `z`、类别 `c`、区间 `[t,s]`，`h=s−t>0`。定义完全冻结的同区间多步映射 `H_K`、`C_K`：前者采用固定高 CFG，后者普通 conditional；两者共用 K 个 Heun 或 midpoint 子步，网络、物理时间、类别不变。

**前向目标。** 先计算并冻结

\[
y=\operatorname{stopgrad}(H_K(z)).
\]

y 是时间 s 的未来 latent；除非 s=1，不叫最终图像。它在反演迭代中不随当前猜测改变，也不借用真实图像或分类器评分。

**实际逆向。** 从 y 沿 conditional 场用负步长及正确时间 `[s,t]` 求一个逆 ODE 初值，再求下列固定目标的局部最小值：

\[
\boxed{x_\lambda=\arg\min_x\;
\frac12\|C_K(x)-y\|^2+\frac\lambda2\|x-z\|^2.}
\]

这里 `λ>0` 时明确是**有偏的正则化前像**，不是声称求到了原 conditional ODE 的精确逆。更适合实现的速度单位变量是 `x=z+h*d`，将整个目标除以 `h²`：

\[
\min_d\frac12\left\|\frac{C_K(z+h d)-y}{h}\right\|^2+
\frac\lambda2\|d\|^2.
\]

对 frozen model 的输入做少量优化即可，不需训练权重。梯度是

\[
\nabla_dJ=J_{C_K}(z+h d)^T\frac{C_K(z+h d)-y}{h}+\lambda d.
\]

可以对 K 步可微 conditional rollout 反传获得 VJP，重计算或 checkpoint 控制显存；`x=z` 是始终可评估的保底状态。反向 ODE 初值只作为候选，若其正则目标比 z 更差，应从 z 开始求解。优化停止依据是固定目标与位移约束，不是较小 cycle residual 就宣称质量更好。

**随后推进。** 从同时间 t 的 `x_λ` 继续已经冻结的普通 CFG 生成过程，先跨到 s，再接原后缀；不接同一区间 C 来制造精确抵消，不迭代更换目标 y。

### 为什么这比“更准地反走”多了一个可检验问题

令 `r=y−C_K(z)`、`A=J_C_K(z)`，局部线性化后的位移为

\[
\delta x_\lambda=(A^TA+\lambda I)^{-1}A^Tr.
\]

普通精确逆在奇异值 σ 方向按 `1/σ` 放大目标差异；正则逆使用 `σ/(σ²+λ)`，限制收缩强的 conditional 流方向所导致的前像放大。**可验证假设**是：强 CFG 未来包含可用的条件信号，但直接反演也放大了其中一些不可靠分量，正则逆可能保留前者而减轻后者。这不是“高 CFG 就是真教师”的结论，λ 也可能过滤掉真正有用信号。

三个精确边界可帮助判定实现：

- 若 `H_K=C_K`，则 z 的目标值为 0，`λ>0` 时它是唯一全局最小点；同场不应凭空改变状态。
- `λ=0` 且局部逆可解并求准，恢复固定未来的离散前像；再接同 C_K 必然回到 y，这只能认证操作，不认证质量。
- 若两场为常向量，`x_λ−z=h*(v_H−v_C)/(1+λ)`，完全等于缩小反射强度。因此不能把收益仅归给“反演结构”；有价值部分须来自有限区间非线性及各向异性的逆敏感性。

### 与原文及仓库旧工作的区别、成本与裁决

RF-Inversion 以外部图像和另一 Gaussian noise 为边界，混合解析场；这里以自己算出的未来 latent 为目标、原状态为有限惩罚中心，求 conditional 多步映射的前像。RF-Solver/FireFlow只提供反向数值积分组件。Z/W2SD 的未正则 reflection 是 `λ=0` 的相关基线；仓库旧 anchored inverse 已经做过冻结多步目标、Picard/Anderson 和逆流，不能把“固定目标”重新声称新颖。本候选新增的研究对象仅是**在相同未来目标下是否应抑制病态逆方向**，其正则形式本身属于标准逆问题技术。

这是比单次 inverse 更贵的候选：K=4 时，生成高 CFG 目标约需 16 次单分支 forward，conditional inverse 初值约 8 次；每次优化还需 8 次 conditional forward 及对应输入反传，另有最终生成。反传不应假记为零 NFE；真实显存/耗时未测，不承诺等成本优势。首次若被授权实现，应只验证一个短区间、小批量与少量优化迭代，不先启动大规模搜索。

**当前裁决：可实现、与反演直接相关，但没有证据称它有把握超过 baseline。** 如果同样改动半径的普通精确反射就得到相同结果，或优势只来自更高预算/更弱有效强度，应直接否定其独立结构收益。若父任务选择了更清晰的反演构造，本候选只保留为研究备忘，不形成第二条实验队列。
