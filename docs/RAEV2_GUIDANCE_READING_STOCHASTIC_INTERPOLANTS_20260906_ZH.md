# Stochastic Interpolants：配对桥、边缘保持与随机动力学的误差控制

2026-09-06。精读 Michael S. Albergo、Nicholas M. Boffi、Eric Vanden-Eijnden 的 *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*，以 [JMLR 26(209):1–80 (2025) 正式版](https://jmlr.org/papers/v26/23-1605.html)、[正式 PDF](https://jmlr.org/papers/volume26/23-1605/23-1605.pdf) 为准。[arXiv 2303.08797](https://arxiv.org/abs/2303.08797) 当前 metadata 标注 v4 为 JMLR Version；没有把多个版本或两个代码库计作多篇论文。

**保留的正面机制是：先由一个明确的随机配对构造边缘路径，再把路径导数投影为当前位置的条件均值速度；正确 score 能把这条路径实现为不同扩散水平的 SDE，而共同扩散产生的 Fisher 耗散能吸收漂移误差。** 这为当前有限配对桥的速度目标提供基础，但不直接给现有两次 midpoint、无 score 的确定性实现添加 KL 保证，也没有推出可直接使用的噪声系数。

此前精读记录的阅读范围：正文 §2.1–2.4、§6、§7；Appendix B.1（Theorems 6–8、FPE）、B.3（KL 恒等式与 Theorem 23）逐式复核；B.2 对照反向时间约定；§4.3、§5.3 相关论述及 Appendix A Gaussian 情形；Appendix C/C.1 的训练和采样开销。未逐项复核 §2.5 所有 likelihood estimator、§3.4 Schrödinger bridge 最优控制证明及全部其他附录。完整 PDF 和逐页文本共80页，已有独立图像为第17、19、52、66、74页；这些图像仅对应上述选定页，不表示每个阅读页都有单独渲染。下文分别标明原文结果与本地补充推导。

归档位置：`/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_stochastic_interpolants_v1`。本轮没有修改任何冻结实验、运行 GPU、训练或生成新图；代数检查不是性能实验。

**归档收尾边界：** 本次接续时正文笔记及24份来源/派生文件已经存在；先将其 SHA、字节数及观测到的文件时间保存为 `closeout_recovery.json`，没有重下载材料。随后另一个代理独立查看上述5张原页渲染，对照第9、12–15、17、19、60–61、63–66页的关键定义和证明，并复算下述两个标量例；不把这次定点复核写成重新通读80页。`closeout_verification.json` 记录收尾 UTC 时间、来源与页码检查、独立代数结果；`manifest.json` 逐文件绑定来源或派生关系，`note_snapshot.md` 保存本笔记快照。

## 1. 配对可以相关，新增噪声的独立性不能省

原文 Definition 1（p.9）定义

\[
U_\tau=I(\tau,Y,W)+\gamma(\tau)\xi,
\qquad (Y,W)\sim\nu,\quad \xi\sim N(0,I),\quad \xi\perp(Y,W).
\]

ν 只须有指定的两端边缘；**不要求 Y、W 相互独立**，ν 也不要求具有联合 Lebesgue 密度。I 的边界为 I(0,Y,W)=Y、I(1,Y,W)=W；要求 I 有相应 C² 正则性与导数矩控制。γ²∈C²、γ(0)=γ(1)=0，且 γ 在内部为正。Assumption 5 另要求两端密度严格正、C²、有有限 Fisher information，并控制 ∂τI 的四阶矩和 ∂τ²I 的二阶矩。它不是任意离散或流形数据均自动满足的陈述。

**应用于本仓库的判断：** 当前 Y=T_native((1−t)X+tε)、W=(1−s)X+sε 使用同一 X、ε，是一个允许的相关 coupling。条件于 X、ε 时两端是确定的，训练条件路径是 Dirac，这本身不妨碍对 joint 取期望后的弱连续性方程。需要区分“条件路径是 Dirac”和“端点边缘本身是 Dirac”：后者若要由良性确定性 ODE 送到非退化分布，确实存在障碍。

当前 Uτ=(1−τ)Y+τW 没有额外 γξ，因此属于原文 Remark 11 的 γ=0 情形。条件均值速度的弱公式仍成立；原文由 Gaussian 卷积导出的内部严格正密度、C∞ 平滑以及简单 noise-to-score 标签不再自动成立。不能把原来的 ε 当作 ξ：ε 已参与 Y、W，且通过非线性 G(Z_t) 进入中间状态。

## 2. 速度与 score 目标从哪里来

原文 Theorems 6–8 的主要内容可以直接从测试函数看清。对于光滑紧支撑 φ，设适用支配收敛且导数可积，则

\[
\frac d{d\tau}E\phi(U_\tau)
=E[\nabla\phi(U_\tau)\cdot(\partial_\tau I+\dot\gamma\xi)]
=E[\nabla\phi(U_\tau)\cdot b_\tau(U_\tau)],
\]
\[
b_\tau(u)=E[\partial_\tau I+\dot\gamma\xi\mid U_\tau=u].
\]

因此 ∂τρ+div(ρb)=0。不是每个配对样本必须沿 b 的轨迹走；interpolant、ODE 与 SDE 只须有相同单时刻边缘，其路径 law 一般不同（Remark 20）。原文 B.1 用 characteristic function 证明同一结论。

原文采用的无目标常数项损失为

\[
L_b(\hat b)=\int_0^1E[\tfrac12\|\hat b(U_\tau,\tau)\|^2
-\hat b(U_\tau,\tau)\cdot(\partial_\tau I+\dot\gamma\xi)]d\tau.
\]

由条件期望的正交投影性质，`ΔLb = Lb(bhat)−min Lb = (1/2)∫Eρ||bhat−b||²`。去掉目标平方不是去掉机制，反而避免把不可约条件方差误称可学习速度误差。当前 R/β 归一化回归与此对应：无额外噪声且线性 I 时，真实速度是 E[R|Uτ]，β 只改变网络输出单位和有限拟合权重。

独立 Gaussian 的分部积分给

\[
s_\tau(u)=\nabla\log\rho_\tau(u)
=-\gamma(\tau)^{-1}E[\xi\mid U_\tau=u],\qquad0<\tau<1.
\]

对应 `Ls(shat)=∫E[.5||shat||²+γ⁻¹ ξ·shat]dτ`，`ΔLs=.5∫Eρ||shat−s||²`。也可回归 η=E[ξ|U]，避免标签显式除 γ；但把 η 变成 s 时仍要除 γ，端点数值问题没有凭空消失。原文 §6.1 的 antithetic ±ξ 能消去损失中 `γ⁻¹ ξ·shat(I)` 的高方差首项，需要两份实际函数评估。

**本地补充：** 当前没有可直接套用的 `−ε/s` 中间 score 标签。无 floor 的原生步可写

\[
U_\tau=\frac{s}{t}Z_t+\beta[(1-\tau)G(Z_t)+\tau X].
\]

条件于 X，若理想 G 可微，其对原 ε 的 Jacobian 是 `s I+(1−τ)βt J_G(Z_t)`；除 τ=1 或特定线性结构外，不是已知标量 times identity。对这个非线性变换用 Gaussian 分部积分会涉及逆 Jacobian 及其导数条件，不能忽略后把原 ε 当新增独立加性噪声。

## 3. “任意扩散都保持边缘”的准确含义

原文 Corollary 10 的代数依据是 `div(ρs)=Δρ`。给定同一条**准确**路径 ρ、准确 b、准确 s 和非负时间函数 a(τ)，

\[
dX_\tau=[b_\tau(X_\tau)+a(\tau)s_\tau(X_\tau)]d\tau
+\sqrt{2a(\tau)}dB_\tau
\]

有 FPE `∂τρ=−div((b+as)ρ)+aΔρ=−div(bρ)`。在初值正确、方程适定与相应正则性/非爆炸条件下，它保持指定边缘路径。这是 a 的**实现自由度**，不是分布边界唯一识别了 a，也不是任意 a 在学习与离散误差下都一样好。

原时间从1向0走时，backward drift 为 b−as。若统一用正向变量 u=1−τ，应写

\[
dZ_u=[-b_{1-u}(Z_u)+a(1-u)s_{1-u}(Z_u)]du
+\sqrt{2a(1-u)}dB_u.
\]

这里噪声和漂移都必须转换时间。正式 PDF 第17页 (2.35) 把噪声写作 `sqrt(2a(u))`，与第15页 (2.24) 的正确 `a(1−u)` 不一致；常数或时间对称 a 不受影响。Appendix B.2 第63页 (B.32) 的证明使用常数扩散，不能消除这个时间依赖情形的差异。本地算例：ρτ=N(0,1)、b=0、a(τ)=1+τ，正确反向是 `dZ=−(2−u)Z du+sqrt(2(2−u))dB`。若沿用未反转的噪声，初始方差导数为−2，立即偏离要求的常方差。

同样，若直接在准确 ODE 速度 b 上加 `sqrt(2a)` 噪声而不加 as，额外的 aΔρ 不会抵消。若用有限 `bhat,shat`，实际漂移误差是 `δb+aδs`；不能把 bhat 对应的 pseudo-score 当当前 q 的 score，也不能从某条 Gaussian 边缘路径单独宣称 a 是唯一自然系数。

γ 与 a 的角色还不同：γ 改变训练 interpolant 的中间 law；a 在准确 b、s 已知后改变实现该 law 的 Markov 过程。将二者绑定为 Brownian bridge 某一常数温度是额外模型选择。本文 §3.4 讨论通过进一步优化 interpolant 恢复 Schrödinger bridge；不等于任意当前 coupling 已是该最优桥。

## 4. L² 漂移误差何时控制终端 KL

原文 Lemma 22/B.3 的核心结果可靠。设两条充分正则的 FPE 有相同初始分布、相同正扩散 a>0，准确漂移 f、近似漂移 fhat；记准确 law 为 ρ、生成 law 为 q，h=log(q/ρ)，δf=fhat−f。分部积分成立、所有项可积时

\[
\frac d{d\tau}D_{KL}(\rho_\tau\Vert q_\tau)
=E_\rho[\nabla h\cdot\delta f]-aE_\rho\|\nabla h\|^2
\le\frac1{4a}E_\rho\|\delta f\|^2.
\tag{A}
\]

积分后得到 `KL(ρ1||q1)≤(1/4a)∫Eρ||δf||²`。扩散提供负 Fisher 项，使速度误差与未知 score 差的内积可以用完成平方消去。权重在**准确 teacher 边缘 ρ**，不是 learned rollout q；KL 方向是 target-to-model，不是本仓库最早有限步笔记的 model-to-target。

**本地直接推广：** 初始 KL 不为零时加上初值 KL；a(τ)>0 的时间依赖版本将右侧换成 `∫Eρ||δf||²/[4a(τ)] dτ`。若 a 允许在端点趋零，必须证明这个加权积分有限及 KL 极限合法；不是把积分形式抄下即可通过。需共同扩散、可比密度、适定动力学、有限熵及积分尾部条件；不存在所有任意 C¹、可能爆炸的漂移都自动有该终端 law 的无条件保证。

对于 `f=b+as`，精确上界还可展开为

\[
\frac{A}{4a}+\frac C2+\frac{aB}{4},\quad
A=\int E_\rho\|\delta b\|^2,
B=\int E_\rho\|\delta s\|^2,
C=\int E_\rho[\delta b\cdot\delta s].
\]

故两种场误差的相关性也进入漂移误差。对固定 bhat、shat 和同一准确路径，常数 a 的理论最优比例是 `sqrt(A/B)=sqrt(ΔLb/ΔLs)`；原文 Remark24 也给出该比例。**不可直接由两份原始 MSE 或当前训练损失求得：** 它们含不同的未知总体最优值、不可约条件方差与时间权重。更大 a 会增加有限步积分成本，本文也明确提醒 a→∞ 不可实际免费实现。

### 正式版 Theorem23 的系数需要更正

B.23/B.30 明确给 `ΔLb=A/2, ΔLs=B/2`。B.44 正确得到 `KL≤A/(2a)+aB/2`，但转为 B.45/(2.45) 时写成 `ΔLb/(2a)+aΔLs/2`，少了2。没有额外误差正交假设时，安全版本应为

\[
\boxed{KL(\rho_1\Vert q_1)\le\Delta L_b/a+a\Delta L_s.}
\tag{B}
\]

这不是只看排版猜测。本地固定反例满足其模型假设：Y,W,ξ 独立 N(0,1)，I=(1−τ)Y+τW、γ²=2τ(1−τ)，所以 ρτ=N(0,1)、b=0、s=−x。取 a=1、bhat=1、shat=−x+1。生成 SDE 为 `dX=(−X+2)dτ+sqrt(2)dB`，X0~N(0,1)，终端 N(2(1−e⁻¹),1)。真实 KL 为 `2(1−e⁻¹)²=0.799152801787456`，原文 (2.45) 右侧仅0.5；Lemma22 和更正后的 (B) 均给1。原文 (2.46)/(B.48) 的 `Lv,Ls` 转换同样少2。**Fisher 耗散机制与最优比例并不因此失效**，只是不能引用印刷版较强常数。

此前代数复算保存在 `analytic_checks.py/json`，无随机试验或拟合。本次收尾代理已独立核对第19、60–61、66页的原始损失及系数，并用80位 Decimal 复算此例及反向时间系数，结果保存在 `closeout_verification.json`。这是本地数学核验，没有声称得到作者确认的勘误。

### 确定性 ODE 的对应边界

a=0 时只能保留原文 Lemma21 的精确恒等式

\[
KL(\rho_1\Vert q_1)=\int E_\rho[(\nabla\log q-\nabla\log\rho)\cdot(\hat b-b)]d\tau
\]

（同初始 law）。负 Fisher 项消失。L² 速度误差小，不能单独推出 terminal KL 小。可在额外 Fisher 控制、全局流稳定性/导数条件下建立相应界，但这些条件没有由当前神经网回归损失自动证明。原文也没有给两次 midpoint 的离散 KL 界。

对当前配对桥，已有精确校准 map/kernel 的数据处理结论仍成立；本论文的随机误差界不能静默替换其中 actual-q transfer 项，也不能改变已冻结的 deterministic arm。辅助训练、独立噪声、score 模型、额外求解步数都需要先给出新的机制和成本设计，再作另一项实验。

## 5. 正则性陈述的使用范围

正式版 Theorem6 文字写出到闭区间端点的任意阶空间平滑，但 Assumption5 只假设端点 C²。B.1 的 Gaussian Fourier 衰减严格支持 0<τ<1 的平滑；γ=0 的端点不能凭空得到 C∞。例如规范化 `exp(−x²)(1+|x|³)` 是严格正 C²、有限 Fisher 的密度，却不是 C³，其端点当然不可能变 C∞。因此本笔记仅使用内部卷积平滑；端点极限另验证。

B.20 的端点条件期望还被写成独立 marginal 平均。对一般 ν，应为 `E[∂τI(0,Y,W)|Y=x]` 和相应 W 条件平均，而不是无条件积分另一个端点。例：Y=W~N(0,1)、I=(1−τ)Y+τW，则 ∂τI 恒为0；独立平均会错误地产生−x。这个印刷/推广缺口不推翻 Definition1 允许任意 coupling，也不推翻用 joint 条件期望作的连续性证明；对本仓库相关配对尤其不能抄该端点简式。

## 6. 数值证据与实际成本

原文 §7 证据可支持“机制有实际效用”，但不支持现成 RAEv2 guidance 或公平成本 FID SOTA：

| 实验 | 保留的观察 | 开销和证明对象 |
|---|---|---|
| 2D checkerboard (§7.1) | 加入 latent γ 可改善学习后的 ODE；同一模型的非零扩散 SDE 通常更贴近目标 | 比较多种 γ 与 a=0,.5,1,2.5；每设定300K生成样本，KDE/log-density 指标；不是无搜索设计 |
| 128D 五模 Gaussian mixture (§7.2) | 在有限拟合下，适量扩散能修正 ODE 过度集中的模式；过大则过度分散。作者观察学习 b 与 noise-denoiser η 通常较好 | SDE固定1000 Heun步，ODE adaptive dopri5；a多值比较。评分为前两坐标 KDE 的 KL，准确/生成各50K建KDE，另50K参考评估；不是完整128D KL |
| Flowers128 one-sided (§7.3) | 同一初始Gaussian状态能随 SDE 随机路径产生不同合理图像；架构能扩展到图像维度 | 图13的 a=1,2,4 对应2000、2500、4000 Heun步，ODE步数自适应；不能当同成本证据 |
| Flowers mirror (§7.3) | 正确 mirror 路径的 ODE回到原图，SDE可在同边缘law内生成变化 | 展示 a=10，图像定性；没有整体分布/FID验证 |

正文 p.51 明确将 ImageNet/FID 的系统研究留待后续。因此本次没有可摘取的 FID 提升表，也没有5%以上目标的实验支持。Appendix Table3 另列 ImageNet32 配置，不等于正文报告了其FID。

Table3 作者列出的 Flowers 参数：batch64、350K训练步、4 GPU；mirror为800K步、2 GPU；均为Ho-style U-Net、EMA0.9999、Adam与预设LR decay。表列训练点315,123，本文没有核验其数据构造，不把它当独立确认的唯一原始图像数。实验使用端点 truncation、部分情况下最终 denoising；这些是作者的数值选择，不是可以自动移植到当前任务的理论窗口。原文未提供完整 GPU 型号/墙钟/联合所有模型训练总成本；模型损失分别对应 b/v/s/η，不能把多场训练当单个冻结模型免费增益。

## 7. 作者实现：有用，但符号与范围要逐项匹配

作者主页明确链接两个代码库；此次仅静态阅读并固定 commit，未复现其图像训练。

- [malbergo/stochastic-interpolants](https://github.com/malbergo/stochastic-interpolants/tree/c9bc956ec5a51c907bcbfb114f96d3539acb70bb)，commit `c9bc956ec5a51c907bcbfb114f96d3539acb70bb`：读取 README、`interflow/stochastic_interpolant.py` 的 interpolant、b/s/η目标和 SDEIntegrator。b/s损失使用±ξ反义样本，实际两次网络计算；`bf=b+eps*s`、`br=b−eps*s`，Heun每步两次 drift评估，若 b/s 各一网则每步四个网络前向。此代码的 `loss_per_sample_eta` 回归**负**噪声，并由 `SFromEta` 作正号除γ；配套后是正确 score。它与正文 η=E[ξ|U] 的命名符号相反，不能拿正文正噪声权重直接塞进该 wrapper。
- [nmboffi/jax-interpolants](https://github.com/nmboffi/jax-interpolants/tree/8e6bf907d11be49da1780b5a3aa7212277365257)，commit `8e6bf907d11be49da1780b5a3aa7212277365257`：读取 README、interpolant/losses/samplers。该实现主要是 Gaussian-base one-sided 路径；`score` 标签为−x0/α，当前 sampler 是两次评估的 ODE Heun；loss 包含另一个 learned time weight。README仍称FID计算待加入。它不是原文任意 coupling+独立 γξ 的完整 SDE复现，更不能拿其一侧Gaussian score公式套当前Y,W非线性桥。

**对下一步研究的贡献：** 保留相关配对的弱连续性机制；把未来随机校正的候选对象明确为“同一指定路径的 velocity 和 score 共同误差”，而非加噪强度试扫；若没有可识别 score 与可估计 excess-risk，a* 仍只是理论关系。当前冻结1K应继续按原确定性机制评价，本阅读不给它补发未证明的KL/FID保证。
