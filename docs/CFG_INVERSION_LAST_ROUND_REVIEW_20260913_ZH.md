# 反演与 CFG 不变性的最终评审

**结论：在现有冻结的类别条件 FM 模型上，尚未得到值得启动新实验的反演方法。这条从重复复制不变性出发的采样改进分支到此结束。** 结论来自对目标、反例、可实施算法和已有研究的联合判断，不是声称所有反演方法都不可能改善生成。

最接近成立的方案是：保持条件和无条件分支各自的生成分布，消除它们在 CFG 混合时产生的无关运动干扰。但是，将这个设想写成具有分布保持保证的算法后，其核心成为已有的加权保守投影与能量蒸馏。反演本身没有带来额外可识别的质量信号，便宜的局部替代也不能继承完整投影的保证。

## 原始复制现象究竟约束什么

给定完整图片 \(x\) 和明确的不修改要求，理想复制操作的目标是

\[
\mathcal L(Y\mid x,\mathrm{copy})=\delta_x.
\]

这要求每轮输出接近最初输入。仅要求相邻两轮越来越相似是不够的：一个把所有图片都变成同一张图的算子，可以在第一轮以后完全稳定。实际多模态系统是否满足这种逐图保持，还取决于源图通道、指令、对话历史和图像编解码；商业模型的观察本身不能隔离这些因素。

源图可以作为独立条件一直保留，内部目标状态仍可有随机性。图像条件的 flow 模型提供这种接口的先例，例如 Kontext；但其接口不意味着严格复制已经被保证。[^1]

现有类别条件 SiT 在采样时只看到当前 latent、时间和类别。把生成模型 \(G\) 的逆当作读图接口，确实定义了一个操作，却没有自动赋予它上述条件任务：

\[
T(x)=G_{\rm out}(E_{\rm in}(x)).
\]

若 \(E_{\rm in}=G_{\rm out}^{-1}\)，任何错误但可逆的生成器都能逐图通过。若换成其他模型的逆，输出变化又同时包含模型差异和参考编码偏好。精确求逆消除了数值误差，不能消除这层目标差异。

这不否定公共参考重建的用途。一个真实分布校准正确的编码器可以提供有效监督；缺的是这个外部锚的正确性，不能靠再多做几次回环来获得它。

## 精确反例：复制评分会否定一次正确的 autoguidance 改进

反例限定在同一个规范 Gaussian FM 构造，不利用任意潜空间旋转。真实目标为 \(P=\mathcal N(0,1)\)，强、弱模型分别精确生成

\[
P_s=\mathcal N(0,1.2^2),\qquad
P_b=\mathcal N(0,1.5^2).
\]

两个模型都使用 \(Z_t=tX+(1-t)\epsilon\)，\(t=0\) 为噪声、\(t=1\) 为数据。对终点标准差为 \(\sigma\) 的模型，记

\[
V_\sigma(t)=(1-t)^2+t^2\sigma^2,
\quad
v_\sigma(z,t)=\frac{\dot V_\sigma(t)}{2V_\sigma(t)}z.
\]

用 \(v_\alpha=(1+\alpha)v_s-\alpha v_b\) 精确积分，终点映射为

\[
G_\alpha(z)=a_\alpha z,\qquad
a_\alpha=1.2^{1+\alpha}1.5^{-\alpha}.
\]

此时真实分布误差是

\[
W_2^2((G_\alpha)_\#\nu,P)=(a_\alpha-1)^2.
\]

在 \(\alpha\simeq0.81705949\) 时，\(a_\alpha=1\)，生成分布完全正确，比强模型的 \(W_2^2=0.04\) 更好。

但若用弱模型的精确逆 \(E_b(x)=x/1.5\) 来评价“回灌应不动”，则

\[
R_b(\alpha)
=\mathbb E_{X\sim P}|G_\alpha(E_b(X))-X|^2
=\left(\frac{a_\alpha}{1.5}-1\right)^2.
\]

当 \(\alpha\ge0\) 时，这个复制风险在 \(\alpha=0\) 取最小值。它将真实最优参数的风险从 \(0.04\) 判成更差的 \(1/9\)。

因此，即使弱模型确实更差、两模型使用相同 Gaussian FM 路径构造与时间约定、反演完全精确、没有新噪声，弱模型回灌偏移也不能普遍决定 guidance 的正确方向。这里两个规范场分别对应受控错误分布，并非同一真实分布的两个精确 Bayes 最优解。这是一个反例，不是对所有独立数据校准方法的否定。

## 两种必须区分的分布不变性

第一种是终点重参数化。若 \(Q_\#\nu=\nu\)，则

\[
(G\circ Q)_\#\nu=G_\#\nu.
\]

在完整先验上，最终生成分布保持不变，因而也不会改变基于该分布的总体质量指标。它可能改变逐图配对和有限样本估计，但不能据此声称总体生成质量提高。若 \(Q\) 改变先验，则需要另一个依据说明改后的先验更好，问题回到真实反演分布或数据学习。

第二种是各个时间的速度场重参数化。它与上一种不同：**先保持两个分支的边缘分布，再混合速度，混合后的分布未必不变。** 这个区别给出了本轮最强的建设性候选，不能用终点先验保持的等式直接否定它。

## 最强候选：分支内无害的运动如何干扰 CFG

对时间 \(t\) 固定，设两个分支的正密度为 \(p_c,p_u\)，其 score 为 \(s_c,s_u\)。加入速度分量 \(r_c,r_u\)，且

\[
\nabla\cdot(p_c r_c)=0,\qquad
\nabla\cdot(p_u r_u)=0.
\]

在相同初始分布和适定性条件下，这些分量不改变各自的边缘密度演化。这类自由度已有严格研究；欧氏旋度非零本身并不等于生成分布错误。[^2]

在可归一化时，定义指定的瞬时倾斜密度和混合附加速度

\[
\pi_w\propto p_c^w p_u^{1-w},\qquad
r_w=w r_c+(1-w)r_u.
\]

乘积求导得到

\[
\boxed{
\frac{\nabla\cdot(\pi_w r_w)}{\pi_w}
=w(1-w)(r_u-r_c)^{\mathsf T}(s_c-s_u).
}
\]

右侧一般非零。两条分支分别无害的运动，可能在 guidance 组合时影响密度。这里 \(\pi_w\) 是指定的检查对象，**不是未经证明的实际 CFG 边缘分布**。

这个机制成立，但不是本轮新发现。仓库已有[完整推导及连续、离散类别反例](FSG_RELEASE_CURL_AND_APG_REVIEW_20260911_ZH.md)，其中甚至保留了相同的交叉项公式。将它重新命名不能形成新的研究贡献。

### 能够成立的修复需要什么

令 \(p_{b,t}\) 是分支 \(b\) 自身实际生成的边缘密度。在其加权平方可积空间中，考虑梯度场闭包上的正交投影：

\[
\nabla\phi_b
=\operatorname*{argmin}_{\nabla\phi}
\mathbb E_{p_{b,t}}\|\nabla\phi-v_b\|^2.
\]

一阶最优条件为

\[
\mathbb E_{p_{b,t}}
[(v_b-\nabla\phi_b)\cdot\nabla\psi]=0
\quad\text{对所有合适的 }\psi.
\]

在分布意义下，这就是

\[
\nabla\cdot\{p_{b,t}(v_b-\nabla\phi_b)\}=0.
\]

因此，精确投影可以保留分支边缘，再以投影后的场组成 CFG。若原边缘确实是规范 Gaussian FM 路径，其规范 Bayes 速度本身属于梯度场，投影能够恢复该代表元。这恢复的是规范 CFG 场，不保证其终点等于真实条件分布或生成质量更好；针对指定倾斜分布的改善，也不必然是针对原条件分布的改善。

**不需要显式知道密度值才能写出训练目标。** 分支前向采样可以提供 \(p_{b,t}\) 的样本，进而拟合标量势。这使它成为原则上可实施的离线蒸馏，而不是不可计算的形式定义。不过有限网络、有限训练和近似采样会破坏精确投影条件；如果用真实加噪数据代替实际分支边缘，分布保持结论也要重新检查。

关键的新颖性问题是，这个目标已经出现在 Thornton 等人的 *Composition and Control with Distilled Energy Diffusion Models and Sequential Monte Carlo*：§3.2 的 Eq.7 给出向量场到保守场的平方投影，后续用于能量蒸馏和模型组合。[^3] 该论文并非这里具体的 SiT 反演实现，但已经覆盖核心投影操作及组合用途。已有先例加上仓库原有机制推导，使当前版本不够构成独立新方法。

而且，反演不是这一步的必要操作：训练数据可以正向获得。仅把采样点改成某个逆路径上的点，并没有证明它能更准确或更便宜地求解投影。

### 为什么只投影 guidance 差场还不够

若把 \(d=v_c-v_u\) 直接投影到 \(L^2(\pi_w)\) 的梯度场，被删掉的余项按定义满足 \(\nabla\cdot(\pi_w r)=0\)。因此它在指定 \(\pi_w\) 上不会改变相应的瞬时连续性残差，不能用这种投影声称消除了前面的非零交叉项。

分支各自的加权投影，与对混合差场做一次投影，是不同的操作。便宜一些并不意味着前者的结论可以直接移植。

## 真正成立的闭环约束，也没有给出新的反演修复

在规范 Gaussian FM 的内部时间 \(0<t<1\)，理想 CFG 差场满足

\[
g(z,t)=v_c-v_u
=\frac{1-t}{t}\nabla_z\log p(c\mid Z_t=z).
\]

因而在固定 \(t\) 的光滑区域内，闭合路径 \(\Gamma\) 上有

\[
\oint_\Gamma g(z,t)\cdot dz=0.
\]

这是一条非空洞的理想模型约束。保持的是同一势函数的路径积分，而非要求任意 guided 图片反演后不动。它有两个重要限制：跨不同时间的回环不能直接使用固定时间的等式；它约束规范 score 表示，不能单独判定生成质量。

利用反演寻找锚点 \(a\)，再构造径向势

\[
\psi_a(x)=\int_0^1g(a+s(x-a))\cdot(x-a)\,ds
\]

看似可以得到便宜的保守修复。对固定锚点，有

\[
\nabla\psi_a(x)
=g(x)+\int_0^1s(J_g^{\mathsf T}-J_g)(a+s(x-a))(x-a)\,ds.
\]

当 \(g\) 真是梯度场时，它保持原场；但它不是上述加权正交投影，不能保证减少真实 score 误差。

一个直接反例是二维 \(g(x)=Ax\)，其中 \(A\) 为 90 度旋转矩阵，真实差场为零。径向势的梯度恰为 \(Aa\)。若评估分布为 \(\mathcal N(0,\sigma^2I)\)，原场误差是 \(2\sigma^2\)，修复后变为 \(\|a\|^2\)，可以任意更大。它把旋转换成了平移。若锚点随 \(x\) 变化而计算时停止其梯度，也不能再宣称整体得到一个全局保守场。

去旋度训练本身也已有 QCSBM 等先例。[^4] 因此，闭环约束成立，并不使“反演锚点加去旋度”自然成为新方法。

## 反演路径能否纠正 CFG 的分布偏差

这条思路具有明确目标：先指定希望采样的分布，再纠正实际运输与目标之间的差异。它比逐图回环位移更有理论意义，但不能把希望的分布与实际 CFG 分布混为一谈。

例如在 \(t\) 从噪声到数据的线性 Gaussian FM 中，记 \(k(t)=(1-t)/t\)、\(v_b=x/t+k s_b\)，权重 \(w\) 为常数。令 \(\pi_t=p_c^w p_u^{1-w}/Z_t\)，按连续性方程直接计算可得

\[
\frac{\partial_t\pi_t+\nabla\cdot(\pi_t v_w)}{\pi_t}
=k(t)w(w-1)\|s_c-s_u\|^2-\partial_t\log Z_t.
\]

这个残差解释了为什么逐时刻混合 score，不等于沿全过程精确采样幂密度。它需要正确的分支 score、正则密度和积分可积性；在端点奇异或估计速度不满足对应关系时不能直接套用。时间变化的权重还会带来额外项。

沿轨迹积分或粒子加权可以处理这类偏差。Feynman–Kac Correctors 已专门研究 guidance、几何平均与乘积目标的修正；2026 年的 *Analytic Distribution of Classifier-Free Guidance for Schedule Design* 又给出确定性 CFG 的路径积分分布表达。[^5][^6] 因此，“反演整条路径后估计累计偏差”没有自动形成空白；若路径已在前向过程中记录，反演也未必提供新信息。

这里的停止理由不是既有方法无效，而是目前没有一个与这些方法实质不同、又能解释计算优势的候选。

## 对各条路线的决定

|候选|能够成立的部分|未达到继续投入条件的原因|
|---|---|---|
|同场精确反演后重建|检查数值可逆性|不能识别生成分布是否正确|
|弱模型反演作为 guidance 校验|定义一个跨模型重建操作|精确 Gaussian 反例中会否定正确改进|
|保持先验的终点重参数化|改变配对而保持分布|总体生成分布没有变化|
|真实反演先验学习|加入真实数据可形成合法目标|已有相关路线；先前当前实现未完成可行性流程|
|分支加权投影后再 CFG|有明确不变性与干扰机制|机制已在仓库推导，核心算法已有能量蒸馏先例|
|反演闭环与径向势修复|理想差场具有零环量|局部替代不等于加权投影，可能放大误差|
|反演路径分布修正|可指定分布并分析连续性偏差|已有 Feynman–Kac 和确定性 CFG 路径公式|
|完整源图条件下的冗余 CFG|目标固定为原图，冗余条件差应为零|需要图像条件接口及训练；并未导出当前纯生成改进|

真正合法的源图条件约束仍然存在：当两条分支都保留完整源图和 no-op 操作，只在冗余描述上不同，理想差场为零。但这是一项明确的图像条件任务，不能当作现有 class-only SiT 已具备的条件。它能否通过多任务训练改善纯生成，是另一项研究问题，当前没有建立独立贡献或收益依据。

最终决定是结束这次反演不变性采样分支，保留定义、反例和已完成结果。未达到的目标仍是超过充分调参且计算成本可比的强 baseline；不把理论自洽、已有数值改进或未经确认的小幅指标变化写成该目标已达成。

## 来源与进一步核查

[^1]: Black Forest Labs 等，[*FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space*](https://arxiv.org/html/2506.15742v2#S3)，2025，§3。支持源图与目标状态分开的图像条件接口，不支持严格 no-op 必然成立。
[^2]: Christian Horvat、Jean-Pascal Pfister，[*On gauge freedom, conservativity and intrinsic dimensionality estimation in diffusion models*](https://arxiv.org/html/2402.03845)，ICLR 2024，Eq.11、Theorem 1。支持密度加权无散自由度及正交分解；CFG 交叉项另见仓库已有直接推导。
[^3]: James Thornton 等，[*Composition and Control with Distilled Energy Diffusion Models and Sequential Monte Carlo*](https://arxiv.org/html/2502.12786)，AISTATS 2025，§3.2 Eq.7、§4、Appendix A.1。与加权保守投影和模型组合的核心操作直接相关，不将论文的具体实验说成当前 SiT 已验证。
[^4]: Chen-Hao Chao 等，[*On Investigating the Conservative Property of Score-Based Generative Models*](https://proceedings.mlr.press/v202/chao23a.html)，ICML 2023，§4。已用 Jacobian 反对称性构造训练正则。
[^5]: Marta Skreta 等，[*Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts*](https://arxiv.org/html/2503.02819v2)，2025。通过权重与粒子方法处理预训练模型组合的目标路径。
[^6]: Enze Jiang、Zheng Ma，[*Analytic Distribution of Classifier-Free Guidance for Schedule Design*](https://arxiv.org/html/2607.19725v2)，2026，v2 Theorem 4.2、4.3。版本标签为 2026-08-06；文内另标 2026-08-24。以固定 v2 和定理定位为准，实际质量结论不由本评审复验。

本地相关证据：[原始不变性重述](INVERSION_IDENTITY_REFRAME_20260913_ZH.md)、[分支混合与旋度的既有推导](FSG_RELEASE_CURL_AND_APG_REVIEW_20260911_ZH.md)、[已有可观测误差投影研究](RAEV2_OBSERVABLE_ERROR_GUIDANCE_20260906_ZH.md)、[Z 实验停止记录](Z_SAMPLING_IDENTITY_RESULTS_20260913_ZH.md)、[反演先验结果](CFG_INVERSE_PRIOR_RESULTS_20260913_ZH.md)。本评审不授权恢复其中任何历史队列。

补充论证：[建设方的候选与修订](research/identity_operator_20260913/last_round_constructive.md)、[理论反例与反驳](research/identity_operator_20260913/last_round_theory_redteam.md)、[原始文献与新颖性核查](research/identity_operator_20260913/last_round_literature.md)。
