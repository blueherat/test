# 从FSG重新看IG：固定点应约束什么，信息由什么承载

2026-09-09，回应用户继续研究FSG/IG等式与信息载体的要求。
本轮没有新增toy实验。本文中的恒等式是标准Bayes、指数族及Gaussian代数的应用；
神经模型实验另行固定协议，不能用代数成立替代真实生成质量。

这轮更具体的判断是：**IG可研究的对象是强弱模型之间的修正证据，而不是让原始
两头对同一输入不再分歧。一次头差只是这份证据的一种读出。** 在理想的Gaussian
观测模型下，latent改写能表达的证据函数有明确限制；局部后验的精度则提供一个
允许两头不同、且可计算的候选平衡式。它仍然需要实际质量检验。

本轮直接生成已判定该近似不宜继续：一维版本FID88.25968，普通IG65.13932，
成本约17倍；二维版本因近奇异放大与约8000次Full/图在192张后中止，不报部分FID。
随后自动微分检查64个真实状态/样本，两头各自的对称投影响应全部正定，
但平衡组合仅21/64正定。数学表达限制仍成立；候选Gaussian平衡式没有成为有效方法。
详见[真实结果与导数审计](SMALL_SIT_PRECISION_BALANCE_RESULTS_20260909_ZH.md)。

## 1. 先区分三种等式

FSG正式论文§3.1式(6)的目标是

\[
\Phi_U^{0\leftarrow t}(\hat z_t)
=\Phi_C^{0\leftarrow t}(\hat z_t),\qquad \hat z_t\in\mathcal M_t.
\tag{1}
\]

这里相等的是完整续生成的结局。原文用较短区间及有限次校准近似它。
“当前两个noise输出相等”“确定性续生成结局相等”“两者对所有可能clean图像的
后验分布相等”是三种不同的要求，不能相互替代。FSG没有证明第三种要求。
[NeurIPS 2025原文](https://proceedings.neurips.cc/paper_files/paper/2025/file/b56d827a2b8433517e722e0272c7f464-Paper-Conference.pdf)

因此，下面研究完整后验的表达限制，不是反驳FSG；它是为了明确我们究竟想保留
IG的哪一部分内容。此前可逆流的\(\Phi_S^{-1}\Phi_G\)确实能编码一个IG终点，
但它没有自动提供一个廉价、能复用强模型知识的读取机制。

## 2. IG缺少的是同一个外部条件吗

原生IG在中间层增加denoising监督，再外推中间层与深层预测。
两头都接收相同的latent、时间和类别，目标也相同；它们主要区别在于表示与计算。
[IG原文§3.2](https://arxiv.org/html/2512.24176v1)

深层信息不能简单解释为“浅层没有得到另一个类别标签”。而且深层状态由浅层状态
继续确定性计算得到，不能套用“越深越有Shannon信息”的嵌套条件期望故事。
对有限读出器，额外计算仍能使原本难以使用的结构成为可用的预测。

因此需要区分：知识在模型参数与计算中；本次输入激活了某些修正；采样更新把
修正的作用写入轨迹状态。这三件事都不等于弱头已经掌握了完整强模型。

## 3. 理想化信息对象：对clean假设的修正函数

为得到可证明的结论，先明确一个额外假设：强弱模型分别对应同一Gaussian
观测通道下的合法先验\(p_S(x\mid c),p_W(x\mid c)\)。真实两头不自动满足此假设。
固定类别并省略\(c\)，令

\[
Z_t=a_tX+\sigma_t\epsilon,\qquad \epsilon\sim\mathcal N(0,I),
\quad a_t>0,\ \sigma_t>0.
\]

相应后验为

\[
q_i(x\mid z)\propto p_i(x)
\exp\!\left[-\frac{\|z-a_tx\|^2}{2\sigma_t^2}\right].
\]

定义修正证据函数

\[
\psi(x)=\log p_S(x)-\log p_W(x).
\]

在共同支持上，Bayes公式直接给出

\[
\boxed{q_S(x\mid z)
=\frac{q_W(x\mid z)e^{\psi(x)}}
{\mathbb E_{q_W(\cdot\mid z)}e^{\psi(X)}}.}
\tag{2}
\]

这份“信息”是对不同clean假设如何重新赋权的函数，不只是一个输出向量。
它是模型相对偏好的描述，不是外部真实质量标签；强模型整体更好并不保证所有
高密度比区域都更好。当前有限网络也不一定能导出这样的跨时间一致密度对。

## 4. latent本身能写入哪一种信息

固定噪声水平，只把弱模型输入从\(z\)改成\(z+\delta\)。展开Gaussian似然得到

\[
\boxed{q_W(x\mid z+\delta)
\propto q_W(x\mid z)
\exp\!\left[\frac{a_t}{\sigma_t^2}\delta^\top x\right].}
\tag{3}
\]

于是有一个准确的判据：

\[
\boxed{q_W(\cdot\mid z+\delta)=q_S(\cdot\mid z)
\iff
\psi(x)=\frac{a_t}{\sigma_t^2}\delta^\top x+\mathrm{constant}
\quad q_W\text{-a.s.}}
\tag{4}
\]

证明只需比较式(2)、(3)并取对数；归一化常数吸收所有与\(x\)无关的项。
该结论对任意选择的\(\delta(z)\)逐个固定输入成立，不要求\(\delta\)本身是线性函数。

**限制的是后验能接受的证据函数，不是latent向量能影响哪些视觉属性。** 对非Gaussian
后验，线性指数倾斜也可能改变方差和模态权重；但它不能任意指定完整的修正函数。
对于Gaussian后验，固定时间改输入只改变后验均值，协方差不随输入改变。

这不是IG独有的不可能性：CFG中\(\psi_c(x)=\log p(c\mid x)\)一般也不线性。
FSG追求特定续生成结局的一致性，比完整后验一致弱得多，故不与式(4)冲突。
同理，式(4)不否定用复杂可逆流把一个确定性终点编码进当前latent。

## 5. 一次头差遗漏了什么

固定当前弱后验\(q\)，考虑小强度的目标\(q_\eta\propto qe^{\eta\psi}\)。
有足够指数矩、二阶矩时，均值的响应为

\[
\left.\frac{d}{d\eta}\mathbb E_{q_\eta}X\right|_{\eta=0}
=\operatorname{Cov}_q(X,\psi(X)).
\tag{5}
\]

它只读出了修正函数与\(X\)的相关部分。若某部分修正与\(1,X\)正交，它可以
在一阶不改变均值，却改变更高阶统计。两个均值或它们的差不能唯一恢复整份证据。
对有限强弱差，沿\(q_u\propto qe^{u\psi}\)积分可得

\[
m_S-m_W=\int_0^1\operatorname{Cov}_{q_u}(X,\psi(X))\,du.
\]

还有一个局部定量版本。用latent位移所允许的线性证据\(b^\top X\)近似\(\psi\)，
若后验协方差\(C\succ0\)，最优系数为

\[
b_* = C^{-1}\operatorname{Cov}_q(X,\psi(X)).
\]

对应两个小倾斜分布的KL距离最低阶为

\[
\min_b\mathrm{KL}\!\left(
\frac{qe^{\eta\psi}}{Z_\psi}\;\middle\|\;
\frac{qe^{\eta b^\top X}}{Z_b}\right)
=\frac{\eta^2}{2}\min_b\operatorname{Var}_q(\psi-b^\top X)+O(\eta^3).
\tag{6}
\]

这里的展开需要对应三阶矩/指数矩与局部正规性；它不是对神经模型FID的界。
式(6)说明：如果期望承载的内容超出了当前观测通道的线性证据族，持续改变位移
的幅度不能消除那一部分表达误差。

## 6. 更丰富的信息可以在哪里

Gaussian似然在clean变量上的自然参数为

\[
h=\frac{a_t}{\sigma_t^2}z,\qquad
\Lambda=\frac{a_t^2}{\sigma_t^2}I,
\quad\log L(x)=h^\top x-\tfrac12x^\top\Lambda x+\mathrm{constant}.
\]

若修正可以局部写成

\[
\psi(x)=b^\top x-\tfrac12x^\top A x+r(x),
\]

则有清楚的对应关系：

- \(b\)：线性证据，可通过改变latent对应的\(h\)写入。
- \(A\)：二次证据，改变精度\(\Lambda\)；标量时间只能控制各向同性那部分。
- \(r\)：更高阶或多模态证据，需要更丰富的读取、表示或在线计算。

当\(\Lambda+A\succ0\)时，前两项可表示为新的Gaussian观测自然参数
\((h+b,\Lambda+A)\)。但当前网络只接受标量噪声时间，不能直接把一个矩阵
塞进去就声称网络已学会读取该观测通道。

所以额外状态可能是局部精度、内部特征或可复用的计算结果；这只是实现候选。
本文没有证明必须增加一个永久memory tensor，也没有证明任意内部feature相加
都会提高质量。此前DDT条件外推的负结果仍然有效。

## 7. 一个允许强弱预测不同的具体平衡等式

给定当前状态的Gaussian后验近似
\(q_S=\mathcal N(m_S,C_S),q_W=\mathcal N(m_W,C_W)\)，
令\(\beta=1+\gamma\)。考虑额外强调强模型的power posterior

\[
q_\gamma(x\mid z)\propto q_S(x\mid z)^\beta q_W(x\mid z)^{-\gamma}.
\]

只有

\[
Q_\gamma=\beta C_S^{-1}-\gamma C_W^{-1}\succ0
\]

时，这个Gaussian候选才可归一化。其均值/众数满足

\[
\boxed{\beta C_S^{-1}(m_\gamma-m_S)
=\gamma C_W^{-1}(m_\gamma-m_W).}
\tag{7}
\]

两边是精度加权的残差。\(m_S\)和\(m_W\)可以始终不同；被求解的是一个新状态
相对于两种预测如何平衡，而不是把两种预测强行变成同一个向量。

标准矩阵代数给出

\[
\boxed{m_\gamma=m_S+\gamma C_S
\big[(1+\gamma)C_W-\gamma C_S\big]^{-1}(m_S-m_W).}
\tag{8}
\]

不要求两个协方差矩阵交换。若\(C_S=C_W\)，式(8)才退化为
\(m_S+\gamma(m_S-m_W)\)。这只是相对于该Gaussian后验目标的比较，
不是声称IG原文必须假定两个协方差相同才能定义其采样器。

式(7)也是一个明确的固定点问题。对二次函数

\[
J(x)=\tfrac\beta2\|x-m_S\|_{C_S^{-1}}^2
-\tfrac\gamma2\|x-m_W\|_{C_W^{-1}}^2,
\]

求\(x=x-\eta\nabla J(x)\)。若\(Q_\gamma\succ0\)且
\(0<\eta<2/\lambda_{\max}(Q_\gamma)\)，这个线性迭代收敛到唯一的式(8)。
这只是冻结局部Gaussian问题的收敛结论，不是原始神经生成过程的收敛或质量定理。

这里完整的局部信息对象是\((Q_\gamma,h_\gamma)\)，其中
\(h_\gamma=\beta C_S^{-1}m_S-\gamma C_W^{-1}m_W\)。
仅输出均值\(Q_\gamma^{-1}h_\gamma\)仍会丢掉精度；实现可以每次重读，也可以
研究如何承载它。当前两组小SiT试验采用每次重读，没有测试跨时间保存矩阵memory。

## 8. 如何从真实网络读取，以及不能跳过的限制

理想Gaussian观测下，后验均值的Jacobian满足

\[
J_zm_i(z)=\frac{a_t}{\sigma_t^2}C_i(z).
\tag{9}
\]

使用denoiser导数估计协方差已有公开方法，如TMPD和Free Hunch。
本文不宣称该读数或Gaussian闭合的新颖性。
[TMPD §3.1](https://arxiv.org/html/2310.06721v3#S3.SS1)，
[Free Hunch](https://arxiv.org/abs/2410.11149)

小SiT的时间为\(z=tX+(1-t)\epsilon\)，clean读出\(m=z+(1-t)v\)。
固定时间中央差分可以获得若干方向的\(Jm\)；式(8)中的共同正标量消去。
本轮固定一维和二维近似，各做一个1K，与已有同分段普通IG比较。
二维方向来自两头响应差的正交分量，不由FID挑选。完整方案见
[冻结协议](SMALL_SIT_PRECISION_BALANCE_PROTOCOL_20260909_ZH.md)。

需要明确四个限制：

- 神经head的Jacobian可能不对称或非正定，不能自动叫作真实posterior covariance。
- 先投影为低维Gaussian再做乘除，一般不等于完整后验乘除后再投影。
- 在每个噪声时刻外推score，不自动等于对clean密度做power tilt后再加噪；
  仓库已有semigroup理论及其负质量结果，本轮没有抹去这一缺口。
- 即使某种模型power posterior能被精确采样，也不自动比原IG更接近真实数据。

本轮代码只在固定正定与数值条件通过时使用该级近似，其余退回低一级/原IG，并完整
记录采用率。它是带适用性规则的近似采样器，不冒称严格实现整条power-posterior路径。

## 9. 当前研究判断

本轮实测将式(7)的适用性问题具体化。即使两个局部响应各自正定，
它们的外推组合也经常不正定；在同一基上使用自动微分仍有43/64个检查点失败。
这不能用“把差分取得更精确”完全解决。它反对的是当前局部Gaussian乘除近似，
不证明真实后验一定单峰、一定多峰或一定不存在某种合法的目标密度。
此次也没有检验跨时间保存精度、内部feature载体或所有有限latent校准算子。

FSG给我们的启发可以保留：先明确要写入的内容和读取者，再定义校准目标。
IG中有意义的下一层问题是：**深层相对于浅层改变了哪些关于clean假设的判断，
其中哪些只能改变当前样本，哪些还需要改变后续读取的精度或内部表示？**

式(4)与式(6)给出一个表达层面的限制；式(7)给出一个不要求原始两头相等、
能落实为真实小模型试验的候选。它们尚未建立有效新方法或真实质量保证。
