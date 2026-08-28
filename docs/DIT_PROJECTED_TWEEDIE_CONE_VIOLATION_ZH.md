# DiT 单路径新候选：Projected Tweedie-cone Violation

日期：2026-08-28
状态：理论与数值实现阶段；尚未打开该候选的质量关联，不能称为有效 bad-case 指标

## 0. 结论先说

Doob consistency probe 得到了一个真实但明显偏向“全局模糊”的弱发现，不能当成融合、错接和结构错位的通用检测器。新的主候选改为：

> **检查冻结去噪器在当前单条生成轨迹上的局部 Jacobian，离精确 Tweedie 后验均值所允许的“对称半正定锥”有多远。**

简称 PTCV（Projected Tweedie-cone Violation）。

它与前面的多后缀方法有三个根本区别：

1. 不生成、比较或选择多个随机后缀；
2. 正常律是逐状态、逐时刻的点态恒等式，而不是坏图经验直觉；
3. 分数是到一个精确定义集合的几何距离，不是把几个启发式量手工加权。

它仍然不是“坏图定理”。一张坏图可以被模型学成稳定且局部合法的模态，此时本量完全可能接近零。它要检验的是更窄、也更可证伪的问题：

> 明显的融合、错接或结构翻转，是否更常经过 learned denoiser 的局部非保守/折叠区域？

## 1. 记号：每个字母是什么意思

类别条件前向加噪过程写成

\[
X_t=\alpha_t X_0+\sigma_t\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I).
\]

- \(X_0\in\mathbb R^d\)：干净 latent 随机变量；在当前 DiT 中 \(d=4\times32\times32=4096\)；
- \(X_t\in\mathbb R^d\)：噪声时刻 \(t\) 的 latent；
- \(x\)：\(X_t\) 的一个具体取值，也就是当前轨迹状态；
- \(t\)：训练扩散的内部噪声时刻；采样日志中的 sampling step 与它方向相反；
- \(c\)：固定类别条件，例如 ImageNet class 602；
- \(\alpha_t>0\)：时刻 \(t\) 保留的信号系数；
- \(\sigma_t>0\)：时刻 \(t\) 的高斯噪声标准差；
- \(\varepsilon\)：标准高斯噪声；
- \(I\)：\(d\times d\) 单位矩阵；
- \(m_t^c(x)\)：精确类别条件后验均值

\[
m_t^c(x)=\mathbb E[X_0\mid X_t=x,c];
\]

- \(\widehat m_t^c(x)\)：冻结 DiT 的 raw class-conditional `pred_xstart`，它只是 \(m_t^c(x)\) 的 learned surrogate；
- \(J_t^c(x)=\nabla_xm_t^c(x)\)：精确后验均值对 noisy latent 的 Jacobian；
- \(\widehat J_t^c(x)=\nabla_x\widehat m_t^c(x)\)：网络 Jacobian；
- \(Q=[q_1,\ldots,q_r]\in\mathbb R^{d\times r}\)：事前固定的低/中频正交投影基，满足 \(Q^\top Q=I_r\)；
- \(r\)：投影维数；
- \(B=Q^\top\widehat JQ\in\mathbb R^{r\times r}\)：网络 Jacobian 在固定子空间中的压缩矩阵；
- \(S=(B+B^\top)/2\)：\(B\) 的对称部分；
- \(A=(B-B^\top)/2\)：\(B\) 的反对称部分；
- \(S_-\)：\(S\) 的负特征值部分；
- \(D\)：\(B\) 到对称半正定锥的归一化平方距离，也就是候选缺陷分数。

## 2. 精确正常律从哪里来

固定类别 \(c\)。高斯似然为

\[
p_t(x\mid x_0,c)
\propto
\exp\!\left[-\frac{\|x-\alpha_t x_0\|^2}{2\sigma_t^2}\right].
\]

对观测坐标 \(x_j\) 求对数梯度：

\[
\frac{\partial}{\partial x_j}\log p_t(x\mid x_0,c)
=\frac{\alpha_t(x_0)_j-x_j}{\sigma_t^2}.
\]

后验密度满足

\[
p(x_0\mid x,c)
=\frac{p_t(x\mid x_0,c)p(x_0\mid c)}{p_t(x\mid c)}.
\]

对后验均值第 \(i\) 个坐标求导。标准的“期望求导等于与 score 的协方差”给出

\[
\begin{aligned}
\frac{\partial (m_t^c)_i(x)}{\partial x_j}
&=\operatorname{Cov}\!\left(
(X_0)_i,
\frac{\alpha_t(X_0)_j-x_j}{\sigma_t^2}
\middle|X_t=x,c
\right)\\
&=\frac{\alpha_t}{\sigma_t^2}
\operatorname{Cov}\!\left((X_0)_i,(X_0)_j\mid X_t=x,c\right).
\end{aligned}
\]

把所有坐标放回矩阵：

\[
\boxed{
J_t^c(x)
=\frac{\alpha_t}{\sigma_t^2}
\operatorname{Cov}(X_0\mid X_t=x,c).
}
\]

由于协方差矩阵必然对称半正定，并且 \(\alpha_t/\sigma_t^2>0\)，所以精确后验均值逐点满足

\[
\boxed{
J_t^c(x)=J_t^c(x)^\top,
\qquad
J_t^c(x)\succeq0.
}
\]

这里不是样本平均、渐近近似或“小 guidance”结论。只要观测真由上面的高斯腐化得到，它对每个 \(x,t,c\) 都成立。

半正定还有一个直接直觉。任取很小的状态扰动 \(v\)，一阶近似为

\[
m_t^c(x+\eta v)-m_t^c(x)\approx \eta J_t^c(x)v.
\]

因此

\[
v^\top\{m_t^c(x+\eta v)-m_t^c(x)\}
\approx\eta v^\top J_t^c(x)v\ge0.
\]

也就是说：沿 \(v\) 增加一点观测证据时，理想后验干净预测不能在同一方向上反向响应。负二次型表示局部映射“折回来”。

## 3. 为什么要投影，而不是构造完整 Jacobian

完整 \(4096\times4096\) Jacobian 既昂贵，也会把大量与肉眼结构无关的高频方向混在一起。事前固定 \(r\) 个正交方向，写成矩阵 \(Q\)。若完整 \(J\) 对称半正定，则

\[
B=Q^\top JQ
\]

仍然必然对称半正定，因为

\[
B^\top=Q^\top J^\top Q=B,
\]

且任意 \(z\in\mathbb R^r\) 都有

\[
z^\top Bz=(Qz)^\top J(Qz)\ge0.
\]

因此只要投影矩阵 \(B\) 违反对称性或半正定性，就已经足以否定“当前网络在这个子空间里像一个合法后验均值”。反过来，\(B\) 合法不能证明完整 \(J\) 合法；这是低成本单向检验。

第一版计划使用固定的 4 维 Hadamard channel mode × 4 个二维 DCT 低/中频方向，共 \(r=16\) 维。基在任何标签、图片和该候选分数打开前写死，并验证 \(Q^\top Q=I\)。

## 4. 为什么分数不是丑陋拼接

把任意实矩阵 \(B\) 分解为

\[
B=S+A,
\qquad
S=\frac{B+B^\top}{2},
\qquad
A=\frac{B-B^\top}{2}.
\]

对称矩阵空间与反对称矩阵空间在 Frobenius 内积下正交。再将

\[
S=U\operatorname{diag}(\lambda_1,\ldots,\lambda_r)U^\top
\]

做特征分解，并定义

\[
S_+=U\operatorname{diag}(\max\{\lambda_i,0\})U^\top,
\]

\[
S_-=U\operatorname{diag}(\min\{\lambda_i,0\})U^\top.
\]

离 \(B\) 最近的对称半正定矩阵恰好是 \(S_+\)。于是

\[
\boxed{
\operatorname{dist}_F^2(B,\mathbb S_+)
=\|A\|_F^2+\|S_-\|_F^2.
}
\]

这里两项的系数都固定为 1，不是经验调参：

- \(\|A\|_F^2\) 是非对称、非保守或局部旋转缺陷；
- \(\|S_-\|_F^2\) 是负曲率、反单调或局部折叠缺陷；
- 两者相加正好是到理论容许集合的欧氏几何距离。

归一化分数定义为

\[
\boxed{
D(B)=
\frac{\|A\|_F^2+\|S_-\|_F^2}
{\|B\|_F^2+\epsilon}.
}
\]

由于零矩阵本身属于对称半正定锥，所以最近点距离不超过到零矩阵的距离：

\[
0\le D(B)\le1
\]

（分母加入极小 \(\epsilon\) 时上界只会更紧）。理想后验均值严格有 \(D=0\)。质量方向在实验前唯一固定为：**高 \(D\) 更可疑**；失败后不能看低尾救结果。

## 5. 单路径如何计算

对第 \(b\) 个单位基向量 \(q_b\)，中心有限差分给出

\[
\widehat Jq_b
\approx
\frac{
\widehat m(x+hq_b)-\widehat m(x-hq_b)
}{2h}.
\]

再与每个 \(q_a\) 做内积：

\[
B_{ab}
\approx
q_a^\top
\frac{
\widehat m(x+hq_b)-\widehat m(x-hq_b)
}{2h}.
\]

这只是在同一个当前状态附近做确定性网络查询，没有随机后缀，也没有候选终点可供选择。

为减少“有限差分步长碰巧选得好”的问题，令

\[
s(x,t)=\sqrt d\max\{\operatorname{RMS}(x),\sigma_t\}.
\]

其中 \(\operatorname{RMS}(x)=\|x\|_2/\sqrt d\)。第一版事前固定半径

\[
h=2^{-9}s(x,t),
\qquad 2h=2^{-8}s(x,t).
\]

分别得到 \(B_h\) 与 \(B_{2h}\)，再用中心差分的 Richardson 外推

\[
\boxed{
B_R=\frac{4B_h-B_{2h}}{3}.
}
\]

若网络局部足够光滑，中心差分主误差为 \(O(h^2)\)，上式消掉该主项。主缺陷分数只由 \(B_R\) 计算；同时记录

\[
R_h=
\frac{\|B_h-B_{2h}\|_F}
{\|B_R\|_F+\epsilon}
\]

作为纯数值稳定性控制。若 \(D\) 只在 \(R_h\) 很大时升高，就更像有限差分失败，而不是模型几何信号。

三个固定检查点不直接平均各自的归一化 \(D\)，而是先合并分子和分母：

\[
\boxed{
D_{\mathrm{path}}
=
\frac{\sum_{k\in\mathcal K}
\operatorname{dist}_F^2(B_{R,k},\mathbb S_+)}
{\sum_{k\in\mathcal K}\|B_{R,k}\|_F^2+\epsilon}.
}
\]

其中 \(\mathcal K\) 是事前固定的三个 sampling checkpoint。这个公式等于把三个 \(B_{R,k}\) 放进一个 block-diagonal 大矩阵后计算同一个锥距离比例，因此仍在 \([0,1]\)，也不会让一个几乎为零、归一化不稳定的 Jacobian 时点获得过大权重。

## 6. 为什么它可能对应融合、错接和结构翻转

这部分是待检验机制，不是已证明事实。

当对称部分出现负特征值时，某个低/中频结构方向 \(v\) 满足

\[
v^\top\widehat Jv<0.
\]

在直觉上，这意味着相邻 noisy 状态的干净结构预测发生反向排序：增强某个空间证据，预测结构却沿相反方向移动。它可能表现为两个区域被映到同一位置、部位连接突然交换，或物体附着关系折回。

反对称部分不对应任何后验协方差。它表示在局部一阶场中含有旋转成分；两个扰动方向的交叉响应不满足互易关系。它可能与局部结构扭转有关，但已有研究也提醒：非保守分量未必总是有害，因此必须让 bad/good 数据决定它是否有质量语义。

纯粹的合法后验不确定性可以让 Jacobian 很大，但只要它仍对称半正定，\(D\) 就保持为零。因此这个分数与“变化幅度大”不同。

## 7. raw conditional 与 CFG 必须分开

主量必须查询 raw class-conditional head：

\[
\widehat m_t^c(x).
\]

不能直接对 CFG 外推后的 `pred_xstart` 套 PSD 定理。CFG 将 conditional 与 unconditional 输出线性外推，系数可以为负；两个合法后验均值的这种外推组合不必对应任何归一化后验，其 Jacobian 也不必半正定。

不过，可以在 CFG 实际生成轨迹到达的状态 \(x\) 上查询 raw conditional head。高斯腐化后的规范后验对所有 \(x\) 都有定义，所以点态 Tweedie 恒等式并不要求 \(x\) 必须由 raw conditional sampler 自己抽到。

因此实验对象应准确写成：

> CFG/DDPM 实际轨迹访问状态上的 raw conditional denoiser admissibility defect。

它不是 CFG sampler 自身违反 PSD，也不是实际 CFG 输出的后验 Jacobian。

## 8. 它能与不能区分什么

如果最终图坏、但整条轨迹的 \(D\) 都低，可能意味着：

- 模型已把错误结构学成稳定的局部合法模态；
- 投影子空间没有覆盖真正的缺陷方向；
- 错误不是局部一阶几何能够发现的。

这类情况不应被本方法强行修正，更像模型能力限制或检测盲区。

如果某条路径先出现高 \(D\)，而独立整轨重采后不再出现且盲评改善，则更像随机轨迹进入了一个可逃逸的 learned-denoiser 折叠区。只有这样的干预响应，才能进一步支持“采样失败”而非“能力失败”。

反例同样明确：

1. 一个总生成同一张坏图的精确后验模型可以处处满足 PSD；
2. 一个生成很好、但 raw denoiser 局部带小的非保守误差的模型可以有高 \(D\)；
3. 投影 \(Q\) 漏掉的负方向不会被发现；
4. 有限差分不稳定可以制造假违例。

## 9. 第一轮实验的停止纪律

第一步只做一个不看标签的数值 smoke test：

- 检查 \(Q^\top Q\)；
- 检查 \(D\in[0,1]\)；
- 检查两步长与 Richardson 结果；
- 检查 raw conditional 查询不改变 baseline state；
- 检查所有输入、源码和输出哈希链。

数值稳定后，先冻结：

- 投影基与频率；
- 投影维数 \(r\)；
- 两个相对步长；
- 三个检查点；
- 路径聚合公式；
- 唯一质量方向 high-is-worse；
- 明确定义的 class/checkpoint robustness，而不是再写含糊的“不能依赖单轴”。

之后才允许完整跑分和打开现有锁定标签。该池仍然只能叫 discovery，因为我们已看过它的视觉标签。失败后不得翻符号、换频率、换时间点或改 subtype 继续救同一池。

## 10. 新颖性边界

Tweedie/Miyasawa 后验恒等式、score 或 denoiser Jacobian 的对称性/半正定性、非保守 score 分析，以及用 Jacobian 特征做 OOD，都已有直接先验工作：

- Manor 与 Michaeli，*On the Posterior Distribution in Denoising*，ICLR 2024：后验协方差与 denoiser Jacobian 的二阶 Tweedie 关系，并用有限差分子空间迭代估计谱；
- Chao 等，*On Investigating the Conservative Property of Score-Based Generative Models*，ICML 2023：score Jacobian 对称性、非保守误差与 QCSBM；
- Hen 等，*Jacobian-Aware Posterior Sampling for Inverse Problems*，TMLR 2026：直接测量 learned denoiser 的负特征值与不对称，并将 Jacobian 非理想性用于逆问题修正；
- Shoushtari 等，*EigenScore*，ICLR 2026：用 denoiser Jacobian 所对应的后验协方差谱做 OOD 检测；
- Cao 等，*Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts*，CVPR 2025：沿生成轨迹检测 artifact 并选择性局部重加噪，但触发量是 temporal score acceleration，不是 PSD 锥距离。

因此不能声称首次发现 PSD 或对称性约束，也不能宽泛声称首次做在线 artifact 检测或选择性重加噪。

目前唯一可能守住、而且必须靠实验成立的贡献边界是：

> 首次把冻结类别条件扩散模型在实际生成轨迹上的投影 Tweedie 容许锥距离，用作成图前的逐样本 artifact 预测量；再以独立校准的低干预预算触发重采，并用可逃逸性区分偶发采样失败与稳定模型能力失败。

在任何独立事件充足验证与因果修复实验之前，这句话也只能是研究假设，不能写成论文结论。

主要来源：

- https://proceedings.iclr.cc/paper_files/paper/2024/hash/d6adf82fa531dcb8bfa53c224ce5e466-Abstract-Conference.html
- https://proceedings.mlr.press/v202/chao23a.html
- https://openreview.net/forum?id=m63GJnhIN2
- https://openreview.net/pdf/1a84a2da67f5ee6586a21c770e045a9f9ce9acc0.pdf
- https://openaccess.thecvf.com/content/CVPR2025/html/Cao_Temporal_Score_Analysis_for_Understanding_and_Correcting_Diffusion_Artifacts_CVPR_2025_paper.html
