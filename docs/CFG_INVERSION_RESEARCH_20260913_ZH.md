# 从反演到 CFG：保持什么、改变什么，以及有限时间引导的研究边界

流匹配确实有一个不需要新高斯噪声的“返回同一输入”操作：固定模型、条件和 guidance 规则，沿同一个 ODE 反向积分，再正向积分。此前把“没有原图条件接口”和“没有保持输入的数学操作”混在一起，说得过于绝对。前者是模型能力问题，后者已有明确答案。

用户提出的“高 CFG 前进，再用普通条件模型反演”也完全可以实施；不过它切换了速度场，目标不再是返回原状态。它是一种**把高 CFG 的未来终点，换算成普通条件流的起点**的操作。这个定义提供了值得研究的连接：局部极限是普通 CFG，有限区间则包含未来模型响应的非线性运输。

基本强前—弱逆—强前构造已经有 Z-Sampling、W2SD 等直接前例；FSG 属于相关算子家族。仓库此前的 lifting 和 FSG 审计也已经研究了其中的重要部分。当前可推进的问题应明确到：**在有效强度、计算量和数值误差得到控制后，有限时间的运输能否提供普通标量 CFG 没有的质量收益？** 本报告给出操作定义、文献边界、独立推导、反例和下一步的判别实验，不将已有反射算法重新命名为新方法。[^1][^2][^3]

## 1. 反演是不是“网络预测速度，然后用负速度走”？

基本理解正确，但必须同时指定状态和时间。全文统一采用 SiT 常用的方向：噪声为 \(t=0\)，图像 latent 为 \(t=1\)。固定条件 \(c\)，写成

\[
\frac{dz_t}{dt}=v_w(z_t,t;c),\qquad
v_w=v_u+w(v_c-v_u).
\]

这里 \(w=0\) 是 null，\(w=1\) 是普通 conditional，\(w>1\) 是外推。其他论文的额外系数 \(\gamma\) 可能满足 \(w=1+\gamma\)，不能混用数字。

反演可以直接把积分区间从 \(1\to0\)：仍预测同一个 \(v_w\)，使用负的步长。也可以定义递增的反向时钟 \(s=1-t\)：

\[
\boxed{\frac{dy_s}{ds}=-v_w(y_s,1-s;c),\qquad y_0=z_1.}
\]

每个反演步骤都在**当前逆向状态、对应原时间**上重新调用网络。这不是将前向时保存的某个速度一直取负，也不是在错误时间 \(s\) 上查询原网络。如果已经使用负步长，不再额外把速度取负。RF-Inversion 明确写出了这种时间反转；流匹配不需要额外训练一个反向网络才拥有这个操作。[^4]

若模型原生预测 epsilon 或 clean image，先按模型的噪声路径转换为实际 ODE 速度；不能直接把任意预测头输出取负。对于线性 FM 路径 \(z_t=tX+(1-t)\epsilon\)，在内部时间且使用相应 clean 预测时，\(v=(\widehat X-z_t)/(1-t)\)。原生 velocity 模型直接使用其速度头，端点处理遵循原采样器。

以下示意代码只说明时间和符号；生产实现还要处理批量时间、模型时间映射及求解精度：

```python
def heun_step(z, t, t_next, field):
    dt = t_next - t              # inverse 时为负数
    k1 = field(z, t)
    z_predict = z + dt * k1
    k2 = field(z_predict, t_next)
    return z + 0.5 * dt * (k1 + k2)

# 生成：遍历 grid；反演：遍历 reversed(grid)。
# field 内使用相同条件和按绝对 t 查询的 guidance schedule。
```

这个说法限定在确定性 ODE。SDE 的分布反向过程不能仅靠把 drift 取负得到。若 CFG-CTRL、动量或缓存参与状态更新，需要把控制器历史也纳入状态；只反走 latent 并清空历史，不是原完整系统的逆。具有离散丢弃或非可逆更新的控制器也不自动有精确逆。

## 2. 反向求解 ODE，不等于精确撤销一条离散更新

设前向 Euler 是

\[
y=x+h v(x,t).
\]

反向 Euler 是

\[
\widetilde x=y-hv(y,t+h).
\]

而精确撤销**原来的 Euler 映射**需要解

\[
\boxed{x=y-hv(x,t).}
\]

未知量出现在网络输入里，所以通常要迭代求解。一种 Picard 迭代为 \(x^{k+1}=y-hv(x^k,t)\)；在适当自映射区域且 \(|h|L<1\) 时可保证收敛。ReNoise 借用的正是反复修正未知逆状态的思想，其名称中的 noising 不应理解为每次独立撒入新高斯噪声。EDICT 采用另一条路线：引入两个耦合 latent，把**新构造的扩展状态离散过程**设计为代数可逆。[^5][^6]

差别不是措辞：对于 \(v(x)=a x\)，精确 ODE 正反组合是 \(e^{-ah}e^{ah}x=x\)，Euler 正反组合却是

\[
(1-ah)(1+ah)x=(1-a^2h^2)x.
\]

因此同场 round-trip 漂移可以完全来自求解器。如果 \(a=-1/h\)，前向 Euler 把所有输入都映射到 0，离散映射甚至不可逆，但对应连续 ODE 仍可逆。Heun 提高精度，仍不等于精确撤销原 Euler 或 Heun 更新。缓存前向速度后做算术减法可以撤销一次更新，但用到了额外记录，不是仅从输出独立反演。

对一般光滑非自治场，反向 Euler 的同场误差为

\[
\widetilde x-x=-h^2(\partial_t v+J_vv)+O(h^3).
\]

更大的 CFG 可能放大局部变化率和误差传播，因为 \(J_{v_w}=J_{v_u}+w(J_{v_c}-J_{v_u})\)。这解释数值困难的一个来源；并不意味着高 CFG 的光滑 ODE 必然失去可逆性，也不意味着反向积分必然收缩。一般 Lipschitz 假设只支持可能随 \(e^{\int Ldt}\) 放大的误差界。

## 3. 为什么它更适合原来的“输入保持不变”问题

把一张图的 latent \(x\) 放到时间 \(\tau\)，有两个不同操作：

\[
\text{fresh 加噪：}\quad z_\tau=\tau x+(1-\tau)\xi,\quad \xi\sim\mathcal N(0,I),
\]

\[
\text{ODE 反演：}\quad z_\tau=\Phi_w^{\tau\leftarrow1}(x).
\]

前者引入新随机变量；仅给模型 \(z_\tau,c\)，通常没有逐图唯一恢复原始 \(x\) 的承诺。后者在同一个确定性模型流里给原图找坐标，不引入新的独立噪声。存在唯一解且正反轨迹均存在时，

\[
\boxed{\Phi_w^{1\leftarrow\tau}\bigl(\Phi_w^{\tau\leftarrow1}(x)\bigr)=x.}
\]

这是一个合法的“不变”对象。原图在这里作为**边界值**进入，无须模型新增图像条件通道。若要求模型像多模态编辑器一样读取源图与“保持不变”的指令，则仍需要源图条件模型或显式控制机制；二者不是同一种接口。RF-Inversion 的受控版本还可显式利用源图目标，因此应与纯 ODE 反演分开。[^4]

但这个恒等式不能给模型质量排序。一个把数据映射得很糟的可逆函数，同样可以有完美的逆；精确反演也会忠实保留输入缺陷。真实图像的逆坐标不保证是典型高斯样本，latent 重建不保证 VAE 编解码逐像素无损。这里改善的是实验问题的定义，不是已经证明生成质量更好。

若做五轮“反演—重建”，应该逐轮保存 \(x_0,\ldots,x_5\)，分别报告 latent、VAE 与像素误差。在纯同场精确流中五轮都恒等；有限精度下测到的是数值与编解码累积。它不能直接复现黑盒多模态模型的指令遵循、历史记忆或语义编辑能力。

## 4. 高 CFG 前进、普通 conditional 反演，实际留下什么

固定同一区间 \([t,t+h]\)，令

\[
g=v_c-v_u,\quad A=v_c,\quad B=v_c+\gamma g,\quad \gamma=w-1,
\]

\[
C=\Phi_A^{t+h\leftarrow t},\qquad H=\Phi_B^{t+h\leftarrow t}.
\]

用户提出的操作是

\[
\boxed{R=C^{-1}\circ H:\quad x_t\longmapsto y_{t+h}\longmapsto x'_t.}
\]

它返回原时间层，但通常不返回原状态。一步显式实现是

\[
y=x+h[v_c(x,t)+\gamma g(x,t)],\qquad
x'=y-hv_c(y,t+h).
\]

因此

\[
\boxed{x'-x=h\gamma g(x,t)+O(h^2).}
\]

用精确子流也有相同的一阶项。例如高 CFG \(w=3\)、逆腿 conditional \(w=1\)，留下约 \(2h g\)。若逆腿换成 null \(w=0\)，留下约 \(3h g\)。它们的动作强度不同；比较时不能将这个差直接归因于“conditional 更理解图片”。

**真正保持的是所指定的未来终点：**

\[
\boxed{C(R(x))=H(x).}
\]

也就是普通条件模型从新起点 \(R(x)\) 出发，会到达高 CFG 从旧起点 \(x\) 出发的同一个局部未来。这里的“保持不变”有了明确位置：保持的是目标 \(H(x)\)，而不是强行要求 \(R(x)=x\)。这是相对流的定义，不要求高 CFG 的目标一定更好。

```mermaid
flowchart LR
    X["原起点 x，时间 t"] -->|"高 CFG 前进 H"| Y["固定未来目标 y，时间 t+h"]
    Y -->|"普通条件反演 C⁻¹"| Z["新起点 R(x)，时间 t"]
    Z -->|"普通条件前进 C"| Y
```

最终接哪条生成腿至关重要：

|操作|精确关系或首阶行为|解释|
|---|---|---|
|强前→同一强场逆|\(H^{-1}H=I\)|同场重建，精确情况下零位移|
|强前→条件逆|\(R=C^{-1}H\)|产生同一时间的起点校正，尚无净前进|
|强前→条件逆→同区间条件前|\(CR=H\)|逆腿被下一条条件腿抵消，等于最初一次强 CFG 前进|
|强前→条件逆→强前|\(HR=HC^{-1}H\)|首阶有效场 \(2B-A\)，对应 \(w_{\rm eff}=2w-1\)|

比如 \(w=3\) 的三腿强—条件逆—强，应先与 \(w\approx5\) 的普通 CFG 对照，而不是只与 \(w=3\) 比。对精确光滑子流，\(HC^{-1}H\) 是对称分裂，局部二阶对应 \(2B-A\)，与该有效场的精确流从三阶开始不同。

若回灌 \(R\) 共 \(K\) 次后用强场推进，首阶有效系数是 \(w+K(w-1)\)；最后用条件场推进则为 \(1+K(w-1)\)。重复次数本身不是新机制。若只有一次完整校正，随后始终沿同一条件流生成到终点，结果也只等于“先走那段强 CFG，再接条件后缀”。

## 5. 有限区间为何可能比一个 CFG 系数包含更多信息

下面是独立推导，用于解释已知算子，不宣称新定理或质量保证。令

\[
X_s=\Phi_B^{t+s\leftarrow t}(x),\qquad
R(s)=\Phi_A^{t\leftarrow t+s}(X_s).
\]

对区间长度求导，得到

\[
\boxed{\frac{dR(s)}{ds}
=D_x\Phi_A^{t\leftarrow t+s}(X_s)\,[B(X_s,t+s)-A(X_s,t+s)].}
\]

因而

\[
R(h)-x=
\int_0^h D_x\Phi_A^{t\leftarrow t+s}(X_s)\,\gamma g(X_s,t+s)\,ds.
\]

一次局部 Euler 式 CFG 更新在当前状态看一次差值；这个算子沿高 CFG 的未来轨迹查看差值，再通过条件逆流的敏感性把影响换算回起点。普通 CFG 在有限区间内也会重新评估未来状态，因此区别不能仅归为“看到了未来”；这里的特殊操作是通过另一条参考流将影响运输回原时间。逆流可以隐式实现这种非线性运输，不必显式构造高维 Jacobian。它使用模型在不同状态上的预测，没有外部新增高斯噪声。

但“包含未来响应”与“能提高质量”之间仍缺实验和目标假设。被跟随的高 CFG 未来可能已经过饱和或丢失类内多样性；运输一个错误目标不会使目标自动正确。

更具体地，假设 \(\gamma\) 在区间内固定，导数均在 \((x,t)\) 处，定义括号

\[
[A,g]=J_gA-J_Ag.
\]

精确回路的二阶展开为

\[
\boxed{R(h)-x
=\gamma h g+\frac{h^2}{2}
\bigl(\gamma\partial_tg+\gamma[A,g]+\gamma^2J_gg\bigr)+O(h^3).}
\]

一阶是原 CFG，二阶包含差值随时间变化、与条件场的非交换作用，以及差值沿自身的非线性变化。仅减去 \(\gamma h g\) 还没有剥离所有“普通 gap 自身的有限步运动”；其中 \(\gamma^2J_gg\) 也可以来自 gap 流自身的曲率。将一个剩余量叫作新语义信息，需要额外证据。

实际的一步 Euler 强前—条件逆则是

\[
x'-x=\gamma h g
-h^2\bigl(\partial_tA+J_AA+\gamma J_Ag\bigr)+O(h^3).
\]

这里即使 \(\gamma=0\) 仍留有同场误差。减去同场 Euler 回路能去掉公共误差的一部分，却不能把结果直接变成上述精确运输公式。发布实现、精确 ODE 解释和新的数值适配器应分别命名、分别验证。

## 6. 哪些论文已经做了相近操作

|工作|实际操作|对当前研究最直接的用途|
|---|---|---|
|DDIM，2020 / ICLR 2021|确定性路径的编码与重建；普通反向离散近似有误差|反演的基本起点，不是首次来自近期 FM[^7]|
|Null-text inversion，2022 / CVPR 2023|低 CFG 构造 pivot，再优化每步 null embedding，使高 CFG 重建该轨迹|两方向 guidance 不同时，需要补偿，不能默认互逆[^8]|
|EDICT，2022 / CVPR 2023|两个耦合 latent 的可逆离散更新|若真要严格离散重建，可以修改状态与算法结构[^6]|
|ReNoise，2024 / ECCV 2024|反复评估未知逆状态，近似求解隐式逆方程|借鉴迭代求逆，而非 fresh 随机加噪[^5]|
|RF-Inversion，2024 / ICLR 2025|纯 RF 逆流与另加控制的反演/编辑|区分原图边界、控制条件和原始同场逆[^4]|
|RF-Solver、FireFlow，2024 / ICML 2025|中点或缓存中点预测，提高反演积分效率；编辑另有特征注入|借鉴数值实现，不能把更准的求解称为更强生成场[^9][^10]|
|Z-Sampling，2024 / ICLR 2025|强 guidance 前进→弱 guidance 反演→强 guidance 前进|用户构造最直接的已有参数化[^1]|
|W2SD，2025|强模型前进、弱模型反演、强模型继续；强弱可来自 guidance 或其他模型差异|把同一构造连接到 autoguidance / internal guidance[^2]|
|FSG，2025 / NeurIPS 2025|选定时刻重复 guided 前瞻→null 反演，再结合 CFG++ 推进|属于相关家族；逆腿、区间和最终推进需逐一对齐[^3]|
|RA-GRPO，2026-09-03 预印本|同一 FM 模型两档 guidance 的反射，再用于偏好训练和轨迹吸收|“反射并蒸馏”也已有直接近例[^11]|

作者源码核查还带来两个实现层面的约束。Z-Sampling 论文公式和文字对 \(\gamma=0\) 的称谓不一致；其公开代码用 \(v_u+w(v_c-v_u)\) 的约定，逆腿 0 是 null，1 是 conditional。Z 与 W2SD 公开实现的逆腿在已推进状态上仍查询旧模型时间，而 inverse scheduler 使用下一时间索引，是一种相邻步近似；不能直接等同于正确新状态、新时间上的精确逆 ODE。FSG 的 scheduler 时钟还需按具体 diffusers 版本重放。详见本次[原文与源码对照](research/cfg_inversion_20260913/fsg_w2sd_prior_art.md)，其中保存固定提交链接。

RF-Solver 和 FireFlow 的 FLUX 示例常以 guidance=1 反演，再以另一 guidance 返回，但这里的 guidance 是蒸馏模型的输入，不能直接当成 SiT 的双分支 CFG。FireFlow 的算法可参考；其仅凭 Lipschitz 推出逆向误差必按 \(e^{-LT}\) 衰减的命题条件不足：\(v=-Lx\) 的逆流就按 \(e^{LT}\) 放大。具体原文位置与反例见[反演文献笔记](research/cfg_inversion_20260913/inversion_literature.md)。这项限制不否定其算法的经验结果。[^9][^10]

Ctrl-Z Sampling 也使用“反演/回退”的名称，但它的探索操作包含新的随机高斯噪声及奖励筛选；它属于相关搜索路线，不是当前要找的确定性同图逆映射。不能只凭方法名字把二者合并。[^12]

## 7. CFG 中可以合法要求保持什么

把所有 \(F_a\) 定义在同一个时间区间，令 \(T_{b\leftarrow a}=F_b^{-1}F_a\)。可要求以下结构恒等式：

\[
\begin{aligned}
T_{a\leftarrow a}&=I,\\
T_{a\leftarrow b}\circ T_{b\leftarrow a}&=I,\\
T_{c\leftarrow b}\circ T_{b\leftarrow a}&=T_{c\leftarrow a},\\
F_b\circ T_{b\leftarrow a}&=F_a.
\end{aligned}
\]

它们分别表示同场返回、参考切换互逆、经中间参考切换与直接切换一致，以及指定未来不变。最后一个等式最贴近当前直觉：可以改变引导场下的起点表示，同时保持选定输出。本次矩阵检查中，这些参考切换等式的误差不超过 \(2.23\times10^{-16}\)。

这些恒等式适合作为实现的内部一致性约束。不能强行要求不同场的 \(T_{b\leftarrow a}=I\)，也不能将满足上述等式直接作为质量奖励。即便所有学习场都很差，只要各自光滑可逆，这些关系仍成立。一个偏离目标 ODE、但自身可逆的离散映射也满足这些代数关系；因此还需要独立的步长收敛检查来评价 ODE 精度。反过来，较小的数值误差也不一定对应更好的样本。

同一个 \((z,t,c)\) 上，CFG 对系数的仿射关系是另一个严格结构：\(v_1=v_c\)、\(v_0=v_u\)、\(\partial_wv_w=g\)。但有限时间终点并不对 \(w\) 仿射，正是这种差别给有限区间研究留下空间。状态依赖的 controller、历史更新、阈值裁剪会改变这套简单结构，应作为额外系统单独讨论。

## 8. 本次核查和仓库已有证据

本次由三个独立分工分别核查反演文献、最近前例及作者代码、算子推导与反例；分歧集中在“返回必须为零”和“反演后仍沿条件流走是否有收益”，最终用明确算子与数值检查裁决。没有启动新的图像扫参。

本次 CPU 检查使用带时间依赖的仿射场及矩阵指数，从而避开把数值 ODE 误差当理论余项。结果见[算子辩论笔记](research/cfg_inversion_20260913/operator_debate.md)和[可执行检查](../experiments/cfg_inversion_20260913/operator_audit.py)：

|检查|结果|能够支持什么|
|---|---:|---|
|精确 mixed loop 减去二阶展开|余项实测阶数约 2.99976|展开含时间项和括号项的实现一致性|
|\(HC^{-1}H\) 对精确 \(2B-A\) 流|局部误差阶数约 3.00245|完整三腿的二阶分裂解释|
|\(CC^{-1}H=H\)|最大误差约 \(2.48\times10^{-16}\)|最后沿同一区间条件流走的准确抵消|
|部分写入 \(C[I+\theta(R-I)]\)|扣除预测二阶差后，余项阶数约 3.00134|它与普通场 \(A+\theta(B-A)\) 的关系|

部分写入的具体差是

\[
C[I+\theta(R-I)]-\Phi_{A+\theta D}
=\frac{h^2}{2}\theta(1-\theta)J_DD+O(h^3),\quad D=B-A,
\]

此处 \(\theta\) 在区间内固定；时间或状态反馈系数还会引入对应导数项。因此仅加一个部分写入系数，也落在既有 lifting/分裂分析内。

反对“准确模型的 mixed loop 应恒等”的检查使用相容的真实联合分布：\(C\sim N(0,1)\)，\(X\mid C=c\sim N(c,1)\)，独立高斯噪声与标准线性 FM 路径；conditional 和 null 都用解析正确速度。在 \(c=0,t=0.2,h=0.15,\gamma=0.8\) 时，mixed loop 把 \(x=1\) 移到约 0.943425。模型没有估计错误，位移仍然存在。它是不同分布流的差，而不是网络应消除的错误。

仓库此前真实 SiT 审计已经给出更强的实践依据：[Jacobian 与目标审计](SIT_FSG_JACOBIAN_AUDIT_RESULTS_20260911_ZH.md)显示，\(H=1/8\) 的精细逆流对固定前向目标的转移相对误差约 0.0376%，但将该有限位移仅用一次局部 Jacobian 解释仍有约 11.29% 的误差。短区间 \(H=1/1024\) 时，精细位移已很接近 \(Hwg\)，方向余弦约 0.999980。这支持“有限反演确实实现非线性坐标运输”，同时说明短区间会退化为普通 CFG。

该审计还发现，用高精度流终点和用算法自己的 Euler 前向终点评价时，FSG 与同长度 CFG 的排序会改变。目标身份必须固定，不能一边生成 Euler 目标、一边用另一条高精度目标批评它的反演；更不能把这类 latent 误差直接写成 FID 收益。

已有[宽强度 lifting 结果](LIFTING_WIDE_SCALE_RESULTS_20260909_ZH.md)也没有建立一致的最佳质量优势：SiT-XL 已观察最优 FID 约 38.0574 对 38.0656，RAEv2 约 38.2642 对 38.4523；lifting 更耗时。这些是有限样本、部分历史搜索的结果，不是当前 conditional 逆腿的最终结论，但足以说明不能从少数同名系数上的改善重启大扫参。

## 9. 最值得推进的两个问题

**问题一：有限时间运输是否存在无法被强度重标定解释的有用方向？**

研究对象保持为 \(R=C^{-1}H\)，先将 numerical defect 与普通 CFG 主项分开，再考察未来重新评估产生的方向。设 \(d=R(x)-x\)，在 \(g\ne0\) 时使用带符号投影

\[
\beta=\frac{\langle d,g\rangle}{\langle g,g\rangle},\qquad
d_\perp=d-\beta g.
\]

同时比较完整 \(d\)、纯平行 \(\beta g\)、去掉或反转 \(d_\perp\)，并保留未来普通 CFG、同 NFE 更细步积分、同场往返等对照。所有位移分支须接同一条预先固定的主推进腿和后缀；若主推进就是同区间 \(C\)，完整 \(d\) 分支严格等于原始 \(H\)，应明确用作等价对照。若需要范数匹配，应另报告匹配后的对照，因为单纯换范数又改变了平行幅度。\(d_\perp\ne0\) 只说明不是当前点上的单标量 gap，不自动排除时间 schedule 或后续普通 CFG 轨迹能实现类似效果。有效强度校准本身已有 Guidance Matters 等前例。[^13]

判定标准是：该有限运输成分在独立噪声、固定质量与语义读出上产生可重复收益，同时不能由更充分调参的 CFG 或同成本求解器解释。若更精确的 inverse 使收益消失，应研究实际离散更新的效应，放弃“精确逆流额外理解语义”的说法。

**问题二：普通 conditional 能否成为更合适的逆流参考，而不只是更小的 guidance gap？**

固定目标有效系数 \(W\)。一轮三腿 reflection 对逆腿系数 \(b\) 使用前腿 \(a=(W+b)/2\)，即可匹配首阶 \(2a-b=W\)。分别用 \(b=0\) 与 \(b=1\)，这样比较的首阶强度相同。还需用实际带符号投影校准有限区间的剩余强度差，并匹配算力；null 和 conditional 单腿本身通常都只需一次分支预测。

这检验的是不同参考场如何运输未来差值，以及在什么时间、什么样本上可靠。高 guidance 不应自动命名为“强模型”；若它的终点更差，精确运输只会更准确地实现坏目标。对 IG，将 \(A\) 替换为明确的弱模型/内部头，将 \(B\) 替换为主模型即可沿用算子分析；但弱头与主头的尺度、时间参数、输出物理量必须一致。

这两个问题具有可证伪性，但尚不能认定其实现或理论尚无前人研究。基础 reflection、多轮、自适应回退和轨迹蒸馏都有相关先例。贡献需要落在明确的新机制证据、可验证的条件或同成本优势上。

## 10. 下一步的最小实验及停止条件

优先利用仓库已捕获的真实 SiT 状态，做一组小而完整的算子诊断：固定相同 \((z,t,c)\)，在 \(t=0.25,0.5,0.75\) 取几个区间长度；实现时间对齐的 Euler 与精细 Heun 逆流，保留一个更细网格验证误差平台。逐个检查同场返回、终点保持、参考切换组合与最终条件腿的抵消。求离散 Euler 逆时，以逆方程残差判定收敛，不用已知原状态充当迭代答案。

随后只对“有效强度匹配的 null/conditional 参考”和“平行/额外方向”进入小规模生成筛查。若仍需五轮回灌，每轮保存同时间 latent、位移和各自固定后缀解码的图；明确图像是中间 latent 的读出，而非又一次像素图输入。完整样本质量筛查与上述算子检查分开，不把一轮的局部重建指标当五轮的最终收益。

满足以下任一情形应停止该候选的大规模质量扫描：差异低于 inverse 求解误差；收益被标量强度或同成本多步 CFG 解释；额外方向在留出样本上没有稳定收益；只有持续提高 CFG、牺牲多样性或语义正确性才改善单项代理分数。这样能尽早识别有效研究空间，避免重复仓库已经排除的解释。

本次完成的是文献核查、数学裁决及 CPU 算子验证；没有新增 GPU 图像实验，也没有宣称找到新的质量增益。检索截至 2026-09-13。三份专题笔记分别为[反演操作与文献](research/cfg_inversion_20260913/inversion_literature.md)、[FSG/W2SD 前例与源码](research/cfg_inversion_20260913/fsg_w2sd_prior_art.md)、[算子推导与反例](research/cfg_inversion_20260913/operator_debate.md)。

## 原始来源

[^1]: Bai et al. [Zigzag Diffusion Sampling: Diffusion Models Can Self-Improve via Self-Reflection](https://arxiv.org/html/2412.10891v2). Algorithm 1、Eq. (2–5)。首发 2024-12，ICLR 2025；[作者源码固定提交](https://github.com/xie-lab-ml/Zigzag-Diffusion-Sampling/blob/eef8bb265deb8f33efd47c53e6ee5506606de360/utils/pipeline_stable_diffusion_xl.py#L1421)。
[^2]: Bai, Sugiyama, Xie. [Weak-to-Strong Diffusion with Reflection](https://arxiv.org/html/2502.00473v3). Algorithm 1、§4.2、Appendix E.2；本文核查 arXiv v3；[作者 reflection 实现](https://github.com/xie-lab-ml/Weak-to-Strong-Diffusion-with-Reflection/blob/c1c160d634837e92dd98851a74b52aa5415f5da5/utils/pipeline_stable_diffusion_xl.py#L1099)。
[^3]: Wang et al. [Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](https://arxiv.org/abs/2510.21512). Algorithm 1、Theorem 1、Appendix C；NeurIPS 2025；[作者实现固定提交](https://github.com/Ka1b0/Foresight-Guidance/blob/012398fae56912f88fd8fec588b4ceb92800d9d6/utils/pipeline_stable_diffusion_xl.py#L995)。
[^4]: Rout et al. [Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations](https://arxiv.org/html/2410.10792v1). §3.2 Proposition 3.1、§3.3 与 §3.5；ICLR 2025。
[^5]: Garibi et al. [ReNoise: Real Image Inversion Through Iterative Noising](https://arxiv.org/html/2403.14602v1). Eq. (1–4)、Algorithm 1；ECCV 2024。
[^6]: Wallace, Gokul, Naik. [EDICT: Exact Diffusion Inversion via Coupled Transformations](https://arxiv.org/html/2211.12446v1). §4、Eq. (10–15)；CVPR 2023。
[^7]: Song, Meng, Ermon. [Denoising Diffusion Implicit Models](https://arxiv.org/html/2010.02502v3). §4.3、§5.4；ICLR 2021。
[^8]: Mokady et al. [Null-text Inversion for Editing Real Images using Guided Diffusion Models](https://arxiv.org/html/2211.09794v1). §3.2、Algorithm 1；CVPR 2023。
[^9]: Wang et al. [Taming Rectified Flow for Inversion and Editing](https://proceedings.mlr.press/v267/wang25ce.html). ICML 2025；[原文 §3.2](https://arxiv.org/html/2411.04746v1)；[作者代码](https://github.com/wangjiangshan0725/RF-Solver-Edit)。
[^10]: Deng et al. [FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing](https://proceedings.mlr.press/v267/deng25c.html). ICML 2025；[正式 PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/deng25c/deng25c.pdf)，§4、Appendix A；理论限制针对 Proposition 3.1 / Appendix B.1 的一般收缩式。
[^11]: Wu et al. [Step Back to Move Forward: Reflection-Aware Preference Optimization for Visual Generation](https://arxiv.org/html/2609.04282v1). 2026-09-03 预印本，§3.2–3.3。核查方法与公式，未复现训练结果。
[^12]: Mao et al. [Ctrl-Z Sampling](https://arxiv.org/abs/2506.20294). 原始 2025-06；本文仅用于区分随机回退探索和确定性逆流。
[^13]: Xie et al. [Guidance Matters: Rethinking the Evaluation Pitfall for Text-to-Image Generation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/559a0998fab1d19b80e7e43a5852401c-Abstract-Conference.html). ICLR 2026。引用用途限定为有效 guidance 投影校准，不能由其有限实验推断所有替代引导都无效。
