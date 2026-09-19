# Z-Sampling：缓存起点的两次离散逆迭代审核

2026-09-13。仅数学、原始文献及仓库只读核查；未写采样器、未运行 GPU 或新实验。裁决：**可以作为原始 Z 的同查询预算数值变体进行有界生成比较。** 不称为新 no-op 操作，也不把更准确求逆等同于质量改善。

## 1. 指定被反演的对象及合法恒等性质

时间 t=0 为 noise，t=1 为 data。固定一个区间 [t,t+h]，所有下列模型查询均用物理左端时间 t。记

\[
A(z)=v_{w_s}(z,t),\quad B(z)=v_{w_b}(z,t),\quad
v_w=u+w(c-u),\quad D=A-B,\quad F_A(z)=z+hA(z).
\]

先作 y=F_A(x)。原始 Z 的本次 FM/Euler 移植为 q_Z=y−hB(y)，最后输出 F_A(q_Z)。这对应作者代码在已推进状态查询旧 t，再 inverse scheduler、再 strong step 的顺序；SDXL scheduler 本身不能直接认作 FM Euler。[原始 Algorithm 1](https://arxiv.org/html/2412.10891v2#S3.SS3)、[固定作者源码 L1434–1496](https://github.com/xie-lab-ml/Zigzag-Diffusion-Sampling/blob/eef8bb265deb8f33efd47c53e6ee5506606de360/utils/pipeline_stable_diffusion_xl.py#L1434)。

候选希望求**实际左端 Euler 映射** F_B 的前像 q*，即 q*+hB(q*)=y，而非把反向显式 Euler 当作 F_B^{-1}。缓存首个 c(x),u(x)，由其免费取得 B(x)，执行

\[
q_0=x,\quad q_1=x+h[A(x)-B(x)],\quad
q_2=x+h[A(x)-B(q_1)],\quad x_{next}=q_2+hA(q_2).
\]

q_1、q_2 都属于返回后的 t 层。B(q_1,t) 使用旧 t 正是此离散逆方程所需的时间，不是漏改成 t+h。y 与 A(x) 在求解内固定，不能随 q 更新。

若 A=B，同场首个差为零，q_1=q_2=x，最终恰为 F_A(x)。这是**回环为恒等、净推进恢复原生一步**的合法契约，不是要求最终输出停在 x。实现需保留 x+h(A−B) 的 delta 运算顺序；直接计算 y−hB(x) 会重新引入浮点消去误差。同场分支也需使用相同预测与运算端点。两场不同时 q≠x 是有意的 guidance，不能惩罚它为零。

## 2. 改变了哪一项，没有证明什么

在 t 固定、场足够光滑、h 小的局部展开中，

\[
q_2=x+hD-h^2J_BD+O(h^3),\qquad
F_A(q_2)=x+h(2A-B)+h^2J_DD+O(h^3).
\]

原始 Z 对应

\[
F_A(q_Z)=x+h(2A-B)+h^2[J_DD-J_BB]+O(h^3).
\]

所以候选相对原版的领先差为 +h²J_BB：去掉原版即使 D=0 也存在的数值自回环项。两者一阶 guidance 完全相同。连续精确流的强—弱逆—强组合，以及非自治有效场的二阶积分，均不是上述 Euler 公式；**整条方案仍通常是全局一阶，不能因两次 Picard 称为二阶采样器。**

若 |h|L_B<1，Picard 具有局部收缩依据，||q_2−q*||≤(|h|L_B)²||x−q*||=O(h³)。一般场没有这个保证，固定两次也不代表已经得到精确逆。其逆方程残差恰为 h[B(q_2)−B(q_1)]；若实际记录它，需要额外的 B(q_2) 查询，不能偷算进 5-call 主采样预算。

可能收益的具体假说是：原 Z 中与 guidance gap 无关的数值运动干扰生成，消去后保留更准确的强弱差效应。但它也可能曾提供有益的偶然正则化。反例：目标输出 N(0,1)，同场模型给出错误的局部 Euler 比例 1+r，0<r≪1。候选忠实得到 (1+r)x；原 Z 得到 (1−r²)(1+r)x，反而离目标单位方差更近。故合法恒等约束、较小逆误差均不推出 FID 改善。

## 3. 当前预定比较精确配平

候选与原 Z 都用 56 个均匀 Euler 主区间：前 42 个 [0,.75) 全部各 reflection 一次，后 14 个为普通 conditional。定义最终希望比较的有效 extra 为 a，则

\[
w_{eff}=2w_s-w_b=1+a,\qquad
w_s=(1+a+w_b)/2.
\]

因为所有 active 区间都 reflection，一阶有效 extra a 在整个前 .75 一致，没有仅部分区间额外增强的混杂。空间常向量场下，这个强度等价精确成立；非线性场中只是首阶配平，h² 以上差异正是待测对象。

固定 a∈{.75,1.125,1.25,1.5,2}、w_b∈{0,1}，共 10 个设置；每个比较原 Z 与缓存 q₂。特意不取 a=1：与 w_b=0 配对时 w_s=1，原 Z 可优化到 3 calls/event，而缓存 q₂ 为 4，不能用冗余查询假造 5-call 配平。现有网格的 w_s 均不等于 0 或 1，两种算法每 event 都为 2+1+2=5 个分支查询。a=.75,w_b=0 时 w_s=.875、strong extra=−.125，合法含义是相对 inverse 更强，不应 clamp 成 ≥1。

|完整采样器|按输出计的实际分支查询|
|---|---:|
|Z 原版 / 缓存 q₂，56 Euler|42×5+14=224|
|普通 CFG，128 Euler|96×2+32=224|
|普通 CFG，64 Heun|48×4+16×2=224|
|已有 APG，64 Heun|48×4+16×2=224|

原版与候选使用相同 noise / labels / a / inverse / 网格；普通 CFG128、CFG64 在相同有效 a 附近调优，APG 同样使用已建立的强基线。224 是模型分支查询相等，不保证 wall time 相等；需要计入额外状态运算。网格不同导致的数值取舍由 Euler128 / Heun64 同预算对照揭示，不能用弱的 Euler56 作唯一基线。

## 4. 仓库重复与胜负标准

[旧 operator_debate §5](operator_debate.md) 已推导真正 Euler 逆的差方程与二阶消去，不能称此次首次发现。experiments/audit_raev2_anchored_inverse.py 已在 RAE 的 noise-time 记号下求解同一个冻结目标 Euler 前像方程，但其范围是 8 条轨迹的数值探针、没有完整 Z 生成质量实验。旧 picard_weak / FWR 在未来时间更新 reference 或速度，其方程与这里固定 y、在旧 t 求 q₂ 后重评估 strong 不同；旧隐式 AG 则解两分支都在新状态求值的 gap 方程。

在此次有界源码与结果检索中，未找到已经执行的、与这套 SiT 56-step / 2+1+2 / cached q₂ 完全等价的图像质量实验；这不构成全面的新颖性检索结论。

可继续的证据必须是：候选在配对设置下优于原 Z，且经调优后对 224-call CFG / APG 仍有竞争力，再以独立生成 bank 确认。只降低 inverse residual、仅胜未配平有效强度的 Z、或仅胜少调用的 Euler56，都不能算目标达成。没有质量收益就结束这一固定构造，不围绕数值契约无限增加反演次数。
