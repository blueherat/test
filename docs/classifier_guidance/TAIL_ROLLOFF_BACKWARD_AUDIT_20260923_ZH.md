# 最新末端回落与反向传播实现核验

2026-09-23，读取至北京时间约13:16的固定快照。核对曲线历史、源码归属、历史真实模型审计，并补充CPU梯度与有限差分检查；没有修改训练或插入GPU任务。

**结论：末端回落是真的，两组都经历了“正峰从最后一格逐渐前移、末格先升后降”。目前证据不支持遗漏Heun分支、截断strong输入Jacobian或截断采样时间步。最新真实checkpoint的尾部有限精度误差，以及有限步采样器和终点损失的效应，仍未排除。**

## 1. 最新曲线和历史

固定数据见 [latest_tail_snapshot.json](../research/self_guidance_schedule_shape_20260923/latest_tail_snapshot.json)。

| 实验 | step / 系数更新数 | 最小额外系数 | 后段峰值及中点 | 最后一格 |
|---|---|---:|---|---:|
| JiT frozen weak | 10354 / 10226 | −1.43405 | 1.84465，t=.93 | .015998 |
| 旧SiT native frozen weak | 30128 / 30000 | −1.13449 | 4.04591，t=.96094 | .431281 |
| 新SiT joint | 9198 / 9070 | .45597 | 尾部.83286，t=.97656 | .791977 |

JiT最后四格：1.84465 → 1.69654 → 1.25830 → .015998。原先“JiT负谷仍在强弱凸组合内部”仅适用于早期快照；现在a<−1，已越过weak。最后a≈0意味着趋近strong，仍正常去噪。

| JiT系数更新数 | 尾峰bin（从0编号） | 最后一格 |
|---:|---:|---:|
| 约2k | 49 | .948 |
| 约3k | 49 | 1.001 |
| 3927 | 48 | .938 |
| 约5k | 47 | .798 |
| 约8k | 46 | .320 |
| 约10k | 46 | .0557 |
| 10226 | 46 | .0160 |

旧SiT也经历峰从bin63移到62、再移到61；末格约15k更新时2.38，20k约2.10、25k约1.50、30k约.431。

这显示边界附近的控制在训练中重新分配。末格并非初始固定为零，也不是仅由EMA产生的绘图断点。

## 2. 实际工程简化与梯度路径

正式JiT请求记录中的关键文件hash与当前sampler、field、feature、梯度累积和训练文件一致。旧native SiT的sampler快照也已找到；差异是后来增加混合Heun/Euler支持和只捕获用到的模式，全Heun反传代数未变。

| 处理 | 实际含义 | 梯度影响 |
|---|---|---|
| 冻结strong/weak参数 | 不求权重梯度，仍求状态输入导数 | 未截断所需一阶链 |
| 每步重算field、激活checkpoint | 避免同时存全部内部激活 | 用重算交换内存 |
| 局部detach后求VJP | 建立局部图，再显式接回cotangent | 不能仅凭detach认定断链 |
| CUDA graph replay | 重放相同算子及VJP | 没有设计上的截断，仍需核验数值 |
| microbatch与双卡 | 累积同步后统一裁剪和Adam | 数学上保留全批目标；浮点累加顺序不同 |
| once_differentiable | 不支持再对backward高阶求导 | 不影响当前所需一阶系数梯度 |
| BF16、部分低精度算子 | 有限精度运算 | 可能影响小梯度 |
| 最后Euler、分母floor、RGB clamp | 定义实际前向映射 | 改变目标映射，不是其backward漏项 |

旧FM训练的Runtime.prefix有no_grad，但当前JiT GAN调用forward_with_intermediate，不经过那个prefix。固定amounts存的是bin索引，真正系数在field内从schedule.parameters取出；所有区间active，零系数也有梯度。

重算与截断应当区分。[PyTorch checkpoint说明](https://docs.pytorch.org/docs/2.14/checkpoint.html)

## 3. Heun两个分支都保留

令

\[
v_1=v(x,t,a),\quad y=x+hv_1,\quad
v_2=v(y,t+h,a),\quad F=x+\frac h2(v_1+v_2).
\]

后续cotangent为p，记 \(J_1=D_xv_1,J_2=D_yv_2\)，则

\[
r_2=J_2^\top\frac h2p,\quad q_1=\frac h2p+hr_2,\quad
p_{\rm prev}=p+r_2+J_1^\top q_1.
\]

记 \(B_1=\partial_av_1,B_2=\partial_av_2\)，系数梯度为

\[
\boxed{\partial_aL=B_1^\top q_1+B_2^\top\frac h2p.}
\]

当前代码逐项保留这些贡献，包括predictor经第二stage回到第一stage的路径。完整反传结束才统一更新系数，没有“先更新末格，再用改后的参数算前格”的顺序更新。

## 4. 最后一格其实不依赖长链Jacobian连乘

JiT最后interval为 \(t=.98,h=.02\)，分母floor=.05。固定进入该步的状态，strong/weak clean predictions记为 \(\widehat x_s,\widehat x_w\)，有

\[
x_{50}=x_{49}+.4[
\widehat x_w+(1+a_{49})(\widehat x_s-\widehat x_w)-x_{49}].
\]

因此

\[
\boxed{\frac{\partial L}{\partial a_{49}}
=.4\langle\nabla_{x_{50}}L,\widehat x_s-\widehat x_w\rangle.}
\tag{1}
\]

这条偏导不需要乘前49步状态Jacobian。前面的步骤决定当前状态和损失，间接影响数值，但最后系数的梯度路径只有最后field和图像反馈。

旧SiT最后是Heun，其最后系数也只涉及最后interval的两次field和区间内链式导数，无须前63步Jacobian连乘。

所以“长链反传越传越坏，特别把最后一格压下去”不是该计算图直接支持的解释。最后Euler只有一个stage也不意味着少算一半梯度：前向本来就只有该stage。

## 5. 已有与本轮核验

| 历史真实模型检查 | 已存结果 |
|---|---|
| 旧SiT ordinary autograd对照 | 端点差0；梯度相对差1.73e−5、8.10e−6 |
| 旧SiT第48格有限差分 | 相对差约.229%；不是最后第63格 |
| JiT初始及人工signed系数 | 梯度相对差约.05044%、.04321% |
| JiT完整GAN速度档 | 梯度最大相对差约.4885% |
| JiT单/双卡 | 梯度相对差约.2086%，一次系数更新最大差5.96e−7 |

这些是历史证据，不是本轮最新checkpoint重跑。

本轮重跑当前相关CPU测试：15通过，4个CUDA测试跳过。另直接使用实际Sampler(graphs=False)，读取step10343的50个真实系数，替换为小型非线性CPU float64 field，保留clean-to-velocity floor、49 Heun加最后Euler、RGB clamp，并对照全Heun：

- 端点与普通autograd完全一致。
- 两种求解器的系数梯度相对差约8.1e−17、1.5e−16。
- 对首部、负谷和末尾六格做两种扰动尺度的中心有限差分，均通过。
- 故意detach Heun predictor，前向不变，梯度相对差变成11.8%、34.9%，说明检查能够识别丢链。

这没有覆盖真实JiT权重、判别器、BF16或CUDA graph。[本轮CPU结果](../research/self_guidance_schedule_shape_20260923/latest_tail_backward_cpu_audit.json)

全向量相对误差小，不保证某个很小的尾部梯度相对误差小或符号正确；数学等价的浮点累加也不保证逐位相同。[PyTorch数值精度说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)

## 6. 末格在受到持续更新，而非冻结

最近500条诊断：

| 位置 | 平均裁剪前损失梯度 | 平均实际系数更新 |
|---|---:|---:|
| JiT末格49 | +3.26e−4 | −1.19e−4 |
| JiT峰格46 | −5.99e−4 | +1.69e−4 |
| SiT末格63 | +6.61e−4 | −1.59e−4 |
| SiT峰格61 | −7.14e−4 | +1.43e−4 |

500条诊断并非500个step：JiT覆盖2368–10352，SiT覆盖17650–30125。实际更新还受Adam和裁剪影响，不能把更新符号与当步梯度机械等同。

另用连续训练日志的相邻系数差检查最近500次实际更新，避开诊断抽样：JiT末格74.6%的更新向下，平均增量−1.773e−4；旧SiT末格89.0%向下，平均增量−2.766e−4。对应峰格46、61的平均增量分别为+3.926e−5、+1.007e−4。所抽样诊断中这些位置的梯度零率均为0。

新SiT joint要单独看：它在生成时间上的最后一格低于倒数第二格，但最近500次训练更新中，最后一格净上升.018108。不能把三组都说成末格随训练下降。

检查的保存点Adam步数恰好等于全局step减128预热，未见年龄重置。系数Adam无weight decay；末格并非被权重衰减拉向零。epsilon相对二阶矩平方根约1e−6量级，没有epsilon压死尾梯度的迹象。无GradScaler或动态overflow跳步；EMA不回灌训练。

日志未发现重复/缺失step或所检索的错误事件。这些支持“训练正在增大峰值、压低末格”，不独立证明该目标等于真实质量，也不完全认证梯度数值。

## 7. 当前应区分的原因

**反传计算错误。** 代数、源码归属、历史真实对照和本轮CPU检查不支持大范围漏链。最新BF16尾部逐格误差仍是未覆盖项。

**正确优化了有限步生成器。** JiT最后Euler、t>.95的floor、RGB clamp，以及共同的有限Heun网格，会改变训练映射。梯度完全正确时，最优系数也可能末端回落。SiT没有JiT的floor/最终Euler，因此这两项不能单独解释共性。

**终点作用重新分配。** 峰前移可能意味着稍早的控制经过后续模型修正后更有价值，而最后直接混入weak的边际作用转差。这与伴随或多模态抵消相容，尚是假说。

Eq.(1)给出可直接检验的解释：最后额外系数的边际损失由终点损失梯度与strong−weak方向的内积决定。该内积为正时，局部梯度下降希望降低a；为负时希望提高a。当前日志的正平均梯度与末格下降相容，但Adam动量和变化中的D意味着不能逐步机械等同。a≈0意味着回到strong输出组合，不意味着没有控制、没有去噪或终点状态回到原始strong轨迹。

普通有限时域最优控制并不自动给出a(T)=0。只有具体终端代价、控制方向与可能的控制惩罚共同满足相应驻点条件时，零系数才可能最优。不能把本轮接近零直接称为已验证的终端边界条件。

两组还共享终点GAN、Inception反馈、图像clamp和Adam等条件。跨模型出现类似形状，不等于排除了共同实现或目标的影响。

训练迭代中的“末格先升后降”和生成时间中的“正负起伏”属于两个时间轴；目前没有据此发现训练参数在迭代上周期振荡。

## 8. 最直接的后续判别

四张GPU在本轮检查时均有任务，没有插入额外实验。应固定最新可用checkpoint、D、噪声和标签，做同一前向映射的梯度对照：

1. CUDA replay与普通checkpoint autograd逐尾bin比较绝对差、相对差、符号；不能只报全向量norm。
2. JiT最后一格用Eq.(1)独立重建；冻结该步输入后做suffix系数差分，无须重复整个长链反传。更一般地，当前逐格独立参数允许缓存并detach尾部之前的状态，然后用普通autograd重建最后K步；此前状态不依赖这K个参数，所以这仍是尾部系数的精确梯度。此结论不适用于共享schedule网络参数或joint weak参数。
3. 使用多个差分尺度，区分实际BF16映射与高精度映射；过小扰动可能被低精度舍入吞掉。
4. 同映射梯度吻合后，再固定物理系数函数加密网格，分别消融末步规则和floor。改变前向是机制实验，不是修正原backward。

**当前判断：回落是被训练更新持续推动的结构，尚未发现反传简化漏项；它的动力学意义与有限步、精度依赖仍需辨识。**

证据：

- [训练历史与配置审计](../research/self_guidance_schedule_shape_20260923/latest_tail_audit.md)
- [反传代码详细审计](../research/self_guidance_schedule_shape_20260923/backward_implementation_audit.md)
- [正式运行源码归属](../research/self_guidance_schedule_shape_20260923/backward_source_hash_audit.json)
- [本轮CPU脚本](../../experiments/theory_self_guidance_20260922/latest_tail_backward_cpu_audit.py)
