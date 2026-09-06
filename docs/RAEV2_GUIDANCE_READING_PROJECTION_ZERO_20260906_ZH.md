# RAEv2 guidance 阅读：APG 与 CFG-Zero⋆

日期：2026-09-06。新增精读两篇此前十七篇目录未覆盖的直接 guidance 原文；既有笔记只曾引用 CFG-Zero 的对照分数。本轮保存了两份完整 PDF、HTML、作者实现和 SHA，阅读正文机制／公式、主要实验、相关附录及实现。**没有新模型、GPU、RAE 数据计算、采样或 FID。**

**最有价值的共同提醒是：投影坐标本身就是机制假设。** APG 在 clean prediction 上抑制径向增益；CFG-Zero⋆ 在 velocity 上保留弱预测方向、放大其正交部分。两者都有清楚的线性代数结构，但都没有从该结构推出真实终点质量改善。直接把它们的投影写在 RAEv2 的 Full/Base clean 输出上，会把两个不同方法混为一谈。

## 1. APG：像素增益直觉准确到哪里

阅读版本：Sadat、Hilliges、Weber，*Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models*，ICLR 2025，arXiv **2410.02416v2，2025-06-03**。[完整原文](https://arxiv.org/html/2410.02416v2)，[PDF](https://arxiv.org/pdf/2410.02416v2)，[OpenReview](https://openreview.net/forum?id=e2ONKX6qzJ)。具体读了 §3–5、Appendix A/B、C.1–C.8、D 及 Algorithm 1；PDF 图 8 另作图像检查。没有把社区 WebUI 复刻当作作者实现；本次未定位到独立作者代码库，以原论文 Algorithm 1 提供的 PyTorch 代码为准。

令 conditional clean prediction 为 `F`，unconditional clean prediction 为 `U`，`d=F−U`，`α=w−1`。论文的原始 CFG 是 `F+αd`。核心分解为

\[
d_\parallel=\frac{\langle d,F\rangle}{\|F\|^2}F,\qquad
d_\perp=d-d_\parallel,\qquad
F_{\rm proj}=F+\alpha(d_\perp+\eta d_\parallel).
\]

**被证明的是分解与增益恒等式；“哪个方向影响质量”主要由干预实验支持。** 若 `d∥` 与 F 同向，平行更新确实把 F 乘以大于一的数。对具有固定数值范围的图像，这解释了增益／对比度增加。论文明确写出同向假设；不能省掉这个条件，把负平行更新也称为增益。论文再以图像、颜色指标和一个 exact-score 500 维双 Gaussian toy 支持过强 guidance 的机制。该 toy 排除了神经估计误差，却没有证明任意 latent 表示或任意时间的平行成分都有害。[§4、Appendix B 末尾](https://arxiv.org/html/2410.02416v2#S4)。

**独立推论：正交更新不等于保持范数。** η=0 时，`||F+αd⊥||²=||F||²+α²||d⊥||²`。它消除了范数的一阶径向变化，有限步仍可增大范数；更没有保证任意非线性 decoder 后的饱和度下降。RAE 的 DINO 表示不是 RGB 数值，原文“image-like latent”直觉在此需要实际验证。变换 clean 坐标原点也会改变投影方向：F、U 同加 a 保持 gap 不变，却把投影参照从 F 变成 F+a。这个坐标依赖没有被正交代数消除。

论文 Appendix A 把 CFG 写成对输出变量 F 的一次梯度上升：

\[
F\leftarrow F+\alpha\nabla_F\tfrac12\|F-U\|^2.
\]

这里把 U 固定，梯度对 **prediction 坐标 F** 求导；它不是对实际采样状态 z 的梯度，也不是数据 likelihood 的梯度。若目标被写成 `f(F(z),U(z))` 再对 z 求导，需要相应 Jacobian。该恒等式为 clipping 和 momentum 提供优化直觉，不提供它们的收敛或 FID 保证。[Appendix A](https://arxiv.org/html/2410.02416v2#A1)。

**作者代码的实际顺序是 reverse momentum → norm clipping → projection。** `m_k=d_k+βm_{k−1}`、`m←m min(1,r/||m||)`，再相对当前 F 分解；因此启用 momentum 后，被投影的是历史滤波结果，不再只是当前 gap。投影先转 FP64，按每个样本的 C/H/W 全维求内积，再转回输入 dtype。Algorithm 1 的函数缺省 η=1、r=0、无 buffer 会退化到 CFG；推荐主实验设置另见 Table 10，不能只调用函数缺省就声称实现了论文 APG。[Algorithm 1](https://arxiv.org/html/2410.02416v2#alg1)。

参数选择是公开的经验设计。η 推荐 0；EDM2-S 用 `(r,β)=(2.5,−0.75)`，DiT 用 `(5,−0.5)`，SDXL 用 `(15,−0.5)`。作者建议按观测 gap norm 选 r，给出 β 的经验范围，并在 Table 9 做消融。这不是任意手工时间 schedule，也不应被歪曲成“极大量调参”；但 r、β 不能由其理论唯一确定，照搬到 `[1024,16,16]` 的 RAE 后再扫参数不符合当前目标。普通 APG 不要求时间窗口；Appendix C.1 的 **IG 是 interval guidance，不是 RAE 的 internal guidance**。[Tables 9–10、Appendix D](https://arxiv.org/html/2410.02416v2#A3.S8)。

**实证有价值，但需要看比较对象。** Table 1 的 EDM2-S 在 w=4 从 FID 10.42 降到 6.49；SDXL 在 w=15 从 26.29 降到 25.35。Appendix D 用 10K 图评估 ImageNet。Table 2 去掉投影／clipping／momentum 分别得到 6.63／7.93／6.85：该组 FID 的主要增益不能只归于“去掉径向分量”。图 8 的 scale 扫描确实包含 w=1 到 5，不能说作者没有比较低 guidance；从图上两方法各自较好区间的接近程度，也不能读取出精确的“最优 baseline 提升 ≥5%”。[§5](https://arxiv.org/html/2410.02416v2#S5)。

论文报告 SDXL 单图 denoiser forward 约 130 ms、guidance 运算约 0.45 ms，RTX 3090。它支持低额外运算成本；不等于本任务 B=8、4090、巨大 DINO latent 上已计时，也未提供 RAE 官方 IG 与 APG 的同总成本比较。额外模型调用数为零不代表 vector reduction、FP64 投影和 momentum 状态成本严格为零。

## 2. CFG-Zero⋆：闭式比例到底优化了什么

阅读版本：Fan、Zheng、Yeh、Liu，*CFG-Zero⋆: Improved Classifier-Free Guidance for Flow Matching Models*，arXiv **2503.18886v2，2025-04-03**。[完整原文](https://arxiv.org/html/2503.18886v2)，[PDF](https://arxiv.org/pdf/2503.18886v2)，[作者仓库](https://github.com/WeichenFan/CFG-Zero-star)。具体读了 §3–5、Eq.4–12、Algorithms、Tables 1–8、Appendix A1/A3，并查看 PDF 第 4、8 页排除公式或表格的 OCR 误读。作者代码固定到 commit `3162be1fba5dd0129ac8423ad6919d928f420a8d`，读了 SD3、Wan 和 Flux 的 guidance 分支。

用 c、u 表示 conditional/unconditional **velocity**，统一 `α=w−1`，方法的非零步骤为

\[
v_s=c+\alpha(c-su),\qquad
s^*=\arg\min_s\|c-su\|^2
=\frac{\langle c,u\rangle}{\|u\|^2}.
\]

**这个动态比例来自一个明确的局部最小二乘，没有手调的时间系数。** 无数值 ε 时可重写为

\[
v_{s^*}=P_uc+w(I-P_u)c.
\]

即保留 c 在弱 velocity 方向上的成分，仅放大其正交成分。这一几何解释比“scalar 修正了估计误差”更精确，因为最小二乘没有访问真实 velocity。[§4.1](https://arxiv.org/html/2503.18886v2#S4.SS1)。

原文推导有两处需要分开处理的错误。Eq.6→7 要求 `ω′=ω−1`，紧随文字却写 `ω′=1+ω`。即使修正该符号，Eq.9 印刷的平方范数上界仍一般不成立。例如标量 `c=u=1, v*=0, α=1, s=0.1`，左侧 3.61，印刷右侧 1.81。这个例子已在 CPU 中复核，PDF 也确实如此。

**不能由此直接否定其投影算法。** 以下是独立修复，令 `e=c−v*`、`d_s=c−su`：

\[
\|e+\alpha d_s\|^2
\le2\|e\|^2+2\alpha^2\|d_s\|^2
\le4\|c\|^2+4\|v^*\|^2+2\alpha^2\|d_s\|^2.
\]

上界有效，未知项都与 s 无关，仍导出同一个 `s*`。**真正未解决的是：更小的上界不保证更小的真实误差。** 取 `c=(2,1), u=(1,0), α=1, v*=(3,2)`，普通 s=1 的 guidance 完全正确；投影 s*=2 把真实误差平方从 0 增到 1，尽管 surrogate 从 2 降到 1。这是否定普遍误差保证的有限维反例，没有声称它就是 SiT/RAE 的实际误差。

还有两个实现边界：正文要求 s>0，闭式解／附录代码却没有非负截断；若内积为负，返回值也为负。代码分母加 `1e−8`，可视为固定数值稳定项，对应有微小 ridge 的解，严格正交恒等式因而只近似成立。代码没有直接计算真实 velocity residual。[Appendix A3；固定版本 SD3 实现](https://github.com/WeichenFan/CFG-Zero-star/blob/3162be1fba5dd0129ac8423ad6919d928f420a8d/models/sd/sd3_pipeline.py#L72)。

**zero-init 的机制是一个需要验证的经验不等式，而不是“真实初速度为零”。** 作者在欠拟合 Gaussian toy 上观察到初速度的估计误差大于用零替代的误差，然后把最初 K 个 solver 更新设为零。等协方差 Gaussian 的精确公式本来就给出 `v*(z,0)=μ−z`；它通常不为零。ImageNet 小模型在训练充分后出现反转：Table 1 的 160 epoch，CFG FID 2.84，zero-init 2.85。正文有“up to 160”与“beyond 160”的宽泛表述；表中反转已经发生在 160。不能把欠拟合解释写成对任何 checkpoint 都有用的保证。[§4.2–4.3、Table 1](https://arxiv.org/html/2503.18886v2#S4)。

K 的选择没有由该不等式确定。算法缺省首一步；Table 7 的 Lumina-Next、SD3 更喜欢前两步，SD3.5 更喜欢一步；作者 README 另推荐总步数 4% 作为起点。这里确实包含经验初始时间窗口，与无需手调的 `s*` 是不同部分。不能因为比例自然，就把整个 CFG-Zero⋆ 称为无窗口方法；也不能因为 zero-init 需要 K，就说 optimized scale 本身依赖手工 schedule。

**成本与效果也要按具体实现拆开。** SiT-XL ImageNet Table 2 报 CFG FID 2.23 → CFG-Zero⋆ 2.10，约 5.83%；这属于该论文的实际优势，不能抹去。正文说明使用未到上述 turning point 的 checkpoint、标准 guidance scale；相关段落没有提供足以复现完整采样数／独立 seed／同总成本的细节，当前作者仓库的上述代码也不是该 SiT 实验 runner。T2I 附录为 200 个 prompt，每个生成 10 张，主要评 Aesthetic/CLIP；这些并非 FID。[§4.3、Appendix A3](https://arxiv.org/html/2503.18886v2#S4.SS3)。

SD3 和 Wan 的固定版本代码都先算模型输出与 `s*`，再在 `i <= zero_steps` 时把 velocity 乘零，因此 `zero_steps=0` 实际跳过一次**状态更新**，却没有省去这一步的模型调用。若另行短路模型调用，会是待验证的新实现。Flux 的当前 `Guidance_distilled.py` 只示范 pure zero-init；对应 pipeline 在 `do_true_cfg` 分支仍用普通 CFG，在另一分支才做 zero-init，虽然函数签名保留 `use_cfg_zero_star`。不能把这一当前示范误写成复现论文 de-distilled Flux 的完整 optimized-scale 实验。[SD3](https://github.com/WeichenFan/CFG-Zero-star/blob/3162be1fba5dd0129ac8423ad6919d928f420a8d/models/sd/sd3_pipeline.py#L1068)，[Wan](https://github.com/WeichenFan/CFG-Zero-star/blob/3162be1fba5dd0129ac8423ad6919d928f420a8d/models/wan/wan_pipeline.py#L535)，[Flux](https://github.com/WeichenFan/CFG-Zero-star/blob/3162be1fba5dd0129ac8423ad6919d928f420a8d/models/flux/pipeline.py#L936)。

Table 8 的额外 FLOPs 数字已对照 PDF：SD3 1024² 写作 `1.6e−3 M`。当前 reduction 的完整操作至少应随 latent 元素数线性增长，论文没有给出足够统计口径来解释这个极小数值；本轮不替作者修正单位，也不据此认领严格成本。定性的“无需额外 backbone 调用、只有点积与乘法”是可信结构判断，RAE 实际成本仍需计时。

## 3. 迁移到 RAEv2 时必须保留的状态项

下面是本轮独立代数，未实现或运行新 guidance。以数据方向时间 `s=1−t`、当前状态 z、Full/Base **同类 clean prediction** F/B 表示 native IG：

\[
G=F+\alpha(F-B),\qquad \alpha=0.78,
\qquad b_F=(F-z)/t,\quad b_B=(B-z)/t.
\]

公式讨论 active interval 内、100 步 native 网格 floor 不生效的情形；实际 BF16 加法次序与区间外回到 Full 的规则仍须另行保持。这里 Base 是同类弱预测，不是 unconditional posterior；两篇的条件／无条件 score 解读不能直接赋予它。

APG 的 clean 投影参照可以明确写成 F，但其“径向增益对应图像过饱和”的假设未在 DINO latent 得证。CFG-Zero⋆ 若忠实在 velocity 上算，则

\[
s^*=\frac{\langle F-z,B-z\rangle}{\|B-z\|^2},\qquad
G_* =G+\alpha(1-s^*)(B-z).
\]

该状态项来自 `G_*=z+t[b_F+α(b_F−s*b_B)]`。直接改成 `s*=<F,B>/||B||²` 再输出 `wF−αs*B` 是另一个方法。CPU 随机固定例中，正确变换恒等式最大误差约 `1e−15`，错误 clean 替换与正确结果相差范数 0.239；这只核对坐标代数，没有模拟 RAE。

上式暂省略数值 ε。若忠实保留作者 velocity 分母里的 ε，则 clean 坐标中的分母应为 `||B−z||²+t²ε`，不是随意在 clean 范数后再加同一个 ε。理论坐标变换与数值稳定项也要一并处理。

## 4. 一个值得优先证伪的机制

**只保留一个具体问题：RAE 当前 checkpoint 的起始 velocity 是否真的比零更不准确？** 这来自 CFG-Zero⋆ 的核心解释，却可以利用 RAE 的独立 linear bridge 写成可识别、无需时窗搜索的边界检验。

在唯一的噪声端点 t=1，`z=ε` 与类内真实 X 独立，精确 dataward velocity 是

\[
b^*(z,1,c)=\mu_c-z,\qquad \mu_c=\mathbb E[X\mid c].
\]

因此，官方起始场比零更差的必要实证依据是

\[
\Delta_0
=\mathbb E\big[\|G(z,1,c)-\mu_c\|^2-\|z-\mu_c\|^2\big]>0.
\]

不用把有限五图 prototype 当作真实 μ：同类 X 与 z 独立时，使用留出真实 X 的配对差 `||G−X||²−||z−X||²`，其期望恰等于 Δ₀，类内真实方差项消去。必须冻结 checkpoint、噪声／类别、native BF16 计算和全部样本，不挑正差类或扫描最初若干步。若 Δ₀ 明确为负，就直接否定“初速度差到不如零”在该边界的解释；不能继续通过调 K 找成功。若为正，也仅支持这个起点误差机制，**不能越过此前已经证明的局部误差／终点质量缺口**，更不自动允许 zero-init、手工窗口或训练。

这里精确的非零 `μ_c−z` 也指出一种更严谨的理论边界条件；它未确定有限样本估计、与后续轨迹的兼容性及 FID 收益。除此以外，本轮没有从 APG 的 clipping／reverse momentum 找到一个无需另立机制假设、可直接获准在 RAE 搜参的设计。

## 5. 归档与核对

数据目录：[reading_projection_zero_v1](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_projection_zero_v1)。`downloads.json` 记录 URL、版本固定路径、最终 URL、字节与下载 SHA；OpenReview 的直接 HTML 请求返回 challenge 页，已明确保留此失败状态，未当作论坛正文读取。权威论文内容来自两篇 arXiv v2 完整原文。

- APG PDF SHA：`89e2f5330484fb18d6f6e9cdf11f65c616a36eff36552b192b4a978d4527ebbf`。
- CFG-Zero⋆ PDF SHA：`362c518e46ec2996fc596b9673a0ee6680247717af9b11054f80b205e1099be0`。
- `cfgzero_code_sources.json` 固定官方 commit、六个下载文件 URL/SHA；`apg_author_algorithm1.txt`/`.html` 直接摘存作者原文代码，不冒称新的官方 Git 仓库。
- `check_math_cpu.py`/`math_checks.json` 保存两个反例、APG 有限范数变化、RAE velocity/clean 变换核对；CPU wall 0.124 秒、CPU 0.435 秒，零 GPU／模型／FID 调用。PDF 渲染与文献下载另属阅读成本，未混作实验采样成本。
- `reading_manifest.json` 记录实际阅读范围、来源与产物的 SHA、失效来源及代码审计行号；没有更改 goal、研究实现或其他笔记。
