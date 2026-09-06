# ERG 精读：检索熵有精确机制，guidance 的误差方向仍需证据

记录日期：2026-09-06。本文只做论文与源码审查、固定小型 CPU 代数核验；没有模型前向、GPU、采样、训练、参数选择或对既有冻结实验的修改。

**结论：ERG 值得借鉴的是“可明确改变哪一种信息”，不能直接借来“改变该信息必然减少生成误差”。** 缩放 logits 对逐行 attention 熵的作用是严格的；固定记忆、兼容 K/V 的 Hopfield 能量下降也有成立的特例，不应因实际模型使用独立投影而全部否定。论文在若干模型上确有超过 5% 的 FID 改善，但参数、层区间和启动时刻来自实验选择。它尚未为 RAEv2 同类 Base/Full 推出免扫参系数或公平成本下的质量保证。

## 1. 来源与作者实际提出的机制

精读对象是 Berrada 等的 *Entropy Rectifying Guidance for Diffusion and Flow Models*，NeurIPS 2025 [正式正文](https://papers.nips.cc/paper_files/paper/2025/file/414f4c9fe9653e5de98fad6964d50315-Paper-Conference.pdf) 20 页，以及[正式补充材料](https://papers.neurips.cc/paper_files/paper/2025/file/414f4c9fe9653e5de98fad6964d50315-Supplemental-Conference.zip)中的 PDF 20 页（印刷页码 21–40）。正文方法/实验、补充 A–G 的理论、实现和实验说明均已阅读；[arXiv 2504.13987](https://arxiv.org/abs/2504.13987)及作者页面仅用于版本和代码出处交叉核对。正式 PDF 优先于有重复段落的 arXiv v2 HTML。

方法在负分支的 attention 中改变检索行为，保持正分支原样：I-ERG 修改图像网络，C-ERG 修改同一提示词的文本编码。正文 Eq.2 的最终预测是 (wD+(1-w)D_{\rm weak})。图像分支采用启动阈值 \(\kappa\)，文本弱 embedding 则全程使用。主算法可写为

\[
Q_{r+1}=(1-\gamma)Q_r+\gamma\alpha\,
  \operatorname{softmax}(\tau\beta Q_rK^\top)V,
\qquad \beta=d^{-1/2}.
\]

这里内循环步数记作 \(m\)，避免与 key 矩阵 \(K\) 混淆；论文将步数也记为 K。原网络对应 \(\alpha=\gamma=\tau=m=1\)。这给出了结构明确的负分支：减弱 query 对不同 key 的选择，或改变检索迭代，而非另训一个任意差模型。作者的正向解释是：同条件弱分支保留部分语义共同项，差分集中于被弱化的关联和细节。但“这一差分纠正了正分支的真实误差”不是上式的代数后果。

## 2. 哪部分 Hopfield 保证成立

**Q 独立于 K 并不破坏固定记忆的 Hopfield 推导。** Q 是可任意初始化的检索状态；关键在于更新返回的 values 是否为同坐标下的 keys，且一次内循环中 K 固定。

令每个 key 为列向量 \(k_j\)，\(K\) 按行存 key，\(c=\tau\beta>0\)、\(\alpha\ge0\)。取与 softmax 公式一致的标准 LSE 定义，有

\[
E(q)=\tfrac12\|q\|^2-\frac{\alpha}{c}\log\sum_j e^{c k_j^\top q},
\quad p=\operatorname{softmax}(cKq),\quad
g=\nabla E(q)=q-\alpha K^\top p.
\]

利用负 LSE 的凹性，直接得到二次上界

\[
E(q+u)\le E(q)+g^\top u+\tfrac12\|u\|^2.
\]

因此 \(q'=q-\gamma g\) 满足

\[
E(q')\le E(q)-\gamma(1-\gamma/2)\|g\|^2,
\qquad 0<\gamma<2.
\]

这是本次独立补出的下降界；\(\gamma=1\) 正是 CCCP 的二次上界最小化步。它也保留了 \(\gamma=1.5\) 在该特例中的下降性质，不能仅因大于 1 就判无效。但下降不等于找到正确记忆或全局最小，更不等于图像/FID 改善。正文 §2.3 的“收敛到全局最小”表述过强：一维 keys \(\{-2,2\}\)、\(\alpha=c=1\)、\(q=0\) 是不动点，能量为 \(-0.693147\)，而 \(q=2\) 的能量为 \(-2.000335\)。

对于一般 values，作者算法使用的场变成

\[
g_V(q)=q-\alpha V^\top p,\qquad
Dg_V=I-\alpha cV^\top(\operatorname{diag}p-pp^\top)K.
\]

该 Jacobian 通常不对称，因而一般不是欧氏坐标中某个标量势的梯度。最小例子为 \(K=(e_1,-e_1)^\top\)、\(V=(e_2,-e_2)^\top\)：\(g_V(x,y)=(x,y-\tanh x)\)，交叉偏导不等。正文“attention 等于 Hopfield 更新再乘线性变换”的说法，可以解释一次输出的关系，但不能省掉该坐标变换，随后直接把 \(pV\) 当作下一轮 query 而仍沿用原能量证明。

有效情形不止严格 \(V=K\)：若存在明确的对称正定 \(M\)，满足 \(V=KM\)，则可用
\(E_M(q)=\frac12q^\top M^{-1}q-\alpha c^{-1}\log\sum_j e^{c k_j^\top q}\)，此时 \(g_V=M\nabla E_M\)，对应 \(M^{-1}\) 度量也有上述下降界。**本地 RAEv2 不自动具有这个条件。** [NormAttention](../external/RAEv2/src/stage2/models/model_utils.py) 分别学习 Q/K/V，Q/K 经过 RMSNorm 与 RoPE，而 V 没有同样变换；后面还有输出投影和块残差。没有证据把其一般 attention 更新视为上述势的下降，更不能把内层检索能量等同于整个 latent 轨迹的能量。

正文印刷的 LSE 定义漏写指数中的温度因子；以上使用与其明确给出的 softmax 更新一致的定义，而不以这个排印问题否定成立的数学特例。

## 3. 独立 Q/K/V 下仍然成立的熵机制

固定一行 logits \(\ell_j=\beta k_j^\top q\)，令 \(p_\tau=\operatorname{softmax}(\tau\ell)\)。无论 V 如何学习，都有

\[
\frac{dH(p_\tau)}{d\tau}=-\tau\operatorname{Var}_{p_\tau}(\ell)\le0,
\qquad
p_\tau=\arg\max_{p\in\Delta}\left\{\langle p,\ell\rangle+\frac{H(p)}{\tau}\right\}.
\]

因此 \(\tau\) 实际是逆温度乘数：小 \(\tau\) 提高检索熵，大 \(\tau\) 降低熵。这是“机制→可控结构”的真实连接。对任意 values 还有

\[
\frac{d\,(p_\tau V)}{d\tau}=\operatorname{Cov}_{p_\tau}(V,\ell).
\]

它把扰动方向具体化为：加强与高 logit 关联的 value、减弱低 logit 关联的 value。这个量比“弱预测更差”更明确，但不自带真实误差的方向。检索熵也不是 denoising 不确定性、分类校准误差或最终样本多样性。

\(\tau\to0\) 是所有 **keys** 等权，输出趋向 mean(V)。它与 SEG 的全局 query 均值 \(JQ\) 不同：后者让不同 **queries** 共用一行通常非均匀的检索分布。两者都删信息，但删去的结构不同；不能把 ERG 的极限当作已验证 JQ 机制的等价替代。单调熵公式只约束固定 logits 的一次修改，也不保证完整多层网络每层的熵都同方向变化。

补充 C 的两种解释也要放在准确边界内。若强弱分支真的是同一时刻的精确保守 score，且归一化常数有限，则 \(w s_F+(1-w)s_W\) 是 \(p_{F,t}^{w}p_{W,t}^{1-w}\) 的 score。这与 KL 正则的指数倾斜变分原理一致；它没有证明所选密度比 reward 有利于质量，也没有证明整条采样 ODE 的终点必然等于逐时刻写出的倾斜密度。C-ERG 的一阶式 \(s(c+\Delta c)\approx s(c)+J_cs\Delta c\) 则解释了指定语义扰动方向的响应，不能据此称它是“最敏感方向”，更不能把大温度扰动当作小扰动定理。原附录 Eq.7–10 的 \(\rho,p_\tau\) 与 reward 分母有记号不一致，因此这里使用重新写清的标准恒等式；附录 Eq.11 的 w 也比正文 Eq.2 的 w 少 1。

## 4. 作者给出的误差解释及实验支撑

补充 D.1 的正向解释是：空条件负分支丢掉过多语义，使初始差分方差偏大；C-ERG 保留同一提示词的一部分结构，可能减少初期过冲。作者用 20 个噪声输入考察弱预测及强弱差分的坐标方差，观察到这种区别。这是一个可检查的解释入口，但这些方差不是真实 denoising 误差协方差；初始速度的方差也不能单独推出终点分布多样性。

补充 D.2 观察到普通 guidance 的某些差分分量后期减弱，ERG 的正交分量仍持续，解释为保留了细节修正信号。D.4 则观察到中间层 attention 更分散，早晚层较多近似确定性检索，因而选择干预中间层。这些结果给“为什么这种弱化可能有用”提供了结构线索；正交信号非零、或某层熵较大，仍不等于误差修正方向正确。RAEv2 的 Base/Full 已经同类，故 C-ERG 相比空条件的优势不能原样搬过来。

以下均为论文自报结果，未复现实验；不能只引用主表中 ERG 的 FID 不佳，也不能只引用有利配置。正文配置通常以多指标平均排名选参，补充另有明确的 FID 单目标选参。来源为[正文 Tables 1–3](https://papers.nips.cc/paper_files/paper/2025/file/414f4c9fe9653e5de98fad6964d50315-Paper-Conference.pdf)与[补充 Tables 8、11–13](https://papers.neurips.cc/paper_files/paper/2025/file/414f4c9fe9653e5de98fad6964d50315-Supplemental-Conference.zip)。

| 场景 | 对照 FID → ERG FID | 该结果支持什么 |
|---|---:|---|
| 主表 COCO，40K，多指标选参 | CFG 12.81 → 13.62 | FID 变差，但 density/coverage/文本一致性提高；ERG+APG 为 11.37 |
| 主表无条件 | 无 guidance 101.50 → 36.25 | 有明显正收益，基线一次前向与 guidance 两次前向不能称同成本 |
| 主表类条件 256 / 512，50K | 3.67 → 3.67 / 5.65 → 4.56 | 512 有约 19.3% FID 改善，256 持平；512 coverage 略降 |
| 补充 FID 单独调优 | CFG 5.25 → 4.93 | 约 6.1% 改善；这是重新选参的结果 |
| 补充 EDM2-XXL，512 | EDM2 2.00 → 1.35 | 约 32.5% 改善；Autoguidance 仍为 1.33 |
| 补充官方 DiT-XL/2 重跑，10 seeds | 2.38±0.05 → 2.15±0.03 | 约 9.7% 改善；这是正面的重复种子证据，非本地 RAEv2 结果 |

所以论文不是“完全没有 FID 支持”，也没有建立跨架构、无选择自由度的 SOTA 保证。

## 5. 哪些选择来自实验，实际实现和成本是什么

理论给出 \(\beta=1/\sqrt d\) 的标准 attention 尺度、熵单调性及特例中的下降区间；**不决定最佳 \(\tau,\alpha,\gamma,m,w\)、层区间或 \(\kappa\)**。补充 A.2 按候选集合中的平均指标排名选参，B 明说不同架构需要调节，D.4 搜索层区间，D.5 搜索启动时刻。主文 Table5 比较多组内循环步数/步长，没有发现稳定的大收益，通常取 \(m=\gamma=1\)，但 T2I 表列 \(\gamma=1.5\) 是明确例外。

补充 Table7 的部分实际设置是：类条件 DiT 用图像层 12:15、\(\kappa=0.3,\tau_i=0.01,\alpha=\gamma=m=1\)；T2I 用图像层 10:17、\(\kappa=0.4,\tau_i=0.01,\gamma=1.5\)，另改文本编码器多层。12:15 在文中对应三层。表的 cutoff 列印为 \(\tau\)，结合上下文才可判断它指 \(\kappa\)；最后两行重复写 SD3-medium，不能擅自改第二行为 SD3.5。FID 最优的附录另外写 guidance scale=1.25、ERG scale=2.0，未清楚指定后者对应哪个算子参数，不能擅自等同 \(\tau\)。完整转录见归档 CSV。

在所查的正式 ZIP、arXiv 链接、第一作者网站和公开仓库列表中，**未找到独立可执行的官方 ERG 仓库**；ZIP 只有补充 PDF 和 macOS 元数据。补充 E 给的是伪代码，而且有两点实质边界：

1. 它使用独立的 Q/K/V 投影、Q/K norm 与 RoPE，随后只缩放图像 Q；没有 K/V 对齐或势梯度计算。
2. 它先将 Q 乘 \(\tau\) 一次，再将这个 Q 放入迭代；正文算法则每次在 logits 内乘 \(\tau\)，保留项用未预缩放的初始 Q。二者在 \(m=1,\gamma=1\) 的图像输出相同，一般 \(\gamma\ne1\) 或 \(m>1\) 时不同。还有调用参数 `lr` 与函数签名 `step_size` 不匹配，不能称为直接可运行的发布代码。

这不否定其单次温度 attention 的可实现性，但使一般步长/多次更新结果的精确复现身份不完整。详见[独立源码审查](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_erg_v1/code_review/implementation_review.md>)。

主表 ERG 与 CFG 都记为每步 2 次 denoiser NFE，常用 \(m=1\) 时 attention 的渐近计算量不变，因此作者“相对普通两分支 CFG 额外开销小”的主张在算子数量上有依据。论文没有实测延迟表；多次内循环增加 attention 工作，C-ERG 还有可按提示词预计算的弱文本编码成本。这个 NFE 口径不能直接等同于 RAEv2 共享编码器的 Base/Full 成本：若扰动共享编码器中层，就必须从该处重放后续计算；若只动两层 decoder，则是不同于论文已选中间层的实例，质量依据尚缺。调参成本也未作完整同预算核算。

## 6. 对当前研究的可迁移问题

可保留一个比“制造坏预测”具体、也容易证伪的假设：**真实 denoising 误差的一部分来自检索选择性不足，且误差与增强 logit 对比产生的网络响应同向。** 在冻结输入/层/上下文的一阶邻域，若负分支采用 \(\tau_w<1\)，那么
\(D(1)-D(\tau_w)\approx(1-\tau_w)\partial_\tau D(1)\)；局部 attention 导数就是上面的 value–logit 协方差，经后续网络传到输出。若真实缺口与此响应反向，外推就会放大错误；若模型已过度自信，这种解释甚至预测反效果。

它尚缺两项关键证据：输出层的真实误差投影，而非 attention 熵或强弱差分大小；以及真实缺口幅度如何由可验证结构确定。特别地，未知的理想 \(\tau_*\) 不会由“熵单调”自动给出，也推不出自然的 w。RAEv2 同类 Base/Full 的差异还包含深度与读出差异，不能预设全都沿这一温度方向。

因此此处只保留机制问题，不启动 \(\tau\)、窗口、层范围或 guidance scale 扫描，也不把 \(\tau=0\) 极限包装成已获准的新方法。若已有固定误差诊断不能支持所需符号/结构，这条解释应被否定；即使局部符号成立，也还需要有限步稳定性和公平成本下的独立质量验证。

归档位于 [reading_erg_v1](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_erg_v1>)，含正式 PDF、全文文本、作者来源、伪代码审查、设置转录、manifest 与 SHA256SUMS。固定小型[代数脚本](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_erg_v1/algebra_check.py>)及[结果](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_erg_v1/algebra_results.json>)核对了梯度、下降界、非保守反例和熵/检索导数；全部通过。实测脚本段 wall 0.005370 秒、进程 CPU 0.005360 秒，排除 Python 启动及结果写盘；正式正文下载与转文本 wall 3.026096 秒，进程 CPU 0.098583 秒（不含 pdftotext 子进程 CPU）。其余阅读/浏览成本没有统一计时，不以这些局部计时冒充总研究成本。模型/GPU 调用均为 0。
