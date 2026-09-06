# RAEv2 guidance 原文深读：分类边界、配对质量与随机校正器

日期：2026-09-06。范围：独立检索后细读两篇原文的推导、实验和附录；不使用本仓库既往实验作为论文结论的依据。本次没有运行 GPU，也没有实现或启动新实验。

**判断先行。** 两篇提供了值得保留的机制，但都没有证明在 RAEv2 的公平总成本条件下超过最佳指导基线 5%。第一篇最有用的启发是“独立学习生成分布的纠偏”，然而最近邻配对本身不保留真实数据的质量分配；第二篇最有用的结论是“同一 guidance 场经过不同采样生成元，可以产生不同分布”。不能把后者改述为加噪必然改善质量。下面明确区分原文结果、独立推导和待证伪假设。

本项目约束始终是：同类别 Full/Base 两头，同一次 forward 得到输出；两头均为近似 flow 场，不得直接当作两个已知分布的精确 score；改动发生在采样过程内；总成本包括辅助训练、配对预处理及推理额外计算；不使用手工时间窗口、大量调参或生成后选图。

**论文一：Studying Classifier(-Free) Guidance From a Classifier-Centric Perspective。** Xiaoming Zhao、Alexander Schwing。公开版本初发于 2025 年，AAAI 2026 收录。核对了 arXiv v1/v2；深读依据为 v2 全文 §3–5、附录 A–D，并查看了 MNIST 的样本与 PCA 图。官方会议信息来自 [AAAI 页面](https://ojs.aaai.org/index.php/AAAI/article/view/38329)，推导及数值来自 [arXiv v2 全文](https://arxiv.org/html/2503.10638v2) 与 [PDF](https://arxiv.org/pdf/2503.10638v2)。AAAI PDF 直连下载失败，因此不声称逐页读过最终排版稿。

原文的实证主张是：低维 CG/CFG 轨迹被推离类别边界；模型生成样本与真实样本做近邻配对，再训练一个 rectified-flow 后处理，可以修复部分低质量样本。这是一篇包含额外训练的研究，满足“至少一种不同于 training-free 调 scale 的路线”。高维结果是间接机制证据，不是直接测量真实决策边界的因果实验。[原文 §3–4](https://arxiv.org/html/2503.10638v2#S3)

其核心建模对象可以写为

\[
\hat x\sim\mu,\quad y\sim K(\cdot\mid\hat x,\mathcal D_c),\quad
z_s=(1-s)\hat x+sy,
\]

其中 \(K\) 为同类真实数据中的最近邻或 top-\(k\) 随机邻居。对该配对训练 flow，目标是学习从生成样本到配对真实样本的移动。注意：按照以上 \(s:0\to1\) 约定，真实速度应是 \(y-\hat x\)。公开 v2 Eq.(9) 的目标写作相反符号，同时正文写 \(\dot z=v_\theta\)、从 \(\hat x\) 出发。复现时必须核对代码中的符号与时间方向，不能照抄这一组印刷公式。[原文 §3.4](https://arxiv.org/html/2503.10638v2#S3.SS4)

**独立审计：不能采用它对 Bayes 分解的强解释。**

论文质疑

\[
q(x_{t+1}\mid x_t,c)=q(x_{t+1}\mid x_t)
\]

是否成立，并以分别训练的条件模型、无条件模型与分类器的轨迹差异支持这个质疑。但若训练加噪就是独立于类别的 Gaussian Markov kernel，条件独立性来自构造。对于这一真实联合分布，直接有

\[
q(x_t\mid x_{t+1},c)
=q(x_t\mid x_{t+1})\,
\frac{q(c\mid x_t)}{q(c\mid x_{t+1})}.
\]

这不要求条件与无条件的 **reverse** 轨迹相同。问题发生在把各自训练的近似项拼接起来，或有限步 reverse kernel 的近似上；它们可以不再对应同一个联合分布。更进一步，即使两个采样器终端分布相同，共享初始噪声下的逐点轨迹也不必相同。以上是本笔记的独立数学审计，不能将论文的轨迹图当作 exact Bayes 恒等式失效的证据。[待审计推导：原文 §3.2、附录 A](https://arxiv.org/html/2503.10638v2#A1)

实验值得保留，但要按其支持范围阅读：

| 原文观察 | 数值或设置 | 能支持到哪里 |
|---|---|---|
| 相同无条件模型及初始噪声，更换线性/非线性分类器，轨迹不同 | 1D Gaussian，Fig.2 | 错误分类器能改变 guidance；不区分所有学习误差来源 |
| 后处理改善 CIFAR-10 EDM 的过强 CFG | scale 2.25：8.016→5.821；2.50：9.402→5.936；2.75：10.75→6.176 | 修复这三个坏基线；同文 scale 1.0 已有 FID 1.850 |
| 高维迁移与开销 | ImageNet 512² 结果不理想；正文称推理时间翻倍 | 尚无同成本 ImageNet 成功证据 |

数值及限制分别来自 [原文 Table 2、§4.3–5](https://arxiv.org/html/2503.10638v2#S4.SS3)。上述范围不能推出“远离类别边界必然降低 FID”。后处理也可能通过一般去噪、改变类别内部模式权重或记忆训练邻居得到收益。

**从原文抽出的新洞见 A：近邻纠偏的首要漏洞是目标质量，而不只是距离是否语义合理。**

这是独立推导。设真实经验分布为

\[
\nu=\frac1n\sum_{j=1}^{n}\delta_{y_j}.
\]

最近邻配对的实际目标边缘分布是

\[
\nu_{\rm NN}=\sum_j\mu(V_j)\delta_{y_j},
\]

\(V_j\) 是 \(y_j\) 的 Voronoi 区域。即便 flow 完美学习且积分精确，终端也只保证到达配对边缘 \(\nu_{\rm NN}\)，不保证到达 \(\nu\)。top-\(k\) 随机配对只是把权重改为

\[
w_j=\int\frac{\mathbf 1[j\in N_k(x)]}{k}\,\mu(dx),
\]

仍然通常不等于 \(1/n\)。因此低 NN 距离可以与严重模式集中共存；同类高置信度也不能消除这个问题。

可证伪预测：若 NN 后处理主要通过少数真实样本或子模式承接过多生成质量而改善表面距离，则配对目标的有效样本量 \(1/\sum_jw_j^2\) 和子模式覆盖应明显下降；改成固定行、列边缘的配对以后，覆盖应恢复，且下降并不必然伴随单对距离改善。若匹配后的有效样本量和覆盖本来健康，或保持目标边缘完全不能改善终端覆盖，则“配对质量错误是主要瓶颈”这一解释应放弃。

一个只需 CPU 的前置否证可以使用已有配对索引：直接统计 \(w_j\)、有效样本量和目标子模式占用，不需要 FID、不需要训练。如果连这些统计都不存在，则论文现有证据不足以把 NN 后处理列为成熟候选。

**对采样内设计的具体启发，仍属未验证提案。**

可以研究在每个噪声水平直接匹配 **当前采样轨迹边缘** \(\mu_t\) 与真实数据按训练桥加噪得到的 \(p_t\)，采用固定行列质量的 coupling；不从终端近邻构造理想未来轨迹。以运输位移为监督，训练一个读取同次 forward 隐状态的轻量纠偏头 \(r_\theta(h_F,h_B,z,t,c)\)，在外层采样中加入该向量。它改变方向，且学习的是分布纠偏，不限于放大 \(v_F-v_B\)。

理论来源仅限于如下冻结时间结果：若在一个事先固定的欧氏度量下，\(T_t\) 是 \(\mu_t\to p_t\) 的二次最优运输映射，且分布足够正则，则沿 \(T_t(z)-z\) 的微小流动，使 \(W_2^2(\mu_t,p_t)\) 的一阶导数为 \(-2W_2^2(\mu_t,p_t)\)。这给出独立方向监督，不涉及 exact score 或类别边界。

但真实训练只会得到有限 minibatch coupling 和近似头；RAE latent 的欧氏距离未必反映图像质量；加入反馈以后 \(\mu_t\) 还会变化。因此这条一阶结论 **不是** 全程收敛、终端 FID 或 5% 收益保证。反馈强度、采样次数与训练成本尚无闭合推导，不能现在就把它注册成无需调参的成熟方法。若需要大量更换特征距离或额外完整网络才能得到纠偏信号，应直接停止这条路线。

**论文二：Classifier-Free Guidance is a Predictor-Corrector。** Arwen Bradley、Preetum Nakkiran。arXiv 初发于 2024 年，作者机构页面列为 TMLR 2025。阅读的是 [arXiv v2 全文](https://arxiv.org/html/2408.09000v2) 与 [PDF](https://arxiv.org/pdf/2408.09000v2)：§2–5、Theorem 1–3、附录 A 的反例及解析解、附录 B、Algorithms 1–2，并查看 Fig.2。发表信息由 [Apple 官方页](https://machinelearning.apple.com/research/classifier-free-guidance) 核对；OpenReview PDF 返回 403，因此不声称该 arXiv 排版等于 TMLR 最终稿。

原文严格区分每一时刻的 powered score 与最终采样分布：Gaussian 加噪与密度幂乘一般不可交换。其主要等价关系是：连续极限下，随机 CFG-DDPM 等于条件 DDIM predictor 加一次 LD corrector，corrector 的指数必须为 \(2\gamma-1\)。它没有证明确定性 CFG-DDIM 与 PCG 等价。[原文 §3–4](https://arxiv.org/html/2408.09000v2#S4)

**核心推导的成立条件。**

采用原文 VP diffusion 和倒序去噪时间 \(dt<0\)，写 \(s_c=\nabla\log p_t(x\mid c)\)、\(s_u=\nabla\log p_t(x)\)：

\[
\begin{aligned}
dX_{\rm predictor}&=-\tfrac12\beta_t(X+s_c)dt,\\
dX_{\rm corrector}&=-\tfrac12\beta_t[(2\gamma-1)s_c+(2-2\gamma)s_u]dt
+\sqrt{\beta_t}\,d\bar W,\\
dX_{\rm total}&=-\tfrac12\beta_tXdt
-\beta_t[\gamma s_c+(1-\gamma)s_u]dt
+\sqrt{\beta_t}\,d\bar W.
\end{aligned}
\]

正确的噪声尺度与漂移系数一起产生等价关系。原文取一个 corrector、步长与 DDPM 噪声相匹配，再令步长趋零；有限大步多次复用旧输出，不等价于在每个新位置重新计算 Langevin 场。漂移代数本身对近似向量场仍能相加，但“其平稳分布就是 powered density”需要真实 score、适当尾部条件和混合性，不能一并移植到近似模型。[Theorem 3 及附录 B](https://arxiv.org/html/2408.09000v2#S4.SS2)

原文还有一个容易误读的界限：PCG 不是保证精确采样 \(p_{0,\gamma}\) 的算法。有限次数 Langevin 没有充分混合；predictor 和 corrector 依循的噪声边缘又不同。把等价分解写成“CFG 在采 powered density”正好撤销了论文自己的反例。

其主要实验支持如下：

| 证据 | 能确认的结论 | 对本任务的缺口 |
|---|---|---|
| 同均值 Gaussian、精确 score，解析求 ODE/SDE 最终方差 | 分布差异不是学习误差或有限步的唯一产物 | 非 RAEv2，但构成严格反例 |
| 双 Gaussian mixture、精确 denoiser | CFG 可改变均值和形状，不是单纯调温 | 无自然图像性能保证 |
| SDXL：200 步 CFG 对 100 predictor + 单次 corrector；另有 1000 predictor steps 的 \(K\) 扫描 | 定性展示理论参数对应及 corrector 效果 | 未给出公平总成本 FID 提升证据 |

来源：[原文 §3、§5.3](https://arxiv.org/html/2408.09000v2#S5.SS3)。作者也明确将 PCG 定位为理解工具。

对解析反例，可以独立复算：当 \(p_0(x)=\mathcal N(0,2)\)、\(p_0(x\mid c)=\mathcal N(0,1)\)，大初始噪声极限下，三种方差分别为

\[
\operatorname{Var}_{\rm powered}=\frac{2}{\gamma+1},\quad
\operatorname{Var}_{\rm SDE}=\frac{2-2^{2-2\gamma}}{2\gamma-1},\quad
\operatorname{Var}_{\rm ODE}=2^{1-\gamma}.
\]

\(\gamma=7\) 时约为 \(0.25,0.153827,0.015625\)。这表明随机性可以大幅改变收缩程度，但它没有告诉我们哪一个更接近所需真实分布。公开 v2 §3.1 的文字把大 \(\gamma\) 时 SDE 方差相对于 powered 方差说成约两倍；按其公式复算应为约一半，本笔记采用公式而非该处文字。

**RAEv2 的桥接必须先写清楚。**

以下是独立推导，采用噪声到数据方向 \(t:0\to1\) 的直线桥：

\[
Z_t=(1-t)\varepsilon+tX_1,\qquad
v^*(z,t)=\mathbb E[X_1-\varepsilon\mid Z_t=z].
\]

若 \(\varepsilon\sim\mathcal N(0,I)\) 且独立于数据，那么在 \(0<t<1\)

\[
s^*(z,t)=\frac{t v^*(z,t)-z}{1-t}.
\]

故速度误差 \(e_v\) 在这个代数 score 代理中变成 \(t e_v/(1-t)\)。在接近数据端时，它可能非常大。Full/Base 即便采用这个变换，也只是两个 score 代理；不能推出它们恰好是 Full/Base 实际轨迹边缘的 score。还有：同类别两头差异不具备通常条件/无条件 Bayes 分类器的含义。

**从原文抽出的新洞见 B：错误随机校正器的机制量是加权散度；curl 既不充分也不必要。**

设正在使用的 guided ODE 为 \(\dot Z=b_t(Z)\)，其真实采样边缘记作 \(q_t\)。尝试加入由 Full 变换得到的 score 代理 \(\hat s_t\)：

\[
dZ=\left[b_t(Z)+\frac{a(t)}2\hat s_t(Z)\right]dt+\sqrt{a(t)}\,dW_t,
\]

这里 \(a(t)\ge0\) 仅依赖时间。把原 \(q_t\) 代入新的 Fokker–Planck 方程，新增项精确等于

\[
R_t=\frac{a(t)}2\nabla\cdot\left[q_t\left(\nabla\log q_t-\hat s_t\right)\right].
\]

因此 \(\hat s_t=\nabla\log q_t\) 是保持原轨迹边缘的充分条件；更一般的条件是上式加权散度为零。去掉普通 curl 既不能保证 \(R_t=0\)，也不是 \(R_t=0\) 的必要条件。例：\(q=\mathcal N(0,1)\)、\(\hat s(x)=-(x-m)\) 完全是保守场，但 corrector 的瞬时均值变化是 \(am/2\)，会系统性平移分布。相反，某些相对于 \(q\) 无散度的环流可以不改变其边缘。

这一条件不要求我们知道 exact score 才能证伪。对光滑、边界项消失的预注册测试函数 \(\phi\)，新 corrector 对期望的即时贡献为

\[
\frac{a(t)}2\,
\mathbb E_{q_t}\left[\Delta\phi(Z)+\hat s_t(Z)\cdot\nabla\phi(Z)\right].
\]

右端只需要实际采样轨迹样本、Full 输出，以及测试函数的一二阶导数。在保存的 latent 上，线性和二次测试函数可直接计算，不必调用 GPU 或估计 \(\nabla\log q_t\)。这不是检查“向量场是否近似保守”，而是在检查它会把 **正在采的分布** 推向哪里。

可证伪预测：在保持漂移–扩散配比的微小随机干预下，上述弱生成元量应预测 held-out 轨迹均值、方差等量的改变方向；只有这些改变指向实测缺失的覆盖时，才有理由期待收益。若减小干预强度后预测仍系统性失败，应优先否定 score 转换、时间方向或状态分布匹配，而不是扩大 \(a(t)\) 的搜索。即便测试通过，它只验证随机校正的机制，不足以预言 FID。

这给采样设计的实质约束是：若未来学习一个小型随机 corrector，应该联合约束漂移和扩散的分布作用，并在 **实际 guided 轨迹分布** 上验证；仅在真实加噪数据上校准误差后直接上线，仍可能因为分布错位失败。若令 \(a=a(z,t)\) 随样本变化，保持密度的漂移还需要 \(\tfrac12\nabla a\)；不能把上面时间标量公式直接变成样本自适应加噪公式。当前没有充分理由为这篇论文单独启动随机性规模扫描。

**两篇合起来，对下一步的约束。**

1. 一个额外 corrector 的“目标密度”必须对应真实配对边缘或正确生成元；理想 score 解释、类别置信度以及更小 NN 距离不能替代这个检查。
2. 第一篇的训练式启发可以发展成当前时间边缘上的质量平衡反馈，但其度量、反馈后的分布变化及额外训练成本尚待解决；这不是终端 PFR、固定点、欧氏方向投影或普通 curl 消除的再包装。
3. 第二篇提供了无需 exact score 的弱生成元否证工具。它的价值先是避免错误解释；通过该检查不是增加随机校正预算的充分依据。
4. 只有预注册的单一实现通过机制检查后，才讨论以固定样本数、相同随机种子、实测总计算成本，对比允许利用节省成本增加步数或训练的强基线。最终门槛仍是 held-out FID 相对下降至少 5%，并检查覆盖退化；本次阅读没有产生满足该门槛的新证据。

本文新增的是两个明确可否证的机制判断与一个条件化训练方向，不宣称得到已经验证的 RAEv2 guidance 方法。
