# 从 Self-Guidance 到受约束的弱分布构造

{{TOPLINE}}

这项研究的核心问题是：能否从一个已训练模型构造参考分布，使外推主要修正模型误差，同时保留已经学对的分布结构。把强分布与其高斯卷积作密度比，在加性噪声坐标下确实落入 SG 的跨噪声对比；单独改变叙事不足以构成方法贡献。可以继续推进的空间在于参考算子的约束、它与真实误差的方向关系，以及它在实际生成轨迹上的效果。

**SG 各版实际改了什么。** arXiv 共列出五版。下表依据逐版 PDF 与正文核对；v3、v4 的 HTML 下载内容相同，但 PDF 的表格数值确有变化，因此对这两版以 PDF 为准。[^1]

|版本|日期|可确认的变化|对研究立论的含义|
|---|---|---|---|
|v1|2024-12-08|§4.1 明确由 CFG 的密度比引出方法；更噪时刻提供更平滑、较平坦的参考分布。AG 已在引言和相关工作中作为需要额外弱模型的对照。|“强分布与更平滑参考分布作外推”已在首版动机中。|
|v2|2025-03-08|把伪影处从高噪声到低噪声的密度下降置于中心位置，强化双峰与双螺旋例子；密度比和 score 差分的核心公式延续。|叙事确有调整，但不足以推断调整的外部原因。|
|v3|2025-05-25|加入 SG-prev，复用前一采样步的输出，并讨论后半段激活以避免额外前向。|这是算法和效率方面的实质扩展，不能全部归为措辞变化。|
|v4|2025-07-03|主要论证沿用 v3；PDF Table 1 中 Flux 的部分 HPS 值更新。例如 CFG 的平均 HPS 从 31.33 改为 31.43，CFG+SG 从 31.43 改为 31.51。|应区分论文结果修订与新算法。|
|v5|2025-09-26|改为期刊版式，增加 SG/SG-prev 双峰比较图及更完整局限讨论，核心差分结构不变。|仍然沿用跨噪声参考与历史近似这两个主体。|

公开检索没有取得能确认“拒稿后改叙事”的会议决定或评审记录。OpenReview 搜索返回了同名论文的 CoRR、TPAMI 导入条目；这些是文献条目，不能当成评审证据，精确标题接口另有访问限制。能确认的后续事实是论文发表于 *IEEE TPAMI*，2026 年 1 月，48(1):781–791，DOI 10.1109/TPAMI.2025.3611831。由 AG 的思想联想到 SG 是合理的研究理解；把作者的改稿动机写成确定历史事实则缺乏依据。[^2]

官方实现已经存在，位于 MAPLE 的 Self-Guidance 仓库。此次沿用已核对的提交 `843bda799bb531ccf8d98164d8ccf1a3b8ce3f6d` 作为原 SG 对照。其 flow 管线直接对不同噪声时刻的模型速度作差；SG-prev 缓存加入 CFG/SG 之前的条件预测。因而必须区分论文的理想 score 密度比解释与代码中的跨时刻速度规则。[^3]

**为什么高斯卷积叙事会与 SG 重合。** 记

\[
H_\tau p=p*\mathcal N(0,\tau I),\qquad s=\nabla\log p.
\]

在加性噪声路径 \(p_u=p_0*\mathcal N(0,uI)\) 上，半群性质直接给出 \(H_\tau p_u=p_{u+\tau}\)。因此

\[
s+\omega\{s-\nabla\log H_\tau p\}
=\nabla\log\!\left[p\left(\frac p{H_\tau p}\right)^\omega\right].
\]

这正是该坐标系中的跨噪声密度对比。对于一般线性路径 \(z_t=\alpha_t x_0+\sigma_t\epsilon\)，要先换到 \(y=z_t/\alpha_t\)、\(u=(\sigma_t/\alpha_t)^2\)，再比较同一个 \(y\) 上的密度；\(\alpha_t=0\) 的端点要单独取极限。直接在相同 \(z\) 上减两个时刻的速度，通常不等于上述准确的卷积比，不能省略输入重标度与 score 系数。

即便 score 完全准确，密度卷积外推也会改变高斯内部方差。若 \(p=\mathcal N(\mu,\Sigma)\)，则

\[
g_{\rm density}(x)
=-\left[\Sigma^{-1}-(\Sigma+\tau I)^{-1}\right](x-\mu).
\]

这个项无条件向均值收缩；它不判断原来的方差是否已经正确。更高密度也不等于更好的感知质量，Density Guidance 对极高密度处细节减少的观察正好提醒了这一点。[^4] 另外，每个时刻的幂密度比可归一化，并不代表一整条修改后的扩散路径具有这些边缘分布，更不能据此直接宣称最终分布就是 \(p_0^{1+\omega}/q_0^\omega\)。[^5]

**先复现的方案：在同一时刻平滑 score，再反向外推。** 令

\[
q_{\log}(x)\propto\exp\{H_\tau\log p(x)\},\qquad
\nabla\log q_{\log}(x)=\mathbb E_\xi s(x+\xi),\quad
\xi\sim\mathcal N(0,\tau I).
\]

这里平滑的是对数密度，不是密度本身。score 平滑、它的插值作用和几何平均密度解释已有文献；MM-SOLD 还使用了正负扰动及矩约束，所以这些部件本身不能重新宣称为首次提出。此次试验考察的是把这个已知算子作为参考项进行反向外推，而不是复现 MM-SOLD 的相互作用粒子采样器。[^6][^7]

一个正负扰动对给出的修正为

\[
\widehat g_{\log}(x)=s(x)-\tfrac12[s(x+\xi)+s(x-\xi)],\qquad
s_{\rm new}=s+\omega\widehat g_{\log}.
\]

若 \(s(x)=Ax+b\)，每个扰动对都严格抵消，不需要无限次 Monte Carlo。这覆盖任意协方差的高斯 score，甚至覆盖不对称矩阵 \(A\) 的一般仿射场。SiT 的同一时刻速度满足 \(v_t(z)=z/t+(1-t)s_t(z)/t\)，所以仿射部分同样抵消，可以直接使用速度残差；无需对不同时间的输出做近似转换。对于真实神经网络，密度解释依赖其近似 score 的程度，但这个仿射抵消的代数事实独立成立。

小尺度展开揭示了它与密度卷积外推的区别。在足够光滑的准确 score 下，

\[
g_{\rm density}=-\frac\tau2(\Delta s+2J_s s)+O(\tau^2),\qquad
g_{\log}=-\frac\tau2\Delta s+O(\tau^2).
\]

后者避开了会作用于高斯线性 score 的 \(J_s s\) 项。但“高斯不变”也带来盲点：如果模型只把一个正确高斯的方差放大了，它完全不能修正这个错误；如果模型已产生尖锐假峰或记忆化，锐化又可能强化它们。Selective Underfitting 与 score 平滑的泛化研究说明，不能把所有平滑都视为应当消除的错误。[^7][^8]

**解析与受控实验。** 对 \(p=\frac12\mathcal N(-2,0.25)+\frac12\mathcal N(2,0.25)\)、\(\tau=0.09\)，独立数值积分复核了所给分析中的三点：

|位置|密度平滑差分|score 平滑差分|
|---:|---:|---:|
|0.25|2.155223|3.191193|
|1.75|0.264706|\(9.2864\times10^{-7}\)|
|2.25|−0.264706|\(3.7156\times10^{-10}\)|

这支持它在远离模式边界时较少改变局部高斯结构，但不是“每个混合分量方差严格不变”的证明。非对角二维高斯的正负残差约为 \(1.4\times10^{-14}\)；矩匹配算子的协方差误差约为 \(1.7\times10^{-16}\)。这些是公式实现检查，不是图像质量证据。

下表的风险是 \(\mathbb E_{p_*}\|s_{\rm method}-s_*\|^2\)，在已知真分布的一维网格上积分，修正强度固定为 1。密度平滑及 log 平滑用 \(\tau=0.09\)；带限方案使用 \(H_{0.09}s-H_{0.18}s\)。

|人为构造的模型错误|原模型|密度卷积外推|log 平滑外推|带限外推|
|---|---:|---:|---:|---:|
|密度过度平滑|0.56407|0.27085|0.54719|0.53079|
|score 过度平滑|0.01638|0.26261|0.00797|0.01189|
|score 平滑 + 高频波纹误差|0.04763|0.38761|0.13297|0.04314|
|额外的中心假峰|0.36147|0.99987|0.76335|0.49352|
|高斯方差膨胀|0.11111|0.08738|0.11111|0.11111|

结论是误差类型决定有效方向：log 外推适合特定的 score 平滑误差，但不能通吃密度模糊、方差错误和假峰。带限方案在“平滑加波纹”的受控例子里优于直接锐化，但在假峰例子上仍退步。以下曲线是**固定时刻归一化后的势函数密度**，不把它们当作实际扩散终点分布。

![受控误差与不同参考算子的瞬时势函数密度](data/weak_reference_20260914/toy/potentials.png)

另一个实验实际积分了 400 步线性路径 ODE，使用 4,096 个标准高斯分位点作为初始粒子。对完全准确的双峰模型，基线到真分布的分位数 W2 为 0.00367，log 外推为 0.01125、密度外推为 0.12886；准确模型本来就不需要修正。把分量方差从真实的 0.25 错设成 0.40 后，基线 W2 为 0.13155，log 外推为 0.12271，矩匹配外推为 0.11724。该试验显示某些误差可被改善，也同时否定了“峰方向正确就一定得到正确分布”的推断。

圆环反例进一步区分了伪影抑制和泛化。真分布是半径 2 的连续圆环加标准差 0.12 的高斯噪声，模型只在圆环上的 12 个点放置同宽高斯。log 外推后，落入这些点的半径 0.15 邻域的质量从 53.27% 增至 61.46%，而真分布仅为 18.35%；真分布到瞬时势函数的 KL 从 1.899 增至 4.022。旋转锐化也加重这个例子的模式间缺口，因此不能仅凭几何不变性保证泛化。

![连续圆环与稀疏记忆化反例](data/weak_reference_20260914/toy/curved.png)

**候选方法一：限制反平滑的频带。** 最直接的 log 外推对 score 函数的高频误差也会放大。可以改为

\[
s_{\rm band}=s+\omega(H_\rho-H_{\rho+\tau})s.
\]

在 score 作为状态 \(x\) 函数的傅里叶域中，增益是

\[
1+\omega e^{-\rho\|k\|^2/2}(1-e^{-\tau\|k\|^2/2}).
\]

它保留仿射不变性，把增强集中在中间尺度，对最高频率的额外增益趋于零。这里的频率是 **score 对整个状态空间的变化频率**，不是直接对生成图像做像素高通滤波。两个尺度的正负配对需要四次参考查询；尺度间共用随机方向以降低差值方差。它本质上使用经典平滑算子，候选贡献应是对可修正误差的频带约束及其验证，而不是“首次使用双尺度”。

这个频带结论针对扰动期望。每步仅抽一个方向时，高频分量的随机估计方差未必随频率消失，共用方向也不能消除这项误差。真实 SiT 补充试验采用扰动标准差 \(0.2(1-t)\) 和 \(0.4(1-t)\)，全程固定强度 1，使用 320 次前向并与 Euler320 比较；它与前述 toy 的两个固定方差不同，也与 DSM 拟合出来的权重分开记录。

更进一步，可采用 \(s+\omega(s-H_{\tau_2}s)-\lambda(s-H_{\tau_1}s)\)，其中 \(\tau_2>\tau_1\)。当 \(\omega\leq\lambda<\omega\tau_2/\tau_1\) 时，可以同时实现低频附近的适度反平滑和最高频率的不放大；还需 \(1+\omega-\lambda\geq0\) 防止高频符号翻转。这是待扩展的正则化反平滑族，不能据此预告真实图像收益。

**候选方法二：保持高斯几何的旋转参考。** 在高维，同一时刻加入 \(\kappa\sigma_t\epsilon\) 会增加预期平方半径约 \(\kappa^2\sigma_t^2d\)。这是相对于局部高斯近似的分析，真实生成状态不必服从单个高斯。即使每个坐标的扰动很小，查询也可能离开模型受过充分监督的区域；对 SiT 的 4,096 维状态和 RAEv2 的 262,144 维状态，影响不应仅按相同 \(\kappa\) 就认为完全一致。

令 \(R\) 正交，\(T_R=\Sigma^{1/2}R\Sigma^{-1/2}\)，构造

\[
\log q_{\rm rot}(x)=\mathbb E_R\log p(\mu+T_R(x-\mu))-\log Z,
\qquad
s_q(x)=\mathbb E_R T_R^\top s(\mu+T_R(x-\mu)).
\]

每次查询保持匹配高斯的 Mahalanobis 半径，并且 \(p=\mathcal N(\mu,\Sigma)\) 时逐查询不变。输出乘 \(T_R^\top\) 是链式法则要求，省掉会变成另一种算法。采用 \(J^\top=-J,J^2=-I\) 和 \(R_\pm=\cos\theta I\pm\sin\theta J\)，可以用随机配对坐标实现低成本小角度旋转。

查询点留在同一椭球面，不等于一般非高斯分布的径向边缘保持不变；对数密度平均及重新归一化仍会改变它。围绕全局类别均值旋转也不保证保留每个局部模式的几何，因而它只是可检查的全局约束。

SiT 原型用既有训练缓存估计的类别均值 \(\mu_0\)，在 \(\mu_t=t\mu_0\) 周围旋转，角度为 \(0.1(1-t)\) 或 \(0.2(1-t)\)。该原型使用球形协方差，**没有声称保留真实图像分布的完整协方差**；一般协方差版本的白化公式只完成了解析检查。速度实现包含 \((\cos\theta-1)\mu_0\) 的均值补偿项，使 \(t=0\) 也有正确极限。范数、逆变换、非高斯势函数的 autograd 链式法则和高斯不变性均已检查。

这与 DSG、DiffRGD 针对外部损失引导步的球面约束有联系，但它们并未因此等同于这里的对数密度旋转参考；Warped Diffusion 的变换一致性引导也属于需明确比较的近邻工作。这里提出的是一个可检验的参考算子原型，尚未建立全面的新颖性结论。[^9][^10][^11]

更直接的近邻是 Tangential Amplifying Guidance：它分解原采样增量的径向与切向部分，再放大切向部分，不增加模型查询；这已经覆盖“保留径向日程、增强几何修正”的宽泛动机。旋转参考的区别在于用变换后的密度与链式法则构造差分，而非直接放大原增量，但是否值得付出额外查询，仍需实验而不是几何类比来回答。[^15]

**候选方法三：矩匹配的密度卷积参考。** 若已知强分布的均值 \(\mu\) 与协方差 \(\Sigma\)，令

\[
Y=\mu+A(X+\sqrt\tau Z-\mu),\qquad
A=\Sigma^{1/2}(\Sigma+\tau I)^{-1/2}.
\]

则 \(Y\) 与 \(X\) 有相同的一、二阶矩，并以匹配高斯为不动点。它的 score 是

\[
s_q(x)=A^{-\top}s_{H_\tau p}\bigl(\mu+A^{-1}(x-\mu)\bigr).
\]

这属于归一化高斯弱化的思路；单个标量协方差版本容易落回 OU/VP 重参数化，研究价值要在结构化协方差和实际误差选择性上证明。参考分布保矩也不等于外推后的整个生成分布保矩，二者不能混写。

SiT 实现使用每类 256 个训练样本的 VAE 后验均值和方差，估计逐坐标方差，包含后验随机性。对路径 \(z_t=tx_0+(1-t)\epsilon\)，设 \(\tau=\kappa^2(1-t)^2\)、\(r=\sqrt{(1-t)^2+\tau}\)，通过 \(t'=t/(t+r)\) 与 \(z'=\{\mu_t+A^{-1}(z-\mu_t)\}/(t+r)\) 查询原模型，再变换回 score。它比直接换时间多了关键的输入和输出变换，但仍仅增加一次模型查询。

这些统计量来自真实训练数据，是模型边缘矩的估计而不是精确值；完整空间相关性未被建模。因此，严格的高斯不变性只针对与采用统计量一致的高斯参考。原型已经完成真实 SiT 图像实验和同计算量对照，不能将统计近似隐藏在“完全无校准”的说法中。

**弱化方向应如何选择。** 设 \(e=s_\theta-s_*\)、\(g=s_\theta-s_{\rm ref}\)，则精确有

\[
\mathbb E\|e+\omega g\|^2-\mathbb E\|e\|^2
=2\omega\mathbb E\langle e,g\rangle+\omega^2\mathbb E\|g\|^2.
\]

正权重改善这个风险必须依赖负的误差内积，“参考更差”本身不够。AG 与 weak-model guidance 的研究强调相近但更强的建模错误，支持把误差对齐作为目标，而非把低质量本身当成目标。[^12][^13]

这里还有一个可辨识性限制。假设能观察到的模型总是 \(\mathcal N(0,1.5)\)，真实分布却可能是 \(\mathcal N(0,1)\) 或 \(\mathcal N(0,2)\)。同一个向中心收缩的自引导，在前一种情形可以纠错，在后一种情形会加剧错误；只访问这个固定模型，无法区分两种情形。这是一个直接反例，说明“人为构造弱分布”必须附带对错误类型的假设、少量真实数据或另外的质量信号，才能说明为何某个方向值得外推。高斯不变性选择保守地不处理这类错误，而没有解决识别问题。

在加噪真实数据上可以用 DSM 目标估计这个方向。线性流中以 \(y=x_0-\epsilon\) 为目标，最小化 \(\mathbb E\|v_{\rm base}+\omega g-y\|^2\) 给出每个时间段的闭式系数

\[
\widehat\omega=\operatorname{clip}_{[0,3]}
\left[-\frac{\sum_i\langle v_i-y_i,g_i\rangle}{\sum_i\|g_i\|^2}\right].
\]

此次真实模型校准使用 1,000 个训练图像、另 1,000 个验证图像，各 8 个时间点。比较白噪声核、单位坐标方差的固定 3×3 低通噪声核和双尺度核，生成器完全冻结。它是所给“学习结构化弱化器”思路的最小实现：**只筛固定核并拟合时间权重，没有训练完整的 \(L_\psi\)**，也不宣称 training-free calibration。

这个校准器在已知错误的 toy 上可以工作：score 过平滑的例子中，真实 score 风险从 0.07633 降到固定强度 1 的 0.05109，训练集拟合强度 3 后在独立测试集为 0.02513；单独加入高频误差时，固定强度 1 把风险从 0.03122 增至 0.13852，校准器选择 0 并保留基线。这些结果说明公式和实现可以识别特定误差方向，却不保证真实模型上存在有利的非零解。

{{CALIBRATION}}

这并不与无 CFG 图像 FID 的改善矛盾。DSM 约束的是加噪真实数据上的局部误差，生成轨迹却可能进入另一组状态。*Learn to Guide Your Diffusion Model* 的附录 C.1 已明确讨论直接 guided score matching 退化到零权重的现象；因此不能把这个校准目标作为未经验证的新贡献。[^5] VSM 从训练中的曲率代理约束出发，是另一个相关方向，但它也没有提供“所有去平滑都改善生成”的保证。[^14]

更值得继续的候选是：**在保持高斯不变性的算子族内，通过短程生成转移的分布匹配学习核与权重**。具体可冻结网络，只学习少量时间段和频带系数；从真实加噪分布出发做少量采样步，将终点分布与对应噪声时刻的真实分布比较，并保留基线作为可选解。它借鉴已有 guidance 学习的分布一致性目标，新增部分必须落在弱化算子的结构约束和可重复的收益上；该扩展尚未进行图像训练实验。

**真实模型的结果与计算量。** 所有配置使用相同类别与初始噪声作配对。SiT 是 ImageNet100 的 SiT-S/2 EMA 800K、SD-VAE 潜空间、FP32；JiT 是 JiT-B/16 EMA；RAEv2 是 DINOv3-L k7 表示上的 DDT EMA。后两者使用既有 FP32 状态、BF16 模型计算和 TF32 设置。SiT 为 ADM ImageNet100 验证参考；JiT/RAEv2 为 nanogen-evals ImageNet1K 256 参考，**不同评估协议的绝对 FID 不作横向排名**。

主 log 原型统一使用 \(\xi=0.2\sigma_t\epsilon\)、一个正负扰动对、\(\omega=1\)。SiT/JiT 的时间从噪声 0 到数据 1；RAEv2 从噪声 1 到数据 0，并沿用 shift=8 的时间网格。CFG 或 IG 保持既有系数和激活区间，参考差分只作用于原始条件分支。

|模型及基线|基础步数|基础前向/图|log 外推前向/图|同前向次数的控制|
|---|---:|---:|---:|---|
|SiT，仅条件模型|64|64|192|Euler 192|
|SiT，CFG extra=1.25，t<0.75|64|112|240|Euler 137，240 次前向|
|JiT，CFG=3|100|200|400|Euler 200，400 次前向|
|RAEv2，IG=1.78|100|100|300|Euler 300，300 次前向|

矩匹配 SiT 为 175 次前向，与 CFG Euler100 对齐；旋转版本为 240 次，与 CFG Euler137 对齐。以上是实际计数的分支前向次数；模型速度、解码和随机扰动开销还需以实测 GPU 时间比较。增加步数并不保证更好，这正是必须实际跑控制组的原因。

{{IMAGE_RESULTS}}

{{DECISIONS}}

这些属于 1K 样本的筛选和有限独立复核，尚不足以支持一个通用方法或可发表的质量提升结论。正向候选在独立种子上复核；旋转、矩匹配和跨模型负向结果没有开展等规模独立复核，不能把筛选失败提升为算法普遍无效的证明。SG 对照固定使用移时 0.01、强度 1，没有对 SG 和新方法开展同预算的完整超参数优化。

没有进行人工偏好或手部结构标注，也没有完整的精度/召回及训练图像近邻审计，因此不能把 FID 的变化直接称为伪影率或记忆化率的改善。固定 ID 的联系图用于观察样本变化，全部展示预先固定的前六个样本，没有根据好坏选择例子。

**复现与证据。** 代码集中在 `experiments/weak_reference_20260914/`。`sampler.py` 是正负 score 平滑，`moment.py` 是矩匹配与准确坐标变换，`angular.py` 是旋转与 score 拉回，`calibrated.py` 是低通/双尺度核，两个 `calibrate_*.py` 分别审计 CFG 和原始条件基线。`toy.py` 包含解析复核、已知误差风险和实际 ODE；`curved_toy.py` 为圆环反例。

[实验说明与完整阶段表](../experiments/weak_reference_20260914/README.md) 列出了所有配置文件、采样器、种子和样本数。最后的数据汇总命令要求阶段表中的所有实验均完成并通过核验。

完整生成数据位于 `/home/zhoushunyu/data/eqvae/experiments/weak_reference_20260914/`，轻量指标、请求清单、核验与联系图位于 `docs/data/weak_reference_20260914/`。请求清单冻结了采样配置、输入和模型/代码哈希；结果检查覆盖样本数、类别平衡、批次覆盖、输入一致性、数值有效性及前向次数。FID 还从缓存特征用 FP64 重新计算，这验证数值计算，**不等同于第二套独立特征提取器**。图表数据另附 [研究数据工作簿](data/weak_reference_20260914/research_data.xlsx)。

典型复现命令如下，已有冻结目录可以恢复未完成批次；改参数应使用新的阶段名，避免覆盖原实验。

```bash
PYTHON=/home/zhoushunyu/miniconda3/envs/myenv/bin/python
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m experiments.weak_reference_20260914.check --model
"$PYTHON" -m experiments.weak_reference_20260914.run prepare --phase sit_screen_1k --config experiments/weak_reference_20260914/configs.json --sampler experiments.weak_reference_20260914.sampler --samples 1000 --seed 2026091407 --batch 16
"$PYTHON" -m experiments.weak_reference_20260914.run controller --phase sit_screen_1k --gpus 0
"$PYTHON" -m experiments.weak_reference_20260914.report --phase sit_screen_1k
"$PYTHON" -m experiments.weak_reference_20260914.toy
"$PYTHON" -m experiments.weak_reference_20260914.curved_toy
```

JiT、RAEv2 与最终数据包的对应命令如下；跨模型检查文件已经归档。复现新环境时，先分别在可用 GPU 上运行 `cross_check --model jit` 与 `cross_check --model raev2`。

```bash
"$PYTHON" -m experiments.weak_reference_20260914.cross_run prepare --phase cross_screen_1k --models jit,raev2 --samples 1000 --seed 2026091407 --batch 32
"$PYTHON" -m experiments.weak_reference_20260914.cross_run controller --phase cross_screen_1k --gpus 2,3
"$PYTHON" -m experiments.weak_reference_20260914.report --phase cross_screen_1k --models jit,raev2
"$PYTHON" -m experiments.weak_reference_20260914.bootstrap
"$PYTHON" -m experiments.weak_reference_20260914.bootstrap --band
"$PYTHON" -m experiments.weak_reference_20260914.build_data
"$PYTHON" -m experiments.weak_reference_20260914.write_report
```

**Sources**

[^1]: Tiancheng Li, Weijian Luo, Zhiyang Chen, Liyuan Ma, Guo-Jun Qi. *Self-Guidance: Boosting Flow and Diffusion Generation on Their Own*. [版本记录](https://arxiv.org/abs/2412.05827)，[v1 PDF](https://arxiv.org/pdf/2412.05827v1)，[v2 PDF](https://arxiv.org/pdf/2412.05827v2)，[v3 PDF](https://arxiv.org/pdf/2412.05827v3)，[v4 PDF](https://arxiv.org/pdf/2412.05827v4)，[v5 PDF](https://arxiv.org/pdf/2412.05827v5)。2024-12 至 2025-09。主要依据各版摘要、引言、方法和表 1。
[^2]: Li et al. *Self-Guidance*. IEEE TPAMI 48(1), 781–791, 2026-01。[PubMed 期刊记录](https://pubmed.ncbi.nlm.nih.gov/40966151/)，[原出版物](https://doi.org/10.1109/TPAMI.2025.3611831)。公开检索未取得拒稿决定，不能由缺少记录反推从未被拒。
[^3]: MAPLE Research Lab. [Self-Guidance 官方代码](https://github.com/maple-research-lab/Self-Guidance)。此次对照提交 `843bda799bb531ccf8d98164d8ccf1a3b8ce3f6d`，重点核对 Flux 和 SD3 pipeline 的预测组合。
[^4]: Rafał Karczewski, Markus Heinonen, Vikas Garg. [*Devil is in the Details: Density Guidance for Detail-Aware Generation with Flow Models*](https://arxiv.org/html/2502.05807). 2025，arXiv:2502.05807；关于密度、细节及极端密度样本的区别。
[^5]: Alexandre Galashov, Ashwini Pokle, Arnaud Doucet, Arthur Gretton, Mauricio Delbracio, Valentin De Bortoli. [*Learn to Guide Your Diffusion Model*](https://arxiv.org/html/2510.00815). 2025-10，arXiv:2510.00815；§3、附录 C.1，分布一致性与 guided score matching 的局限。
[^6]: Zhenyu Yao, Daniel Paulin. [*Training-Free Generative Sampling via Moment-Matched Score Smoothing*](https://arxiv.org/html/2605.14276). 2026-05，arXiv:2605.14276；§2–3、附录 C，log-domain smoothing、矩约束和正负扰动估计器。与此次反向外推的动力学不同。
[^7]: Zhengdao Chen. [*On the Interpolation Effect of Score Smoothing in Diffusion Models*](https://arxiv.org/html/2502.19499). 首版 2025-02，arXiv:2502.19499；score 平滑的插值与泛化解释。
[^8]: Kiwhan Song, Jaeyeon Kim, Sitan Chen, Yilun Du, Sham Kakade, Vincent Sitzmann. [*Selective Underfitting in Diffusion Models*](https://arxiv.org/html/2510.01378). 2025-10，arXiv:2510.01378；监督区域与生成时外推区域的区分。
[^9]: Lingxiao Yang, Shutong Ding, Yifan Cai, Jingyi Yu, Jingya Wang, Ye Shi. [*Guidance with Spherical Gaussian Constraint for Conditional Diffusion*](https://arxiv.org/html/2402.03201). 2024，arXiv:2402.03201；球面约束用于外部损失引导的相邻方向。
[^10]: Jia-Wei Liao, Li-Xuan Peng, Mei-Heng Yueh, Min Sun, Cheng-Fu Chou, Jun-Cheng Chen. [*DiffRGD: An Inference-Time Diffusion Guidance Through Riemannian Gradient Descent*](https://arxiv.org/html/2606.28417). 2026-06，arXiv:2606.28417；球面优化与保持潜变量高斯结构。
[^11]: Giannis Daras, Weili Nie, Karsten Kreis, Alex Dimakis, Morteza Mardani, Nikola Borislavov Kovachki, Arash Vahdat. [*Warped Diffusion: Solving Video Inverse Problems with Image Diffusion Models*](https://arxiv.org/html/2410.16152). 2024，NeurIPS 2024，arXiv:2410.16152；变换一致性与 equivariance self-guidance，作为旋转参考的相关工作边界。
[^12]: Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, Samuli Laine. [*Guiding a Diffusion Model with a Bad Version of Itself*](https://arxiv.org/html/2406.02507v1). 2024-06，arXiv:2406.02507；AG 的建模误差、弱模型及分布覆盖讨论。
[^13]: Nikolas Adaloglou, Tim Kaiser, Damir Iagudin, Markus Kollmann. [*Guiding a Diffusion Model with Itself Using Sliding Windows*](https://arxiv.org/html/2411.10257). 2025-08 v3，arXiv:2411.10257；早期版本标题为 *The Unreasonable Effectiveness of Guidance for Diffusion Models*。§3 关于相似且更强的建模错误。
[^14]: Mahesh Bhosale, Naresh Kumar Devulapally, Abdul Wasi, Chau Pham, Vishnu Suresh Lokhande, David Doermann. [*Score-Control for Hallucination Reduction in Diffusion Models*](https://arxiv.org/html/2606.00377). 2026，arXiv:2606.00377；VSM 是训练中的曲率代理约束，不是此次采样外推。
[^15]: Hyunmin Cho, Donghoon Ahn, Susung Hong, Jee Eun Kim, Seungryong Kim, Kyong Hwan Jin. [*TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling*](https://arxiv.org/html/2510.04533). 首版 2025-10，ICML 2026；[作者项目页](https://hyeon-cho.github.io/TAG/)。这里的 TAG 是 Tangential Amplifying Guidance，与 arXiv:2510.11057 的 Temporal Alignment Guidance 是不同工作。

补充近邻文献如下。它们分别约束“平滑”“结构化弱化”“保留生成器权重”等宽泛动机的新颖性；操作对象不同，需要比较具体公式和目标。

- Susung Hong. [*Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention*](https://arxiv.org/html/2408.00760). 2024，arXiv:2408.00760。SEG 平滑 attention energy。
- Seyedmorteza Sadat, Jakob Buhmann, Derek Bradley, Otmar Hilliges, Romann M. Weber. [*CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling*](https://arxiv.org/html/2310.17347). 2023，ICLR 2024。对条件加入退火噪声，并讨论与 score 平滑的关系。
- Boseong Jeon. [*SPG: Improving Motion Diffusion by Smooth Perturbation Guidance*](https://arxiv.org/html/2503.02577). 2025，arXiv:2503.02577。沿运动时间轴构造平滑参考。
- Xinyu Zhou, Jiawei Zhang, Stephen J. Wright. [*Smoothing the Score Function to Enhance Generalization in Diffusion Models*](https://arxiv.org/html/2601.19285). 2026，arXiv:2601.19285。score 平滑及泛化的相关研究。
- Susung Hong, Gyuseong Lee, Wooseok Jang, Seungryong Kim. [*Improving Sample Quality of Diffusion Models Using Self-Attention Guidance*](https://arxiv.org/abs/2210.00939). 2022，ICCV 2023。SAG 已使用高斯模糊和注意区域退化来构造参考。
- Donghoon Ahn et al. [*Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance*](https://arxiv.org/abs/2403.17377). ECCV 2024。PAG 使用注意力退化形成参考。
- Donghoon Ahn et al. [*Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models*](https://arxiv.org/html/2506.10978). NeurIPS 2025。HeadHunter 研究头级扰动选择，所以“选择结构化退化”本身不是空白。
- Donghoon Ahn et al. [*A Noise is Worth Diffusion Guidance*](https://arxiv.org/abs/2412.03895). 2024，ICLR 2026。NoiseRefine 训练初始噪声映射以吸收引导效果，操作对象和目标不同于每步的正负 score 查询。

所给中文分析作为方法线索使用，原始材料为本地附件 `pasted-text.txt`；其中的版本判断、公式、数值例子和文献指向已分别复核。论文副本、下载 URL、PDF SHA256 与版本差异记录保存在 `readings/weak_reference_20260914/`。以上网页及版本核对截至 2026-09-14。
