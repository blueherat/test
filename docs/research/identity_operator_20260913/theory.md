# 原样输出迭代的数学对象、反例与可识别条件

核心判断：此前 `0.25 × 原 latent + 0.75 × 新高斯噪声 → class-conditioned ODE 后半程` 是一个**破坏后重新采样的转移核**，不能作为“输入完整图片、要求不作修改”的替代。完美的 flow 模型在这项旧操作下也会快速失去逐样本对应关系。另一方面，同模型精确反演再生成可以对任何可逆的错误模型都完美恒等，所以把旧操作直接换成“正逆循环”也不足以得到生成质量指标。

必须先独立确定：模型每次实际得到什么信息；指令定义的正确输出是什么；随机数、历史、隐变量是否保留；我们衡量逐图忠实度还是边缘分布质量。

本文中“文献结论”均就近列明原始来源；解析例子和可识别性判断标为本文推导，不声称是新定理或研究创新。

## 1. 原观察应怎样形式化

### 1.1 完整图片作为条件的无编辑算子

设 `u=0` 表示“忠实返回当前输入图片，不作编辑”，`x` 是这次请求上传的完整图片，`ξ` 是模型内部的采样随机性。定义

\[
Y=A_\theta(x,u=0;\xi),\qquad
K_{\theta,0}(x,B)=\Pr[A_\theta(x,0;\xi)\in B].
\]

逐轮把这次输出变成下次输入：

\[
X_{r+1}\sim K_{\theta,0}(X_r,\cdot).
\]

如果指令要求严格逐像素相同，正确目标是

\[
K_0^*(x,dy)=\delta_x(dy).
\]

如果只要求视觉上不变，应先固定允许的等价关系或距离，例如内容、几何、颜色、纹理分别使用哪些量，写成

\[
K_{\theta,0}\bigl(x,\{y:d(y,x)\leq\varepsilon\}\bigr)\geq1-\eta.
\]

阈值 `ε,η` 是任务规范，不能看到结果后移动。语义特征相同不意味着构图相同，更不意味着逐像素相同。

**独立采样噪声本身不否定这个目标。** 当干净 `x` 作为额外条件一直可见，算法可以忽略随机性，也可以用随机轨迹到达同一个终点。要求 \(p(Y\mid x,u=0)=\delta_x\) 完全自洽。

这是本文建议作为“原样输出回灌”主定义的操作，名称可用 **iterated no-op editing / recursive identity-conditioned editing**。最近的直接相关文献也把 no-op 编辑理想化为输入的固定点，并做十轮 no-op 编辑；它支持该操作定义，但不证明这种分数能排序通用生成能力。[1]

### 1.2 同一对话保留历史是另一个实验

若同一会话累计先前图片、文本、编辑记录或模型隐藏状态，实际更新应写成

\[
X_{r+1}\sim K_\theta(\cdot\mid X_r,H_r,u=0),\qquad
H_{r+1}=U(H_r,X_r,X_{r+1},u=0).
\]

仅看 `X_r` 一般不再是时间齐次 Markov 链；扩充到 `(X_r,H_r)` 才能得到完整状态描述。历史可能携带第一轮后已经从图片中丢失的信息，因此它可以绕过 image-only 链的信息瓶颈。

例如当前图已少了一个小物体，历史仍有原图；下一次输出恢复它，不能据此断言模型只根据当前图就恢复了丢失信息。反过来，若最新指令明确要求复制“当前这张图”，历史诱导模型恢复旧图也可能违反当前指令。

因此至少分开报告：

| 条件 | 每次模型可见什么 | 研究对象 |
|---|---|---|
| 清空历史 | 当前一张图片、相同 no-op 指令 | 图片自身经重复无编辑处理的稳定性 |
| 保留历史 | 当前图片及之前完整对话 | 会话系统的长期忠实度 |
| 固定原图锚点 | 当前图片、最初原图、明确角色 | 带外部记忆的约束编辑 |

第三项是合理的工程方案，但不能混入第一项分数。当前原观察是否保留历史尚未确定，不能替它假定。

### 1.3 接口接收完整图，不等于生成核心没有信息瓶颈

上面 \(\delta_x\) 是**接口任务的规范**，并非“任何原生图像编辑架构都能严格实现它”的定理。如果模型真正可用的条件仅是 \(C=\psi(x)\)，且存在 \(x_1\neq x_2\) 但 \(\psi(x_1)=\psi(x_2)\)，同一个条件输出核不可能同时等于 \(\delta_{x_1}\) 和 \(\delta_{x_2}\)。VAE、图像tokenizer、缩放、裁剪和量化都可能构成这样的瓶颈。

因此需要分别检查 API 收到的文件、图像预处理、实际模型条件和输出codec。原生图像条件仍是最贴合原观察的接口，但不能据其名称就承诺零像素误差。若复制旁路能保留原文件，那在 no-op 产品任务上是合法解决方案；它不能证明生成先验更强。

codec-only 基线有助于定位损失，但不能在任意非线性距离（例如 LPIPS）上直接做“总误差减codec误差”并声称完成因果分解；交互项、不同输入状态和非线性解码会破坏可加性。

## 2. “不变”至少有五个不同含义

| 性质 | 数学表达 | 它实际保证什么 | 它没有保证什么 |
|---|---|---|---|
| 逐图恒等 identity | \(T(x)=x\) 或 \(K(x,\cdot)=\delta_x\) | 当前输入保留 | 去伪影、生成自然度、编辑能力 |
| 确定性幂等 idempotence | \(T(T(x))=T(x)\) | 第一次变化后不再变化 | 第一次输出接近输入；不坍缩 |
| 核幂等 | \(K^2=K\) | 一步与两步的条件输出分布相同 | 同一条轨迹的图片相同 |
| 边缘平稳 stationarity | \(\pi K=\pi\) | 输入若来自 \(\pi\)，输出总体仍来自 \(\pi\) | 任意原图被保留；慢混合 |
| 路径可逆 round-trip | \(D(E(x))=x\) | 某一编码-解码配对能恢复输入 | `D` 的先验采样分布正确；中间结果自然 |

**解析反例 1：幂等而没有任何内容保留。** 令 \(T(x)=c\) 为常量图片，则 \(T^2=T\)，但全部输入变成 `c`。随机版本令 \(K(x,dy)=\pi(dy)\)，每次独立重画：它同时满足 \(K^2=K\) 与 \(\pi K=\pi\)，却完全忽略原图。这个例子说明“分布幂等”比“逐图不变”弱得多。

**解析反例 2：恒等而没有任何去伪影能力。** 令 \(T=I\)。它完美保留清晰照片，也完美保留黑屏、错误手指、压缩块和涂鸦。若任务就是 no-op，这当然是正确行为；但不能从它推导一个通用的自然度分数。

图像恢复文献已明确分开配对失真和输出边缘分布的感知质量；其 posterior sampling 结论说明总体分布可以正确，同时与具体原图仍有不可避免的差异。[2] 这里不把该 tradeoff 误用于完整 `x` 可见的严格复制任务：完整条件让 posterior 退化时，复制的失真可以为零。

## 3. 直接匹配旧 SiT 操作的精确反例

下面是本文推导的一维 Gaussian 例子，逐维扩展到任意维。它使用旧实验相同的线性插值路径和确定性 ODE 采样，因此无需把真实实现误称为 posterior sampler。

设真实数据与基础噪声都是标准高斯，且独立：

\[
X\sim\mathcal N(0,1),\quad E\sim\mathcal N(0,1),\quad
Z_t=tX+(1-t)E.
\]

记

\[
s(t)^2=t^2+(1-t)^2.
\]

对流匹配平方损失，精确最优边缘速度为

\[
u^*(z,t)=\mathbb E[X-E\mid Z_t=z]
=\frac{2t-1}{t^2+(1-t)^2}z.
\]

因为 \(u^*(z,t)=\frac{s'(t)}{s(t)}z\)，精确 ODE 从时刻 `t` 到 `1` 的解为

\[
F^*_{t\to1}(z)=\frac{z}{s(t)}.
\]

将旧实验的每轮操作代入：

\[
X_{r+1}=F^*_{t\to1}\bigl(tX_r+(1-t)E_r\bigr)
=\rho X_r+\sqrt{1-\rho^2}\,E_r,
\quad\rho=\frac{t}{\sqrt{t^2+(1-t)^2}}.
\]

只要 \(X_0\sim\mathcal N(0,1)\)，每轮 \(X_r\) 都精确服从真实数据分布，模型和求解器均无误差。但

\[
\operatorname{Corr}(X_r,X_0)=\rho^r,\qquad
\mathbb E[(X_r-X_0)^2]=2(1-\rho^r).
\]

旧实验 `t=.25` 时，\(\rho=1/\sqrt{10}\approx0.316228\)。五轮后

\[
\operatorname{Corr}(X_5,X_0)=10^{-5/2}\approx0.003162,
\quad \mathrm{MSE}\approx1.993675.
\]

也就是**生成分布始终完美，五轮后与原样本却几乎不相关**。在直接使用原始坐标作为特征的 Gaussian FID 中，人口层面的 FID 始终为零；有限样本估计会有采样误差。该例只是反驳“完美生成模型应在旧操作下逐图不变”，不是对真实图像相关系数的数值预测。

Flow Matching 原文区分按数据样本条件化的路径与对其求平均得到的边缘向量场，并证明后者实现边缘概率路径；它没有承诺每条新造出的 noisy-data 直线都能被边缘 ODE 逆回原终点。[3] 上面的显式 Gaussian 系数是本文独立推导。

### 3.1 只把 fresh noise 改成 fixed noise 仍然不是逆操作

继续同一精确例子，若每轮都重用相同 `e`：

\[
x_{r+1}=\rho x_r+\sqrt{1-\rho^2}\,e,
\qquad
x_\infty=\frac{\sqrt{1-\rho^2}}{1-\rho}e.
\]

它会收敛到一个依赖 `e`、基本忘掉初始图的固定点。固定随机种子改变了跨轮 coupling，却没有构造出原 ODE 轨迹的逆映射。“最终稳定”在这里恰好可以是内容遗忘。

若真的保存了实际噪声 `e` 且把它作为额外信息传给解码器，那么 \(x=(z-(1-t)e)/t\) 可直接算出。这又成为携带 side information 的可逆编码任务，与只让 denoiser 看 `z` 不同。

## 4. Posterior mean 和 posterior sampling 也不能混用

**解析反例 3：完美 posterior mean 会收缩。** 令 \(X\sim\mathcal N(0,\sigma_x^2)\)，\(Y=X+N\)，\(N\sim\mathcal N(0,\sigma_n^2)\)。精确最小 MSE 去噪器为

\[
m(y)=\mathbb E[X\mid Y=y]=a y,
\quad a=\frac{\sigma_x^2}{\sigma_x^2+\sigma_n^2}<1.
\]

如果把干净样本直接当作观测输入并反复用 `m`，得到 \(m^r(x)=a^rx\to0\)。去噪器在其训练任务上完全最优，却不使所有真实样本成为固定点。

如果每轮重加噪再取 mean，链为 \(X_{r+1}=aX_r+aN_r\)。它的稳态方差是

\[
\frac{a^2\sigma_n^2}{1-a^2}
=\frac{\sigma_x^4}{2\sigma_x^2+\sigma_n^2}
<\sigma_x^2,
\]

所以这也不等于 posterior sampling 的平稳分布。Alain–Bengio 的最优重建公式及小噪声 score 展开支持“去噪重建会沿数据密度改变输入”，不能把其位移统一视为模型错误。[4]

**解析反例 4：完美 posterior sampling 保分布而不保原样本。** 用同样的 corruption `Y=X+N`，但重建时采样

\[
X'\mid Y\sim\mathcal N(aY,\tau^2),\quad
\tau^2=\frac{\sigma_x^2\sigma_n^2}{\sigma_x^2+\sigma_n^2}.
\]

新采样与原 `X` 在给定 `Y` 后独立时，\(X'\) 的边缘分布仍为 \(\mathcal N(0,\sigma_x^2)\)，但

\[
\mathbb E[(X'-X)^2]=2\tau^2>0.
\]

这是 corruption–posterior Gibbs 转移的正常行为。Bengio 等关于 denoising autoencoder 的生成性定理要求真实条件重建分布及适当遍历条件，并保证渐近边缘分布，而不保证逐图固定。[5] 该结论适用于概率重建核；旧 SiT 后缀是 ODE，不能直接援引它断言旧链必然平稳，旧链的反例已在第 3 节另外给出。

## 5. 信息论上怎样修正“噪声带来新信息”

判断方向正确：仅将原图与大量新噪声混合，再要求模型恢复唯一原图，改变了任务。更精确的表述是：独立噪声带来新的随机自由度，但不携带关于原图的独立语义证据，即 \(I(X;E)=0\)。如果解码器只看到 `Z=aX+bE`，原图经破坏信道后通常不可唯一识别。

对于只依据 `Z` 与独立内部随机数输出 `Y` 的解码器，有 Markov 结构 \(X\to Z\to Y\)；数据处理不等式给出 \(I(X;Y)\leq I(X;Z)\)。连续变量的恒等映射可能有无穷互信息，严格零误差条件更宜直接写为 \(\operatorname{Var}(X\mid Z)=0\) 或 posterior 为点质量。Gaussian 例子在任意非零噪声下条件方差大于零，因此一般做不到几乎必然逐样本恢复。

**必要的可识别条件，本文推导：** 在没有和真实 `X` 特殊关联的 side information 时，要让 `Y=f(Z,R)=X` 几乎必然成立，需要 `X` 几乎必然是 `Z` 的可测函数；对于离散量可表述为 \(H(X\mid Z)=0\)，对平方可积连续量可表述为 \(\operatorname{Var}(X\mid Z)=0\)。独立 `R` 不能补回具体丢失的样本信息。

相反，当模型额外得到完整干净图 `x`，条件集合变成 `(Z,x)`，上述瓶颈不适用。一个极简单的精确 image-conditioned flow 为

\[
u(z,t\mid x,u=0)=\frac{x-z}{1-t},\quad t<1.
\]

它的轨迹为 \(z_t=tx+(1-t)\xi\)，终点对每个 `ξ` 都是 `x`。终点场写法有奇性，但轨迹极限明确。**所以“使用独立生成噪声”与“要求复制完整条件图”完全兼容；旧操作的问题是干净源图没有作为单独条件保留。**

## 6. 为什么同模型反演再生成不能直接选 CFG / IG

令 `F_g` 是某个 guidance 配置对应的精确可逆流，定义 `E_g=F_g^{-1}`。则

\[
T_g=F_g\circ E_g=I
\]

对任意 `g` 成立，与 `F_g` 从标准噪声产生什么分布无关。

**解析反例 5：错误生成器也可以循环完美。** 真实数据取 \(\mathcal N(0,1)\)，生成器 `F(z)=100z` 将标准噪声变成 \(\mathcal N(0,10000)\)，显然不是目标分布；但 `E(x)=x/100` 令 `F(E(x))=x` 完全精确。甚至 `F=I` 可以在图像任务中只生成高斯噪声，同时拥有零循环误差。

因此同模型 round-trip 能可靠衡量的主要是：离散求解误差、逆算法近似、精度、codec 损失或隐藏状态是否恢复。若数值不稳定伤害实际任务，这当然有工程意义；但其含义需停在可逆实现层面。ODE 概率流文献证明特定场在理想条件下实现指定边缘分布，这不等于任何可逆场都分布正确。[6]

换成强模型反演、弱模型生成，\(F_w\circ E_s\) 不再代数恒等，确实可测相对 transport mismatch；但需要一个额外前提才能把 mismatch 称为错误：`s` 的耦合必须有独立理由被视作正确参考。两种生成器可能具有同样正确的边缘分布、不同的样本耦合。

**解析反例 6：两个完美分布模型仍有巨大的交叉循环差。** 标准二维高斯下，令 `F_1=I`，`F_2=R` 为旋转矩阵。两者先验采样分布完全相同且正确；但 `F_2(E_1(x))=Rx` 通常不是 `x`。这也说明“强弱模型差异”不是天然的误差方向。

反向挑战也成立：让模型专门编码不可见信息，可以获得很低循环误差，却不保证语义上正确的中间翻译。CycleGAN 的原始分析曾实证循环一致性目标诱导隐藏高频信息，说明配对可恢复性仍可能被旁路满足。[7]

## 7. 固定点稳定性怎样才有解释力

设确定性 no-op 算子在 `x` 附近写成 \(T(x)=x+b(x)\)。局部迭代误差受到偏置 `b` 和 Jacobian `J_T` 共同影响，不能只检查“是否收敛”。

- 若任务是保留所有允许输入，则理想 `T=I`，切向导数应为 `I`，并非越收缩越好。
- 若任务是恢复某个外部定义的数据流形，理想投影才可能沿法向收缩、沿切向保留；但这已是 restoration / projection 任务，需要指定哪些变化是损坏、哪些是合法细节。
- 如果数据流形由同一个待评价模型定义，“回到自己的流形”可能只是自证。错误的低维流形也能形成稳定吸引子。

对 CFG / IG 的直接后果：若把任何反复重建的漂移都惩罚掉，可能把原本需要的类别外推、细节探索或去伪影一并压掉；若只奖励收缩，又可能推动 mode collapse。方向增益大于 `1` 也不能仅凭这一点判定坏：合法编辑放大、细节恢复或坐标尺度都可能产生放大。

## 8. 当前最可信的研究桥接及其反驳

### 8.1 可以立即成立的：编辑任务的零编辑校准

使用真正接收完整图像的编辑模型，先测 `u=0` 下的条件忠实度，再研究改变 CFG / IG 时多出来的非请求变化。这明确评价 instruction-conditioned editing 的校准，而不是 class-only 生成的自然度。

反驳：若只有 no-op，直接复制输入就能满分。必须同时有可验证的小幅非零编辑任务，例如指定局部颜色变化、局部对象属性变化，并同时检查请求完成率和未请求区域保留。测试需记录当前源图、实际请求、目标区域和允许变化，防止“关闭全部编辑”伪改善。

### 8.2 只能作为诊断的：固定外部反演参考

固定一个 reference inverter `E_ref`，比较 `F_g(E_ref(x))`。它避免每个候选自己反演自己导致的零误差恒等，但得到的是对 reference transport 的保持程度。

反驳：`F_ref` 在结构上有天然优势；优化该误差可能只学回 reference。跨多个参考、独立样本及外部任务验证可以减少偏差，不能彻底自动把参考一致性变成生成质量。应把它命名为 reference reconstruction consistency，保留这个解释边界。

### 8.3 需要新证据才能成立的：从 no-op 缺陷推导 class-only CFG / IG

需要先证明某个独立测到的缺陷信号与外部生成质量或特定伪影存在稳定关系，再考虑在线干预。至少要控制简单降低 guidance、增减步数、固定随机性、codec 更换和输出平滑等混淆因素。

反驳：即使相关也未必有可干预的因果关系；把指标直接优化后可能失效。更严格的验证应让缺陷定义、信号模型、调参样本、最终评测互相独立。当前 1K FID 的小幅改善不承担这些证明。

### 8.4 一个可以先验证的 IG 误差结构诊断

下面是本文推导，限于**最终输出的线性外推**，不等同于当前采样器逐步 velocity IG。

对同一图 `x`、同一 no-op 指令、同一实际信息集，预先明确强弱模型的联合随机 coupling，写最终输出为

\[
Y_s=x+e_s,\qquad Y_w=x+e_w.
\]

若做输出外推 \(Y_\alpha=Y_s+\alpha(Y_s-Y_w)\)，则

\[
\mathcal L(\alpha)=\mathbb E\|Y_\alpha-x\|^2
=\mathbb E\|e_s\|^2+2\alpha B+\alpha^2 A,
\]

其中 \(A=\mathbb E\|e_s-e_w\|^2\)，\(B=\mathbb E[e_s^\top(e_s-e_w)]\)。当 `A>0`，无约束最优值 \(\alpha^*=-B/A\)；要求正 guidance 时为 \(\max(0,-B/A)\)。若 `A=0`，差分为零，`α` 不可识别且没有作用。

因此正向外推能减小 no-op MSE 的必要充分局部条件是

\[
\mathbb E[e_s^\top e_w]>\mathbb E\|e_s\|^2.
\]

**弱模型自己的 MSE 更大完全不够。** 若 `e_w` 与 `e_s` 独立且前者零均值，则 `B=MSE_s>0`，正外推反而更差；若 \(e_w=k e_s, k>1\)，则 \(\alpha=1/(k-1)\) 可以精确抵消。这把“差模型沿同类错误偏得更远”从口头直觉变成一个可测的联合误差条件。

它的局限同样关键：`same seed` 并不自动给跨架构模型提供唯一自然的 coupling；像素 clipping、非线性 VAE 解码或 LPIPS 等非二次距离不满足这条精确二次公式。velocity IG 的改动还要沿整个采样轨迹传播，涉及 Jacobian 和高阶项，不能把最终输出线性外推公式直接当作其效果证明。该诊断应在独立样本上估计 `B`，联合非零编辑任务，而不是再凭弱模型更差就开始大规模 guidance 搜索。

## 9. 建议的最小判别实验，不启动新大网格

先做一个**操作诊断矩阵**，而非再次搜索十种 guidance：

| 算子 | 干净源图是否单独保留 | 预期无误差行为 | 该项的用途 |
|---|---|---|---|
| 像素恒等复制 | 是 | 完全恒等 | 文件格式、评价代码基线 |
| VAE encode-decode | 编码中经过瓶颈 | 接近输入或一次投影后稳定，取决于codec | 分离codec损失 |
| 真实编辑器 no-op | 是，作为图像条件 | 尽量恒等，允许误差需事先声明 | 对应原观察的主实验 |
| 同模型 ODE inverse-forward | 通过配对可逆latent保留 | 理想严格恒等 | 求解/反演基线 |
| 新噪声破坏后 class-only 生成 | 否 | 允许逐图漂移；理想边缘可平稳 | 原实验的重新命名与对照 |

no-op 主实验至少五轮，每轮分别保存图片；比较每轮相对上轮、相对初始图的误差。将 fresh seed / fixed seed、fresh session / retained history 分开记录，不能用固定 seed 宣称恢复了 inverse。若资源紧张，先做少量有文本、细边、重复纹理、特定小物体和人脸细节的图片，它们能快速暴露哪类信息在何处丢失。

同时保留一小组明确非零编辑，用于排除 copy bypass。若只能调用现有 class-only SiT，又没有图像条件模块，则它不能完整实现主实验；此时应承认模型接口不足，而不是继续用大噪声 img2img 替代语义。

## 10. 辩论记录与已解决分歧

| 被挑战的最强主张 | 反例或缺失前提 | 修正后成立范围 |
|---|---|---|
| “只要是原生图像条件，no-op 必能精确 \(\delta_x\)” | 非单射 \(\psi(x)\) 使两个不同输入的模型条件完全相同 | \(\delta_x\) 是接口规范；架构可达性取决于保留信息、旁路和codec |
| “同模型反演循环最贴合原图，可用来挑更强guidance” | `F(z)=100z, E(x)=x/100` 生成分布错误，循环仍完全恒等 | 同模型循环测反演与求解的实现误差；需独立外部任务才评价质量 |
| “固定第三方反演器就获得客观质量标准” | 标准高斯上 `I` 和旋转 `R` 的边缘都完美，但交叉重建不同 | 它测指定reference的transport保持；多个reference与独立任务缓解锚定偏差 |
| “有更多随机噪声就不能要求no-op” | 给定完整源图的 \(u=(x-z)/(1-t)\) 可从任意噪声到达 `x` | 无可恢复干净条件的破坏信道不能一般性保原样本；独立随机性本身不禁止复制 |
| “弱模型错误更大，所以IG外推可纠错” | 独立零均值弱误差使正外推MSE严格上升 | 需要联合错误相关结构满足 \(E[e_s^Te_w]>E\|e_s\|^2\)，且此处仅为输出外推诊断 |

1. **反演方案质疑：** 正逆循环是否最合理替代？结论：它是可逆性基线，但对同一场可代数恒等，不能独立测试生成质量。反演方向研究代理同意，并找到 EDICT 对“可逆不自动保证 realism / faithfulness”的明确说明；跨域循环亦同理。
2. **压缩/投影方案质疑：** 幂等能否解释强模型更稳？结论：它可解释某些codec迭代，但第一次失真独立，常数投影是退化解。跨领域研究代理同意，且找到直接 no-op DiT 漂移研究，与单纯 VAE 解释相互制衡。
3. **外部 reference 桥接质疑：** 固定参考是否已经打破自证？结论：打破同模型逆映射的代数恒等，但仍有参考锚定偏差；得到相对coupling一致性，需要外部任务判定好坏。
4. **随机性分歧修正：** 不能笼统说随机噪声使复制不可能。完整图像条件使退化目标 \(\delta_x\) 可实现；不可识别来自破坏后丢失干净条件，而不是采样随机性的存在本身。

## 11. 对 Drift Kernel 推导的独立审查

*The Drift Kernel* 的 §3.1.1 从前向加噪出发，假设解码器输入可写为均值加各向同性 Gaussian 扰动，经一阶 Taylor 得到 drift 随扰动方差的近似，并进一步将其解释为 no-op 漂移的结构原因。其实验使用固定 seed 与 img2img strength 扫描。[8]

**可接受的范围：** 如果已经有独立理由假设**最终解码器输入**的扰动是均值零、协方差 `σ²I`，且解码器在附近近似线性，那么这个局部敏感性计算有意义。

**关键缺失：** 前向加噪的协方差不自动等于最终解码器输入的残余协方差。实际 denoiser、条件、guidance 和 scheduler 均影响终端分布；实验参数 img2img strength 也不是自动等于终端latent的 Gaussian 标准差。文本条件只影响均值、不影响方差的普遍断言没有从这一 Taylor 展开推出。

更一般的一阶近似，本文推导：设编码latent为 `z`，`a=D(z)−x` 是codec残差，最终latent扰动 \(\delta=b+\eta\)，其中 \(E\eta=0\)，\(\operatorname{Cov}(\eta)=\Sigma\)。令 \(J=J_D(z)\)，则

\[
\mathbb E\|D(z+\delta)-x\|^2
\approx\|a+Jb\|^2+\operatorname{tr}(J\Sigma J^\top).
\]

`b`、`Σ`、局部 `J` 及选取的展开点都可能随条件、guidance、noise schedule 和 strength 变化，因此不能无条件约成同一常数 `c` 加固定斜率乘 strength²。上式本身仅是一阶解码线性化；若要完整保留关于扰动的二阶项，且 `a≠0`，还需codec偏置与解码器 Hessian 的交叉贡献。

第 5 节的完整源图条件 flow 进一步提供反例：生成轨迹可以有任意随机起点，但终点条件方差为零。因此前向噪声并不是一个对所有带图像条件系统都不可消去的终端方差下界。该反例使用终点极限或投影，不声称有限时间有界 Lipschitz 可逆流可以把连续分布精确压成点质量。

论文的实测 drift 可以保留；但“某个固定 pipeline 在某段 strength 扫描中拟合某个函数”与“所有 no-op 编辑器都满足该函数且无法消除 drift”是两个强度不同的主张。固定一个随机 seed 对所有输入作平均，也不等于为每张输入估计定义中的随机轨迹期望。必须先核查实际输入通道和统计协议，不能因为题目写了 no-op 就把普通 class/text img2img 当成完整源图条件的原样复制任务。

## Sources

1. Xiaoce Wang et al. *Why Do DiT Editors Drift? Plug-and-Play Low Frequency Alignment in VAE Latent Space*. arXiv:2605.08250v1, 2026-05-07. 直接阅读 §3.1–3.4 与实验协议；本文只用它支持 no-op 固定点定义和组件拆分，不把其特定模型的漂移归因泛化。[原文 HTML](https://arxiv.org/html/2605.08250v1)
2. Yochai Blau, Tomer Michaeli. *The Perception-Distortion Tradeoff*. CVPR 2018. 直接阅读 §3–4、Theorem 3 及 Appendix F：paired distortion 与 marginal perception 不同，posterior sampling 保边缘分布，其 MSE 等于 MMSE 的两倍。[原文 PDF](https://arxiv.org/pdf/1711.06077)
3. Yaron Lipman et al. *Flow Matching for Generative Modeling*. ICLR 2023. 直接阅读 §3、Theorems 1–2、§4：条件路径的边缘化与对应向量场。[原文 HTML](https://arxiv.org/html/2210.02747v2)
4. Guillaume Alain, Yoshua Bengio. *What Regularized Auto-Encoders Learn from the Data-Generating Distribution*. JMLR 15:3743–3773, 2014. 直接阅读 Theorem 1 及 §3.4：最优 Gaussian DAE 重建与 score 关系。本文中的完整 Gaussian AR 链方差为独立推导。[原文 PDF](https://jmlr.org/papers/volume15/alain14a/alain14a.pdf)
5. Yoshua Bengio, Li Yao, Guillaume Alain, Pascal Vincent. *Generalized Denoising Auto-Encoders as Generative Models*. NeurIPS 2013. 直接阅读 §2.3 Theorem 1：条件重建和遍历假设下的渐近平稳边缘。[原文 PDF](https://papers.nips.cc/paper_files/paper/2013/file/559cb990c9dffd8675f6bc2186971dc2-Paper.pdf)
6. Yang Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021. 官方研究页面支持 probability-flow ODE 与 SDE 的边缘分布关系；本次全文 PDF 访问受限，因此不从此源引申未经直接阅读的细节。相应条件场推导可交叉核对 [3] Appendix D。[官方作者机构页面](https://research.google/pubs/score-based-generative-modeling-through-stochastic-differential-equations/)
7. Casey Chu, Andrey Zhmoginov, Mark Sandler. *CycleGAN, a Master of Steganography*. NeurIPS 2017 Workshop Machine Deception. 直接阅读原文，支持循环损失可通过隐藏高频信息满足的经验反例；不声称现代图像编辑API都使用这种机制。[原文 PDF](https://arxiv.org/pdf/1712.02950)
8. Gokul Srinath Seetha Ram, Rashmi Elavazhagan. *The Drift Kernel: Why Diffusion Models Change Even When Told Not To*. CVPR 2026. 官方 PDF 实际下载并阅读 §3.1.1 与 §4.1；浏览器工具返回403，shell下载成功。这里只引用并审查其已陈述的局部假设，不否定其所有实测数据。[官方 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ram_The_Drift_Kernel_Why_Diffusion_Models_Change_Even_When_Told_CVPR_2026_paper.pdf)
