# 用 B 状态门约束跨尺度路径证据：严格推导与下一独立池协议

## 技术结论

这条路线可以收成一个很小、可以直接否证的内部方法：

1. 保留已经固定的 \(B\) 量，问“模型在第 69–149 步给出的最终草稿是否持续偏糊”。
   它是内部、预终点统计量，但不是鞅。
2. \(B\) 不冒充概率。它只在抽取下一步随机数以前，决定操作性备择
   \(Q^\star\) 此刻是否开启、看哪个局部区域。
3. 门开启时，用当前尺度与更高噪声尺度的 score 差给方向，再用局部 mask 和
   整条路径 \(K_{\rm total}=2.0\) 的 KL 预算真正缩小均值偏移。这个预算是在
   纯 Gaussian matched-Q 功效门否决 0.5 后、任何真实 screen 采样前固定的。得到的
   \(E_k=dQ^\star_{0:k}/dP_{0:k}\) 是相对于实际 Gaussian sampler 的精确正鞅。
4. 这个“精确”只属于新定义的操作性 \(Q^\star/P\)。B 门、局部 mask 和缩放都
   改变了备择语义，因此它不再等于理想热流的
   \(p_{v+\Delta}(x)/p_v(x)\)。
5. 纯 B、B-gated exact e-process，以及去掉 B 状态门的 exact ablation 必须分开。
   后者只回答“序贯证据有没有增量”，不能在主候选失败后顶替它。

当前只完成推导、label-free 核心和合成检验。没有打开第三池逐行标签、reviewer
结果、真实 B/C 分数或 endpoint embedding，也没有真实质量结论。

## 一、直觉：B 选检查位置，Gaussian LR 才是证据

已有两种现象各有一半是对的：

- B 的直觉很直接：正常成图时，中途预测的干净草稿通常会逐渐形成清楚局部边缘；
  明显模糊或软融合的轨迹可能长期没有形成这些边缘。
- 路径 likelihood ratio 的数学很硬：只要在下一步随机数出现前写好两个正规化
  转移核 \(P_k,Q_k\)，累乘 \(Q_k/P_k\) 就是正鞅。

不能把它们混成“B 大，所以 B 是 likelihood ratio”。正确连接是：

> B 只决定当前历史下可预测的实验设计；真正进入 e-process 的仍是下一步
> Gaussian innovation 在 \(P_k\) 与预先写好的 \(Q_k^\star\) 下的精确密度比。

直观上，B 像在化验前决定重点检查哪个部位。可以依据此前病史调整检查位置，
但不能看完本次化验结果再挑最异常的位置，然后仍声称原来的误报保证不变。

## 二、所有字母的含义

### 1. 采样器

| 字母 | 含义 |
|---|---|
| \(i\) | 第 \(i\) 条生成轨迹 |
| \(k\) | 生成方向的采样步；本协议观察 \(69,79,\ldots,149\) |
| \(t\) | DiT 内部扩散时间，\(t=249-k\) |
| \(\mathcal F_k\) | 抽取第 \(k\) 步 innovation 前已见的全部历史 |
| \(x_k\) | 第 \(k\) 步转移前的 \(4\times32\times32\) latent |
| \(\mu_k\) | baseline 实际 Gaussian 均值 |
| \(\sigma_k\) | baseline 实际逐坐标标准差 |
| \(\Sigma_k=\operatorname{diag}(\sigma_k^2)\) | baseline 对角协方差 |
| \(Z_{k+1}\) | 第 \(k\) 步实际抽到的标准 Gaussian innovation |
| \(P_k\) | 实际转移核 \(\mathcal N(\mu_k,\Sigma_k)\) |

实际一步为

\[
x_{k+1}=\mu_k+\sigma_k\odot Z_{k+1},\qquad
Z_{k+1}\mid\mathcal F_k\sim\mathcal N(0,I).
\]

\(I\) 是单位协方差，\(\odot\) 是逐元素乘法。learned variance、CFG=4 和
离散化都已包含在实际 \(\mu_k,\sigma_k\) 中。

### 2. 中途草稿与 B

| 字母 | 含义 |
|---|---|
| \(\widehat x_{0,k}\) | 第 \(k\) 步预测的最终干净 latent，即 pred_xstart |
| \(D_{\rm VAE}\) | 冻结的 SD MSE VAE 解码器 |
| \(I_k\) | 临时解码并裁到 \([0,1]\) 的 RGB 草稿 |
| \(Y_k\) | 草稿的灰度图 |
| \(j\in\{1,\ldots,16\}\) | 固定 \(4\times4\) 网格的 row-major tile |
| \(V_{k,j}\) | tile \(j\) 的灰度空间方差 |
| \(g_{x,k},g_{y,k}\) | 平滑灰度图的水平、垂直 Sobel 梯度 |
| \(\ell_k\) | 同一平滑灰度图的离散 Laplacian |
| \(q_{k,j}\) | tile 二阶边缘能量与一阶边缘能量之比 |
| \(\mathcal A_k\) | 当前方差最大的 8 个 active tiles |
| \(\mathcal W_k\) | \(\mathcal A_k\) 中 \(q\) 最低的两个 weakest-edge tiles |
| \(B_k\) | 当前草稿局部模糊严重度；越大越偏糊 |
| \(B_{\rm pers}\) | 九个 \(B_k\) 的平均 |
| \(b_{c,k}^{\rm gate}\) | 类别 \(c\)、检查点 \(k\) 的冻结 label-free 门阈值 |
| \(a_k\) | B 状态门，\(a_k=1\{B_k>b_{c,k}^{\rm gate}\}\) |

解码为

\[
I_k=\operatorname{clip}\left(
\frac{D_{\rm VAE}(\widehat x_{0,k}/0.18215)+1}{2},0,1
\right).
\]

对 tile \(j\)，

\[
q_{k,j}=
\frac{\operatorname{mean}_{(h,w)\in j}\ell_{k,h,w}^2}
{\operatorname{mean}_{(h,w)\in j}(g_{x,k,h,w}^2+g_{y,k,h,w}^2)+10^{-12}}.
\]

在边缘总量相近时，模糊过渡的二阶变化通常更弱，所以 \(q\) 较小。于是

\[
B_k=-\log\left(Q_{0.25}\{q_{k,j}:j\in\mathcal A_k\}+10^{-12}\right),
\]

\[
B_{\rm pers}=\frac{1}{9}\sum_{k\in\mathcal K}B_k,\qquad
\mathcal K=\{69,79,89,99,109,119,129,139,149\}.
\]

它到第 149 步模型求值后就完全可得，所以是 preterminal、predictable 的内部量。
但它没有条件期望守恒律，因此不是 e-process。

### 3. 跨尺度方向

| 字母 | 含义 |
|---|---|
| \(\bar\alpha_t\) | DiT schedule 的累计 signal power |
| \(\nu_t=(1-\bar\alpha_t)/\bar\alpha_t\) | 规范化 additive-heat 方差 |
| \(d\in\{1,4\}\) | 两个冻结 heat shift，\(\Delta\nu_d=1\) 或 4 |
| \(t_d^+\) | 最接近 \(\nu_t+\Delta\nu_d\) 的更高噪声离散时间 |
| \(\rho_{d,k}\) | \(\sqrt{\bar\alpha_{t_d^+}/\bar\alpha_t}\) |
| \(\epsilon_k\) | 当前实际 CFG 模型的四通道 epsilon 输出 |
| \(\epsilon^+_{d,k}\) | 在 \((\rho_{d,k}x_k,t_d^+)\) 上的 epsilon 输出 |
| \(s_k\) | 当前 raw VP score，\(-\epsilon_k/\sqrt{1-\bar\alpha_t}\) |
| \(\theta_{d,k}\) | 拉回当前坐标的跨尺度 score 差 |

同一规范化热坐标位置在高噪声网络输入中是 \(\rho_{d,k}x_k\)，所以

\[
\theta_{d,k}
=-\rho_{d,k}\frac{\epsilon^+_{d,k}}{\sqrt{1-\bar\alpha_{t_d^+}}}
+\frac{\epsilon_k}{\sqrt{1-\bar\alpha_t}}.
\]

两个模型输出、当前 pred_xstart、VAE 草稿及 tile 选择都在 \(Z_{k+1}\) 前完成，
因此都是 \(\mathcal F_k\)-可测的。

### 4. 操作性 Q* 与路径证据

| 字母 | 含义 |
|---|---|
| \(m_k\) | 把 \(\mathcal W_k\) 映射到 latent 后的二值 mask |
| \(w_{d,k}\) | 未缩放 whitened mean shift |
| \(K^{(0)}_{d,k}\) | 未缩放一步 KL，\(\tfrac12\|w_{d,k}\|^2\) |
| \(K_\star\) | 每尺度整条窗口总 KL 上限，固定为 2.0 |
| \(R_{d,k}\) | 到第 \(k\) 步前该尺度尚未使用的 KL 预算 |
| \(\beta_{d,k}=R_{d,k}\) | 当前 gate 可用的 allowance |
| \(\gamma_{d,k}\) | 把一步 KL 压到 allowance 的缩放 |
| \(u_{d,k}\) | 最终 whitened Q* mean shift |
| \(\delta_{d,k}\) | latent 坐标均值差，\(\sigma_k\odot u_{d,k}\) |
| \(Q^\star_{d,k}\) | 操作性同协方差 Gaussian 备择 |
| \(\Delta\ell_{d,k}\) | 一步 log likelihood ratio |
| \(E_{d,n}\) | 尺度 \(d\) 的累计路径 likelihood ratio |
| \(\pi_d\) | 固定 mixture weight，两个尺度各 \(1/2\) |
| \(E_n^{\rm BG}\) | B-gated 两尺度 mixture e-process |
| \(\alpha_e\) | baseline 总报警预算，固定为 0.10 |

定义

\[
w_{d,k}=a_k\,m_k\odot\sigma_k\odot\theta_{d,k},\qquad
K^{(0)}_{d,k}=\frac12\|w_{d,k}\|^2.
\]

若尺度映射是 identity 或 B 门未开启，则 \(w_{d,k}=0\)。否则

\[
\gamma_{d,k}=\min\left(1,\sqrt{\frac{R_{d,k}}{K^{(0)}_{d,k}}}\right),
\qquad u_{d,k}=\gamma_{d,k}w_{d,k}.
\]

当 \(K^{(0)}_{d,k}=0\) 时取 \(\gamma_{d,k}=1\)。观察本步 innovation 后更新

\[
R_{d,k+1}=R_{d,k}-\frac12\|u_{d,k}\|^2.
\]

闭门时花费为 0；第一次开门若 raw direction 足够强，可以使用全部剩余预算；若
raw direction 较弱，残余预算按同一个事前规则留给以后 checkpoint。每次用于构造
\(u_{d,k}\) 的 \(R_{d,k}\) 都在本步 innovation 前已知，而且只会下降，因此

\[
\sum_{k\in\mathcal K}\frac12\|u_{d,k}\|^2\le K_\star=2.0.
\]

最终

\[
Q^\star_{d,k}
=\mathcal N\left(\mu_k+\sigma_k\odot u_{d,k},\Sigma_k\right).
\]

它的准确语义是：当前草稿相对同类正常路径已偏糊时，在最弱的两个有效局部区域，
下一步更倾向沿冻结高噪声跨尺度方向演化，而且强度受控。它不是未经修改的全图
高噪声边际过程。

还要把“同坐标 tile”的局限说清楚：VAE decoder 有跨层卷积和非局部感受野，图像
tile 与 latent tile 不是严格的一一因果对应。把图像中的两个 weakest-edge tiles 按
归一化坐标映射到 latent，只是事前冻结的局部化启发式；它不能保证均值偏移只改变那两
块图像，也不能证明该图像块就是伪影根因。这个局限不破坏 Gaussian LR 的精确性——
(Q^\star) 本来就是以该 latent mask 定义的——但会限制其“局部模糊备择”的语义解释，
必须由独立池消融而不是公式来验证。

## 三、为什么 B 状态门之后仍是精确 e-process

给定 \(\mathcal F_k\)，\(u_{d,k}\) 已是固定向量，而
\(Z_{k+1}\sim\mathcal N(0,I)\)。同协方差 Gaussian 的密度比为

\[
\Delta\ell_{d,k}
=u_{d,k}^{\top}Z_{k+1}-\frac12\|u_{d,k}\|^2.
\]

由标准 Gaussian 矩母函数，

\[
\mathbb E_P[\exp(\Delta\ell_{d,k})\mid\mathcal F_k]=1.
\]

所以

\[
E_{d,n}=\exp\left(\sum_{r<n}\Delta\ell_{d,k_r}\right)
\]

是相对于实际 \(P\) 的正鞅。允许 \(u_{d,k}\) 随历史改变；关键只是它不能看本步
innovation。因而当前 B 门、当前 tile、当前 score gap 和当前 sigma 都可进入
\(Q^\star\)。

固定 \(\pi_1=\pi_4=1/2\) 后，

\[
E_n^{\rm BG}=\frac12E_{1,n}+\frac12E_{4,n}
\]

仍是正鞅。Ville 不等式给出

\[
P_P\left(\sup_n E_n^{\rm BG}\ge\frac1{\alpha_e}\right)
\le\alpha_e=0.10.
\]

这控制 baseline 全体轨迹的总体报警率，不是 clean-good 条件 FPR。质量功效仍要由
独立盲化终点评价否证。

## 四、投影、分块和 mixture 的严格边界

### 一般可预测投影

把 transition latent 展平为 \(D\) 维。令 \(A_k\in\mathbb R^{r\times D}\) 和
\(\lambda_k\in\mathbb R^r\) 都是 \(\mathcal F_k\)-可测的，且
\(C_k=A_kA_k^\top\) 可逆。只观察 \(Y=A_kZ\) 的精确 Gaussian LR 为

\[
L_k=\lambda_k^\top C_k^{-1}Y
-\frac12\lambda_k^\top C_k^{-1}\lambda_k.
\]

它对应完整空间均值偏移

\[
u_k=A_k^\top C_k^{-1}\lambda_k,
\]

并保持投影正交补上的 \(P\) 条件律。因此 \(\exp(\sum L_k)\) 仍是正鞅。
若 \(A_k\) 行正交归一，则 \(C_k=I_r\)。

### rank 1 不自动解决 collapse

令完整 whitened shift 为 \(w_k\ne0\)，取

\[
a_k=\frac{w_k}{\|w_k\|},\qquad \lambda_k=\|w_k\|.
\]

则

\[
\lambda_ka_k^\top Z-\frac12\lambda_k^2
=w_k^\top Z-\frac12\|w_k\|^2.
\]

形式上是一维，LR 却逐点完全相同。它没有降低
\(D_k=\|w_k\|^2\)，所以没有解决 collapse。

真正降低 \(D_k\) 只能：

1. 用事先定义的真子空间或局部 mask 丢掉一部分方向；
2. 显式令 \(u_k=\gamma_kM_kw_k\)、\(\gamma_k<1\)。

二者都会改变 \(Q\)，必须承认语义随之改变。本候选同时使用局部 mask 和 KL cap，
而不是拿“rank 1”当免费解法。

| 操作 | 是否仍有 e-process | 原因 |
|---|---:|---|
| 当前 B 门决定 \(u_k\) 是否为零 | 是 | 在 innovation 前可测 |
| 当前草稿决定 tile mask | 是 | 在 innovation 前可测 |
| 固定尺度、符号或 change-point mixture | 是 | 非负固定权重的鞅 mixture |
| 看完 innovation 后挑内积最大的 tile | 否 | 选择偏差 |
| 对多个 tile 的 \(E\) 取最大值 | 否 | 最大值不是固定 mixture |
| 动态重写已有分量的 mixture 权重 | 一般否 | 不自动满足 self-financing |
| 用 \(u'=\gamma u\) 且补偿为 \(-\gamma^2\|u\|^2/2\) | 是 | 定义了缩小后的新 Q |
| 只把旧 log-LR 乘 \(\gamma\) | 不是同一个 Q/P | 二次补偿项错误 |
| 把 B、斜率、绝对内积或平方内积叫 LR | 否 | 没有正规化密度比 |

## 五、如何真正控制高维 LR collapse

未经控制时令

\[
D_{d,k}=\|\sigma_k\odot\theta_{d,k}\|^2.
\]

在 \(P\) 下，

\[
\Delta\ell_{d,k}\mid\mathcal F_k
\sim\mathcal N\left(-\frac12D_{d,k},D_{d,k}\right).
\]

高维时典型 log evidence 会快速下沉，期望 1 只由极少数巨权重维持。
本方法的三种控制作用不同：

- B 状态门：普通状态令 \(Q^\star=P\)，不消耗预算；
- 两个 weakest-edge tiles：只保留与模糊直觉直接相关的局部子空间；
- 总 KL cap：无论 mask 后还有多少维，都强制每尺度
  \(\sum_k\|u_{d,k}\|^2/2\le2.0\)。

若 \(K_{d,k}=\|u_{d,k}\|^2/2\)，则

\[
\mathbb E_P\left[
\left.\left(\frac{E_{d,k+1}}{E_{d,k}}\right)^2\right|\mathcal F_k
\right]=e^{2K_{d,k}}.
\]

所以

\[
\mathbb E_P[E_{d,T}^2]\le e^{2K_\star}=e^4\approx54.6.
\]

固定 mixture 由平方凸性也满足相同上界。这个界比原先设想的 \(K_\star=0.5\)
明显更松，但仍是与 latent 维数无关的有限上界。为什么接受这个代价，必须由下面的
Q-side 功效审计解释，而不能只谈校准。

## 六、先过 matched-Q 功效门，避免“精确但永远不响”

在匹配的单个 Gaussian \(Q^\star\) 下，若完整使用总 KL \(K\)，终点 log-LR 为

\[
\log E_T\sim\mathcal N(K,2K).
\]

若没有 mixture 稀释，\(K=0.5\) 越过 \(\log 10\) 的终点概率约 3.6%。本方法还把
两个尺度各赋一半权重；若另一个分量暂时为 \(E=1\)，匹配分量需要达到

\[
\frac12E_{\rm matched}+\frac12\ge10
\quad\Longrightarrow\quad E_{\rm matched}\ge19.
\]

于是 \(K=0.5\) 的解析参考功效只有约 0.73%。这意味着原先“二阶矩很漂亮”的 0.5
实际上会重演 likelihood-ratio collapse 的另一种形式：不爆炸，但也几乎不可能触发。

因此在任何真实 screen 之前冻结一个纯 Gaussian 必要门：

- 两个尺度用正交标准 Gaussian 坐标表示；
- 数据依次来自每个匹配 \(Q_d^\star\)；
- 匹配分量在第一次合法 gate 使用完整 K，另一坐标保持 P；
- 固定两尺度各半 mixture、\(\alpha_e=0.10\)；
- 400,000 次，PCG64 seed 2026082808；
- 两个 matched component 的 anytime power 最小值必须至少 0.30。

结果是：

| 设计 | 最小 matched-Q anytime power | 判定 |
|---|---:|---|
| \(K_\star=0.5\) 的弱设计复核 | 0.007933 | 事前否决 |
| \(K_\star=2.0\) 的冻结设计 | 0.320155 | 通过必要门 |

\(K=2\) 下，把另一分量固定为 1 的解析终点参考为 0.3184，与模拟一致。这个门只说明
“理想地用满预算且方向匹配时不至于先天无功效”，不说明真实 B gate 会打开，也不说明
真实 theta 有足够 raw K。因此在任何标签连接前还要在至少 60 条路径上做第二个
label-free 门：

- 每尺度至少 50% 路径出现过一次 B gate；
- gate-open 路径中至少 50% 实际使用到 \(K\ge1.5\)。

任一失败就 STOP，不得打开标签或事后调 K、alpha、门阈值。这里的
\(\alpha_e=0.10\) 始终是总体干预预算，不是好图 FPR。

## 七、为什么只固定两个尺度

冻结 DiT-250 schedule 决定：

- \(\Delta\nu=1\) 在前四个 B checkpoint 是 identity，后五个真正移动；
- \(\Delta\nu=4\) 只在第一个 checkpoint 是 identity，后八个移动。

这只看 schedule，没有看图、标签或分数。固定 \(\{1,4\}\) 各半权重，覆盖较短和较长
的结构生命周期，同时把 family 限在两个分量。确认池不能再选“最好尺度”或加入第三个
尺度救结果。

## 八、四个输出必须分开

| 输出 | 对象 | 精确 e-process | 作用 |
|---|---|---:|---|
| \(B_{\rm pers}\) | 九步草稿模糊均值 | 否 | 纯内部 heuristic anchor |
| \(E^{\rm BG}\) | B 门、B 局部 mask、跨尺度 Gaussian LR | 是，针对 \(Q^\star/P\) | 新主候选 |
| \(E^{\rm no\ gate}\) | 去掉状态门，其余相同 | 是，针对另一 Q/P | 固定机制消融 |
| \(S_{\rm slope}\) | 九个 \(B_k\) 对时间的 OLS 斜率 | 否 | 边缘形成诊断 |

底层 \(E_k\) 是鞅；running-max log E 是用于排序的回顾性分数，不是鞅。在线规则仍是
首次 \(E_k\ge10\)。

增量价值要单独检验：

\[
\Delta_{E-B}=\operatorname{AUC}(E^{\rm BG})
-\operatorname{AUC}(B_{\rm pers}),
\]

\[
\Delta_{\rm gate}=\operatorname{AUC}(E^{\rm BG})
-\operatorname{AUC}(E^{\rm no\ gate}).
\]

只有第一项大于 0 且预注册的配对 seed-block 单侧置换检验通过，才能说路径证据增加了
B 以外的质量信息。若第二项不大于 0，就不能说 B 状态门提高了功效。

## 九、阈值不使用质量标签

每个最终入选类别用 20 条、与确认 seed 不重叠的 label-free calibration 路径：

- 每个 checkpoint 的 B 状态门取第 17 个升序 \(B_k\)，严格大于才开启。在一个固定
  checkpoint、无 ties 的 exchangeability 理想下，fresh path 严格越过它的 rank
  概率是 \(4/21\)。这不是 across-time 报警保证。
- 纯 B 报警取第 19 个升序 \(B_{\rm pers}\)，严格大于才报警；fresh path 的边际
  触发上界为 \(2/21\)。
- e-process 不用经验分位数，固定在 \(E_k\ge10\) 首次报警。

这些都是总体干预预算，不是好图 FPR。视觉标签只能在阈值、trace、源代码和 reviewer
可靠性全部锁定后，由另一个 evaluator 连接。

## 十、与现有 event-rich v3 的协议冲突

现有 event-rich scientific v3 已冻结 B/C 两个 co-primary，并分别按 B 风险和 C 风险
选择类别。本文的 B/E family 与它不兼容，不能悄悄加入 v3。

截至本文冻结候选时，真实 event screen 仍为 0，因此只有两个诚实选择：

1. 在任何真实 screen 采样前显式冻结 scientific v4：把 blur 线改为 B/E，重写
   selector、Holm family、event gates、label-free products 和 evaluator；C 必须降为
   明示诊断或进入独立 family。
2. 保留 v3 不动，让 E 进入以后一个 calibration 和 confirmation seeds 都独立的新池。

在 v3 按 B/C 选出的类上事后算 E，只能叫探索，不能叫 B/E confirmation。本方法锁
因此标记为 execution_ready=false。

## 十一、独立池的硬否证门

打开候选值前，外部盲评层至少要有 15 个 clear blur/soft-fusion bad、60 个 clean-good、
覆盖至少 3 类，并通过另一个冻结 reviewer 可靠性门。事件不足就停止。

事件门通过后只检验两个主候选，并对两个原始 class-matched permutation p-value 做
Holm 校正：

- 纯 B：AUC 至少 0.75、Holm \(p<0.05\)、冻结阈值处 TPR>FPR；
- B-gated E：AUC 至少 0.70、Holm \(p<0.05\)、\(\alpha_e=0.10\) 处 TPR>FPR，
  且至少 3 个 blur/fusion 正例越界。

以下任一条直接否证 v1：

1. gate、tile、方向、尺度、权重或缩放看到了本步 innovation 或未来状态；
2. 观察 hook 改变 baseline latent、RNG、transition draw 或 endpoint pixel；
3. 任一尺度总 KL 超过 2.0；
4. matched-Q synthetic power 门或后续 label-free gate-open/K-utilization 门失败；
5. 独立池中 E 的 TPR 不大于 FPR，或 AUC/p-value 门失败；
6. 失败后在同一确认池换 tile 数、窗口、尺度、符号、通道、聚合或阈值。

## 十二、当前实现与下一工程桥

label-free 核心：
[observe_dit_blur_focused_eprocess.py](../experiments/observe_dit_blur_focused_eprocess.py)

冻结候选数值：
[dit_blur_focused_eprocess_v1.json](../experiments/configs/dit_blur_focused_eprocess_v1.json)

合成测试：
[test_observe_dit_blur_focused_eprocess.py](../tests/test_observe_dit_blur_focused_eprocess.py)

CPU 合成检验覆盖：

- sharp/blur 的 B 顺序；
- Gaussian LR 与直接 Normal 密度比一致；
- 保留原范数的 rank-1 重写与完整 LR 相同；
- KL cap 确实改变 \(Q^\star\)；
- Monte Carlo e-value 与固定 mixture 均值接近 1；
- 看完 innovation 后取最大分量明显失准；
- 每尺度总 KL 不超过 2.0；
- 弱 \(K=0.5\) 设计的 matched-Q power 被事前否决，\(K=2.0\) 通过 0.30 门；
- label-like poison input 被拒绝。

replay adapter 已实现于
[replay_dit_blur_focused_eprocess_inputs.py](../experiments/replay_dit_blur_focused_eprocess_inputs.py)。
它只从既有 trace 加载九个 checkpoint 所需的 state、pred_xstart、sigma、innovation
和 schedule 数组；不会加载 final_latents、decoded_images 或 endpoint PNG。它重新做
当前与两个 shifted DiT forward、临时解码草稿，并要求所有 model/VAE 观察前后的 CUDA
RNG hash 完全相同。

无标签阈值校准器已实现于
[calibrate_dit_blur_focused_eprocess.py](../experiments/calibrate_dit_blur_focused_eprocess.py)。
它只从既有 preterminal visual product 加载行标识、九个 checkpoint 轴和
`decoded_local_blur_severity`；同一个 NPZ 中共存的 ResNet/其他诊断 track 不会被索引，
CSV 中的 endpoint 路径也不会被打开。它要求每类恰好 20 条、所有类共享同一 20-seed
cohort，然后冻结逐类逐步第 17 个升序门阈值和逐类第 19 个升序纯 B 阈值，并把源
manifest、time-series hash、实际加载数组 hash 和 calibrator source hash 一起写入
自校验 JSON。这个产物依然只是 label-free calibration，不是质量评价。

目前尚未对任何真实 trace 运行该 adapter。真实运行前仍必须冻结兼容的 scientific v4
或另建独立池，并做有无观察路径的 latent/RNG/endpoint pixel 审计；不能因为 adapter
已经写好就绕过协议冲突。

## 十三、最值得下一池验证的主候选

\[
\boxed{
E_k^{\rm BG}
=\frac12E_k^{(\Delta\nu=1)}
+\frac12E_k^{(\Delta\nu=4)}
}
\]

每个分量只在当前 \(B_k\) 超过 label-free 状态阈值时开启，只作用于当前两个
weakest-edge active tiles，并受每尺度 \(K_{\rm total}=2.0\) 约束。

选择它不是因为公式漂亮，而是它让各对象各司其职：

- B 指出“现在像不像正在形成模糊或软融合”；
- cross-scale 方向提出“若继续依赖高噪声支撑，下一步会往哪里走”；
- Gaussian LR 判断实际 innovation 是否连续支持这个事前备择；
- Ville 与总 KL 分别约束报警率和高维权重退化；
- 独立盲评只负责最后否证质量关系，绝不进入方法。

若 E 不能显著超过纯 B，路线应收缩为“B 是便宜的内部 preterminal blur detector”；
若 E 的 TPR 都不高于 FPR，就应放弃这版跨尺度质量解释，而不是继续扫描更多差分。
