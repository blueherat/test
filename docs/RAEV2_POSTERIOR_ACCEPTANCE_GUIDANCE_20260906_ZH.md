# RAEv2：具有独立 KL 证书的 posterior acceptance guidance

**状态：已退出，原因为研究范围错误，不是 FID 检验失败。** 用户指出本方法只是生成后拒绝采样，不是其要求的 guidance。四个 GPU 任务已停止，共记录 832 个 proposal、321 个接受结果，尚无完整配对 cohort，也未计算 FID；不继续此方向。下文保留的 KL 重加权推导在其条件下成立，但它没有解释 RAEv2 的 guidance 误差，也没有导出采样轨迹中的引导机制，不应冒充本任务所需的理论贡献。正文机制与固定架构在 4K 特征提取期间写定，结果于第 9 节追加。此前两条 radial 候选在 seed 066 的 1K 图像 FID 为 38.4084 / 38.4086，均略差于对应 baseline 38.3978。

本方法在冻结的官方 RAEv2 图像 proposal 之后，用一个拟合的线性真实/生成分类器决定是否接受图像。训练分类器和被拒绝的完整 proposal 都计入成本。它属于生成后的分布引导，有额外训练与采样开销；没有时间窗、外推幅度、温度或接受率搜索。

## 1. 固定设计与数据角色

设类别数 K=1000，目标类别先验 π_y=1/K，真实分布和冻结 proposal 分别为

\[
p(x,y)=\pi_y p_y(x),\qquad q(x,y)=\pi_y q_y(x).
\]

图像进入冻结 raw DINOv3-L 的最后一层 CLS，记其 unit-L2 特征为 φ(x)。本轮固定使用这一特征、一个共享的线性 logistic head，不使用类别输入，不比较其他层、pooling、网络宽度或正则强度。特征提取和归一化的数值实现须一致；理论只需 ‖φ(x)‖≤1，零向量可定义为零。

\[
D(x)=\sigma(w^\top\phi(x)+b),\qquad
(\widehat w,\widehat b)=\arg\min_{w,b}
\left\{\sum_{i=1}^{2000}\operatorname{BCEWithLogits}
\big(w^\top\phi(x_i)+b,t_i\big)
+(1/2)\|w\|^2\right\}.
\]

真实标签为 1，生成标签为 0；训练数据为 real A1000 和 official seed 066 的 1000 张图，均每类一张。使用 **sum BCE**，bias 不罚。这是 w∼N(0,I) 与不惩罚 intercept 的 MAP：对任意固定 unit 特征，先验随机 logit wᵀφ 的方差等于 1。这里的“1”是预先固定的建模约定，有明确尺度含义，但没有唯一最优性的定理；不能把它称为完全没有建模选择。用 mean BCE 加同样 .5‖w‖² 会改变该先验相对数据的强度，不等价。训练中两类都存在，ridge 保证 w 有界，目标也在 |b|→∞ 时发散，因此存在有限最优解；还须记录实际优化收敛。

训练完成后，只用 train fake 冻结

\[
m=\frac1{1000}\sum_{i=1}^{1000}D(x_i^{q,\mathrm{train}}),\qquad
f(x)=\log D(x)-\log m.
\]

独立检验使用 real C1000 和 official seed 067 的 1000 张图，仍各类一张。w、b、m、特征处理、置信界规则必须在查看该检验函数值前冻结。C 不用于选 checkpoint、特征、regularizer、temperature 或 m。验证 AUC、BCE 可报告为诊断，不能代替下文的 C(f) 证书。

## 2. 密度重加权直接给出正向 KL 的变化

对任意有界可测 f，令 q_f=q e^f/Z，Z=E_q e^f。在 p≪q 且 KL(p‖q)<∞ 的条件下，

\[
\begin{aligned}
\Delta\mathrm{KL}
&=\mathrm{KL}(p\|q_f)-\mathrm{KL}(p\|q)\\
&=\log\mathbb E_qe^f-\mathbb E_pf\\
&\le \mathbb E_q(e^f-1)-\mathbb E_pf=:C(f).
\end{aligned}
\]

等式只用 Radon–Nikodym 比值；上界只用 log z≤z−1。因此，对**已经冻结的任何分类器**，C(f)<0 都足以证明改善，不需要先证明 D 接近 Bayes classifier 或近似准确密度比。把一个分类器训练得能区分真假，本身并不保证这个条件。

记 a=E_q D。代入 train-only m 得到可用独立均值估计的函数

\[
\boxed{C_m=\frac{a}{m}-1-\mathbb E_p\log D+\log m.}
\]

对应准确的 pooled 重加权变化是 log a−E_p log D，两者差为

\[
C_m-(\log a-\mathbb E_p\log D)
=\frac am-1-\log\frac am\ge0.
\]

所以训练 m 的估计误差会增加保守程度，但不会破坏上界。m 只服务于证书，实际接受概率不除以 m。

## 3. 逐类接受保持类别先验，Jensen 给出同一充分证书

部署对每个 y 独立反复生成 x∼q_y，抽取独立 U∼Uniform(0,1)，当 U<D(x) 时接受，并结束该类。接受后的准确条件分布为

\[
q_{D,y}(x)=\frac{q_y(x)D(x)}{Z_y},\qquad Z_y=\mathbb E_{q_y}D.
\]

每个类别各输出一张，因此实际 joint 分布是 q_D(x,y)=π_y q_{D,y}(x)。它不同于把 pooled q(x,y) 用单一 a 归一化的分布；后者通常会改变类别比例。对实际逐类目标，有

\[
\begin{aligned}
\Delta\mathrm{KL}_{\mathrm{joint}}
&=\sum_y\pi_y\log Z_y-\mathbb E_p\log D\\
&\le\log\!\left(\sum_y\pi_yZ_y\right)-\mathbb E_p\log D\\
&=\log a-\mathbb E_p\log D\le C_m.
\end{aligned}
\]

因此一个 pooled 的 **C_m<0 证书也足以保证保持类别先验的平均条件 KL 改善**。不必从每类一张的数据估计 1000 个 log Z_y；但不能据此宣称每个单独类别均改善，也不能直接改称无标签 image marginal KL 必然改善。后者的 KL 链式分解还有标签后验项；本证书的准确目标是联合分布或等价的平均条件 KL。

接受概率就是 D∈(0,1)，无需 MH mixing、density-ratio envelope、burn-in 或人工接受率。保持 proposal 和接受 RNG 相互独立，每次 rejection 使用新的 proposal。有限参数下 d_min>0，逐类循环几乎必然终止，期望总 proposal 数为

\[
\mathbb E N_{\mathrm{proposal}}=\sum_{y=1}^K Z_y^{-1},
\]

一般不等于 K/a；K/a 只是由 Jensen 给出的下界。不能达到拒绝次数上限后强制接受，否则目标分布改变；资源结束时可保存未完成状态并继续。用于最终 FID 的接受序列使用独立于 train/test 的 proposal 与接受随机数，报告 proposal 总数、完整网络调用、decoder/feature 成本和 wall time。

## 4. 特征不可逆不损坏这个 KL 机制

令 V=φ(X)。因为接受权重只依赖 V，q_D(x|v,y)=q(x|v,y)。在相关 KL 有限、正则条件分布存在时，链式分解给出

\[
\mathrm{KL}(p_{X,Y}\|q_{D,X,Y})
-\mathrm{KL}(p_{X,Y}\|q_{X,Y})
=\mathrm{KL}(p_{V,Y}\|q_{D,V,Y})
-\mathrm{KL}(p_{V,Y}\|q_{V,Y}).
\]

这不是把特征距离经 decoder 逆映射提升到像素距离，而是同一个图像密度比的恒等式。DINO 可以不是单射；特征纤维内部的条件生成错误保持原样，因此该方法不能纠正 φ 看不见的缺陷，但也不需要为 KL 差异证明逆 Lipschitz。

这一点只为作用机制提供理论范围，不预设本轮 unit-L2 CLS 保留了有用的真实/生成偏差信号。若 p_V=q_V，任何仅依赖 V 的 reweighting 都没有可利用的总体证据；拟合误差仍可能产生看似有效的训练分类器。

## 5. 理想 posterior 给出 harmonic 目标，而不是直接还原 p

先考虑不带类别的 image 或 feature 分布 P、Q。相等的真/假训练先验下，理想 logistic posterior 为

\[
D^*(v)=\frac{P(v)}{P(v)+Q(v)},\qquad
Q_D(v)\propto\frac{P(v)Q(v)}{P(v)+Q(v)}.
\]

这是一种 harmonic 重加权，并非 Q·P/Q=P。它以有界 posterior 自然限制单次重加权，避免为 odds 估计全局 envelope；此处并未另加温度。若有限线性 head 无法表达该 posterior，oracle 结论不自动适用，仍由独立 C_m 检验承担验证。

在 oracle 下，a=∫PQ/(P+Q)≤1/2，而且 Jensen 给出 E_P log D*≥−log2，因此 log a−E_P log D*≤0。当 P≠Q 时 a<1/2，故上界严格为负。对我们的 pooled feature oracle，此结论经第 3 节也保证平均条件 KL 不增；但实际每类 q_y(x)D*(φ(x))/Z_y 不是逐类的 harmonic p_yq_y/(p_y+q_y)，不能混用两个 oracle。

神经判别器辅助 rejection 已有明确先例：[Discriminator Rejection Sampling](https://arxiv.org/html/1810.06758v3) 基于判别器 odds 并处理 envelope/实际接受率问题。[Resampled Priors for Variational Autoencoders，§3](https://proceedings.mlr.press/v89/bauer19a/bauer19a.pdf) 直接使用有界 learned acceptance，目标为 proposal×acceptance/Z。本轮不能声称首次用判别器筛选生成图。待检验贡献范围是 RAEv2 的固定 posterior 结构、类别先验保持与独立有限样本 KL 证书的组合；上述密度恒等式和 rejection 原理本身是经典机制。

## 6. 解析范围来自固定特征与权重，不手动 clip

冻结 r=‖w‖，令

\[
\ell_-=b-r,\quad\ell_+=b+r,\quad
d_-=\sigma(\ell_-),\quad d_+=\sigma(\ell_+).
\]

则 log D∈[log d_-,log d_+]，e^f=D/m∈[d_-/m,d_+/m]。对同类的独立 real/fake pair，定义

\[
U_y=\frac{D(X_y^q)}m-1-\log D(X_y^p)+\log m,
\qquad \widehat C=K^{-1}\sum_yU_y.
\]

准确的全局包络为

\[
L=\frac{d_-}m-1-\log d_++\log m,\qquad
H=\frac{d_+}m-1-\log d_-+\log m,
\]

\[
\boxed{R=H-L=\frac{d_+-d_-}m+\log d_+-\log d_-.}
\]

这些界对所有输入成立，不从验证样本的最小值/最大值估计范围。log D 用稳定的 log-sigmoid 计算，避免数值下溢；不通过手动 clipping 改变 D。若数学上 R=0，D 为常数且 m=D，C_m=0，接受后的分布恒等；不能把浮点舍入产生的微负均值当作严格改善。

## 7. 每类一张：独立异分布的有限样本证书

条件于已经冻结的训练数据与模型，假设不同类的验证 pair 相互独立，类内样本分别服从规定的 p_y、q_y。U_y 通常不具有相同分布，但

\[
\mathbb E\widehat C=\frac1K\sum_y\mathbb E U_y=C_m.
\]

所以统计总体是**均匀类别先验的平均条件图像分布**。固定完整类别列表没有把类别重新 iid 抽样；并不需要这样做。真实图必须是规定类内抽样框的随机样本，生成图使用独立 proposal 随机性。文件互不重复本身不足以证明独立抽样；如抽样框仅为某个有限数据子集，证书首先针对该抽样框。不能把 1000 个固定经验原子直接当成与连续 q 具有有限 KL 的真实总体。

### 7.1 Hoeffding

对 n=1000 个独立、范围长度均≤R 的 U_i，任意预定 δ∈(0,1)，

\[
\boxed{C_m\le \widehat C+
R\sqrt{\frac{\log(1/\delta)}{2n}}=:\operatorname{UCB}_{\mathrm H}(\delta)}
\]

以至少 1−δ 的概率成立。独立异分布的形式来自 [Hoeffding 1963，Theorem 2](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)。若所有 real 与 fake 也分别独立，直接把两组均值作为 2n 个有不同 range 的独立项，可得到更紧的同类界：将上式余量替换为

\[
\sqrt{\frac{\log(1/\delta)}2
\left(\frac{R_q^2}{n_q}+\frac{R_p^2}{n_p}\right)},\quad
R_q=(d_+-d_-)/m,\quad R_p=\log d_+-\log d_-.
\]

配对版本不需要 real/fake 在同一 pair 内独立，只有跨 pair 独立的要求，故实现更保守也有效。

### 7.2 严格的 empirical Bernstein

定义 s_U²=Σ_i(U_i−Ĉ)²/(n−1)。[Maurer–Pontil 2009，Theorem 11](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf) 明确针对 independent、无需 identically distributed 的 [0,1] 随机变量。把 U 缩放到 [0,1] 后得到

\[
\boxed{C_m\le\widehat C+
\sqrt{\frac{2s_U^2\log(2/\delta)}n}
+\frac{7R\log(2/\delta)}{3(n-1)}
=:\operatorname{UCB}_{\mathrm{EB}}(\delta).}
\]

这不是将该论文只适用于 iid 的 Theorem 4 直接套到分层样本。Theorem 11 使用的 pairwise sample variance 恰等于上述 n−1 分母的方差；类间均值差异进入该方差，使界保守，不能因为每类只有一个样本便声称能够单独估出所有类内方差。训练决定的 R、m 在条件于训练后固定，因此允许用于该界。

### 7.3 正式判定与失败解释

正式准入使用事先冻结的单侧上界是否小于 0；例如固定 δ=.05 的 EB 界。若希望 Hoeffding 或 EB 任意一项通过即可准入，必须给两项各分配 δ/2，再取两 UCB 的较小值；不能对各自 δ=.05 的两界事后取最有利者仍称总错误率 .05。若追加其他界或复测，同样要记录相应的错误概率预算与独立数据角色。

解析 R 可能较宽，1000 pair 下即使点估计 Ĉ<0，严格 UCB 仍可能≥0。这表示当前样本量和固定范围未证明改善；不是实际 C_m 已被证明非负，也不能换成正态近似、AUC>0.5 或训练集负值来冒充严格证书。正态近似可以明确标作 diagnostic。若当前独立检验失败，不据其结果回调 critic；本方法尚无图像 FID，不提前把一次失败解释成所有 acceptance 方法无效。

## 8. 理论边界与最终图像检验

有限参数给出 d_->0，因此实际 q_D 保留 q 的支持，但不能创造 q 原来缺失的模式。KL 改善的解释以 p≪q、KL(p‖q)<∞ 为前提；C_m 是有界函数期望，能在独立样本上检验，却不能仅凭这个检验建立支持覆盖或有限 KL。若原 KL 无穷，不能把恒等式写成有意义的 ∞−∞ 改善。证书不要求在验证集上估计像素密度，但也没有消除这个总体前提。

即使 KL 证书严格成立，也没有普适的 FID 单调定理。一个有限、正支持的反例：在位置 (−2,−1,0,1,2) 上，

\[
p=(0,1/4,1/2,1/4,0),\quad
q=(1/32,1/8,11/16,1/8,1/32),\quad
D=(.1,2/3,8/19,2/3,.1).
\]

两分布原均值都为 0、方差都为 .5，按恒等特征定义的 Gaussian-moment FID 为 0。接受后 KL(p‖q_D) 从 0.1873467 降至 0.0512322，准确归一化的 C_m=−0.1361146，但输出方差成为 0.4145127，FID 增至 0.0040044。这是 CPU 直接枚举的数学边界，不是 RAEv2 实验。它说明理论应证明机制能改善一种真实分布误差，同时把具体 FID 目标交给真实图像检验。

本轮可证伪链条是：固定 critic 是否学到独立可验证的概率重加权信号；固定接受规则的样本是否准确实现 q_yD/Z_y；在报告完整成本、保持每类一张的独立图像评测中，FID 是否实际达到预定改善。任一环节没有证据时都按实际状态记录，不把前一环节的数学保证改称后一环节已经成功。

## 9. 首次独立验证：固定模型通过，真实 FID 未测

固定结果位于 [posterior_cls_probe_v1/summary.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/posterior_cls_probe_v1/summary.json)，逐类观测见 [heldout_per_class.csv](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/posterior_cls_probe_v1/heldout_per_class.csv)。源代码与 request 已存档；正式 gate 是单一 EB、单侧 α=.05，Hoeffding 和正态近似仅作并列诊断，没有取最有利的界。

| 项目 | 固定结果 |
|---|---:|
| n | 1000 个完整类配对 |
| ‖w‖、b | 16.6417271875、1.1545403364 |
| train-only m | 0.37949835069551946 |
| 优化 gradient ∞-norm | 9.6113×10⁻⁸ |
| Ĉ | −0.3347951835493776 |
| s_U²、SE | 0.1391403421056933、0.0117957764520 |
| 解析 range R | 18.12224404116305 |
| 预定 EB UCB(.05) | **−0.14661417986477235** |
| Hoeffding UCB(.05)，仅诊断 | +0.3665768998593139 |

独立审阅直接从 CSV 重算全部 U_i、样本方差、解析范围和 EB 上界，与存档值逐项一致；probe SHA256 与冻结记录一致。运行代码先写入 probe 与冻结记录，之后才加载检验特征；审阅没有重新训练。即使把归一化允许的数值半径保守扩为 1+10⁻¹⁰，上界也仅变为 −0.1466141798504，结论远离浮点误差边界。

这个上界针对第 7 节规定的类内抽样总体。实际 C 的 [selection.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/heldout_clean_c_current_fp32/selection.json) 记录：对每类，从完整 Packed ImageNet train 排除 A、原 B 的全部 2048 个去重行后均匀抽取一张，seed=202609064，不按模型损失、图像、latent 或 FID 选样。因此首先得到的是剩余总体 p_rest 的证书。

若要把目标精确扩回完整有限训练集，记每类排除比例 e_y=n_excluded,y/N_full,y，且 p_full,y=(1−e_y)p_rest,y+e_y p_excluded,y。固定 D 的 log-range R_p=15.4871870203 给出确定性转移界

\[
C_{\mathrm{full}}-C_{\mathrm{rest}}
=\frac1K\sum_y e_y
\left(\mathbb E_{p_{\mathrm{rest},y}}\log D
-\mathbb E_{p_{\mathrm{excluded},y}}\log D\right)
\le\underbrace{\frac1K\sum_y e_y}_{\bar e}\,R_p.
\]

因此使用 UCB_full=UCB_rest+ē R_p；这是已知抽样框的确定性 slack，不需要再分配 α，也不改变 m 或 critic。它不能反过来证明该有限数据集代表所有现实图像；有限 KL/支持条件仍按第 8 节保留。

独立 CPU 元数据审计核对完整 Packed train 的 1,281,167 行：976 类各排除 2 行，24 类各排除 4 行；每类总数范围 732–1300，逐类 total−excluded 与冻结 selection 中的 eligible 数完全相等。准确结果为

\[
\bar e=0.001604484310399357,\quad
\bar eR_p=0.024848948586331722,
\]

\[
\boxed{\operatorname{UCB}_{\mathrm{full}}
=-0.12176523127844063<0.}
\]

因此完整 class-balanced 有限训练集的证书在补偿排除偏差后也通过。核查用 int64 class-count 数组 SHA256 为 `20aa5c22f74dd8111124e4f7ab7f320c3098d7ff043bb1e4a164ef6ffdd2c436`，excluded-count 数组为 `fb53fdaf6c3fe08a3c2cc3b6f3a2af525dd420895d9baff7060cea95ecb0d894`，selection 文件为 `5780251ef1c53c6ddee7bee53f76b11da36be6e8e06116c06fbf526f59a477a8`。这项通过允许继续检验已固定接受机制，不等于已经达到图像 FID 目标。
