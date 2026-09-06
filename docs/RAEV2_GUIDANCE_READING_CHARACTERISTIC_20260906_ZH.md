# Characteristic Guidance：干净预测共识与 Gaussian 收敛条件

日期：2026-09-06。**这篇论文提供了一个不同于同点输出外推的结构：在不同查询上对齐两条场所指向的干净端。** 本轮将其转换到 RAEv2 参数化，独立推导了 Gaussian 特例的收敛条件；现有 792 个方向导数条目没有违反该必要条件。它保留了一条可继续研究的机制线索，尚不构成新 RAEv2 sampler 或质量结论。

## 原文与实现核查

Zheng、Lan，[Characteristic Guidance: Non-linear Correction for Diffusion Model at Large Guidance Scale，ICML 2024](https://proceedings.mlr.press/v235/zheng24f.html)。采用 [27 页会议正式 PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/zheng24f/zheng24f.pdf)，阅读正文 §§1–9、推导附录 A–E、构造与实验附录 F–I；阅读展示附录 J 的图注，不将图注阅读称为全部图像的视觉质量审查。正式 PDF SHA 为 `4c1ee6a74c066831eba50b0e4b09b063a9ce9fa5ee056a199501b454351ed453`。检索所得 arXiv v3 是 2024 年 1 月的旧版，章节和 latent 投影向量写法不同，本笔记以会议版为准。

论文的机制是先指定 clean 端的 power target `p_F^w p_B^(1−w)`，再处理其加噪路径。score 的 Fokker–Planck 方程有非线性项，同一点的 score 线性组合一般不再满足该路径方程。作者在三条理想 score 都满足 harmonic 假设 `Δs=0` 时，通过特征线构造两个移位查询，恢复这一特定端点目标的加噪 score。干净端的边界最好以 score 表述；零噪声处 ε 本身退化为零，不能仅凭 `ε(0)` 的平凡等式确定目标。[§§3–5、附录 C–D](https://raw.githubusercontent.com/mlresearch/v235/main/assets/zheng24f/zheng24f.pdf)

这保留了一个正向设计原则：若目标是某个干净分布，先从这个目标推出整条路径，而不是分别在每个噪声层定义 power density。但指定 `p_F^w p_B^(1−w)` 仍需归一化，且它不自动等于真实数据分布。论文的精确结论还要求适用的 score、harmonic 条件和精确隐式解；正则投影和有限迭代不是这些条件的一部分。

实验边界清楚：CIFAR-10 表 1 中 DDIM 最低 CF/CH FID 为 `4.52/4.46`，DPM++2M 最低为 `3.33/3.35`；较大 guidance scale 上的改善不能当作优于最佳常数基线。ImageNet latent 实验正文 §6.4 和 Figure 6 明确报告 CH 的 FID 更高、IS 更好，主要是有效控制增强。本任务不据此声称存在同成本 5% 收益。附录 I 对 CIFAR 采用 RMSprop 求根、对 ImageNet 采用 Anderson acceleration；迭代容差、步长、投影、历史长度均有具体设置。全文所读相关实验段未给出可直接复用的 RAEv2 成本结论。

当前作者 [WebUI 实现](https://github.com/scraed/CharacteristicGuidanceWebUI/tree/d88752ddcaef68ff2712ccdccb55dfc8f0effffb)固定到 commit `d88752ddcaef68ff2712ccdccb55dfc8f0effffb`。已读核心 `CharaIte.py`、包装接口及 README：代码采用带正则的子空间最小二乘、随噪声变化的正则目标、Anderson 更新及未收敛时的零修正。它是作者正式链接的工程扩展，不能当作没有参数的原始 `P=I` 理论算法，也不能把这些选择搬进本项目。没有安装或运行下载的代码。

## RAE 参数化下的精确代数

以下转换与收敛分析为本次独立推导。令 `α=1−t`，RAE 的 bridge 为 `z=αX+tε`，`w=1.78`、`g=w−1=.78`。对任意有限 clean 预测 H 定义

\[
\epsilon_H(z,t)=\frac{z-\alpha H(z,t)}{t}.
\]

先不使用论文的经验投影，即 `P=I`。将原式转换到 RAE 坐标得到

\[
\delta=t\{\epsilon_B(z+w\delta,t)-\epsilon_F(z+g\delta,t)\}.
\]

这个转换可先用 `ρ=√(α²+t²)` 将 RAE 状态变为 OU 状态 `z/ρ`；同时 `σ_OU=t/ρ`、`δ_OU=δ/ρ`，所以乘回 ρ 后正好得到上式，不缺额外噪声比例。

右侧化简为 `δ+α(F_shift−B_shift)`。因此对 **0<t<1**，它严格等价于

\[
F(z+g\delta,t)=B(z+w\delta,t).
\]

两个输入满足 `w(z+gδ)−g(z+wδ)=z`；系数相加为一，故最后 guided clean 为

\[
wF(z+g\delta,t)-gB(z+w\delta,t)
=F(z+g\delta,t)=B(z+w\delta,t).
\]

这里的共识是不同查询上的预测相同，不是在同一点强行缩小原 gap。这个代数对有限 H 也成立，但密度路径的解释不随之自动成立。若将 I 换成正交投影 P，固定点处只能推出投影共识 `P(F_shift−B_shift)=0`，不能继续声称完整 F/B 相等。任意线性算子 P 不具备这个推论；本轮不通过挑 P 补救。

## 一个可证明的 Gaussian 收敛特例

假设两支真是原 bridge 下满秩 Gaussian prior `N(μ_H,Σ_H)` 的精确 posterior mean，并且干净 power target 的 precision

\[
P_*=w\Sigma_F^{-1}-g\Sigma_B^{-1}\succ0.
\]

它们的 clean Jacobian 为常数对称正定矩阵

\[
J_H=\alpha\Sigma_H(\alpha^2\Sigma_H+t^2I)^{-1}.
\]

定义共识方程的矩阵 `A=wJ_B−gJ_F`。由

\[
wJ_F^{-1}-gJ_B^{-1}
=\alpha I+\frac{t^2}{\alpha}P_*\succ0
\]

及正定矩阵取逆反序，可得 `J_F < (w/g)J_B`，即 `A≻0`。这个论证**不要求两协方差可交换**。又因为 `0<J_B<I/α` 和 `J_F>0`，有 `0<αA<wI`。原 fixed-point 映射的 Jacobian 为

\[
T'(\delta)=I-\alpha A,
\qquad\lambda(T')\in(1-w,1)=(-.78,1).
\]

在这个 Gaussian affine 特例中，它是全局线性收敛的映射。共识值正好等于 power target 的 Gaussian posterior mean，而不是另造一个局部 surrogate；这一点已用非交换二维协方差独立核验。

但不存在由这条结论给出的统一小迭代数。固定算例在 `t=.99` 的最慢特征值为 `0.9998912`，shift 范数约 `34.51`；算法仍收敛，却可能极慢且查询远离原状态。t=1 时原方程是 `δ=δ`，没有唯一 shift，也不能从此直接断言两头共识。若 Gaussian prior 保留 RAE 已有公共仿射约束，应限制到共同切空间求逆；ambient A 只需半正定，法向 fixed-point 特征值可以为 1，不能套用满秩唯一根结论。

正 A 也不是目标可归一化的充分条件。固定反例 `Σ_F=4I,Σ_B=I` 给出 `P*=−.335I`，而 `t=.1` 时 A 仍为正、迭代仍收敛。这防止将一个看起来稳定的求根器当作合法目标分布的证明。

## 固定缓存检验：必要方向未被反驳

GPU 前向和 VJP 均复用此前已经完成的 `density_transport_audit_seed202609074_v2/per_sample_step.csv`。它保存同一 detached 原始 gap 方向 v 上的 `q_full=vᵀJ_Fv` 与 `q_base=vᵀJ_Bv`。在计算新组合前，先写入 [request.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/characteristic_gaussian_audit_v1/request.json)，固定全部 800 行、唯一组合及边界排除规则。这是已审查历史数据上的描述性后续检验，不是未见数据的预注册实验。

检查量为

\[
\frac{1.78q_B-.78q_F}{\|v\|^2}.
\]

预先只排除 8 条 `t=1`，其余 **792=99×8 条全部保留**，不按旧 density ratio、原 mixture 正性或 covariance test 成败过滤。所得全部为正，最小 **0.4948893304**，最大 **16.2061406640**。[完整结果](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/characteristic_gaussian_audit_v1/results.json)及逐条组合已保存。

独立 [review.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/characteristic_gaussian_audit_v1/review.json)用原 CSV 十进制字符串的有理数运算逐条复算，结果和行身份一致；它另从目标 posterior 与两支逆映射出发核对四个固定 Gaussian case，没有调用生产脚本的 A-solve 路径。生产构造中的共识/精确 posterior 误差小于 `4e−16`，原噪声方程误差小于 `5e−16`，矩阵逆恒等式最大误差约 `5.69e−14`。独立复核对应的共识、噪声方程和逆恒等式误差分别不超过 `4.45e−16`、`7.11e−15`、`1.14e−13`，均通过固定 `1e−10` 容差。

**通过只意味着所测必要方向未被反驳。** 一个方向不能证明矩阵正定；同状态 Jacobian 也只在 affine 特例中等于移位根处的 Jacobian。缓存来自 8 个相关 FP32/no-autocast/TF32off rollout，不是 BF16 根求解结果；它不能证明实际 F/B 对应某对 Gaussian 或 exact score，也不能建立完整 harmonic 条件、目标正性、一般 CH 收敛或 FID 提升。

这条结果保留共识/自然参数方向的进一步研究价值，但不支持立即采用作者工程参数或启动新的 1K。下一步需要解决实际有限头在移位查询下的可解性、动力学语义和计算成本，而不是把“隐式方程有理论”当作质量保证。

## 归档与成本

论文及官方源码在 [reading_characteristic_v1](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_characteristic_v1)；本地检验代码为 [audit_raev2_characteristic_gaussian.py](../experiments/audit_raev2_characteristic_gaussian.py)。固定新组合与构造验证的执行为 wall `0.039430 s`、CPU `0.159427 s`，计时始于导入之后，不含阅读和序列化退出。下载请求有逐项 wall 记录；人工阅读、文本转换及未计时浏览不拼成虚构的总成本。本次新增模型、VJP、图像、FID 调用均为零。
