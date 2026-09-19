**端点对抗训练弱头：代码、目标和研究判断（2026-09-15）**

核心结论：这个目标作为“冻结强模型、通过最终生成结果训练一个辅助弱头”的后训练方法是成立的。没有发现损失符号反了、生成样本被错误 detach、或后半段冻结主干切断梯度这类主链错误。它学习的是适合实际采样器的修正场；要进一步声称学到了某个规范扩散族的最优弱分布，需要额外条件。首轮结果尚未证明有效。

本次只新增独立审查脚本和报告，没有改动训练器、检查点或原队列。审查对象主要为 `endpoint_gan_v1` 的冻结快照及其 200 步检查点；审查期间已有独立启动的 `endpoint_gan_v2`，从 200 继续到 600。18:10（北京时间）的记录为 230/600，详见[状态快照](data/adversarial_weak_audit_20260915/run_status.json)。本报告中的质量数字全部属于 v1。

**1. 代码实际上优化什么**

训练入口是 [train.py](/home/zhoushunyu/eqvae/experiments/adversarial_weak_training_20260915/train.py:27)。冻结 SiT-S/2、VAE、Inception，只训练 depth4 Context MLP 弱头与类别条件判别器。弱头从完整训练集上完成标准去噪训练 50K 的 EMA 初始化。首轮对抗训练为 200 次 D 更新、184 次弱头更新，另外 16 次是 D 预热；不能把它称作 50K 对抗训练。

设强速度场为 S，弱头为 W_phi，共享浅层特征为 H_S。实际部署的场是

\[
F_\phi(t,z,c)=S(t,z,c)+a(t)\{S(t,z,c)-W_\phi(H_S(t,z,c),t,c)\}.
\]

这里 `coefficient=1.0` 表示 a 的峰值为 1，不是“不外推”：活跃区间等价于总外推系数 w=2。前四分之一 a=6/7，第二个四分之一 a=1，后半程 a=0；Heun 第二次场查询沿用该步左端点的开关。源码见 [sampling.field](/home/zhoushunyu/eqvae/experiments/guidance_dynamic_50k_20260915/sampling.py:38)。

令 T_phi 是完整 64 步 Heun 的离散映射，B 是 VAE decoder，f 是固定 Inception2048，则被训练的“生成器”实际为

\[
\epsilon\longmapsto z_1=T_\phi(\epsilon,c)
\longmapsto x=B(z_1)\longmapsto f(x).
\]

判别器不对逐张图片做像素 MSE。它对四个来源进行交叉熵分类：P=真实图像编码后验的解码样本；G=冻结强模型的端点库；Q=当前弱头单独积分的端点；R=当前强弱外推的端点。四个来源均有相同的类别标签、相同解码和特征流程。P/G 从训练数据中抽取；Q/R 每步重新采样，并使用同一批噪声。这种 Q/R 配对在当前逐样本判别器中不改变各自边缘分布。

记四个 logit 为 l_P,l_G,l_Q,l_R。判别器最小化

\[
L_D=\tfrac14\sum_{i\in\{P,G,Q,R\}}
\mathbb E_{x\sim i}\operatorname{CE}(l(f(x),c),i)
+\tfrac\gamma2\mathbb E_{x\sim P}
\|\nabla_{f(x)}(l_R-l_P)\|^2.
\]

弱头最小化

\[
L_W=\mathbb E_{x\sim R_\phi}
\operatorname{softplus}(l_R(f(x),c)-l_P(f(x),c)).
\]

因此，“弱模型最小化判别器”准确地说是：**让外推样本更容易被判为真实**。若让弱头也最小化把 R 正确分类为 R 的 D-loss，方向会反；当前代码没有犯这个错误。这是非饱和生成器更新，与原始 GAN 的饱和 minimax 生成器损失并非逐步相同。[GAN 原论文](https://arxiv.org/abs/1406.2661)

**2. 主梯度链的核对结果**

| 项目 | 结论与证据 |
|---|---|
| 外推负号 | S+a(S-W) 对 W 的直接导数为 -a；实际 autograd 包含该负号，独立标量检查方向正确 |
| D 更新 | generated_features.detach() 只用于 D-step，不向弱头回传 |
| W 更新 | 冻结 D 参数后重新算 logit，输入梯度保留；VAE/Inception 同样冻结参数而保留输入梯度 |
| 冻结强模型 | 生成路径中的主干调用没有整体 no_grad，状态和 h4 的导数保留 |
| 后半程 | 虽然 a=0，不再调用弱头，仍计算冻结强模型的状态雅可比 |
| 更新次序 | W 的整段 forward/backward 之间没有修改 W；更新 D 不会污染 W 的已保存采样状态 |
| 多卡 | 等量 local batch，先平均梯度再裁剪和更新；已有检查点显示三卡 head/critic 参数与 buffer 指纹一致 |
| EMA | 只移动参数，固定归一化/位置 buffer 不变；检查点显示 W 优化器 184 步、D 优化器 200 步 |
| 前向口径 | 完整梯度检查中的采样结果与同设置部署函数逐位一致；首轮质量评估也核对过原基线批次 |

实现采用逐离散步重算 VJP：[discrete_adjoint.py](/home/zhoushunyu/eqvae/experiments/adversarial_guidance_endpoint_20260915/discrete_adjoint.py:14)。对于 z_(k+1)=Psi_k(z_k,phi)，正确梯度是

\[
\lambda_k=(\partial_{z_k}\Psi_k)^\top\lambda_{k+1},\qquad
\nabla_\phi L=\sum_k(\partial_\phi\Psi_k)^\top\lambda_{k+1}.
\]

后半程直接的 partial_phi 为零，但 partial_z 一般不为单位阵，不能省。当前实现保留了它。

这次另外使用**真实 200 步在线弱头、真实训练后的判别器、完整 VAE/Inception 终点损失**，与 PyTorch 普通链式反传（activation checkpointing 仅节省内存）比较了整个 64 步，batch=1，独立噪声，类别 17。没有用随机线性判别器替代实际 D。

| 数值设置 | 完整弱头梯度相对 L2 误差 | 解释 |
|---|---:|---|
| 当前默认 TF32/注意力内核 | 2.988e-3 | 约 0.299%，梯度余弦 0.9999956；第一次探查约 0.198%，未通过当时 2e-4 的严格阈值 |
| 关闭 TF32，保留默认注意力 | 3.574e-4 | 误差变小，仍未过上述严格阈值 |
| 关闭 TF32，普通 math attention | 5.899e-6 | 完整链式反传对照通过；初始噪声梯度误差 3.245e-6 |
| CPU float64 非线性链，独立方向有限差分 | 绝对误差 2.784e-11 | 验证重算导数与链式法则及外推方向 |

这支持“反传公式正确，默认数值内核存在可测的长链梯度误差”，不支持宣称默认 GPU 梯度逐位精确，也不能据此认定 0.3% 的误差造成了质量变化。不同数值设置下的前向一致性是**各自设置内部**的比较，不意味着 TF32 和 math attention 之间样本逐位一致。

所有检查前后，强模型、VAE、Inception、弱头和判别器的状态指纹完全不变。本次临时检查只取得空闲 GPU 的同协议锁，已退出；没有更新任何训练权重。原始证据：[默认精度](data/adversarial_weak_audit_20260915/gpu.json)、[严格 FP32](data/adversarial_weak_audit_20260915/gpu_strict_fp32.json)、[math attention](data/adversarial_weak_audit_20260915/gpu_math_attention.json)、[CPU 与检查点](data/adversarial_weak_audit_20260915/cpu.json)。

**3. 思路成立，但“弱分布自己寻找”的含义要精确**

最可靠的表述是

\[
\min_\phi\mathcal D(P,R_\phi),\qquad
R_\phi=\operatorname{Law}(B(T_\phi(\epsilon,c))).
\]

W 不再接收预设的弱分布监督，确实比“先假设高斯平滑弱分布，再看外推有没有效”更直接。若能产生降低终点损失的参数方向，即使这个方向让 W 的去噪 MSE 或单独生成 FID 变差，也可能是正确更新。

在精确去噪 MSE 的驻点附近，小参数扰动对原 MSE 的一阶变化为零，而对当前终点损失的一阶变化不必为零。因此“去噪已经收敛”并不排除还存在有用的终点改进方向。有限模型的 50K 初始化是否真的达到驻点，当前记录没有证明。类似地，W=S 只意味着引导场回到 S；若 G 仍不同于真实分布且浅头存在有效修正方向，复制强模型并不自动最优，不能仅靠网络容量推断它一定会发生。

然而当前 W 是一个自由速度读出。只要其单独数值积分有良好定义，便能定义一个隐式端点分布 Q_phi；这**不等于** W 是 Q_phi 经某个指定前向加噪过程后的精确 score 或对应 flow-matching 条件均值。当前代码没有施加这种跨时刻的一致性。不能据此直接写成“找到了与 P、G 满足固定密度幂次关系的 q*”。

相对初始化 W_0，存在严格的函数恒等式

\[
F_\phi-F_{\phi_0}=-a(t)(W_\phi-W_0).
\]

所以这同时是一种**用共享浅层特征参数化的残差后训练**。若函数容量足够，对任意希望实现的活跃区间场 F*，都可以形式上取

\[
W^*=\frac{(1+a)S-F^*}{a},\qquad a\ne0.
\]

它不必“弱”，也没有天然理由必须是平滑后的强分布。这是代数解释，不保证 depth4 MLP 能表达该 W*，也不覆盖 a=0 的后半程。它提醒我们：真正的研究贡献要落在有限浅头的参数效率、共享特征、初始化与终点优化效果上。

即使两个弱场独立采样得到相同 Q，它们在不同时刻沿途的行为也可能不同，从而产生不同的 R。反过来，多个弱场也能得到相同 R。因此 endpoint objective 一般不能唯一确定弱场或 Q。仅分析 Q 的两个峰如何移动，仍不足以完整解释实际引导过程。

**4. 当前目标有三个实质限制**

第一，判别器只学习固定 Inception 后面的 MLP。即使这个 MLP 容量无限且优化完全，最多保证所观察到的特征分布匹配：f#P=f#R；它不会自动保证 P=R。固定特征可能看不到某些局部结构、纹理或多样性缺陷。用预训练特征做判别有成功先例，但多尺度信息的处理本身就是方法的一部分，当前实现只使用最终 pooled 2048 维。[Projected GANs](https://arxiv.org/abs/2111.01007)

因此当前方法较准确地叫“固定视觉特征上的自适应判别反馈”。判别器在适应，但底层图像表示没有学习。评价器虽是另一个 ADM/TensorFlow 实现，仍主要观察 Inception 体系；这不算一个与训练表征完全独立的质量验证。

第二，四源分类的理论消元有条件。无正则、无限容量、内层优化充分、来源先验正确时，最优分类器满足 C_i=pi_i p_i/sum_j(pi_j p_j)，P/R 成对概率可还原 p/(p+r)，G/Q 从比值消去。当前等量来源实现的两个 logit softmax 在这点上正确。但有限共享 MLP、谱归一化、特征 R1、每轮一次 D 更新都使该理想条件不成立。辅助 G/Q 可能帮助表示，也可能让 D 优先解决容易的来源分类，不能用理想比值恒等式证明四源一定优于 P/R 二分类。

更具体地，Q 完整采样是每步必须完成的操作，而 [weak_integrate](/home/zhoushunyu/eqvae/experiments/guidance_dynamic_50k_20260915/endpoints.py:13) 在 Q 发散或数值超阈值时直接抛错、终止训练。端点优化本来允许 Q 独立质量下降；即使 R 良好，Q 发散仍会让实现停止。这是辅助分支给主目标施加的额外运行约束，目前没有证据表明首轮触发过。

第三，训练目标 P 是“真实训练图像编码后验的 VAE 重建分布”，不是原始真实像素分布。统一四源 decoder 避免 D 仅靠有无 VAE 重建痕迹区分来源，这是有理由的工程取舍；但评价参考是原始验证图像，两个目标仍有差别。

此外，[训练 decode](/home/zhoushunyu/eqvae/experiments/adversarial_weak_training_20260915/data.py:38) 没有 clamp，而[部署 decode](/home/zhoushunyu/eqvae/experiments/lifting_scale_sweep_20260909.py:282) 会裁剪并量化。量化前使用连续代理通常是必要的；省掉 clamp 则还改变了可见值域。在本次一个样本上，越界元素约 0.0010%，严格 FP32 下 clamp 使特征相对变化约 3.6e-5、损失变化约 1e-6，尚未看到明显利用越界的证据。不能把单样本推广成全数据结论，也不应把该差异直接定为本轮 FID 不提升的原因。

**5. 首轮效果和优化幅度**

相同 coefficient=1、相同 1K 噪声与标签、同一 ADM 评价口径：

| 模型 | 1K FID |
|---|---:|
| 标准完整数据 diffusion 50K 弱头 | 64.140582 |
| 端点 GAN 200 步 EMA 弱头 | 64.219546 |
| 差值，后训练减基线 | +0.078963 |

这组 1K 是 ImageNet100，每类 10 张；不是 ImageNet1000 每类 1 张。它仍只能作为开发信号，不能凭这个差值判定方法成功或失败，也不能与此前 5K 的绝对 FID 混比。本轮还没有新的 5K 质量结果。

已逐项核对日志与检查点：

- 184/184 次 W 更新触发阈值 1 的梯度裁剪；裁剪前范数最小 6.20，中位 27.23，最大 199.84。
- 在线头相对初始可训练参数的 L2 改变量约 0.0280%；EMA 约 0.0179%。这是参数空间比例，不等价于输出只变化这么多。
- 最后 50 步的平均 P/R 判别准确率，真实约 73.25%、引导约 67.67%；D 仍能区分两者。
- 平均 generator loss 从第 17–66 步的 0.8852 到最后 50 步的 0.9013。因为 D 同时变化，这不是固定目标下的单调性测试。
- 首批固定 8 对图像确实有局部变化，但看不到一致的结构改善；它们不足以代表总体质量。[原有配对图](/home/zhoushunyu/data/eqvae/experiments/adversarial_weak_training_20260915/endpoint_gan_v1/quality/step000200_ema/c0040/paired_first8.png)

不能由“都裁剪了”推导出“Adam 的实际步长缩小了 27 倍”：Adam 的一二阶矩会部分抵消梯度的整体缩放。现在需要关注实际参数更新、函数输出变化、不同批次梯度方向和同输入质量，不宜仅根据 D-loss 或 norm 武断调大学习率。

全局 batch=24，首轮 D 看过约 4800 个真实样本抽取；初始归一化另外抽取 384 个。数据池覆盖完整 126689 张，然而这不代表 200 步遍历过整个训练集。小批量与短预算的联合效果尚未被分离。

**6. 独立于研究目标的具体工程问题**

| 优先级 | 位置与触发条件 | 实际后果；当前结果是否受影响 |
|---|---|---|
| P2 | [evaluate.py:49](/home/zhoushunyu/eqvae/experiments/adversarial_weak_training_20260915/evaluate.py:49) 在 a=0 的 1K 分支只复用 metrics；[99 行](/home/zhoushunyu/eqvae/experiments/adversarial_weak_training_20260915/evaluate.py:99) 扩展任何 5K 都读取本地 n1000/summary.json | 若 a=0 进入最优两个点，生成 5K 后会因缺少 summary 失败。当前只有 a=1 的 pilot，不受影响。修复应同时复用/核验基线的样本记录，或为 a=0 单独提供完整 5K 复用路径 |
| P2 | [train.py:128](/home/zhoushunyu/eqvae/experiments/adversarial_weak_training_20260915/train.py:128) 只要 start!=0 就启用 W 更新 | 若预热未完时恢复，会提前更新 W。v2 从 200 恢复且 warmup=0，不受影响。应按明确的全局预热边界判断 |
| P2 | [common.py:55](/home/zhoushunyu/eqvae/experiments/adversarial_weak_training_20260915/common.py:55) 只归档两个本地目录 | 实际场、Adapter、Head、decoder/特征提取器、底层模型代码等没有完整纳入当前 run 的快照/校验。14 个源码 hash 匹配不能覆盖整条科学实现。审查已补记关键依赖的当前 hash，不能追认这些依赖过去从未改变 |
| P3 | train.py 结束只写 latest/complete，不把 progress.phase 改成 complete | v1 的 progress.json 仍显示 training，容易误报 GPU 还在训练；应综合 latest、完成标记和进程状态 |
| P3 | [train.py:59](/home/zhoushunyu/eqvae/experiments/adversarial_weak_training_20260915/train.py:59) world size 不同时不恢复旧数据 RNG | 不能称不同卡数恢复为逐步精确续训。当前 v1/v2 均为三卡，未触发 |

本次开始时 v1 的 14 个冻结源码都匹配；审查期间另一个执行流程更新了 launch.py，为 v2 补充 initial-head/resume 来源记录。v2 的请求与现存源码匹配，train.py、采样场、梯度实现没有随之改变。审查没有覆盖或撤销该更新。上述问题在报告中明确列出，未直接修改正在使用的冻结实验源码。

**7. 与已有研究的关系**

| 研究 | 与当前工作的关系 |
|---|---|
| [DRaFT](https://arxiv.org/abs/2309.17400) | 终点可微反馈穿过采样过程更新扩散参数已有先例；当前变化在于更新共享浅头、D 在线适应、保持多步部署 |
| [AlignProp](https://arxiv.org/abs/2310.03739) / [后续 VADER](https://arxiv.org/abs/2407.08737) | 终点奖励反传与省显存已有研究。AlignProp 原 arXiv 已标记由后续论文涵盖，引用时应注明状态 |
| [Projected GANs](https://arxiv.org/abs/2111.01007) | 固定预训练视觉特征加可学习判别头已有先例；当前只取最终 pooled 特征的限制需实测 |
| [ADD](https://arxiv.org/abs/2311.17042) / [DMD2](https://arxiv.org/abs/2405.14867) | 对抗真实数据与扩散蒸馏的结合已有先例；DMD2 同时包含 distribution matching，不能把本方法称为 DMD2 等价实现 |
| [APT](https://arxiv.org/html/2501.08316v1) | 扩散预训练之后针对真实数据做对抗后训练是最直接相关路线。其一步模型、训练规模和判别器结构与本实验明显不同，不能照搬迭代数或批量结论 |
| [Discriminator Guidance](https://arxiv.org/abs/2211.17091) | 学习判别器修正 score 也有先例，但该方法推理时使用判别器梯度；当前 D 只在训练时出现 |
| [Improving Discriminator Guidance](https://arxiv.org/abs/2503.16117) | 指出 CE 判别准确不自动意味着引导后的 KL 改善；其定理针对判别器引导，不可直接移植成当前端点 GAN 必然失败的证明 |
| [Variational Control](https://arxiv.org/abs/2502.03686) | 为终点代价与沿途控制的联系提供已有框架；支持用受限修正场解释当前机制，而非仅由弱终点密度解释 |

这些文献使“GAN loss 接到 diffusion 终点”本身不足以构成新贡献。本次检索没有建立“此前无人用这种确切的共享浅头外推结构”的结论；没有找到完全相同实现不等于证明新颖性。

本方法更值得验证的命题是：**普通去噪目标已训练充分的浅头，其特征空间中是否还存在能够稳定改善实际终点的方向；能否通过很少的可训练参数，在推理成本不增加的条件下利用这些方向。**

**8. 我建议怎样继续检验，而不是同时更改多个部分**

先完成已有 v2 的同协议比较；一条曲线的短期不提升不足以推翻目标。保存在线头和 EMA，分清对抗更新次数、实际样本数和耗时。APT 的实验也表明更久并不必然更好；它的训练曲线和批量观察只能提供警示，不能直接指定本实验最优预算。[APT 的训练分析](https://arxiv.org/html/2501.08316v1)

下一项最有信息量的消融是：**同一个 W_0、同一个 coefficient、同一个采样器与预算，仅使用 P/R 二源判别，去掉辅助 G/Q。**它直接对应用户最初的目标，同时消除 Q 必须独立稳定的运行约束。先保持 Inception 和其他配置不变，才能判断四源是否真的帮助；不要同一轮同时换特征、loss、时间窗和学习率。

随后应增加与训练 Inception 不同的结构/多样性检查。若只有 Inception FID 改善而配对结构与另一种视觉表征变差，需要先解决目标盲区。DMD2 的消融出现过更低 FID 与更弱文本/审美表现并存，说明这种区别在相关系统中有实证，不是当前模型已经发生了同样问题。[DMD2 消融](https://arxiv.org/html/2405.14867v2)

若更新不稳定，可考虑在实际 on-policy 轨迹上约束相对初始化的场改变量

\[
\mathbb E_{t,z_t}\|a(t)(W_\phi-W_0)\|^2\le\rho,
\]

并自适应控制更新幅度。它控制的是已有生成过程被改变多少，不规定 Q 必须是高斯平滑或某个预设弱分布。它仍然是一项新的正则选择，需要与不加约束版本单独比较；仅有当前“梯度都裁剪”的记录，尚不足以断定必须加入。

研究对照还应包含参数和计算预算相近的普通残差适配。因为 F_phi-F_phi0=-a(W_phi-W_0)，必须证明初始化、共享浅特征或时间结构带来实际优势，才能把结果归因于“学习适合外推的弱模型”，而非所有小参数对抗后训练都能获得的收益。

复现此次检查：`python -m experiments.adversarial_weak_audit_20260915.audit cpu`；GPU 模式会自行尝试取得一张空闲卡的锁，`gpu --strict-fp32 --math-attention` 是严格链式法则对照。检查脚本位于 [audit.py](/home/zhoushunyu/eqvae/experiments/adversarial_weak_audit_20260915/audit.py)，不会续训或启动质量扫描。
