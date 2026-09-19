# 按最终引导预测训练参考读出

本轮固定 strong/backbone，重新训练原有参考读出，使它对最后被使用的引导预测负责。IG 保留一次主干前向；CFG 保留原条件、null 前向。没有独立弱前缀，没有在线局部探针、分量系数、额外生成模型或强度网格。

令 $Y=X-\epsilon$ 为原始 flow-matching 目标，$S$ 为冻结的强预测，$W_\phi(H)$ 为参考读出，$a(t)>0$ 为既有额外引导系数。原参考训练损失是 $\|W-Y\|^2$；本轮候选训练

$$\left\|W_\phi(H)-\left[S+\frac{S-Y}{a(t)}\right]\right\|^2
=\frac1{a(t)^2}\|S+a(t)(S-W_\phi(H))-Y\|^2.$$

右式是按时间归一化的引导预测损失，不是教师 guided 输出蒸馏。训练目标来自真实 clean/noise 对及同状态冻结强预测。只续训相同读出、目标仍为 $Y$ 的 native 组是必要对照。

在固定 t、无限可表示读出、平方可积条件下，$W_0=E[Y|H]$，新最优读出

$$W_*=E[S|H]+a^{-1}E[S-Y|H],\qquad W_*-W_0=(1+a^{-1})E[S-Y|H].$$

把两者用于同一引导公式，人口平方风险之差为 $(1+a)^2 E\|E[S-Y|H]\|^2$。这只解释一种参考可预测偏差为何被原训练目标保留，不预设模型误差共线。完美 Bayes 强预测下，tower property 使该差为零；有限读出、有限优化及 guided rollout 分布均不满足自动质量保证。

IG：ImageNet100 SiT-S/2 800K EMA、原 depth4_v 50K EMA。复制的仅为读出，全部强主干参数冻结。CFG：复制原 final_layer 作为 null 专用输出读出；null 的完整特征来自既有无条件前向，条件前向继续用原 final_layer。新读出参数只存储一份原大小输出头，不改变网络深度。

四项训练为 ig_native、ig_direct、cfg_native、cfg_direct，各 1500 步、batch32、AdamW lr=1e-4、betas=(.9,.999)、weight_decay=0、clip=1、EMA=.995，固定最后 EMA；种子 2026120913。两条线分别使用相同输入、噪声、时间和原图抽样流。IG 时间均匀 (.01,.5)，CFG 时间均匀 (.01,.75)，仅覆盖实际引导激活区间。IG 保留 .1 标签 dropout；CFG strong 使用真实类别，参考始终使用 null。IG 的 $a(t)$ 为 t<.25 时 .8×6/7，之后 .8；CFG 为常数 1.25。这些数值来自已有基准，不按本轮质量重选。

训练使用已有互斥 real-data VAE moments：train 25600 图，validation 3200 图，继承原始数据 ID 与文件哈希。使用 FP32 冻结特征计算与 FP32 小头训练，避免候选与实际 FP32 推理之间额外引入 BF16 主干差异。记录训练主循环时间、可训练参数、强模型 state hash、前后验证读出 MSE 及 guided MSE；验证仅检查，不选 checkpoint。理论等式用有限条件概率的数值例子检查，不用玩具质量替代真实生成。

八组配对 1K：原 IG、续训 native IG、direct IG；原 CFG、续训 native CFG、direct CFG；既有 APG CFG（a=2、beta=-.5）及 ADG IG（a=.8）。所有主比较沿用 Heun64、原时间区间，IG 每图 128 full / 0 prefix；CFG 每图 224 full / 0 prefix。原型、native 与 direct 同一条线的求解器、强度、输入、解码和评价参考相同。训练成本另列，记录实际调用及 GPU 秒，不能仅凭 NFE 相同声称延迟相等。

探索沿用已有 bank seed=202610100；同一噪声已用于历史探索，故不称独立确认。若候选比原生、native 续训及本线已知低成本参考均至少低 .5 FID，则固定权重与配置，在新 bank 上进行配对 5K 确认；否则停止当前构造，不扩强度、训练步数或深度。阈值是推进规则，不是统计显著性标准。若验证风险降低而生成质量恶化，应判定此种 teacher-state 风险目标不足以指导质量改进。

运行前检查包括：IG 共享输出与原实现精确一致；CFG null 替换前后原头恢复、初始化输出精确一致；目标损失代数和梯度一致；原 strong state hash 不变且无梯度；所有候选零引导回到相同 strong 轨迹；四个既有基线完整首批 latent 精确重放。全部正式输出验证覆盖、原始哈希、FID/sFID/IS，并从缓存 Inception 特征独立重算 FP64 FID/sFID。

最近邻约束：IG、SGG、MG/GFT 已研究把 guidance 纳入强模型训练；SSG 已研究冻结 backbone 的适配器监督；OPD 已分析 guided matching 的分支不可识别。本轮冻结强分支、只改变参考读出目标，但不能将平方损失回归恒等式或“guidance-aware training”宽泛概念认作新颖性。相关原文：[IG](https://arxiv.org/html/2512.24176v1)、[SGG §4.3](https://arxiv.org/html/2603.20584v1#S4.SS3)、[SSG](https://arxiv.org/html/2607.29122v1)、[OPD](https://arxiv.org/html/2607.24731v1)。

旧大队列的暂停标记保持不变。此轮协议不授权重启被取消的独立弱前缀路线。
