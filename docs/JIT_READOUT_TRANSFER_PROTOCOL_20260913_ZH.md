# SiT有效读出向JiT-B/16的固定迁移检验

用户要求在JiT采样验证已在SiT有效的MLP中间读出。本轮保持一次共享主干前向的IG预算，迁移读出结构和固定训练预算，不迁移SiT权重。原独立弱前缀路线继续停止。

使用本地官方JiT-B/16 model_ema1、256像素、ImageNet-1K；原IG弱头为既有depth4、50K EMA。固定depth4、宽度768；MLP直接复用SiT有效结构：token线性映射、condition线性映射、8维固定位置映射、SiLU及patch输出。输出改为JiT原生16×16×3 clean patch。训练使用真实图像的clean-output velocity loss，分母max(1-t,.05)，t=sigmoid(N(-.8,.8²))、噪声scale1、标签丢弃.1、水平翻转.5。数据和参数化沿用JiT，不能照搬SiT的velocity输出与uniform时间。

两个新头同批训练：MLP从零输出初始化；原RMSNorm+AdaLN+Linear读出从官方初始化开始，其中RMSNorm权重保留为1，输出和AdaLN末层置零。全部主干和原末层冻结，无蒸馏、无strong生成数据。固定3000步、batch32、AdamW(lr3e-4,wd1e-4)、EMA.995，取最终EMA。两头共用32个训练批估计MLP固定逐通道均值与标准差，然后开始参数更新。训练seed2026121371。每类固定抽48张真实训练图，排除原弱头保留的1000张验证图；更新分两次独立打乱遍历，共96000张次，统计批单列。预测验证只作记录，不选择checkpoint；旧验证图也不称为新的独立验证集。

主比较固定6臂、每臂1000个新噪声、每类1张：MLP IG、同训练原结构IG、旧原IG、旧头ADG、原条件模型Euler100、官方CFG3 Heun50参考。所有初始噪声及标签完全相同，numpy seed2026121381，batch4。前三个IG沿用JiT已有alpha=.3、前50步启用、后50步关闭、Euler100；ADG沿用相同窗口与额外强度.3。它们每图100次full、0prefix，单条路径。原条件模型也为100次full。

CFG参考使用官方实现的Heun50末步Euler、区间(.1,1)、CFG=3、t_eps=.05，每图198次full。这是额外主干预算的实用参考；不把不同求解器和预算的差异归为读出结构效应。本轮不另叠加IG与CFG，不搜索两种guidance的组合系数。

通过forward hook在原模型第4层结束时计算选中的弱头，原全模型继续执行。hook关闭时不计算弱头，也不再计算被替换的原弱头。预检要求共享前向的强输出与官方模型逐项相同，旧弱输出与原features接口相同，零guidance回到原模型，旧IG完整轨迹与旧实现公式逐项相同，CFG完整轨迹与官方stepper逐项相同。模型层调用和完整调用计数实际记录。

所有六臂都完成后才看结果。固定门槛：MLP的FID比旧IG及ADG都低至少1，且IS不低于旧IG的90%，才视为值得独立5K确认；另记录它是否比等训练原结构低至少1，防止把训练变化当成结构收益。未通过则停止这次迁移，不扫训练步数、学习率、深度、宽度或引导强度。1K每类一图不提供类别内可靠统计或多种子显著性。正结果也不是跨模型普适性或新理论的证明。

记录替换后参数净差和同GPU、batch4、轮换预热后三次完整采样及像素量化耗时。训练时间单列。所有输入、权重、源码、批次图像/状态和评价资源保存SHA。用既有nanogen evaluator的imagenet_256_fid_stats评价，缓存特征FP64复算FID并核查样本覆盖、标签、输入SHA与实际调用数，不当作独立特征提取器验证。展示固定前四张全部对照。

实现来源：[JiT官方代码](https://github.com/LTH14/JiT)、[SiT读出证据](CONTEXT_REFERENCE_5K_RESULTS_20260912_ZH.md)、[等训练控制](IG_READOUT_MATCHED_CONTROL_RESULTS_20260913_ZH.md)。[SSG](https://arxiv.org/html/2607.29122v1)已有冻结中间adapter先例，本实验不会把小MLP本身称为新贡献。
