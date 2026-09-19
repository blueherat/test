# 参考目标实验：记录、实现与空闲 GPU 执行协议

2026-09-14。用户已授权实现上轮两个候选，并在 GPU 空闲时执行。研究依据为 [目标推导](WEAK_REFERENCE_LOSS_DESIGN_20260914_ZH.md)。本协议将研究规格变成固定 SiT-S/2 实验；CFG 与跨模型迁移在取得首轮信号后另立协议。

第一轮训练 real、self、gaussian、mixture 四个 Context 头，第二轮训练 excess 与类内打乱权重的 shuffled。使用已有真实／强模型生成连续 latent bank，各 2K、100 类各20张。强模型的生成 bank 为无 guidance 的 Heun64 端点，不冒充 SSG 百万 CFG 样本复现。

全部新头同一结构、同一初始化、同一冻结归一化统计，depth4，hidden384，输出16维patch速度。固定3000步、batch32、AdamW lr3e-4、weight_decay1e-4、betas(.9,.999)、EMA.995，时间均匀(.01,.99)，沿用 Context 训练的不丢标签设置。以最终EMA生成，不按loss挑权重。每个头恢复完全相同的数据、时间、噪声和整图混合标记随机流；共同随机流是对照配对，不意味着不同目标的输入完全相同。

Gaussian 使用解析积分后的速度监督：d=sqrt((1-t)^2+t^2*tau^2)，z=tX+d*eps，target=X+(t*tau^2-(1-t))/d*eps。tau在质量评估前固定为生成bank的全局每维中心RMS的一半。Mixture按整图Bernoulli(.5)选择普通或Gaussian训练对，不向头提供标记，不裁剪额外噪声。普通分支保留原FM运算顺序。

Excess的离线比值仅在固定表征中估计：f(X,c)为同一冻结SiT第4层在输入X、t=.99、类别c上的token均值和标准差；这是确定的端点表征，不是声称X来自t=.99的实际轨迹。真实／生成各半，按类内图像下标奇偶两折，StandardScaler只在训练折拟合，L2 logistic regression C=1、max_iter2000，无超参数选择。对留出折生成端点预测eta，使用w=1+[(2eta-1)/eta]_+，原权重在[1,2]。类内以整个冻结训练bank的均值归一化。Shuffled只在同类别内打乱同一组权重。分类器不在推理时调用。

这个权重定义一个合法的表征加权生成分布，不能宣称它等于完整像素空间的真实密度差。若归一化权重标准差小于1e-3，则记录“权重退化、目标近似均匀”，跳过两个等价训练，保留全部结果；不改C、深度或特征寻找分类信号。来源AUC仅描述离线估计，不作为生成有效性标准。两折特征与权重哈希落盘。

采样保持原Heun64、IG额外系数.8、.25之前6/7倍、.5之后关闭，每图128次完整前向、0额外prefix。新头仅读取正在执行的主干特征；新头生成路径不计算旧弱头。原IG对照使用原弱头。沿用既有Context3000步作为强控制，另包含真实2K同预算训练控制。

第一轮固定7臂：real/self/gaussian/gaussian_half/mixture/context/native；第二轮只增加excess/shuffled并复用同bank的self。每臂新1K、每类10图，同噪声同标签，seed2026091462，batch8。pure Gaussian半强度是唯一额外系数对照。Mixture必须比real/self/gaussian/gaussian_half/context/native均低至少1 FID；Excess必须比real/self/context/native/shuffled及已有平滑结果均低至少1。两者均要求IS不低于native的90%。这是资源推进标准，不是显著性检验。

若有通过者，只取FID最低的一个，冻结该头与最强对照、native，在seed2026091463上做独立5K、每类50图。最多3×5K，不追加训练或参数搜索；确认结果如实记录，无自动认定长期目标完成。全部未通过则结束本实验。

执行优先级：先等待当前RAEv2后训练头→IG+SG的监督进程退出／完成，避免在两阶段之间的短暂空档启动。再要求选定GPU连续空闲（无计算进程、使用显存<600MiB、利用率<=5%，三次检查间隔20秒）。每次训练、特征提取与采样任务启动前重新检查空闲。只占用一张空闲GPU，ADM评估使用CPU，单个评估进程4线程。监督器和GPU采用本实验独立的锁，不修改已有队列。

CPU预检现在执行；依赖真实GPU的共享特征、原生轨迹、调用数、零引导检查在空闲后作为第一项任务执行，未通过不训练。代码、权重、训练bank及协议冻结哈希。训练每500步保存optimizer/EMA/数据RNG，采样每批持久化并校验恢复；同一实验STOP_AFTER_CURRENT请求会在当前步骤／batch结束后保存并退出。训练错误不自动重试。

原始产物：`/home/zhoushunyu/data/eqvae/experiments/weak_reference_loss_20260914`。源码：[实验目录](../experiments/weak_reference_loss_20260914)。本轮新推导的其他loss尚未加入这个固定队列，避免研究过程改变待运行实验的比较对象。
