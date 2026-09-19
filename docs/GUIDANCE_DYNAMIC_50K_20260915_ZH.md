**本轮全部重新训练：恢复原50K弱头的完整真实数据流，SiT与JiT逐idea执行。**

用户已明确要求本轮所有候选重训，并选择为生成分支建立与真实训练集同量级的大样本库，每个训练batch重新采时间和噪声。随后追加要求：原四级系数扫描保持不变，将1K表现最好的两个不同系数各扩展到5K；JiT也必须执行。

SiT采用原depth4原生弱头的run_config：完整126,689张ImageNet-100真实训练图像、5,000张原验证图像。直接复用原NpyMomentsDataset、DistributedSampler、可恢复数据遍历与sample_sdvae_posterior；每步重新采VAE posterior、扩散噪声、均匀[0,1)时间。四卡DDP，全局batch256、每卡64，AdamW lr1e-4、betas(.9,.999)、weight decay0、EMA .9999、BF16前缀与头计算、50,000优化步。保持源模型冻结与depth4 Context MLP结构。

JiT采用原50K读出训练的数据与参数化：完整ImageNet-1K训练库，排除既有每类1张的1,000张验证图像。每次读取原图；训练时随机水平翻转.5、标签丢弃.1，t=sigmoid(N(-.8,.8²))，新高斯噪声；clean-output velocity MSE分母max(1-t,.05)。四卡、全局batch256、lr1e-4、weight decay0、EMA .9999、BF16、50K。MLP输出原生clean patch；向量差值损失除以原生velocity分母，不把SiT的velocity输出直接搬到JiT。

各模型的MLP从零初始化，并用完整真实训练流的32个全局batch重新估计输入归一化统计；不加载旧3K或小bank 50K头的可训练参数。主头、混合均值辅助头和来源判别辅助头均重新训练50K；每500步保存optimizer、EMA、各rank RNG与数据位置，保留3K/10K/20K/30K/40K/50K快照。验证只检查训练，不选择训练终点。

生成来源strong、独立self50K弱头、原生IG、原生CFG各自冻结一轮。每个来源的train/validation样本数与该模型的真实train/validation相同，标签频率逐一匹配；噪声种子与质量采样不同。weakmix=(strong+weak)/2。SiT保存连续float32 latent；JiT使用模型正常输出的uint8图像，按lossless PNG打包，读取后映射至[-1,1]并做原生动态训练增强。这是像素图像分布，不声称保存JiT裁剪前连续终点。生成分支不会加入不存在的VAE posterior，也不会每个训练batch重新跑整个生成过程。

两折辅助学习按完整训练数据ID奇偶分割，只在另一折训练，在留出折预测；真实验证集和生成验证库不参加任何辅助模型或scaler拟合。Excess分类器仍是固定depth4均值/标准差特征上的两折logistic regression，改用完整大库；权重按完整训练类均值归一化，shuffled只在同类别、同train/validation分区内打乱。Gaussian/mixture维持原Rao–Blackwellized平滑目标，tau由新strong训练库的每维中心RMS的一半确定。

每个idea执行顺序：SiT 50K训练→0.4/0.2/0.1/0.025四级1K扫描→前两个系数各扩展5K→JiT同样流程→下一idea。先做标准diffusion loss作为完整协议对照。最优两区间仍以端点平均FID排序，并补齐两个区间之间的整个连续范围；每级只改一个guidance系数。初始范围[0,2]，不自动扩展边界。旧MLP、原生IG不另开扫参轮。

5K扩展先锁定1K排名前两个不同系数，保留其原1000张及原噪声/标签，追加同一组4000张配对噪声/标签。每组最终共5000张；不是另生成5000张，也不是独立于选择集的验证。若有效系数不足两个则明确记录实际数量，不伪造第二个系数。SiT沿用Heun64及原窗口；JiT沿用Euler100、前50步弱头引导，CFG路线采用原Heun50末步Euler和(.1,1)窗口。SiT用原ADM统计，JiT用原nanogen ImageNet256统计。

四卡共同训练一个模型/idea，采样按原batch边界分成四份。检查完整真实数据、DDP同步、真实生产loss下的优化器与数据恢复、零系数与强模型轨迹一致、强模型始终冻结。训练和指标失败会保存当前任务并停止相关队列，不悄悄换超参数继续。

原小bank实验与扫描目录保留为历史。新目录为`/home/zhoushunyu/data/eqvae/experiments/guidance_dynamic_50k_20260915`。为容纳JiT的大生成库与5K评估，质量图像使用无损压缩；每个来源最后一个消费者完成后只清理本轮可按冻结种子重建的生成大文件，保留生成清单、哈希、种子、模型与所有质量结果，绝不清理真实数据或旧实验。
