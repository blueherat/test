# RAEv2 冻结主干后训练：第4层输出头与第8层MLP，统一50K

用户要求所有头达到50K，并明确“扫参”是采样时强弱预测的外推系数。训练预算按用户要求确定，不以生成FID选择中间checkpoint。

原模型为DINOv3-Lk7 RAEv2 EMA。原生IG取28层encoder的第8层，使用SiLU(t_base+h8)后接DDTFinalLayer(1440,1,1024)；强分支使用28层encoder、2层DDT decoder及最终输出层。主干和原生输出完整冻结，decoder和归一化资产保持原来源。

新增第4层两种头：与原生base_final_layer完全同类、同维度、同初始化的DDTFinalLayer，以及仓库guidance_distribution_20260912.local_head.Head的Context MLP。两者从原架构初始化重新训练，并非不经训练直接移植第8层权重；共享完全相同的真实clean latent、噪声、时间序列。新头seed2026091431，batch8，AdamW lr3e-4、weight decay1e-4、EMA .995，t~U(.01,.99)，clean-latent MSE，50,000步。MLP以32个训练批次估计特征归一化统计。沿用现有5,000个真实训练latent与1,000个互斥验证latent。每个验证样本的噪声与时间固定；验证仅作诊断。保存在线头、EMA、AdamW及数据随机流的可续训checkpoint，最终固定50K EMA用于采样。

第8层Context MLP从既有20K latest.pt继续30K步至总计50K；恢复实际在线参数、EMA、AdamW、数据／CPU／CUDA随机数状态，不从EMA重新初始化优化器。核对恢复的EMA与已归档20K最终head.pt逐tensor一致。原始训练种子、特征归一化、数据流和超参数沿用原训练。记录固定训练／验证集的预测误差，但不把误差下降作为FID改善。

先验证第4层共享主干捕获特征与独立prefix逐位一致；原生第8层输出重建逐位一致；主干参数无梯度且不更新。训练与采样来源、模型和数据哈希保存。正式生成前验证零外推恢复强分支、原生IG与既有实现一致、新头不增加独立prefix调用。

采样外推统一定义为v_weak+w*(v_strong-v_weak)，w=1为无外推；原生IG为w=1.78。数值实现为既有FP32 velocity算术full+(w-1)*(full-weak)，不更换BF16 clean混合次序。每个头各自搜索w，原生IG接受同等搜索；层位、训练超参数、SG时间偏移不属于此系数扫描。先在共享筛选bank选定系数，再冻结选择与独立5K初始噪声。扫描网格、细化规则和5K方案会在生成前单独冻结并完整报告。用户后续要求的SG扫描在本头实验完成后执行。
