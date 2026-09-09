> 最新用户指令：lifting 暂时搁置。只执行训练、depth4 IG调参、PFR；下文lifting部分保留为暂停方案，不执行。

# JiT 按用户指定顺序：训练→单depth IG调参→PFR→lifting

此协议覆盖此前直接固定gamma比较IG/PFR/OU的计划；OU不列入当前队列。
等待当前小SiT历史配置lifting及五组扩展全部完成，再恢复JiT 24000步检查点
至50000步（已有两个头继续训练，避免改变冻结训练协议）。生成实验只用depth4。
不按heldout预测误差选择depth，不同时扫描depth8。

配对1K复用JiT seed202609831、B4、labels0..999，FP32状态/BF16/TF32、
原官方像素量化。主100步Euler，clean输出按原t_eps=.05转换为velocity。
原生data-time前半[0,.5)、后半[.5,1]，不与RAE的noise-time混淆。
IG为F+gamma*(F-W)，类别条件不移除。既有Full/官方CFG评价复用。

先在后段gamma=0下扫描前段0,.35,.6,.9,1.2（0/0复用Full）。
固定首轮最优前段，扫描后段0,.15,.3,.5；再固定选出的后段，检查其余前段。
结果只称该坐标搜索中已评估配置的最好，不称全局最优。全部噪声相同，存在选择偏差。
若前段最佳为0，第二阶段仍按协议检验后段引导；不因误差大小跳过生成实验。

IG配置冻结后，再比较canonical projected PFR：h=min(1/32,.5-t)，仅前半程
且gamma>0时查询。G=F+gamma*(F-W)，C=(1+gamma)*(F-W)，q=z+h*Proj_ray(G)C，
PFR=G+(1+gamma)*(W(z,t)-W(q,t+h))；后半沿用选定IG。不单独优化PFR强度。
然后才比较lifting m=1/2：选定IG前后gamma分别作为alpha，block最多4个
Euler细步、不跨.5；同block重复m次alpha/m写入后Strong Euler推进。
若某段gamma=0则仅Strong，不额外加nativeIG；目标/逆流Heun。

四卡协作一个候选，候选顺序；每组1K后官方评价和配对输入核验，保存调用及耗时。
开始前GPU验证共享Full接口精确复现官方Full，并复现4张已有Full像素。
若接口/输入/数值检查失败停止队列，不能自动忽略错误或继续污染比较。
只有固定最优IG与PFR/lifting的配对比较才能讨论方法增量；官方CFG为实践参照。
有质量信号后仍需独立更大bank确认，不写论文，不据此单独归因RAE多层平均。
