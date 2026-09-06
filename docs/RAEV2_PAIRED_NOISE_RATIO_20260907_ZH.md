# 共享噪声的边界一致判别器：推导与一次固定拟合

2026-09-07。semantic_orthogonal独立5K已失败，故启动本方向。三个解析测试及数据身份已验证；现已完成一次预定2048步拟合，独立验证通过预定入口，两个固定1K均已完成且未达3%。早先未训练准备的状态由文末实际拟合记录更新。

出发点是 [Discriminator Guidance](https://proceedings.mlr.press/v202/kim23i.html) 的noisy real/fake density ratio。它与此前在预测clean上对冻结DINO判别器作Gaussian posterior tilt不同；后者的两个固定1K没有实用改善。以下是针对RAEv2线性噪声坐标的本地推导，不是该论文已给出的训练公式，也未完成新颖性检索。

## 1. 先在参数化中消除纯噪声奇点

记a=1−t，真实/生成clean分别为X_p、X_q。同一类内的noisy marginals为

    Z_p=a X_p+t ε,  Z_q=a X_q+t ε.

分类器logit设为 d_θ(z,t,c)=a f_θ(z,t,c)。于是t=1处logit对所有输入精确为0。沿RAE的score→clean坐标换算，t<1有

    δG = (t²/a) ∇_z d_θ = t² ∇_z f_θ.

右式对平滑有限网络可以直接延拓到t=1，无需除零、clamp或手工关闭若干步。固定a>0时，f的这一参数化仍可表达任意有限logit；对有限矩的共同Gaussian加噪分布，高噪声端的log density ratio一阶项确实为a乘一个关于z的线性均值差，因而该边界不是任意加窗。

但d(·,1)=0不单独识别f(·,1)；需要近端训练及连续性。它也没有证明实际native IG就是模型终点重新加噪后的score，见文末边界。

## 2. 配对噪声使信号归一化的训练梯度保持有限

普通BCE在高噪声处的可分信号消失。对同一个a、类别和ε计算 f_p=f(Z_p,t,c)、f_q=f(Z_q,t,c)，使用

    L_a = [softplus(−a f_p)+softplus(a f_q)−2log2] / (2a²).

同一a>0上，减常数及乘正数不改变Bayes density-ratio最优解。真/假的边缘各自正确，共享噪声仅改变损失估计的coupling，不改变其期望。

直接展开softplus：

    L_a = (f_q−f_p)/(4a)
          + [log cosh(a f_p/2)+log cosh(a f_q/2)]/(2a²).

第二式以FP64算标量部分，避免先减去两个近似log2造成严重消减。模型本身必须使用FP32而非BF16/TF32，否则极小输入差会破坏配对抵消。当前100步最小非零a约0.00126；不将a=0直接传入此正信号损失。

在f及其所需导数连续、可交换极限且有可积控制时，t→1有

    f_q−f_p = a ∇_z f(ε,1,c)·(X_q−X_p)+O(a²),
    L_a → 1/4 ∇f(ε,1,c)·(X_q−X_p)+1/8 f(ε,1,c)².

若两边使用独立ε，第一项一般为O(1/a)随机量；共享ε使其有有限极限。对Gaussian ε用Stein恒等式，边界总体目标的最优函数是

    f*(ε,1,c) = (E[X_p|c]−E[X_q|c])·ε.

这与真实高噪声log-ratio的一阶系数一致。相应光滑性/矩条件下，参数梯度也获得同样抵消；没有声称任意ReLU网络或有限精度都自动满足严格方差界。

三项CPU测试通过：稳定式与正信号BCE恒等；已知Gaussian密度比的参数梯度在Gauss-Hermite积分下小于1e−12；高噪声线性模型梯度收敛到上述极限，共享噪声消去发散项。测试不包含RAEv2图像质量。

## 3. 已准备的数据与模型

真实训练5000张、验证1000张，原始ImageNet source row没有重叠。生成训练采用历史seed20260801的全部5K IG1.78 endpoint，生成验证来自seed20260802的5K。两组历史数据已用于先前研究，验证仅对未来这次参数拟合保留，不冒充从未见过的最终质量确认。真实编码为FP32后存FP16，历史生成B4且终点存FP16，与当前B8实际trajectory的差异保留。

小模型为128宽、2层、4头Transformer，输出f；类别输入使用原RAEv2 checkpoint中冻结的1440维类别embedding，投影后与6个连续时间特征一起输入。输出层零初始化。没有逐时刻外推系数、没有FID拟合。以上是准备时状态；之后主路线的独立5K失败，本模型按第5节的单次固定预算完成训练。

已固定模型/data source SHA、class embedding身份及所有real/fake文件SHA，见 [数据manifest](../experiments/results/raev2_guidance_20260907/paired_ratio_data.json)。具体代码为 `raev2_paired_ratio_loss.py`、`raev2_paired_ratio_model.py` 和 `prepare_raev2_paired_ratio_data.py`。

## 4. 尚未解决的质量条件

训练负例是模型终点重新加噪的bar_q_t，不自动等于原Euler轨迹实际q_t；native IG场也不自动等于bar_q_t的score。即使分类器最优，修正后还剩 s_G−∇log(bar_q_t) 的自洽误差。这个明确缺口与[旧DG阅读](RAEV2_GUIDANCE_READING_DISCRIMINATOR_20260906_ZH.md)一致，没有被边界参数化或新的配对loss消除。

有限数据、CE函数值与输入梯度的泛化差异、raw latent与decoded质量之间的差别也仍在。因此后续必须检验held-out方向和真实FID，不能从本节的解析训练性质直接申报3%成功。这个自洽缺口是后续实际轨迹分布诊断的重点，见第8节。

## 5. 单次固定拟合与准入结果

训练开始前冻结：2048次更新，global B64（4×16对），AdamW lr1e−4、weight decay .01、betas(.9,.999)、clip-norm1，无学习率搜索或时间窗，最后一次update作为唯一checkpoint。上述优化器值是常规工程设置，不宣称理论唯一；没有按FID选择训练预算。训练时间均匀覆盖原100-step shifted Euler中索引1..99的正信号输入点；共享每对的类别/时间/noise，类内真实与生成endpoint独立均匀抽取。纯噪声端由连续参数化延拓，不向正信号loss传入a=0。

独立验证使用seed20260802的全部5000个fake endpoint与每类1张held-out real；原99个时间点平衡排列后固定shuffle。只在最后checkpoint上验证，不作best-checkpoint选择。预定准入条件是scaled loss均值加2个按1000类cluster计算的标准误低于零判别器。实测均值−10.1495059，SE .3469588，上界−9.4555883，满足条件；这不能证明输入梯度泛化，更不证明FID。

训练耗用308.0984 GPU秒、端到端约87.35秒；762753可训练参数。固定clip在rank0的2048步中触发2022次，保留这个优化限制，未重新训练或改阈值。checkpoint SHA为 `9570104c61fa9b551c95ae93e28e82a7b6a077d466746eed58a7ed3191525560`。所有数据SHA和源文件快照在运行首尾核验。采样预定为原native BF16 G + t²∇f，强度1、全部100时刻，seed202609071、1000图；通过真实梯度检查与original official8逐像素一致性后才执行。

[完整固定fit计划](../experiments/results/raev2_guidance_20260907/paired_ratio_fit_plan.json)。早先未训练状态保留为研究历史，不代表当前执行状态。

## 6. 实际梯度与完整轨迹验证

首次纯FP32中心差分在t=.5出现3.25%相对误差，预检立即停止，当时尚无新图像；该失败日志保留。随后对完全相同权重和输入以FP64求梯度及差分：部署FP32梯度相对FP64梯度的最大L2误差4.35e−5，FP64差分在h=.001时最大相对误差约2.54e−4。小导数的FP32函数值相减分辨率不足解释了原问题，未更改权重/采样梯度公式，也未放宽原FP32数值阈值来掩盖失败。t=1的修正RMS约.0149564，t≈.07477为1.46e−7，均有限。零输出头产生精确零修正。参见 [梯度审计](../experiments/results/raev2_guidance_20260907/paired_ratio_gradient_audit.json)。

完整8图的original official输出与最初anchor逐像素相同；paired_ratio轨迹有限，每图仍100次主模型调用，另100次小判别器输入梯度。固定1K已启动，未改变最后checkpoint、强度1或全时间区间。失败预检和修正后检查分别保存在continuation与continuation_v2，后者没有重启训练或重采任何已生成图像。参见 [8图记录](../experiments/results/raev2_guidance_20260907/paired_ratio_smoke.json)。

## 7. 单个概率温度的理论、校准与负结果

未校准1K完成后，分类风险仍明显非驻点，故检查一次全局概率校准：对已固定f，求 alpha≥0 下原paired_scaled_logistic(alpha f_p, alpha f_q, a)的最小值。该目标是一维凸函数；二阶导为 [f_p² sech²(a alpha f_p/2)+f_q² sech²(a alpha f_q/2)]/8，非退化时严格为正。求导根只是在固定分类数据上做一个标量的似然拟合，不逐个温度生成图像或评估FID，不引入分时参数。Bayes proper-scoring结构只保证此函数族内的风险最优，不保证任意受限分类器都是精确log density ratio。

固定偶数类别估计alpha，奇数类别仅验证。在原99点FP32时间网格上，alpha=1.5041111779，按类sandwich SE=.0733203；奇数类别相对alpha1的损失差−1.375493，按类SE=.266750，满足预先说明的正向校准条件。采样仅有这一校准值，仍用全部100时刻、同一个2048步checkpoint；原strength1结果保留。原方案FID38.442135（+0.115985%），校准方案38.567599（−0.210008%），均未达标。推理成本、训练成本、两条样本SHA和独立FID复算见 [归档](../experiments/results/raev2_guidance_20260907/paired_ratio_screens.json)。没有因校准分类损失通过而宣称质量成功，也不继续改变温度。

## 8. 实际采样分布上的失败定位

同一个固定critic在新seed202609089、64个预定散布类别的所有99个positive-signal节点上，对重新加噪终点保持负loss（−7.511846），但对原生实际轨迹变为+3.566283，已差于零判别器。差11.078128，按类SE1.275236。对应64条native轨迹、真实/历史生成endpoint都共享新独立初始noise；后两者是合法加噪边缘，实际轨迹直接由原模型推进。该检查没有生成/筛选图像或拟合新参数。它支持把当前的q_bar与实际q的混淆作为下一实验的具体问题，不能据此声称已经识别最优score差。详见 [诊断与实际状态数据方案](RAEV2_ACTUAL_RATIO_20260907_ZH.md)。
