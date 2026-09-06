# 共享噪声的边界一致判别器：推导与未训练准备

2026-09-07。当前仅完成推导、三个解析测试、数据身份准备和小模型代码；没有训练或采样FID。由于固定semantic_orthogonal已经取得1K +2.0318%的更强信号，优先完成其独立5K及必要成本对照，本方向暂不占用GPU。

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

小模型为128宽、2层、4头Transformer，输出f；类别输入使用原RAEv2 checkpoint中冻结的1440维类别embedding，投影后与6个连续时间特征一起输入。输出层零初始化。没有逐时刻外推系数、没有FID拟合。模型目前未训练，训练预算/优化器尚未作为正式实验启动；当前主路线通过则无需执行它。

已固定模型/data source SHA、class embedding身份及所有real/fake文件SHA，见 [数据manifest](../experiments/results/raev2_guidance_20260907/paired_ratio_data.json)。具体代码为 `raev2_paired_ratio_loss.py`、`raev2_paired_ratio_model.py` 和 `prepare_raev2_paired_ratio_data.py`。

## 4. 尚未解决的质量条件

训练负例是模型终点重新加噪的bar_q_t，不自动等于原Euler轨迹实际q_t；native IG场也不自动等于bar_q_t的score。即使分类器最优，修正后还剩 s_G−∇log(bar_q_t) 的自洽误差。这个明确缺口与[旧DG阅读](RAEV2_GUIDANCE_READING_DISCRIMINATOR_20260906_ZH.md)一致，没有被边界参数化或新的配对loss消除。

有限数据、CE函数值与输入梯度的泛化差异、raw latent与decoded质量之间的差别也仍在。因此后续必须检验held-out方向和真实FID，不能从本节的解析训练性质直接申报3%成功。当前优先事项仍是已经获得实际1K收益的固定类别补充。
