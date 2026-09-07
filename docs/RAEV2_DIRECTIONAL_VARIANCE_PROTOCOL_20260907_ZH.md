# 第4轮：沿原生双头分歧的单标量条件协方差

第5轮结果更新：固定1K与5K均已完成独立审核。1K FID38.610371（相对official −.3211%），5K FID6.910567612851594（相对official +.564060%，相对原全局方差 +.495737%）。推理成本约1.014倍。未达到3%，不再增加协方差方向或比例；下文保留采样前的协议和阶段记录。

这是最后八轮内的一个固定扩展：保留已经训练完的条件方差头，只拟合一个全局正数。若原条件方差5K已达到3%，先做成果审核，不启动本扩展；否则提取一次方向统计，按预定留出门槛决定是否完整做1K/5K。此项之后不再变更协方差方向、维数或倍率。本协议写于该标量数据提取及采样之前。

## 两个理论不能直接叠加

本轮重读仓库的[Stochastic Interpolants原文核对](RAEV2_GUIDANCE_READING_STOCHASTIC_INTERPOLANTS_20260906_ZH.md)、[CFG++正式版与代码核对](RAEV2_GUIDANCE_READING_CFGPLUSPLUS_20260906_ZH.md)、[动态控制](RAEV2_GUIDANCE_READING_ADAPTIVE_CONTROL_20260906_ZH.md)、[反馈与双差](RAEV2_GUIDANCE_READING_DOUBLE_DIFFERENCE_20260906_ZH.md)，并再次核实[PMLR covariance原论文](https://proceedings.mlr.press/v162/bao22d.html)和[JMLR Stochastic Interpolants](https://jmlr.org/papers/v26/23-1605.html)正式来源。没有增加新论文篇数，也没有声称重新通读全部原文或运行其代码。

当前桥方差来自给定Euler均值的条件矩匹配；再加一个同点Langevin补偿会改变这个均值。以X~N(0,4)、t=.6、s=.5为例，Z_t方差1，精确posterior均值1.6Z_t、方差1.44；Euler条件均值1.1Z_t，补上q² posterior variance=.04后，下一点方差恰为1.25。若再加`.5*.04*score_t(Z_t)=−.02Z_t`，方差变为1.2064，反而破坏精确结果。这否决的是直接叠加两种更新，不是否认另行推导的SDE。CFG++的一阶重组又可落回时间增益；FBG及控制论文中的幅度/窗口未由本地误差识别，因此不把这些文献直接改造成新扫描。

## 同一固定均值下，改用一个有方向的协方差族

原生已有F和B，定义d=F.float()−B.float()，u=d/||d||。方向不选层、不按样本挑候选，也不另定义质量reward；它是预训练模型已有的两种深度预测分歧。零分歧时不定义一个任意方向，直接使用原球形核。

令G保持原native BF16 IG算术，m(z,t,c)>0是已经通过独立风险及像素验证的条件方差头，权重固定为SHA `4b714004cd57addae8c833970574c13fd6f3c3addfb9b119a913ae0767108faa`。候选为

    C_kappa(z,t,c) = m(z,t,c) [I + (kappa−1) u u^T]，kappa>0，
    Z_s = native_Euler(Z_t,G) + q C_kappa^(1/2) ξ，q=(t−s)/t。

沿u的特征值是kappa*m，其余D−1维是m，因此这是SPD族。kappa=1严格退化到当前条件球形方法。每图每步仍仅一次原生主前向，无输入反传，无额外前缀；只做一个向量投影和同一Gaussian噪声的线性变换。它没有保持trace的附加重缩放，新增方差与成本如实报告。

对真实forward配对e=X−G，固定G、m、u。秩一矩阵的determinant和inverse给出相对kappa=1的Gaussian NLL差：

    Δℓ = .5 [log kappa + beta (1/kappa−1)]，
    beta = (u^T e)^2/m。

这是完整D维NLL差，除以D才是每坐标值；不把两种单位混在一起。置h=log kappa，训练风险的二阶导为`.5 E(beta) exp(−h)>0`，唯一最优解直接为

    kappa_hat = mean_train beta。

只在d非零的记录上求该均值；d=0时协方差与kappa无关，风险差为0。均值误差和其方向性都保留在e中，不把它称作纯Bayes posterior covariance。该理论与Bao等固定不完美均值的covariance最优性同源；选用原双头方向是本地假说，不是原论文证明这个方向有效。

## 唯一数据与预定裁决

复用原64K train/8K validation的真实latent、initial epsilon、全部100个查询时刻和native B8分片，不重训练条件方差头。重新做每样本一次teacher前向，仅增加方向投影残差；新提取的全部原生特征和总MSE必须与上一阶段已保存数组逐位相同。m由同一冻结头在这些同一特征上计算。已完成输入SHA核验按不可变原件记录复用，新增数组完整散列。旧验证集在历史研究中已用过，按类SE仅作描述性门槛，不称全新未触及验证。

直接取train beta均值，**没有优化器、正则、参数网格、强度回扫或best checkpoint**。验证用全部8K记录：相对kappa1的NLL差按1000类汇总，均值加2SE须低于0；全部值必须有限。这个门槛只决定是否采样，不随结果改变。各time统计全部报告，但不挑有利time，也不拟合100个系数。

若通过，只测kappa_hat这一个候选。先做原official8、原条件方差8、kappa1新代码8的全像素与输入身份对齐；之后原seed202609071的1K和202609072的5K都完整执行，不因1K负号取消5K。与原official、历史interval、条件球形控制比较；独立FID与完整样本身份审核，报告新增72000个teacher查询、原头准备费及实际推理费。小样本FID和两个规模的seed差异继续披露。

可能失败的原因很具体：Full/Base分歧可能主要是有益guidance偏置，向该方向补方差会削弱它；一个全局比例可能无法描述时间/类别变化；真实forward的方向残差未必迁移到实际guided状态；最优Gaussian NLL也不等于FID。失败后不改成分时比例或另找方向，最多第8轮完成归档和Git收束。

## 已完成的准入和实现检查

固定72000个teacher前向已完成，全部新特征和总MSE与先前缓存逐位相同，train/validation非零方向分别64000/8000；额外提取worker耗时513.533273秒（含输入读取），CPU取均值及验证 .038951秒。未改原条件方差头。训练给出唯一 **kappa=27921.89940172958**，按类sandwich SE100.036234；留出完整D维NLL差−13954.626952，按类SE104.268161、两SE上界−13746.090629，原门槛通过。每坐标差−.0532326773。D=262144，因此沿一个方向的较大比例对应总体trace增加 **10.65097786%**，不是将整个噪声方差放大27922倍，也不是手调guidance外推系数。

三个CPU检查通过：有限Gaussian桥与错误补偿的反例、确定性正负基底的协方差精确积分、logdet/inverse形式与秩一似然差的等价及均值最优点。原official8与旧anchor完整像素相同；原条件方差8保持原样；新kappa1代码的8图也与该条件方差anchor完全相同。采样补丁只在方向统计父进程退出后按SHA应用。实际唯一候选8图轨迹有限，固定1K启动，随后同参数5K。

[解析检查](../experiments/results/raev2_guidance_20260907/directional_variance_analytic_checks.json)、[完整矩数据身份](../experiments/results/raev2_guidance_20260907/directional_variance_moments_complete.json)、[唯一标量与所有时间统计](../experiments/results/raev2_guidance_20260907/directional_variance_calibration.json)、[全像素检查](../experiments/results/raev2_guidance_20260907/directional_variance_smoke8.json)。锁目录manifest中的`applied:false`保存应用前计划；真实激活状态见执行记录，不覆盖历史计划。所有似然数值仍不能替代正式FID结果。

## 固定1K结果与5K状态

唯一kappa的1K已完成并独立审核：FID **38.610371195124856**，相对原official38.486773927092486恶化 **.3211421884%**，也差于球形条件方差38.54711465462702与历史interval38.335024。实测推理成本为原official **1.0130297038倍**；不同阶段的微小耗时变化不证明新增投影让推理变快。样本SHA `1507e1f52abe3310e34439afd19b91fb20519ab85bca0bf9d5297ba54796f9b3`。独立Gram谱FID、全部合并像素、初始noise与类别、原头/单标量/配置/decoder及源码快照审核均通过。

[1K完整审核](../experiments/results/raev2_guidance_20260907/directional_variance_screen1k_audit.json)。8K验证的100个时间组都降低该Gaussian NLL，但没有因此获得1K FID改善；这尤其提示固定均值的误差二阶矩不等于对生成质量有益的随机校正。仍按预定计划做同一kappa、同一条件方差头、全部100点的seed202609072完整5K；第4轮结束时它正在运行。没有裁剪27921.8994或把它改成逐时刻比例。
