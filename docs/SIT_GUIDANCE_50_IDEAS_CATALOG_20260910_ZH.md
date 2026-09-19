# 已完成 50 idea 队列的候选说明

以下从冻结请求 `4589f50aac4b3932e95fc8bdb64a96c6430ba403ee1287bae83d0f79fbf3c011` 导出，参数与实际完成的 600 组候选一致。这是运行后的可读说明，不把新文档冒充预先冻结的协议。每项 4 个 strength × 3 个结构参数；完整结果见 [结果报告](SIT_GUIDANCE_50_IDEAS_RESULTS_20260910_ZH.md)。

IG strength=(.5,.65,.8,.95)，t<.25 乘6/7、.25≤t<.5 取峰值、其后0；CFG strength=(.5,1,1.5,2)，t<.75启用。所有候选使用64步Heun；历史只在接受步后更新，同一步两个stage共享扰动。原生IG另测12点Dopri5；CFG对照在cutoff=.5/.75各测12点。

S/W 是条件强/弱速度，U 是 null Strong，D=S−W，Dc=S−U，b=1−t，m=z+bS，C 是既有浅层特征预测 gap；N_D 为恢复到原 gap 范数。第36项方向诊断基于替代类别gap；第23项Strang无普通RHS诊断。第49项原始对称线性误差场保守，cap后不保证保守。所有理论依据均是局部代数或建模假设，不是FID保证。

## 01. CFG：带反向动量的clean投影

键：`cfg_apg_momentum`；beta=[-0.75, -0.5, -0.25]。

构造：`M=Dc+beta*Mprev; u=P_m^perp clip(M,2||Dc||); v=S+a*u`。

依据：反向动量压低相邻预测共有的增益，clean正交投影去掉一阶径向放大；检验过饱和是否是当前CFG的限制。

仓库近邻：[旧记录](RAEV2_GUIDANCE_READING_PROJECTION_ZERO_20260906_ZH.md)。变化：旧记录仅精读APG且刚才球面方法没有动量；本次用真实null分支、完整动量顺序和相对范数上限。

限制：投影依赖latent坐标原点；不保证保范数、语义或FID；属于已有方法适配。

参考：[APG](https://arxiv.org/html/2410.02416v2)。

## 02. CFG：向1收缩的弱速度回归

键：`cfg_zero_ridge`；ridge=[0.0, 0.1, 0.5]。

构造：`s=(<S,U>+r||U||²)/( (1+r)||U||² ); u=S-s*U`。

依据：这是min_s ||S-sU||²+r||U||²(s-1)²的闭式解；向原CFG收缩可减少小弱速度下的比例过拟合。

仓库近邻：[旧记录](RAEV2_GUIDANCE_READING_PROJECTION_ZERO_20260906_ZH.md)。变化：保留CFG-Zero的正确velocity坐标，增加有目标函数的ridge；不采用有训练阶段依赖的zero-init。

限制：最小化局部上界并不保证真实误差更小；r=0是文献组件的明确对照。

参考：[CFG-Zero*](https://arxiv.org/html/2503.18886v2)。

## 03. CFG：保通道均值的方差回缩

键：`cfg_channel_rescale`；mix=[0.25, 0.5, 0.75]。

构造：`mg=m+b*a*Dc; R=mean(mg)+std(m)/std(mg)*(mg-mean(mg)); G=(1-r)mg+rR`。

依据：分别处理颜色均值与通道内增益，避免全图范数恢复把两种误差一起压缩。

仓库近邻：[旧记录](SMALL_SIT_NONLINEAR_PREDICTION_RESULTS_20260909_ZH.md)。变化：旧sphere约束whole-image长度且全部失败；此次保持guided均值，仅对每个latent通道的中心方差做收缩。

限制：有限Strong中心方差不是数据真实方差；这是可检验的增益校准假设。

参考：[Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891)。

## 04. CFG：局部修正的凸预算

键：`cfg_patch_budget`；radius=[0.5, 1.0, 2.0]。

构造：`u_i=Dc_i*min(1,r*rms(Dc)/rms(Dc_i)); i为2x2 latent patch`。

依据：投影到逐patch欧氏球是唯一最近修正，可抑制少数位置耗尽全图guidance预算的离群行为。

仓库近邻：[旧记录](RAEV2_SPATIAL_ENERGY_BALLS_RESULTS_20260906_ZH.md)。变化：旧方法约束整批样本的DC/AC状态能量；本次不耦合样本，只限制单图guidance增量的局部尾部。

限制：强修正也可能是真实细节；球半径是调参量，不冒充真实posterior置信域。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)。

## 05. IG：通道椭球中的修正约束

键：`ig_channel_metric`；ridge=[0.1, 0.5, 2.0]。

构造：`C=cov_spatial(m)+r*tr(C)/4*I; u=C^(1/2) clip(C^(-1/2)D, per-pixel RMS budget)`。

依据：四个VAE通道的尺度和相关性不同；正定度量内的球投影避免把高方差与低方差通道同等惩罚。

仓库近邻：[旧记录](RAEV2_GUIDANCE_READING_COVARIANCE_MISMATCH_20260906_ZH.md)。变化：替换旧全latent球形或低秩precision假设为带显式ridge的4维通道度量；不求两posterior精度之差。

限制：度量来自当前预测空间统计，不能称真实条件协方差；颜色结构可能受损。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)。

## 06. IG：Haar细节的稀疏近端修正

键：`ig_wavelet_shrink`；threshold=[0.25, 0.5, 1.0]。

构造：`u=Haar^-1(LL(D), soft(detail(D),r*MAD(detail)))`。

依据：正交Haar域的soft threshold精确解细节L1正则近端问题，保留低频并压掉小而散的差值。

仓库近邻：[旧记录](SPECTRAL_SELF_GUIDANCE_LITERATURE_THEORY_PLAN_ZH.md)。变化：旧静态频带乘法与能量选层失败；改用位置相关的非线性稀疏修正，检验尾部结构而非静态频谱能量。

限制：细节小不等于误差；Haar网格可产生块效应；不声称自动识别语义高频。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)。

## 07. IG：对不可预测差值使用有界影响

键：`ig_huber_innovation`；threshold=[0.5, 1.0, 2.0]。

构造：`R=D-C; u=N_D( R/sqrt(1+(R/(r*rms(R)))²) )`。

依据：pseudo-Huber影响函数对大坐标增长有界；只对已移除浅层可预测部分的创新应用，避免让拟合离群值旋转整图方向。

仓库近邻：[旧记录](SMALL_SIT_PREDICTABLE_GAP_RESULTS_20260909_ZH.md)。变化：旧残差只做整图范数恢复，未处理坐标尾部；本次保留原gap总幅度，改变尾部支配的方向。

限制：保范数会重新分配能量；有界影响不是对实际生成误差的统计识别。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)。

## 08. IG：按预测边界平滑差值

键：`ig_edge_resolvent`；penalty=[0.1, 0.5, 2.0]。

构造：`u≈(I+r*L_m)^-1 D; L_m为由相邻clean预测差定义的对称加权图Laplacian`。

依据：二次图平滑约束同一局部区域的修正一致，边界处权重较小；正定resolvent无需不稳定精度相减。

仓库近邻：[旧记录](IMAGENET100_SIT_MULTISCALE_GUIDANCE_RESULTS_ZH.md)。变化：旧Fourier控制不随样本结构变化；本次根据当前clean边界构造空间度量，用固定Jacobi迭代求解。

限制：VAE latent边界不一定对应RGB语义边界；有限迭代只近似解目标。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)。

## 09. IG：部分移除浅层可预测差值

键：`ig_partial_residual`；subtraction=[0.25, 0.5, 1.5]。

构造：`u=N_D(D-r*C)`。

依据：完整减去线性预测要求其幅度校准正确；收缩估计能在可预测偏差和拟合方差间折中。

仓库近邻：[旧记录](SMALL_SIT_PREDICTABLE_GAP_CONFIRMATION_RESULTS_20260909_ZH.md)。变化：明确挽救旧固定r=1的小幅5K正结果，联合调减除率与guidance强度；不把同一算子重命名为新理论。

限制：这是参数化优化的旧路线；已看过的训练和旧5K不算本轮独立证据。

参考：[AutoGuidance](https://arxiv.org/abs/2406.02507)。

## 10. IG：仅移除可预测方向的重叠

键：`ig_predictable_subspace`；removal=[0.25, 0.5, 1.0]。

构造：`u=N_D(D-r*P_C D)`。

依据：当C的方向比幅度可信时，只删除其张成子空间中的分量，可避免直接相减的标定要求。

仓库近邻：[旧记录](SMALL_SIT_PREDICTABLE_GAP_RESULTS_20260909_ZH.md)。变化：旧parallel_norm把新方向投回原D；这里反过来，从D中消除C子空间，二者代数与可失败预测不同。

限制：C子空间也可能含有有效guidance；投影本身没有质量保证。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)。

## 11. IG：对创新的重复误差模式降权

键：`ig_residual_whitening`；ridge=[0.1, 0.5, 2.0]。

构造：`u=N_D((Sigma_R+r*tr(Sigma_R)/16*I)^(-1/2)(D-C))`。

依据：从独立训练缓存估计16维patch创新协方差，对反复占优的相关方向降权；ridge限制小特征值放大。

仓库近邻：[旧记录](SMALL_SIT_PREDICTABLE_GAP_RESULTS_20260909_ZH.md)。变化：旧残差仅用标量norm；本次使用训练缓存的通道/patch形状信息，且不在FID bank上估计矩。

限制：创新协方差不是Strong误差协方差；白化可能放大本来无用的方向。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)。

## 12. IG：按可解释能量的广义方向降权

键：`ig_explained_eigenfilter`；penalty=[0.5, 1.0, 2.0]。

构造：`A=Sigma_D^-1/2 Sigma_C Sigma_D^-1/2; u=N_D(Sigma_D^1/2(I+r*A)^-1 Sigma_D^-1/2 D)`。

依据：在统一gap尺度内，优先压低浅层预测反复解释的方向；正定收缩不要求Sigma_D-Sigma_C为正定。

仓库近邻：[旧记录](SMALL_SIT_PREDICTABLE_GAP_RESULTS_20260909_ZH.md)。变化：用二阶可解释结构替代逐状态C相减，区别于旧精度平衡的非正定矩阵逆。

限制：可解释不等于有害，两个协方差的估计偏差仍会改变排序。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)。

## 13. IG：按回归外推风险保护原gap

键：`ig_leverage_residual`；leverage_scale=[0.5, 1.0, 2.0]。

构造：`ell=x^T(Gram+lambda*I)^-1x; u=N_D(D-C/(1+ell/(r*median_train(ell))))`。

依据：ridge预测方差随设计杠杆增大；生成特征远离训练支撑时，减少对新线性读出的依赖。

仓库近邻：[旧记录](SMALL_SIT_FEEDBACK_READER_RESULTS_20260909_ZH.md)。变化：旧低训练MSE读出在闭环FID恶化；此次不扩大网络，以训练设计矩阵给出状态相关收缩。

限制：杠杆只是固定线性模型的外推指标，不检测所有闭环偏移。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)。

## 14. IG：以真实配对残差校正Strong

键：`ig_target_error_readout`；error_mix=[0.25, 0.5, 1.0]。

构造：`Ehat(h)=ridge(target-S|h_weak); v=S+a*(D+r*cap(Ehat,||D||))`。

依据：真实paired target-S的条件均值能识别Strong的系统误差；保留原IG，只对估计误差添加受限修正。

仓库近邻：[旧记录](SMALL_SIT_FEEDBACK_READER_RESULTS_20260909_ZH.md)。变化：旧双MLP同时改变强弱头而破坏guidance；此次用一个闭式线性误差读出，固定原双头并联合调收缩。

限制：局部MSE改善不蕴含FID改善，仓库已有反例；这是显式挽救而非新保证。

参考：[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)。

## 15. IG：分噪声层的可预测差值

键：`ig_time_residual`；time_bins=[2.0, 4.0, 8.0]。

构造：`C_j(h)=ridge(D|h,t in bin_j); u=N_D(D-C_bin(t)(h))`。

依据：共享一个线性读出可能平均掉不同噪声段的系统偏差；分层估计允许偏差随时间变化。

仓库近邻：[旧记录](SMALL_SIT_PREDICTABLE_GAP_RESULTS_20260909_ZH.md)。变化：旧6160系数跨整个t∈[0,.5]共享；新读出只按预先固定时间分箱拟合，不用FID挑训练样本或边界。

限制：更多分箱增加估计方差与参数；不是自动发现最优guidance时间窗。

参考：[AutoGuidance](https://arxiv.org/abs/2406.02507)。

## 16. IG：对差值做因果时间降噪

键：`ig_gap_ema`；memory=[0.2, 0.5, 0.8]。

构造：`u=(1-r)D+r*EMA_previous(D); EMA只在完成采样步后更新`。

依据：若短尺度波动主要是预测误差，因果平均降低方差；保留强场实时更新，避免完整预测的语义滞后。

仓库近邻：[旧记录](IG_HISTORY_SEPARATED_RESULTS_20260908_ZH.md)。变化：旧影子轨迹将当前状态也分开；本次只存同一路径的gap，不额外演化另一条图像。

限制：时间相关误差不会按独立样本方差下降；快速真实变化会被拖慢。

参考：[HiGS](https://arxiv.org/html/2509.22300v1)。

## 17. IG：补偿EMA的线性趋势滞后

键：`ig_gap_linear_trend`；memory=[0.2, 0.5, 0.8]。

构造：`L=(1-r)D+r(Lprev+Tprev); T=(1-r)(L-Lprev)+r*Tprev; u=cap(L+T,2||D||)`。

依据：Holt趋势状态分开估计水平与局部线性变化，检验普通时间平均的偏差是否限制效果。

仓库近邻：[旧记录](SPECTRAL_SELF_GUIDANCE_LITERATURE_THEORY_PLAN_ZH.md)。变化：不是把任意旧预测当更弱模型；只预测gap的下一步趋势，明确限制外推幅度并使用实际步长。

限制：趋势只是局部假设，转弯时可能放大误差；与EMA属于相关但可区分的假设。

参考：[HiGS](https://arxiv.org/html/2509.22300v1)。

## 18. CFG：有界clean历史创新

键：`cfg_higs`；history_gain=[0.025, 0.05, 0.1]。

构造：`G=m+b*a*Dc; Gnew=G+r*a*cap(G-EMA_previous(G),b||Dc||)`。

依据：用guided clean预测的历史差检验当前新细节能否补充CFG；限制额外创新避免把历史差当无限外推方向。

仓库近邻：[旧记录](SPECTRAL_SELF_GUIDANCE_LITERATURE_THEORY_PLAN_ZH.md)。变化：旧文档只提出HiGS重合警告；本次明确列为文献启发适配，固定EMA=.8和活动区间，不声称完整复现原论文过滤器。

限制：历史不是独立观测，新增细节未必真实；与原文结果不能直接混用。

参考：[HiGS](https://arxiv.org/html/2509.22300v1)。

## 19. IG：利用已完成步的clean二阶变化

键：`ig_clean_curvature`；curvature_gain=[0.025, 0.05, 0.1]。

构造：`A=m_k-2*m_(k-1)+m_(k-2); u=D+r*cap(A/b,||D||)`。

依据：在等距网格上二阶差分对仿射clean变化为零，隔离相邻预测的转弯而不是重复一阶增益。

仓库近邻：[旧记录](RAEV2_PFR_TEMPORAL_CURVATURE_RESULTS_20260908_ZH.md)。变化：旧future curvature额外查询跨噪声时间；本次用实际接受的过去clean点，不增加模型调用，固定小信任幅度。

限制：没有曲率朝向真实误差的普遍保证；这是旧曲率路线的低成本、有界挽救。

参考：[HiGS](https://arxiv.org/html/2509.22300v1)。

## 20. IG：带回拉的影子差值

键：`ig_leaky_shadow`；tether=[1.0, 4.0, 16.0]。

构造：`dy/dt=S(y,t)+r(z-y); u=N_D(S(y,t)-W(y,t))`。

依据：若局部Strong Lipschitz常数小于回拉率，状态分离有受迫稳定界；降低旧影子参考持续漂移的混杂。

仓库近邻：[旧记录](IG_HISTORY_SEPARATED_RESULTS_20260908_ZH.md)。变化：旧影子完全不回拉且FID恶化；此次强弱仍在同一影子状态读取，但显式控制影子与主路径分离。

限制：未证明实际Lipschitz条件成立；增加完整前向并可能丢掉有效历史差。

参考：[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)。

## 21. IG：把影子差值局部运回主状态

键：`ig_shadow_transport`；transport_mix=[0.25, 0.5, 1.0]。

构造：`u=N_D(D(y)+r*(D(y+eps*(z-y))-D(y))/eps); shadow tether=4`。

依据：一阶Taylor搬运显式补偿参考读取位置差；在有界二阶导数下剩余误差随两状态距离平方增长。

仓库近邻：[旧记录](IG_GUIDANCE_HISTORY_20260908_ZH.md)。变化：旧跨状态读数混入大公共历史响应；本次搬运同状态gap并固定回拉，有限差分eps=.1。

限制：Taylor界并未在真实模型上全局验证；额外双头查询必须计费。

参考：[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)。

## 22. IG：带信任上限的隐式差值读取

键：`ig_implicit_gap`；horizon_steps=[0.5, 1.0, 2.0]。

构造：`u0=D(z); u_(j+1)=D(z+r*h*u_j), j=0,1; u=cap(u2,2||D||)`。

依据：rho*h*Lip(D)<1时固定点迭代为压缩映射，近似隐式gap更新；检验显式guidance的反馈放大。

仓库近邻：[旧记录](SMALL_SIT_PRECISION_BALANCE_RESULTS_20260909_ZH.md)。变化：旧precision逆近奇异并严重变差；这里不做精度相减或矩阵逆，以两次有界前向迭代尝试隐式反馈。

限制：只在局部条件下收敛，两次迭代不是精确解；隐式稳定也不保证质量更好。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)。

## 23. IG：对称组合Strong流与gap流

键：`ig_strang_split`；gap_substeps=[1.0, 2.0, 3.0]。

构造：`Phi_S(h/2) composed with Phi_(aD)(h) composed with Phi_S(h/2); each subflow uses Heun`。

依据：对称Strang分裂消除非对称Lie分裂的一阶交换子误差；检验之前finite write失败是否含明显排序偏差。

仓库近邻：[旧记录](SMALL_SIT_CARRIER_FLOW_RESULTS_20260909_ZH.md)。变化：旧write-before/after是单侧组合；本次使用对称半Strong步，主时间只在Strong子流推进，gap时间固定在中点。

限制：分裂更准确不等于更接近真实分布，成本明显更高；不能忽略子流查询。

参考：[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)。

## 24. IG：只对差值做水平反射平均

键：`ig_flip_average`；average_mix=[0.25, 0.5, 1.0]。

构造：`u=(1-r)D(z)+r*(D(z)+flip^-1 D(flip(z)))/2`。

依据：当数据/模型目标近似水平反射对称，Reynolds平均消除差值的非等变部分，保留当前Strong构图。

仓库近邻：[旧记录](RAEV2_AFFINE_REFLECTION_RESULTS_20260906_ZH.md)。变化：旧反射作用于RAE公共仿射法向；本次是图像平面水平反射且只平均guidance，不直接平均Strong输出。

限制：ImageNet类别并非严格反射不变；降低非等变风险不是FID保证。

参考：[AutoGuidance](https://arxiv.org/abs/2406.02507)。

## 25. IG：平均patch网格相位差

键：`ig_patch_phase_average`；shift_pixels=[1.0, 2.0, 4.0]。

构造：`u=(D+shift^-1 D(shift(z))+shift D(shift^-1(z)))/3`。

依据：对相反空间移位平均，可减弱固定patch分割相位的伪影；与镜像针对不同的非等变误差。

仓库近邻：[旧记录](IMAGENET100_SIT_MULTISCALE_GUIDANCE_RESULTS_ZH.md)。变化：旧frequency乘法不改变tokenization相位；本次用1/2/4 latent像素平移后拉回的实测模型响应。

限制：周期边界是假设，位置编码会破坏严格平移对称；与反射平均共享数学工具而非独立新理论。

参考：[AutoGuidance](https://arxiv.org/abs/2406.02507)。

## 26. IG：双头差值的对称扰动平均

键：`ig_antithetic_query`；noise_fraction=[0.02, 0.05, 0.1]。

构造：`u=(D(z+r*b*xi)+D(z-r*b*xi))/2; xi在每个实际采样步固定`。

依据：对称查询消除Taylor展开的奇数项，近似局部平滑；重点检验gap的查询噪声而非单头偏移。

仓库近邻：[旧记录](PFR_POSTERIOR_QUERY_NOISE_PROTOCOL_20260908_ZH.md)。变化：旧单侧弱头随机查询混入公共变化；本次正负扰动同时查询强弱，且两次Heun阶段复用同一扰动。

限制：有限平滑带来O(r²)偏差；xi不是已识别的真实posterior样本。

参考：[AutoGuidance](https://arxiv.org/abs/2406.02507)。

## 27. IG：固定预测signal的噪声旋转

键：`ig_noise_rotation`；rotation=[0.1, 0.2, 0.4]。

构造：`e=z-t*S; z±=t*m+b*(sqrt(1-r²)*e±r*xi); u=(D(z+)+D(z-))/2`。

依据：若预测noise为独立标准Gaussian，该旋转保持noise边缘，减少单纯增加噪声方差的混杂。

仓库近邻：[旧记录](SIT_OU_OUTPUT_CONTROL_PROTOCOL_20260908_ZH.md)。变化：旧OU/PFR改弱参考时间或输出；本次同时间保预测signal，旋转只作用noise并在差值中消公共响应。

限制：模型预测noise与clean通常相关且不标准；保边缘结论只在明确理想条件下成立。

参考：[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)。

## 28. IG：保留跨频带的差值响应

键：`ig_cross_band_response`；deletion=[0.02, 0.05, 0.1]。

构造：`q=z-r*t*P_low(m); E=P_high(D(z)-D(q))/r; u=D+.1*cap(E,||D||)`。

依据：线性平稳、频率对角的模型中跨带响应为零；排除直接锐化同一频带的零假设。

仓库近邻：[旧记录](SPECTRAL_SELF_GUIDANCE_LITERATURE_THEORY_PLAN_ZH.md)。变化：旧记录提出过跨带诊断但quality流程主要是Strong blur response；此次处理双头gap的非对角响应并限制其幅度。

限制：非对角非零只证明非平稳/非线性作用，不能把它自动命名成有用语义。

参考：[AutoGuidance](https://arxiv.org/abs/2406.02507)。

## 29. IG：保边界的TV近端差值

键：`ig_tv_resolvent`；tv_weight=[0.025, 0.075, 0.2]。

构造：`u≈argmin_u .5||u-D||²+r*rms(grad D)*TV(u), 固定12次对偶迭代`。

依据：TV惩罚支持分段平滑且容许跳变，与Fourier线性平滑对边缘的处理不同。

仓库近邻：[旧记录](SPECTRAL_SELF_GUIDANCE_LITERATURE_THEORY_PLAN_ZH.md)。变化：旧静态频带方法没有局部非线性近端运算；此项直接检验空间不连贯的小修正是否有害。

限制：块状偏差及过平滑是明确风险；12次迭代为近似解，不声称精确目标最优。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)。

## 30. IG：补偿有限写入的交换子

键：`ig_commutator`；commutator_mix=[-1.0, 0.5, 1.0]。

构造：`B=J_D S-J_S D, 用固定小差分估计; u=D+r*h*cap(B,||D||/h)/2`。

依据：BCH展开把Strong/gap执行次序的首个差异写成交换子；有符号对照检验这个差异是否影响旧有限写入结果。

仓库近邻：[旧记录](SMALL_SIT_CARRIER_FLOW_RESULTS_20260909_ZH.md)。变化：旧先/后write只改顺序；本次单独显式控制交换子并保留其相反符号，能区别方法效果与排序误差。

限制：数值截断误差的修正没有FID定理；负号对照也不是事后选解释。

参考：[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)。

## 31. IG：隔离深度与类别的交互

键：`ig_condition_interaction`；unconditional_removal=[0.5, 1.0, 1.5]。

构造：`I=(S_c-W_c)-r*(S_u-W_u); u=N_D(I)`。

依据：四个读数形成深度×条件的factorial contrast；r=1精确去掉不依赖类别的容量差主效应。

仓库近邻：[旧记录](RAEV2_SEMANTIC_COMPLEMENT_RESULTS_20260907_ZH.md)。变化：旧方法把Full条件差直接加到IG；这里消除的是unconditional的Strong-Weak gap，不是再加一个CFG向量。

限制：交互项不等于因果语义或真实误差；null弱头的校准可能弱于条件弱头。

参考：[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598), [AutoGuidance](https://arxiv.org/abs/2406.02507)。

## 32. IG：减少与类别强化重复的部分

键：`ig_remove_condition_overlap`；overlap_removal=[0.25, 0.5, 1.0]。

构造：`u=N_D(Di-r*P_Dc Di)`。

依据：在局部类别score代理方向上减少重复放大，检验IG能否更多改善类内质量并保留覆盖。

仓库近邻：[旧记录](RAEV2_SEMANTIC_COMPLEMENT_RESULTS_20260907_ZH.md)。变化：旧semantic_orthogonal是把CFG垂直IG的分量加进来；此项从IG删除与CFG重合的分量，方向相反。

限制：仅局部内积可解释；不保证全程类别概率或类内多样性保持。

参考：[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)。

## 33. CFG：避免重复容量锐化

键：`cfg_remove_capacity_overlap`；overlap_removal=[0.25, 0.5, 1.0]。

构造：`u=N_Dc(Dc-r*P_Di Dc)`。

依据：若CFG和内部容量差共同承担过度锐化，保留类别差中的独立分量可能改善质量/覆盖折中。

仓库近邻：[旧记录](RAEV2_SEMANTIC_COMPLEMENT_RESULTS_20260907_ZH.md)。变化：旧方法固定IG再附加语义；此次以纯CFG为主要基线，IG只定义要消除的重复方向。

限制：与32共享投影工具，但检验对象与主基线不同；不将两个方向视作两个新定理。

参考：[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)。

## 34. IG：满足局部类别方向的最小改动

键：`ig_condition_guard`；alignment_floor=[0.0, 0.25, 0.5]。

构造：`u=argmin ||u-Di||² s.t. <u,Dc> >= r*||Di||*||Dc||`。

依据：半空间投影是满足指定局部方向约束的最近修正；只在IG与类别方向冲突或不足时干预。

仓库近邻：[旧记录](RAEV2_SEMANTIC_COMPLEMENT_RESULTS_20260907_ZH.md)。变化：旧语义补充无论是否冲突都添加固定量；新约束在未触发时严格保持原IG，并记录触发率。

限制：constraint可能不活跃，保留该负机制证据；半空间不等于真实类别概率约束。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html), [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)。

## 35. IG/CFG：只处理冲突的双信号合成

键：`ig_cfg_pcgrad`；cfg_ratio=[0.25, 0.5, 1.0]。

构造：`若<Di,Dc><0，分别去掉指向对方负半空间的分量；u=Di_corrected+r*Dc_corrected`。

依据：借用多目标优化的冲突投影，检验固定语义补充失败是否含有两个局部方向互相抵消。

仓库近邻：[旧记录](RAEV2_SEMANTIC_COMPLEMENT_RESULTS_20260907_ZH.md)。变化：旧直接/正交语义补充固定.15且5K失败；现在保留正相关分量，只处理冲突，并联调相对比例与总强度。

限制：模型差不是两个已知目标的精确梯度，不能继承全局多目标收敛或FID保证。

参考：[Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)。

## 36. CFG：独立类别的几何参考

键：`cfg_alt_class_reference`；reference_classes=[1.0, 2.0, 4.0]。

构造：`u=S_c-mean_j S_(c_j), c_j从非目标类别独立选择并固定每条轨迹`。

依据：平均不同条件score对应几何参考的梯度，检验null token的学习偏差是否限制当前CFG。

仓库近邻：[旧记录](RAEV2_GUIDANCE_READING_INDEPENDENT_CONDITION_20260906_ZH.md)。变化：旧记录精读ICG/TSG而未在当前小SiT完成此扫参；这里固定同输入、真实非目标类别和实际额外NFE。

限制：几何参考不是无条件混合分布，不称恢复真正unconditional score。

参考：[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)。

## 37. CFG：条件嵌入的对称方向响应

键：`cfg_symmetric_condition_response`；radius=[0.25, 0.5, 0.75]。

构造：`e±=e_u±r(e_c-e_u); u=N_Dc((S(e+)-S(e-))/(2r))`。

依据：中心差分消除条件响应中的偶数阶项，检验condition曲率导致的偏移能否与一阶类别响应分离。

仓库近邻：[旧记录](FSG_CLASS_TOKEN_CARRIER_RESULTS_20260908_ZH.md)。变化：旧FSG直接搬运class token并改变读取时机；新方法在同状态同时间做对称响应，再匹配原CFG幅度。

限制：负方向embedding在训练域外，Taylor近似只在小半径可信；不称概率条件插值。

参考：[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)。

## 38. CFG：在不同条件强度读取方向

键：`cfg_condition_secant`；condition_scale=[0.5, 1.5, 2.0]。

构造：`u=N_Dc(S(e_u+r(e_c-e_u))-S(e_u))`。

依据：网络对条件embedding的映射是非线性的；保持最终向量范数，可检验改变查询位置是否带来独立于scale的方向收益。

仓库近邻：[旧记录](FSG_CONDITION_HANDOFF_RESULTS_20260908_ZH.md)。变化：旧handoff改变条件携带路径；本次只在同点换条件secant长度，原Strong和null锚点不变。

限制：embedding外推不保证更强语义，且r=2可能偏离训练支撑；不能与输出线性scale等同。

参考：[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)。

## 39. IG：温和identity-attention弱前缀

键：`ig_soft_pag`；identity_mix=[0.1, 0.3, 0.5]。

构造：`前4层attention output改为(1-r)A*V+r*V，弱头重读；u=N_D(S-Wpert)`。

依据：保留token自身信息而减少跨token检索，为当前同类弱头建立可控的结构退化。

仓库近邻：[旧记录](RAEV2_GUIDANCE_READING_ATTENTION_WEAK_20260906_ZH.md)。变化：不直接复跑旧DDPM/PAG；仅扰动训练过depth4头的共享前缀，保留原Strong，额外成本为prefix。

限制：更坏的弱模型未必与Strong误差兼容；这是PAG在IG参考上的明确适配。

参考：[Perturbed-Attention Guidance](https://arxiv.org/abs/2403.17377), [AutoGuidance](https://arxiv.org/abs/2406.02507)。

## 40. IG：软空间局部化的弱前缀

键：`ig_local_attention`；locality=[0.5, 2.0, 8.0]。

构造：`Apert_ij ∝ A_ij exp(-r*distance(i,j)²); 前4层读Wpert`。

依据：减少远程token依赖而不裁掉输入或改变张量形状，可把弱化程度连续化。

仓库近邻：[旧记录](RAEV2_GUIDANCE_READING_SWG_20260906_ZH.md)。变化：旧crop/window删除信息且可能破坏边界；本次使用对称软距离核，在原位置网格读取原生弱头。

限制：全局信息也可能是有效条件；soft locality本身不是AutoGuidance误差兼容性的证明。

参考：[Sliding Window Guidance](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_26/paper.pdf)。

## 41. IG：提高检索熵的弱前缀

键：`ig_entropy_reference`；logit_scale=[0.8, 0.6, 0.4]。

构造：`前4层q→r*q，固定k/v；读取Wpert，u=N_D(S-Wpert)`。

依据：对固定logits，降低正比例系数单调提高attention熵，提供方向明确的检索弱化旋钮。

仓库近邻：[旧记录](RAEV2_GUIDANCE_READING_ERG_20260906_ZH.md)。变化：旧ERG阅读讨论全模型改造；本次只弱化原生浅层reference且范数匹配，避免混淆Strong改造与guidance强度。

限制：熵不决定真实预测误差，属于文献机制的受限迁移而非新方法宣称。

参考：[Entropy Rectifying Guidance](https://arxiv.org/abs/2504.13987)。

## 42. IG：保查询重心的对比退化

键：`ig_query_contrast`；contrast_removal=[0.1, 0.3, 0.5]。

构造：`q→mean_tokens(q)+(1-r)*(q-mean_tokens(q))，k/v不变`。

依据：保留公共query而收缩不同空间位置的检索差异，可区分全局检索温度与位置特异信息。

仓库近邻：[旧记录](RAEV2_ATTENTION_CONTRAST_GEOMETRY_20260906_ZH.md)。变化：旧文档仅给出独立推导，没有该小SiT quality sweep；本次实现重心保持的弱前缀并对照固定温度路线。

限制：query重心没有自动语义解释；可证明的仅是中心化对比收缩。

参考：[Entropy Rectifying Guidance](https://arxiv.org/abs/2504.13987)。

## 43. IG：保attention权重的值对比退化

键：`ig_value_contrast`；contrast_removal=[0.1, 0.3, 0.5]。

构造：`v→mean_tokens(v)+(1-r)*(v-mean_tokens(v))，q/k不变`。

依据：A行和为1时输出变为mean(v)+(1-r)(Av-mean(v))；隔离被检索内容的多样性而非检索熵。

仓库近邻：[旧记录](RAEV2_GUIDANCE_READING_ERG_20260906_ZH.md)。变化：旧检索温度改变A；本次固定A只改变V，测试不同的误差退化来源。

限制：该代数不证明有益，数个中间层共同修改还会改变后续attention。

参考：[AutoGuidance](https://arxiv.org/abs/2406.02507)。

## 44. IG：按head降低弱参考容量

键：`ig_attention_head_dropout`；heads_dropped=[1.0, 2.0, 3.0]。

构造：`前4层每层固定丢1/2/3个attention value head，mask按输入和实际步生成`。

依据：减少独立检索通道而保留其余heads的完整空间交互，检验结构容量退化与整体gain缩小的差异。

仓库近邻：[旧记录](INTERNAL_GUIDANCE_LITERATURE_REPO_AUDIT_ZH.md)。变化：旧综述已包含HeadHunter/S2先例；此项是当前训练弱头上有共同随机数的prefix容量干预，不做按FID挑head。

限制：随机弱模型有方差且兼容性未知；同一步两次Heun查询必须复用mask。

参考：[AutoGuidance](https://arxiv.org/abs/2406.02507)。

## 45. IG：只降低弱前缀MLP残差

键：`ig_mlp_contraction`；contraction=[0.05, 0.15, 0.3]。

构造：`前4层mlp输出乘1-r，attention路径保持原样，读取Wpert`。

依据：把逐token非线性计算容量与跨token检索容量分开，避免所有weakness都被解释为注意力退化。

仓库近邻：[旧记录](INTERNAL_GUIDANCE_LITERATURE_REPO_AUDIT_ZH.md)。变化：旧skip-block同时删除两条残差支路；这里只收缩MLP且保留原训练弱读出。

限制：残差收缩改变feature分布，不能由“更弱”直接推出正确引导方向。

参考：[AutoGuidance](https://arxiv.org/abs/2406.02507)。

## 46. IG：轻度错配位置与检索内容

键：`ig_value_permutation`；permutation_mix=[0.05, 0.15, 0.3]。

构造：`v→(1-r)v+r*permute_tokens(v)，q/k不动；permutation按步固定`。

依据：保留value集合与其均值，破坏检索权重对应的位置内容；与同时置换K/V的精确不变操作不同。

仓库近邻：[旧记录](INTERNAL_GUIDANCE_LITERATURE_REPO_AUDIT_ZH.md)。变化：明确借鉴token perturbation；使用轻度value-only混合的原生弱前缀，避免把等变置换误算成新方法。

限制：这不是TPG全流程复现；位置错配可能制造完全不兼容的弱错误。

参考：[Token Perturbation Guidance](https://arxiv.org/html/2506.10036v1)。

## 47. IG：用正定响应代价稳定方向

键：`ig_response_metric`；response_penalty=[0.1, 0.5, 2.0]。

构造：`Q=orth(D,C); R=J_m Q; u=Q*(I+r*R^T R/mean_diag)^-1*Q^T D`。

依据：输出响应能量的二次代价给出正定正规方程，抑制会令Strong预测剧烈变化的状态方向。

仓库近邻：[旧记录](SMALL_SIT_PRECISION_BALANCE_RESULTS_20260909_ZH.md)。变化：旧rank2精度差近奇异；改成I+r*R^T R的稳定收缩，不将两确定性头当独立Bayes观测。

限制：稳定小矩阵不等于正确posterior；只探测两个方向，有限差分有误差和额外Full调用。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)。

## 48. IG：用正规方程搬运clean目标修正

键：`ig_inverse_response`；ridge=[0.1, 0.5, 2.0]。

构造：`Q=orth(D,C); R=J_m Q; coeff=(R^T R+r*I)^-1 R^T(bD); u=cap(Q*coeff,2||D||)`。

依据：在可观测二维响应中，显式求使Strong clean变化接近目标bD的最小二乘状态修正，避免默认latent写入会原样保留。

仓库近邻：[旧记录](SMALL_SIT_CARRIER_FLOW_RESULTS_20260909_ZH.md)。变化：旧carrier flow直接写D并产生负FID；本次以Strong实际有限差分响应作局部搬运，使用ridge和信任上限。

限制：局部目标仍是模型内IG修正，不是真实终点目标；不能由正规方程认领质量保证。

参考：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)。

## 49. IG：拟合保守的patch误差场

键：`ig_conservative_error`；error_mix=[0.25, 0.5, 1.0]。

构造：`Ehat(z,t)=A_t*z+b_t, A_t=A_t^T; v=S+a*(D+r*cap(Ehat,||D||))`。

依据：真实配对target-S的对称仿射回归是二次势的梯度，可在小函数类内修复系统偏差而不引入任意旋转场。

仓库近邻：[旧记录](RAEV2_REFERENCE_LINEAR_FIT_PROTOCOL_20260908_ZH.md)。变化：旧任意线性/MLP参考改变预测且未稳定改善FID；这里固定Strong/Weak，解带对称约束的Sylvester正规方程并限制修正。

限制：保守性只是结构条件；训练态残差与实际生成误差可能不同，局部MSE仍不决定FID。

参考：[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)。

## 50. IG：同时读取多深度的稳健共识

键：`ig_robust_head_consensus`；huber_radius=[0.1, 0.3, 1.0]。

构造：`D_l=N_D4(S-W_l), l=4,6,10; u≈argmin_u sum_l Huber_r(||u-D_l||)`。

依据：几何Huber位置估计降低单个不兼容头的影响，检验同时的可靠方向是否比预设深度轮换更有效。

仓库近邻：[旧记录](IMAGENET100_SIT_MULTISCALE_GUIDANCE_RESULTS_ZH.md)。变化：旧depth progression及频谱选层已失败；本次三个已有v-head同状态共享一次主干，不把深度当连续求解迭代。

限制：三头误差相关且可能共同错误；稳健位置不创造新信息或posterior样本。

参考：[AutoGuidance](https://arxiv.org/abs/2406.02507)。
