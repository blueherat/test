# RAEv2 仿射反射平均：固定配对 1K 协议

本协议在任何新候选 GPU 前向与 FID 之前记录。目标仍是理论机制导出采样过程内 guidance，并在公平成本下获得至少 5% 的相对 FID 降低；本次只作固定配置的质量筛查。

## 误差机制与可实现结构

当前 DINOv3-L K7 encoder 移除了 LayerNorm 的仿射参数，输出是多个经过 channel normalization 的 token 特征的均值，再加末个选定层的空间均值。因此理想算术下每个原始 token 特征 a 的通道和为零。它没有固定半径，不能套用球壳约束。

设官方标准化 latent 为 X_j=(a_j−μ_j)/σ_j，σ_j=√(var_j+10⁻⁵)，j 为空间位置。令 u_j=σ_j/‖σ_j‖，c_j=−(∑_channel μ_j)u_j/‖σ_j‖，P_N 为逐空间位置沿 u_j 的正交投影。所有真实 clean latent 满足 P_N(X−c)=0，即已知仿射子空间 H。这里的 μ、var 就是官方 decoder 所必需的既有 normalization stats，没有新数据校准或训练。

Gaussian bridge Z_t=(1−t)X+tε 的法向分量仅含已知平移与无信息高斯噪声。因此精确 posterior mean 不应响应法向噪声的反射。有限模型及其 guidance 可能产生此类伪响应；纠正对象是这个违反已知对称性的分量，不是任意削弱整个 guidance。

定义 R_t z=z−2P_N(z−(1−t)c)，Π_H f=f−P_N(f−c)。令 G 为完整原生 guided clean prediction，本次唯一候选为

    G_sym(z,t,y) = Π_H ((G(z,t,y)+G(R_t z,t,y))/2).

平均权重由二元素群 {I,R_t} 的均匀测度决定；整体法向反射不依赖子空间基的选择。它不是对 2²⁵⁶ 个独立翻转或 O(256) 的完整积分，仍可保留偶次法向依赖。两个查询使用同一状态、时间、类别；不生成候选完整图片再挑选。修正全部 guided field，包括 Full 与其原有外推差分；仅平均差分不具有下面的风险收缩恒等式。

## 理论保证与边界

对 teacher bridge 的联合分布 (Z_t,X)，反射不改变联合 law，X∈H。任意平方可积预测 f（不要求精确 score、保守场或 Gaussian clean 数据）满足

    E‖f(Z)−X‖² − E‖(f(Z)+f(RZ))/2−X‖²
      = ¼ E‖f(Z)−f(RZ)‖².

再投影到 H，风险额外减少 E‖P_N((f(Z)+f(RZ))/2−c)‖²。真实 posterior mean 本身不变；理论去掉的是已知无效的反对称与法向预测成分。这个保证针对 denoising 风险，不是 FID 或最终 decoder 特征风险。

另有闭环离散保证：理想算术下 G_sym(R_tz,t)=G_sym(z,t) 且 G_sym∈H。当 Euler 的分母等于 t、下一时刻 s<t 时，

    E_{t→s}(z)=(s/t)z+(1−s/t)G_sym(z,t),
    E_{t→s}(R_t z)=R_s E_{t→s}(z),
    P_N z_t=(1−t)c+t P_N z_1.

因此从标准 Gaussian 起点，完整候选轨迹保持这项反射对称性，且法向 bridge 精确。100-step shift8 的最小正时间约 .074766>.05，所以所有候选步骤适用。最终 z_0∈H。浮点实现会有有限误差，parity 记录而不迭代修补。该结构不保证切向终点更接近真实分布，也不保证整个 teacher posterior 路径；必须实测质量。

此前 normal-noise 审计的 teacher MSE 改善仅约 .033475%，它使用不同的 FP32 guidance 混合与 TF32 设置，且未跑这个闭环候选。那个数不是 FID 上界，也不是已完成的质量否证。此次不按它选择系数或时刻，而以新的固定质量检验裁决。

群平均并非本研究新创的基本定理。相关原始工作包括 [Diffusion Models under Group Transformations](https://proceedings.mlr.press/v258/lu25a.html) 的推理时 score 对称化，以及 [Robustness and Structure Preservation in Flow-Based Generative Models via Wasserstein Path-Space Divergences，v2](https://arxiv.org/html/2410.01244v2) §7 的风险正交分解与条件性 flow 误差上界。本次特化使用 RAE 已知的仿射支撑及无信息法向噪声。令 q=z−(1−t)c 后，输入反射化为固定正交变换 A=I−2P_N；本候选额外有 clean 输出投影，所以不能从纯 vector-field 群平均的“Lipschitz 常数不增”直接声称完整候选对原始官方速度有更紧的终点上界。上面的风险恒等式和 100 步闭环推导不依赖这项额外声称。

## 冻结实现与对照

- 模型为官方 strict DINOv3-L K7 EMA step100080，原有 config、decoder 和 stats，完整身份写入 plan/request。
- 固定 seed202609131；一次 CUDA FP32 randn([1000,1024,16,16])，1000 个类别各一张，ID=label=0…999。所有臂完整 noise、RNG、label SHA 一致。
- B8，FP32 状态/Euler；BF16 backbone 与 decoder，TF32 关闭。窗口内原生 BF16 `B+1.78*(F−B)`，窗口外 Full，随后才转 FP32。两个查询分别运行 B8，不能改为 B16。反射、两头平均和 Π_H 在 FP32 执行；u,c 由 FP64 stats 计算后转 FP32。
- 保留原有 IG1.78、窗口 [.1,1]、100-step shift8、原生 t_eps=.05；候选在每步应用同一公式，包含窗口外的 Full。没有额外幅度、层选择、随机方向、手工时间调度或校准样本。
- 官方 100 步、候选 100 步，以及依据下述计时规则确定的官方 K 步。所有臂为生产式 batch-major 完整轨迹；候选每样本 200 次 stage2 前向，官方每样本 K 次。Base/Full 在一次共享 stage2 前向内返回。
- 原生 BF16 decoder→clamp→BF16×255→uint8 NHWC，保留全部 1000 张图像。

正式采样前运行必要 CPU 结构检查，以及固定 16 图 GPU off-branch parity：相同噪声下与 production.sampling_step(use_potential=False) 的完整 100 步终点、原生像素逐位相同。同时记录实际 stats 的反射/投影数值残差。另固定取这组初始噪声的首 B8、t=1，实际计算 G_sym(z) 与 G_sym(R_tz)，合计 4 次独立 B8 原生 stage2 前向；记录反射二次作用残差、输出法向残差、G_sym 反射不变性残差，以及首个 Euler successor 的法向 bridge 残差。这些有限精度量只检查有限性并完整记录，不选阈值、系数或迭代修补；CPU 测试核验理想恒等式。额外 4 次前向及诊断成本与 off-branch 的 400 次 B8 前向、4 次 B8 decode（32 个样本解码）分开记录；这些图不用于挑选参数或判断质量。源码、测试、输入身份与本协议在 GPU 开始前冻结到 plan。若实现检查失败，保留失败及修复身份；不得改公式来掩饰失败。

## 成本规则：先计时，后看 FID

三个臂顺序使用同一张 RTX4090（固定 GPU UUID），避免以不同卡的速度差作方法收益。记录完整轨迹（含全部双查询、反射、投影及循环成本）、decode/uint8、噪声和几何准备、加载、传输、哈希、I/O、峰值显存、前向次数及整个运行 wall time。无额外训练或参考数据编码成本；已有主模型与官方 stats 的训练/获得成本与基线共享。研究诊断和论文阅读成本单列，不冒充推理成本或隐去。

定义 T 为同步的完整轨迹加 decode/uint8 耗时；W 为 runner 从 main 开始至写最终 summary 前的完整 wall time，包含准备、加载与输出审计。先完成 official100 与 reflection100，在任何本轮 FID 计算前冻结

    K=max(200, ceil(100*T_reflection/T_official100),
               ceil(100*W_reflection/W_official100)).

K≥200 同时保证官方获得不少于候选的 stage2 前向预算。若 officialK 实测 T 或 W 尚低于候选，在看 FID 前单调补足

    K_next=max(K+1, ceil(K*T_reflection/T_officialK),
                    ceil(K*W_reflection/W_officialK)).

每次必须是真实更高步数的官方采样，保留所有成本记录，不通过空等、无效计算或 FID 选步数。K≥154 可能触发官方 .05 floor，保持原实现并披露，不能修改 floor 以另选 solver。T/W 是特定运行条件的计时点估计，不是硬件抖动为零的承诺；报告预算超出比例以及 cold 与 inference 两个口径。

使用统一 nanogen official Inception/FID evaluator、固定 reference 与 evaluator SHA。每臂同样保存特征，可重建评价；本轮 FID 只在成本对照完成后计算。报告候选相对 official100 与最终 officialK 的变化，不以较差的对照替代另一条结果。

若无质量信号，结束这个固定实现，不搜索反射强度、子空间组合、平均权重、窗口或 seed。若有值得确认的信号，保持公式不变用独立噪声及足够规模验证。1K 的 5% 不是最终规模改善的必要筛选门槛；1K 也不能证明 SOTA。最终成功仍需公平成本下至少 5% FID 改善及独立确认。
