# 有限采样的终点误差与跨时间抵消：研究起点

2026-09-07。用户重新授权以 ICLR 级论文为目标开展研究。此文件只冻结第一项探索，
不把旧停止记录删除，也不将当前假设当作突破。此前最后三轮和 3% 目标的失败保留。

## 核心问题

瞬时误差即使在真实边缘上可观测，仍可能在不同时间相互抵消。减少局部误差
能否破坏这种抵消，并解释已有同头 oracle gate 与 safe gate 的质量反序？
只有解析反例不足以回答；需要训练模型上的可重复机制和独立干预预测。

## 有限步、无导数的精确对象

令时间朝数据方向增加，p_k 是已知加噪桥边缘，T_k 是实际有限步更新，
S_k=T_{K-1}∘…∘T_k 为实际 suffix（S_K=identity）。对固定终点特征 φ，定义

    m_k = E_{Z_k~p_k} φ(S_k(Z_k)),
    d_k = m_k − m_{k+1}。

于是 sum_k d_k = E_generated φ − E_data φ，且

    d_k = E[φ(S_{k+1}(T_k(Z_k))) − φ(S_{k+1}(Z_{k+1}))]。

Z_k、Z_{k+1} 可使用同一真实 X 和 Gaussian ε 构造。无需知道 Bayes 速度、
密度或输入梯度；不假设连续 ODE 或小扰动。可用时间块代替单步，代数不变。
随机更新时 suffix 表示条件期望；本 pilot 只研究确定性 Heun。

这是标准 telescoping / backward-observable 思路的应用，**不认领恒等式的新颖性**。
相同样本桥上全时刻求和的闭合只是实现检查，不是预测成功。

对有限随机 Fourier 特征，令 D=sum d_k，则

    ||D||² = sum ||d_k||² + 2 sum_{i<j}<d_i,d_j>。

保留有符号交叉项，避免局部平方和丢掉抵消。这里 ||D||² 是有限特征下的
经验均值差，不能直接叫总体 MMD 或 FID；也不由分解本身证明干预的因果效果。

## 第一项实验（冻结后运行）

- 使用已有连续螺旋 checkpoint，三 seed 20260831/20260901/20260902，D512/H128。
- 同一 D0 双头上的 D2、D3 Bayes oracle 和 D4 gate；不训练、不选择 checkpoint。
- 沿用原 200 步 Heun、全部端点处理和 denominator floor。
- 10 个等步数块；每模型先用 512 个全新配对 bridge 样本。
- 固定 192 个 Fourier 坐标（三尺度 .1/.3/1.0 各 64），feature seed 73001。
- 同时保留 intrinsic/off-subspace 指标，不能让二维投影掩盖 ambient 损伤。
- 记录完整 restart 特征、分块 signed defect、总量/对角/交叉项、逐块 teacher
  Bayes risk，以及相同模型的普通采样。另用独立真实 reference 计算 intrinsic SWD。
- 所有采样、特征、checkpoint 和依赖源码身份保存到数据目录。原资产只读。

pilot 判断：需要先确认既有 oracle/safe 排序在新样本中大体存在，再检查 cross-time
交叉项是否系统影响两者排序；只有显著且跨 seed 一致的现象才设计独立干预。
任何结果都不追溯修改此 pilot 为确认实验。若机制不支持，不继续包装此解释。

## 必须补齐的论文证据

1. 新颖性：与 weighted Hodge、dual weighted residual、performance difference lemma、
   Stepwise-Flow-GRPO、adjoint matching 的原文比较；单纯重述 error≠quality 不够。
2. 因果：只改变预先识别的时间误差，检验质量符号；归因曲线不是充分证明。
3. 前瞻：在独立 seed/模型上优于 teacher risk 和瞬时 observable-risk 的符号预测。
4. 实用性：清楚说明完整 suffix 的成本；如果不能得到可用方法，必须判断机制贡献
   自身是否充分，不能宣称 ICLR 水平已经达成。

## 本轮已核对的相关原文入口

- https://arxiv.org/html/2606.06179v1 ：weighted Hodge 与瞬时 gradient error；
  本轮阅读 introduction、贡献及摘要，后续需完整证明核对。
- https://arxiv.org/html/2603.28718v1 ：Stepwise Credit Assignment for GRPO；
  本轮取得原文，尚未完整阅读，不增加已读论文数。
- 仓库旧反例：`RAEV2_GUIDANCE_READING_PATHS_20260906_ZH.md`。
- 旧失败伴随控制：`RAEV2_ENDPOINT_ADJOINT_RESPONSE_RESULTS_20260906_ZH.md`。

## 2026-09-08 pilot 结果

三训练 seed、每条件 512 个新样本已完成，普通采样实现逐 tensor parity 为零；
独立参考下 oracle/safe 的 SWD 反序再次出现。但九个条件的无偏 cross-time
energy 全为正，safe 的对角项和交叉项同时下降，没有支持“更强抵消带来收益”。
因此不继续将本例包装为跨时间抵消机制。此结论只否定本 pilot 中的解释，
不否定解析抵消反例的存在。

初始经验均值平方含有限样本偏差，已补上 distinct-pair U-stat Gram：
`G_ij = (n <mean d_i,mean d_j> - mean_a <d_ia,d_ja>)/(n-1)`。
配对样本是独立单位；不可把不同 restart 当独立样本。原始结果不覆盖，修正
汇总保存在 `experiments/results/terminal_defect_20260908/temporal_pilot.csv`。
triangle cancellation_fraction 不能代替有符号交叉项证据。

四项解析/估计器检查通过。原始数据、输入与依赖源码在
`~/data/eqvae/experiments/terminal_defect_20260907/pilot_seed*/`。

后续转向 [子空间控制竞争](SUBSPACE_GATE_RESEARCH_PROTOCOL_20260908_ZH.md)。
同时复核 LPL 文档发现条件均值保持的 `J^T W J e` 预条件已由仓库提出，且
最新 [PFM](https://arxiv.org/html/2607.03524v1) 已讨论感知回归改变条件目标；
不把重新发现这些内容计为本轮创新。
