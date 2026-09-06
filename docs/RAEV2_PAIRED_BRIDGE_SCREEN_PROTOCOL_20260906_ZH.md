# RAEv2 配对桥流：固定 1K 质量筛查

本协议在本轮 parity、完整图像采样与 FID 前冻结。承接[有限步理论](RAEV2_ACTUAL_ROLLOUT_FINITE_TRANSPORT_20260906_ZH.md)、[冻结训练协议](RAEV2_PAIRED_BRIDGE_PILOT_PROTOCOL_20260906_ZH.md)和[已完成的机制实验](RAEV2_PAIRED_BRIDGE_PILOT_RESULTS_20260906_ZH.md)。当前目标仍为理论导出 guidance，在公平总成本下至少降低 5% FID；本次是按推理成本配对的探索性质量筛查，不能单独完成最终目标。

## 进入筛查的理由和保留的反证

已完成的固定验证中，candidate 的速度回归风险比零预测差约 0.082%–0.084%，没有学到有效速度场的直接泛化证据。两次修正后的固定 2048 维边缘统计均值偏差相对官方降低约 7.30%，但相对均值场对照的净优势约 98.51% 来自最后一步。32 张实际轨迹均有限，未解码或评价 FID。这些结果只支持一次固定质量检验，不能认定 covariance 机制成立，不能据此只保留最后一步或选择窗口。

## 固定机制、权重与输入

真实 bridge 的配对端点为 Y=T(Z_t)、W=Z_s，R=W−Y，U_τ=Y+τR。条件速度 w_τ(U)=E[R|U_τ=U,t,s,c] 在适当正则性下运输 T#p_t 至 p_s。精确有限输运不放大相对真实边缘的 KL，可逆时为等号；有限网络和两次查询 midpoint 不继承精确输运保证。此条件回归目标可由弱连续性和 L2 正交分解独立导出，见[Flow Matching 精读](RAEV2_GUIDANCE_READING_FLOW_MATCHING_20260906_ZH.md)。

唯一 candidate 在每个原生 Euler successor Y 后执行：

    β=(t−s)/max(t,.05)
    k1=β f_candidate(Y,t,s,0,c)
    U_mid=Y+k1/2
    z_s=Y+β f_candidate(U_mid,t,s,1/2,c).

control 使用同架构、同数据和固定预算训练的均值场 f_control，第二次调用仍输入实际 U_mid，但两个 τ 输入均为 0。两个臂全 100 步均执行同一公式；没有新 gain、裁剪、时间窗口、再训练或 checkpoint 选择。权重固定为 `paired_bridge_v1/train/final.pt`，SHA256 `5013cbf075ddffa5c2ae021fc916be0615ca41d9817d3af91e6b3e46c86e1983`，第 2048 update，candidate/control 各 3,652,224 参数。网络仅接收当前状态 U、t、s、τ、类别，不输入配对原图或未来 W。

- 官方 strict DINOv3-L K7 EMA step100080；原 config、decoder、stats、IG1.78[.1,1]、CFG1、shift8 和 t_eps=.05。
- 固定新 seed202609151，单次 CUDA FP32 randn([1000,1024,16,16])；ID=label=0…999，每类一张。所有完整臂的噪声、RNG 和标签 SHA 一致。
- B8，CPU FP32 构造时间网格再取标量；原生 BF16 main/IG，FP32 状态、Euler 和辅助场；TF32 关闭。
- 各臂 batch-major 完整轨迹；原生 BF16 decoder→clamp→BF16×255→uint8 NHWC；全部样本保留，没有完整图像选择、排序或拒绝。
- official 只驻留 stage2 和 decoder；candidate/control 仅额外驻留自身辅助场。训练身份的 SHA/来源审计计入各臂 W；official 不反序列化辅助权重。共同来源审计是本实验的复现开销，不宣称最小部署基线的准备成本。

## 实现核验

正式采样前，固定 16 个输入完整运行四套 100 步轨迹：production official、新 sampler official、零输出 candidate、零输出 control。逐步状态、终点和原生像素均要求逐位一致。另只对首 B8 的第一步记录训练后两个场的有限响应，不用它选阈值或判断质量。四套 parity 的全部实际主模型、辅助场和 decoder 调用及加载、输出时间单列计费。

必要 CPU 检查涵盖 floor 区间的原生算术、零场完整轨迹、当前 midpoint 输入、真实 forward hook 计数、B8 限制和 archive 格式。协议、代码、测试、训练链及评价身份冻结到 screen_v1/plan.json；失败应保留日志和已用成本。不得以实现修复名义改变已冻结公式或权重。

## 先成本、后 FID 的固定顺序

同一物理 RTX4090，UUID `GPU-7d3e4e7d-abfa-e06e-c264-796052797949`。顺序为 parity → official100 → candidate100 → control100 → officialK → 统一评价。记录每个子进程 PID、起止时间、退出码、外层 wall 和 GPU 状态；driver 使用独立会话运行，重入拒绝覆盖，不能因前端会话消失重采样。

定义 T 为每批同步的完整轨迹至 decode/uint8 返回的 wall 之和，包含两次场查询、全部修正和中间检查；W 为 runner 从 main 入口至最终 summary 前的 wall，包含模型/来源核对、加载、噪声、传输、输出与哈希。imports 和进程开销由外层 wall 补充。T/W 为实际运行的点估计，须报告预算超出比例；不声称硬件抖动为零。

完成前三个完整臂、尚未计算任何本轮 FID 时，取

    K=max(100, ceil(100*T_candidate/T_official100),
                ceil(100*W_candidate/W_official100)).

K=100 且 official100 已同时满足两个预算时，复用其完整结果；否则运行新的 officialK。若其实测 T 或 W 低于 candidate，则仍在 FID 前单调补足

    K_next=max(K+1, ceil(K*T_candidate/T_officialK),
                    ceil(K*W_candidate/W_officialK)).

不因 FID 或图像结果选 K，不用空等或无效计算补预算；所有成本不足的中间官方臂保留并计入研究投入。官方步数只按这个成本规则变化；原有时间 shift、IG 和 floor 均不改。更高步数可能进入 .05 floor，需披露，不能事后当作质量原因。

candidate/control 每图各 100 次 stage2 前向加 200 次小场前向；这不是 300 次 stage2 NFE。各模型实际调用由 hooks 核验，另记录 decoder 调用、显存、冷启动和纯推理耗时。control 为机制对照，不用于确定 candidate 的预算。official100 是较便宜的可靠参照，最终 officialK 是实测推理预算参照；两者都报告，不只选较差的一个。

## 评价与完整成本边界

只有 cost_match.json 落盘后才统一使用原 nanogen official Inception/FID evaluator。固定 evaluator commit `19dfb4c2705333eb8b97e454fb354d47d1fe135b`、ImageNet256 reference SHA `925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac` 和原 Inception 权重；保存全部特征。各臂同批量 64、相同 evaluator seed2020。同规模 FID 的模型相关偏差不自动相消，1K 不证明独立种子或大规模改善。

当前已知前置成本：6000 张真实图像的历史编码 113.860739042 s；固定 pilot/train/validate/rollout 四个 runner wall 合计 741.310023239 s，其中共同训练两场 490.104689422 s。共同训练成本不得随意除以二。CPU 汇总约 13.897838 s、此次 parity、采样、评价与所有失败/中间成本另外披露。历史数据源选取成本未闭合，训练 driver 外层精确 wall 因会话中断缺失但有 runner 测量和 UTC 区间，见机制结果文档。以上不能等同完整且精确的总成本。

本轮 K 只匹配推理成本；训练、数据创建及研究诊断不会被藏入“免费辅助场”。若质量信号值得继续，保持结构不变，再完善可复现实验的总成本边界并在独立噪声、足够规模上比较。1K 相对改善不足 5% 不构成普遍淘汰定理；明显不支持质量的结果则结束当前固定实现，不开展 gain、窗口、训练步数或种子扫描。最终只有真实独立验证且公平总成本下至少 5% FID 改善，才能完成 goal。
