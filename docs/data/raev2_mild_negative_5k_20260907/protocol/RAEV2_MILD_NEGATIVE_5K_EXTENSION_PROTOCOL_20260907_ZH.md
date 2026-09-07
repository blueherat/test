# 轻微阴性1K补至5K：用户授权的固定补测

用户在四轮结题后明确要求：“补一些恶化较小的1k，补到5k，因为可能1k恶化但是5k提升”。本次据此重新开启**有限补测**。此前停止决定作为历史保留；不新增方法、训练、系数、窗口、校准数据或checkpoint选择。

## 在新样本前固定选择

| 候选 | 原1K官方→候选FID | 相对恶化 | 原数据根目录 |
|---|---|---:|---|
| global proximal | 38.397787497→38.408363290 | 0.027543% | proximal_seed202609066/global |
| 旧全局energy ball | 38.397787497→38.408604162 | 0.028170% | energy_ball_seed202609066/energy_ball |
| native全局energy ball | 38.159880814→38.161144351 | 0.003311% | spatial_energy_balls_v1/global100 |
| affine reflection | 38.251591886→38.288640821 | 0.096856% | affine_reflection_v1/reflection100 |

根目录R=`/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906`。前三项是最轻微的固定阴性结果，反射提供不同机制。已有5K阴性的势函数不重跑；两个空间球恶化约0.412%，本次未选。配对桥轻微正结果也不扩大，本次聚焦用户所指轻微阴性项。选择依据是已看过的旧1K，因此合并5K不是完全未选择的独立测试。

## 样本与方法保持

每个family完整保留原1K，再加入四个新的固定1K，seed依次202609171、202609172、202609173、202609174。每组类别0…999各一图，B8；总计5K每类5图。另独立报告新增4K（每类4图），区分原筛查样本与新证据。新增seed在当前仓库检索未发现既有使用；实际noise/labels哈希仍逐组核验，不能只凭seed不同断言无重叠。

能量球的归约cohort始终N=1000，按五个cohort拼合评价；不改成一次N=5000的相互作用采样。旧global与旧energy保持FP32 guidance/Euler、TF32=true和逐B8 CUDA randn；native global与reflection保持BF16内部IG、FP32状态、TF32=false和完整1K CUDA randn。它们之间不因seed相同就混用图片。旧global/energy共享各自legacy official；两native方法只有在实际噪声/标签、模型数值路径及parity核对后才共享新增official100。

保持原config、EMA100080、decoder、stats、IG1.78[.1,1]、CFG1、shift8、t_eps=.05。原global校准与两套energy真实参考均逐字节固定，不重估。所有候选仍100步；反射每步两次主模型，其余每步一次。全部图像保留，不筛选或提前根据FID停止某个cohort。

新native包装器仅更改固定seed、plan/protocol与来源记录，原采样和校正函数保持不变。每个新增cohort各自执行16图官方parity；原legacy sampler直接使用其已有seed CLI。旧样本来自各自原始目录，先绑定sample/request/summary/batch manifest身份。输出目录为R/scale_extension_5k_v1，拒绝覆盖历史或自动重启失败作业。

## 运行、成本与解盲

四块分别使用物理GPU0/1/2/3的RTX4090；同一新增块的候选与基线在同一GPU顺序运行。记录每个父/子PID、起止UTC、退出码、外层wall和GPU身份。不同块可以并行；GPU wall求和是总计算投入，不能把四卡并行总时长当作单卡成本。

每块运行：两种native parity、legacy official100/global proximal/旧energy、native official100/global ball/reflection，以及native official201。旧反射已有official100与成本选择所得official201，全部保留。先完成全部新采样及成本核对，再统一FID，禁止按图像/FID挑步数。

反射先沿用原成本对照201步。汇总原1K和新4K的轨迹至解码T及runner W，若official201未同时覆盖候选，则在FID前固定更高统一K：`max(K+1,ceil(K*T_candidate/T_officialK),ceil(K*W_candidate/W_officialK))`。此时全部五个块都使用同一个K，旧seed对应的新K也补跑；不得混合201和另一个K却称单一官方K。所有成本不足臂保留计费。该校正不改变候选参数。

legacy的历史脚本只有从runner开始、含加载/准备/采样/解码/I/O的elapsed及主模型调用数，没有分离T。保持这些原口径并补新进程外层wall；100步基线与候选匹配主模型NFE，校正/准备等额外wall如实列出，不能伪称精确墙钟或公平总成本已经闭合。native global也报告100步实测开销、原100与已有更贵成本对照，不隐藏投影/归约费用。本次首先回答固定方法扩大样本后是否反转，不据此跳过总成本与独立确认条件宣布SOTA。

## 合并、评价与结论

CPU合并核验每个原1K的全部125个batch和完整样本一致，逐组方法/基线noise与labels配对；ID重映射为0…4999，label和像素顺序保持，保存cohort及原local ID。每臂产生pooled5K与new4K，统一使用原nanogen evaluator commit19dfb4c2705333eb8b97e454fb354d47d1fe135b、ImageNet256 reference SHA925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac、原Inception权重、batch64、seed2020，保存特征。

最终表并列旧1K、新4K、合并5K及相对各自基线的差；相对改善定义为(FID_base−FID_method)/FID_base。不跨不同精度/旧seed family直接排名绝对FID，也不把五个interacting cohort的粒子说成5000个完全独立重复。必须保留所有候选结果，包括继续恶化的项；新4K支持复现程度，5K仍有有限样本偏差。样本、成本、结果、来源和新代码追加归档/Git，保持之前结题记录不变。
