# RAEv2 两个空间能量球：冻结 1K 结果

日期：2026-09-06。**本候选没有改善 FID，结束该固定实现。** 配对 1K 中，两个空间球相对官方 100 步恶化 0.411658%，相对同参考全局球恶化 0.408333%。未达到至少 5% 的研究目标，不进入新种子、扩大规模或半径、分量数、强度、窗口搜索。这是一次固定有限实现的阴性筛查，不是所有空间引导均无效的结论。

[冻结协议](RAEV2_SPATIAL_ENERGY_BALLS_PROTOCOL_20260906_ZH.md)及采样代码保持原样；本文件记录执行后状态。运行目录为 [spatial_energy_balls_v1](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spatial_energy_balls_v1)。

## 图像质量与比较条件

| 方法 | 1K FID ↓ | 相对官方的 FID 降幅 | 轨迹实测耗时 |
|---|---:|---:|---:|
| 官方 100 步 | 38.1598808139 | — | 635.282860 秒 |
| 全局能量球 100 步，同真实参考 | 38.1611443514 | −0.003311% | 633.260509 秒 |
| 两个正交空间能量球 100 步 | 38.3169688602 | −0.411658% | 626.382556 秒 |

三个臂各保留全部 1000 张图，ID、类别与初始噪声相同。候选和全局对照均在采样轨迹内处理完整 N=1000 cohort，B8 只是计算微批；没有多生成图片后挑选。所有臂使用同一 strict DINOv3-L K7、EMA step100080、decoder/stats、原生 BF16 guidance/像素量化及 FP32 Euler，关闭 TF32。各臂恰好 12500 次 B8 主模型前向、100000 次样本级前向，以及 125 次 B8 解码；Full/Base 在同一次主模型前向中返回。没有额外训练或反传。

统一评价沿用 `evaluate_raev2_official_samples.py` 和 nanogen-evals commit `19dfb4c2705333eb8b97e454fb354d47d1fe135b`，Inception-v3-compat、batch64、评价 seed2020，reference 为 `imagenet_256_fid_stats`。三个档案及 reference/特征权重的 SHA 在评价请求中固定。原始 [FID CSV](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spatial_energy_balls_v1/fid_results.csv) 和 [完整数值 JSON](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spatial_energy_balls_v1/fid_results.json)已保留。

独立[身份与成本复核](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spatial_energy_identity_cost_audit_v1/audit.json)确认三臂的 125 批 noise/labels 摘要、完整噪声/RNG 记录、初始噪声与最终 RNG 数组、所有档案 ID/label 值及逐臂 13 份来源快照一致；大模型/参考文件按已记录身份对齐，没有将复核描述成再次独立重算所有大文件摘要。[解盲复核](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spatial_energy_identity_cost_audit_v1/fid_metadata_audit.json)另外确认 CSV/JSON、档案摘要、evaluator commit 与冻结时间顺序；该 agent 没有重算 FID 或读取像素/特征。Root 在首次评价启动前重新计算过三个完整样本档案的 SHA。机器可读的[本候选完成记录](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spatial_energy_balls_v1/result_summary.json)明确保存 `research_goal_achieved=false`。

这是单个配对 1K 的观察值，没有显著性或总体等效性结论。候选使用 cohort 相互作用，不把 1000 张输出当作 1000 次独立控制实验；没有追加图像级 bootstrap 或用局部近似置信区间将它升级为独立重复。旧历史分析对“1K 必须先超过 5%”的更正仍成立；这里结束的理由是固定筛查没有正向质量信号，并非将 1K 的 5% 门槛重新设成必要条件。

## 成本先于 FID 冻结

三臂采样均以 exit0 完成后，先依据 `K=max(100,ceil(100*T_spatial/T_official100))` 得到 **K=100**，并核对实测官方轨迹时间不少于候选。[cost_match_before_fid.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spatial_energy_balls_v1/cost_match_before_fid.json)在评价开始前写入；随后首次评价于 09:08:20 UTC 开始，09:08:42 完成。没有根据 FID 选择步数或省略一个已要求的额外官方臂。

候选的 626.382556 秒包含完整主模型/Euler、空间归约、同步、全 cohort 写回及逐步诊断。其中投影连同诊断为 3.427330 秒，已包含在总轨迹计时中，不能重复相加或减去它再报成本。官方采用生产式 batch-major 轨迹，没有附加逐步能量归约；候选和对照采用 time-major。

| 另行记录的执行成本 | 官方 | 全局球 | 两个空间球 |
|---|---:|---:|---:|
| 解码与原生 uint8 转换，秒 | 3.503464 | 3.455531 | 3.433245 |
| 含输出的采样过程，秒 | 643.371145 | 640.868925 | 634.011863 |
| 入口内总 wall，含加载及来源校验，秒 | 667.406795 | 665.712182 | 659.313282 |
| 外层进程 wall，含启动/退出轮询，秒 | 674.732075 | 672.726543 | 666.717028 |
| 采样期峰值 allocated bytes | 5496272896 | 6528312832 | 6528312832 |

这些计时字段存在包含关系，不将各行相加。三张同型号 4090 的一次并发运行只给出当前运行条件下的预算比较：候选不含投影的 model/Euler 段本身就比另一张卡上的官方更快，**不能据此宣称投影算法提速或严格时间相等**。设备 UUID 和启动负载保存在运行档案中。

16 图 GPU parity 与全部 14 项 CPU 检查在 1K 前完成。GPU 检查中，无投影的 time-major 与 batch-major 的最终 latent 和原生像素均逐位一致；它验证循环顺序与算术，不是质量或吞吐对照。该 parity 有 400 次 B8 主模型前向、4 次 B8 解码；本轮三臂合计 37500 次 B8 主模型前向、375 次 B8 解码。统一三臂评价另花费 22.026739 秒外层 wall。

真实参考取历史固定 bank 的全部 5000 张；取得这些真实 latent 的历史编码成本尚未完整闭合。此前两套 bank 能量审计的 CPU 处理为 55.054054 秒，但它既不等于单套参考准备的最低成本，也不包含历史编码，不能拿来代替完整准备费用。既有缓存不代表免费。因此本轮提供机制和质量筛查，**没有完成总成本意义上的达标证明**。

## 对机制的含义

独立[逐步审计](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spatial_energy_balls_v1/diagnostic_review_v1/review.json)完成 2500 项核对：100 个 successor time、每步完整 N1000、预算和实际 FP32 λ 均符合冻结实现。空间球有 58 步至少一个分量发生收缩；DC/AC 的最小 λ 分别为 0.99597305 / 0.97814834。DC 的正预算浮点残差最多 1.10e−8，AC 最多 5.11e−8。日志中 `constraint_note` 沿用了全局球的文字说明，审计已注解；实际 mode、计算公式及数值正确，冻结文件未改。

空间球最后一步投影前的 DC/AC 能量为 0.48888406 / 0.51024044，投影后为 0.48495456 / 0.51024044，预算为 0.48495455 / 0.51643954。因此该步只收缩 DC，AC 已低于上界。**这些投影前状态属于此前已经受控的轨迹，不是另一条官方轨迹的 endpoint**；不能将历史未受控 IG 的“DC 不足”直接套到这里。实际反馈随轨迹变化，不能仅凭历史终点分量偏差预测全程投影方向或图像结果。

[两套历史 5K 的能量分析](RAEV2_SPECTRAL_ENERGY_AUDIT_20260906_ZH.md)确实发现了总能量中 DC 不足与 AC 超额的抵消；闭凸球投影也确实对当前经验分布和满足预算的目标给出单步 W₂ 不增保证。这两项仍成立，不能用本次阴性 FID 倒推它们为假。

但后续动力学、decoder 和评价特征不保留该单步排序，历史潜变量也不等于本次全部实际轨迹状态。实验未支持“限制这两个空间分量的超额即可提高 RAEv2 图像质量”这一有限设计。它没有定位失败必然来自哪个环节，也不能把旧 decoder reversal 升级为本轮失败的已证实原因。后续机制必须连接实际轨迹纠偏与解码后的质量，而不能只把一个可重复的 latent 偏差重新命名为生成错误。

## 主要档案摘要

完整记录在运行目录。以下摘要方便复查冻结顺序，不取代原文件：

| 档案 | SHA-256 |
|---|---|
| 冻结协议 | `5ff6e86d26f3c24dc1304dd34ec8c9f566daa5e7bccf17862726eb7ac8df9df9` |
| 冻结采样实现 | `e68c4d5197f40ba4cbffff54349edd619a172e171b8265682def8e108cc0ab34` |
| 采样执行记录 | `89734d83f2fe4b2d1f16b515e2c93aca7d663190dcb62e2affb0261864bcbc57` |
| 看 FID 前的成本决定 | `a5cb1aa678de4ab3b259a94576f133e387a91eb6730eeb060e226affc067c897` |
| FID 请求 | `8a428b25756e29e6ca40e83c0d39a7f867bb7fcea2ca3ab4de440cfff8806d46` |
| FID 完整结果 | `01f75536729e653ab14ce8c9c1e1da5589242e755977609c5aecd008b077bfd7` |

完整噪声 SHA 为 `8847ee5c667e23f413fad195234728890c4a72260fdde5ef0cd04b2762762de0`，CUDA RNG 状态为 `3c342629c8a0ddbda83c73e11470a7c9d7ca357c74231108872c059de23c92e1`，标签为 `702746827e553786bb026ac120cb58745fef3d3f554c33891809001cc37639f0`。这些在三臂一致；seed 为 202609121。
