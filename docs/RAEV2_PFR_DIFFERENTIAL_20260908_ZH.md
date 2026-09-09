# 回到独立 PFR 路线：RAEv2 强弱时间差分

近期 AG/IG 载体变体未取得实质收益，本轮暂停该路线，按照用户保留的两条路回到 PFR，不与 FSG 信息搬运混合。

已有正式5K：RAE mild PFR 的 FID 恶化主要来自均值项，SiT 原PFR均值项改善；共同时间响应在两者均存在。clean输出对照和canonical投影对照未解释 RAE 失败。这些不是共同时间响应有害的因果证明，但给出具体方法试验。

记同状态 z 当前/未来时刻完整模型的速度为 S,W 与 S',W'。原PFR修正 W-W'。本版采用：

    R = (W-W')-(S-S')
    v_new = v_native_IG + 1.78*R

固定 future=t-min(1/32,t-.5)，仅 t>.5 修正；query始终是当前 z。rho=1沿用已有canonical查询试验。其余原生IG继续，无范数恢复、投影、额外载体、窗口或参数扫描。

R=(S'-W')-(S-W)，所以原生IG活动时刻该式精确等于 W+1.78*(S'-W')。它保留当前弱速度作transport anchor，用未来的深浅差替换当前差，并不是完整模型预测搬运。这种 difference-in-differences 思路在旧 SiT 资产中失败过（PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md §相关结果），不是全新算子或通用贡献。这里检验其 RAE 特异效用；如果两侧都失败，就不能用简单删除强时间响应解决迁移。

固定100步shift8，IG1.78原窗口，B4，FP32状态/BF16+TF32，seed202609413，1000类各1图。100当前Full+89未来Full=189完整前向/图，无prefix；仅本方法四卡协同采样，逐batch保存。复用已有native38.264239、raw canonical52.942740等结果，不重采样或重评对照。若有质量收益仍须更大独立样本及计算价值验证。不写论文。
