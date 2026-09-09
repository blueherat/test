# 历史分离的 IG：固定四卡 1K 方法

历史 IG_GUIDANCE_HISTORY_20260908_ZH.md 做过8条轨迹的机制比较，没有测试本方法的生成质量。本次按用户“方法先于机制门槛”要求实施质量试验。旧机制结果没有证明历史会耗尽 guidance，本文也不采用这个前提。

当前轨迹 zG 与 Full-only 影子轨迹 zS 从同一噪声与类别出发。每个 IG 活动时刻：

    fG = Full(zG,t)
    fS,bS = Full(zS,t), Base(zS,t)  # 同一次完整 forward
    guided_clean = fG + .78*(fS-bS)
    zG 按 guided Euler 前进；zS 按 Full-only Euler 前进

实际用 reference=fG-(fS-bS) 对原 zG 转 velocity 后走 native 外推算术。影子轨迹提供有历史的纠正信号载体，当前 Full 仍读自己的状态，guidance 保持到原活动区间结束。没有使用 fG-bS 的跨状态内容差，不是把弱参考预测直接从另一张图搬过来。但影子轨迹的局部方向也未必适合当前状态，质量收益是假设；同状态 score ratio 解释不直接适用于该历史依赖场。简单双轨迹本身不构成新颖性主张。

初始 t=1 两轨相同，复用当前 Full/Base。其余98个活动时刻额外完整 forward；最后非活动步不再查询影子。每图198 Full，无 Base前缀；近两倍调用成本。若只取得小幅增益，不能证明同算力价值。

固定100步 shift8、IG1.78与原窗口、seed202609413、B4、FP32状态/BF16+TF32、全1000类各1图。只有本方法四卡协同采样，各 batch 原子保存并记录模型/源码/请求/噪声哈希。已有 native IG 数据和评估结果直接复用，不重跑对照。不扫描融合系数、reset频率或窗口。不写论文。1K只是方法筛查，不作为可靠成功证据。
