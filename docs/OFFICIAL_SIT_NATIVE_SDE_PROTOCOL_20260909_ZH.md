# 官方 SiT-XL 强基线：原生 SDE、CFG 与 IG

本实验在读取任何新原生 SDE 质量结果前固定。它补足此前 Euler115 / PFR100 配对探索与作者发布配置之间的距离；不提出新方法，不代替 PFR 对称曲线，也不宣称 1K 复现了作者的 FID-50K。

## 配置与源码依据

直接调用固定版本 `Internal-Guidance/SiT/samplers.py` 的 `euler_maruyama_ig_sampler`，不重写随机动力学。官方 `gen.sh` 给出 SDE250、CFG=1.35、IG=1.4、CFG high=.7、IG low=0、VAE=ema；`generate.py` 默认 IG high=1。脚本变量名 `sg_val` / `sg_guidance_low` 与已声明变量不一致，Python 还引用不存在的 `args.sg_scale`；这里将上述明确数值作为结构化参数直接传给原生 sampler，避免运行有缺陷的启动脚本。

模型为已核对的官方 800EP SiT-XL/2 EMA，联合 depth-8 弱头，FP32 网络与原生 FP64 SDE 状态。TF32 关闭、B4、同一初始噪声与平衡 1000 类标签；这两项不同于发布脚本的默认 TF32 开启、随机标签大批次。因此本实验是发布超参数的受控实现，不能称逐像素复现作者图像。

只生成一个原生 SDE 轨迹集合，分别使用 `stabilityai/sd-vae-ft-ema`（发布脚本配置）和 `stabilityai/sd-vae-ft-mse`（此前本地 Euler/PFR 配置）解码。两者原始权重文件和实际 VAE 状态均记录哈希。双解码用于区分解码器差异与采样器差异，不能从中隐蔽择优。

## 样本与计算

复用 `official_sit_baseline_control_20260909/inputs.npz` 中精确的 1K 输入；仍属探索银行。每个全局 B4 批次的 Brownian seed 由固定 namespace `official_sit_native_sde_20260909/brownian/batch/{index}` 的 SHA-256 前八字节转换为 63-bit 整数，记录全部 seed。每个批次重设生成器；不将一个名义 seed 的相邻偏移当成独立复现。原生 sampler 每条轨迹产生 249 次 FP64 Gaussian 增量，最后一步不加随机噪声。

四张卡 round-robin 分配 250 个全局 B4 批次。包装模型仅统计真实前向输入大小、调用次数和时间区间；每图有 250 个采样步，由于 CFG 双分支，完整网络计算不等于 250 次单图前向。预先从同一时间网格计算调用数，并与实测逐项核对。保留推理、双解码时间，不将较昂贵 SDE 与便宜 Euler 直接称为同成本比较。

## 验证与输出

运行前核对旧银行、checkpoint、源码及 VAE 文件。前四个批次各在对应卡以相同 Brownian seed 重跑一次，要求原生 sampler 的 FP64 终点逐位一致；包装调用数满足网格计数。解码检查有限值和 uint8 范围。全部 1000 个索引必须恰好覆盖一次，合并终点与各片段逐位相同。

输出两组真实图像与 FID/IS、原始特征、终点潜变量、源码快照和请求记录；之后用同一特征的 float64 样本空间谱公式复算 FID。该数值核验不等于独立抽样或另一评估模型。此次没有新 PFR 臂，后续若比较 PFR，必须用同一原生配置、配对随机增量与明确计算预算，不能直接拿旧 Euler 的结果充当对照。

此基线用于约束论文方法应超过的水平。仅完成基线、发现更低 FID，或复原作者的采样行为，都不构成用户所要求的实质新 idea 或机制证据。
