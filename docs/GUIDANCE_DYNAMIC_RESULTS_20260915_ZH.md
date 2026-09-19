**完整数据动态采样重训：先 SiT 全部 idea，再 JiT 全部 idea；当前两卡训练，全局 batch 256**

队列状态：paused。当前 sit_small / self：search。GPU任务占用 0/4。 待启动：sample__sit_small__self__c0064__1000__3, sample__sit_small__self__c0080__1000__0, sample__sit_small__self__c0080__1000__1, sample__sit_small__self__c0080__1000__2, sample__sit_small__self__c0080__1000__3（等待所需空闲资源）。

每个模型内部：一个 idea 训练 50K → 四级 1K 扫描 → 前两名各扩展 5K → 下一个 idea；SiT 全部完成后进入 JiT。

|idea|模型|状态|1K最佳系数|1K最佳FID|5K结果|
|---|---|---|---:|---:|---|
|real|sit_small|complete|1.0|64.14058231741768|1: 36.9095; 0.975: 36.9241|
|real|jit|training||||
|guided_weak|sit_small|complete|1.075|63.6736167185403|1.075: 36.7326; 1.05: 36.6888|
|guided_weak|jit|queued||||
|guided_weak_deployed|sit_small|complete|1.0|65.02714227303295|1: 37.3977; 1.025: 37.3767|
|guided_weak_deployed|jit|queued||||
|self|sit_small|search||||
|self|jit|queued||||
|mixture|sit_small|queued||||
|mixture|jit|queued||||
|gaussian|sit_small|queued||||
|gaussian|jit|queued||||
|excess|sit_small|queued||||
|excess|jit|queued||||
|shuffled|sit_small|queued||||
|shuffled|jit|queued||||
|contrast_s|sit_small|queued||||
|contrast_s|jit|queued||||
|contrast_m|sit_small|queued||||
|contrast_m|jit|queued||||
|contrast_weak|sit_small|queued||||
|contrast_weak|jit|queued||||
|covariance|sit_small|queued||||
|covariance|jit|queued||||
|contrast_null|sit_small|queued||||
|contrast_null|jit|queued||||
|real_residual|sit_small|queued||||
|real_residual|jit|queued||||
|weakmix_fm|sit_small|queued||||
|weakmix_fm|jit|queued||||
|weakmix_contrast|sit_small|queued||||
|weakmix_contrast|jit|queued||||
|ig_residual|sit_small|queued||||
|ig_residual|jit|queued||||
|ig_contrast|sit_small|queued||||
|ig_contrast|jit|queued||||
|cfg_residual|sit_small|queued||||
|cfg_residual|jit|queued||||
|cfg_contrast|sit_small|queued||||
|cfg_contrast|jit|queued||||

旧固定小bank结果保留为历史；本队列不复用旧训练头或旧系数选择。5K包含原1K，不是独立留出集。

数据与训练协议：[说明](GUIDANCE_DYNAMIC_50K_20260915_ZH.md)。原始记录：/home/zhoushunyu/data/eqvae/experiments/guidance_dynamic_50k_20260915
