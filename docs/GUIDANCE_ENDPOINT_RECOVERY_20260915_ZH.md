**2026-09-15：强模型端点库生成因标签 dtype 报错；独立运行时修复将标签转为 int64。**

在 16:05（北京时间）查询时，原队列已于 15:42:57 自动暂停。三个已启动的 `generate__sit_small__strong__train` 任务均在首批失败，SiT self 训练尚未开始，端点库中没有已完成或部分保存的批次。

根因已由日志和实际数据确认：SiT `data/train_labels.npy` 与 `validation_labels.npy` 是 int16，原 `endpoints.generate` 从 NumPy 直接创建 CUDA ShortTensor。SiT 的类别 embedding 只接受 int32/int64 索引。此前质量评估使用的独立标签 bank 为 int64，因此本次失败不使已经完成的 1K/5K 结果失效。JiT 对应训练/验证索引为 int64。

原始失败日志：`/home/zhoushunyu/data/eqvae/experiments/guidance_dynamic_50k_20260915/logs/generate__sit_small__strong__train__0.log`。

修复位于 `experiments/guidance_endpoint_recovery_20260915`，不修改原始冻结源码。它只包装端点生成模块的两个积分入口，在调用既有 strong/native/CFG 积分器或独立 weak 积分器之前执行 `labels.long()`。标签值、端点保存格式、初始噪声、采样步数、模型输出和系数不做改变。已有训练与质量结果继续复用。

修复清单位于实验根目录 `endpoint_repair_20260915/request.json`，包含代码哈希并继承原科学清单及 guided_weak_deployed 清单。恢复前需要完成实际 GPU 全批采样检查，并归档本次暂停和失败记录；只重新排入本次失败的三个生成任务。

检查已通过：实际 SiT 首批 64 个标签及对应固定初始噪声，修复后的 int16 输入与原积分器接收 int64 输入得到逐位相同的最终端点，均为 64 步 Heun、128 次完整模型调用。四类端点生成路线的标签转换和两模型实际训练/验证标签值也已检查。记录在 `endpoint_repair_20260915/validation.json`。

北京时间 16:12:47 已恢复控制器 PID 1036300。旧状态及失败记录保存在 `endpoint_repair_20260915/before_resume`，恢复记录在 `resume.json`。16:13:48 确认 GPU 0、1、3 正在生成强端点，已有 24 个已保存批次、1536 个训练端点，未出现新的失败或停止标记；GPU 2 被其他任务占用。当前仍是 self 的样本库准备阶段，尚非 self 训练。

**本次同时获得的生成结果。** 下表均为完整数据、弱头 50K、四级 1K 扫描后选两个系数扩展到 5K 的相同评估协议。FID 越低越好；5K 包含选择用的 1K。

|SiT 方法|最低 1K FID|两个 5K 中最低 FID|该 5K 系数|
|---|---:|---:|---:|
|普通真实数据弱头 real|64.1406|36.9095|1.0|
|原 guided_weak|63.6736|36.6888|1.05|
|guided_weak_deployed|65.0271|37.3767|1.025|

guided_weak_deployed 将训练时间限制在既有引导区间，并将 SiT 的早期幅度曲线写入 loss。它相对原 guided_weak 的最好 5K 差 0.6878，相对 real 差 0.4672；这次跟进没有产生改善。因为同时改变训练时间分配和曲线，不能从这个对照单独定位是哪项造成退化。

guided_weak_deployed 另一系数 1.0 的 5K FID 为 37.3977232940。结果来自 `sit_small/points/guided_weak_deployed__c0040/n5000/metrics.json` 和 `guided_weak_deployed__c0041/n5000/metrics.json`，该方法于北京时间 15:42:37 完成全套评估。

三分布参考候选仍处于核心公式与接口实现状态，完整判别器训练及弱快照反馈尚未接入运行；不能把接下来已有 self 所需的强端点库生成称为三分布方法已训练。
