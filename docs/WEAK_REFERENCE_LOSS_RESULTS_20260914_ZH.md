# 参考分布训练目标：执行与结果

执行状态：`paused`。当前完成1K评估0组、独立5K评估0组。

这是固定目标比较，不是参数扫描。新方法只有完成生成评估后才能判断质量。

[冻结协议](WEAK_REFERENCE_LOSS_PROTOCOL_20260914_ZH.md) · [目标推导](WEAK_REFERENCE_LOSS_DESIGN_20260914_ZH.md)

|阶段|方法|样本数|FID↓|IS↑|完整/额外前缀调用|
|---|---|---:|---:|---:|---|

当前尚无新目标的图像质量结果。GPU预检也必须在空闲后完成。

原始产物：`/home/zhoushunyu/data/eqvae/experiments/weak_reference_loss_20260914`。训练和采样均可按同一请求校验恢复，STOP_AFTER_CURRENT仅作用于本实验。

来源bank为各2K连续latent，所有新头固定3K；原有Context控制来自既有3K训练。所有新采样均为独立于训练bank的新噪声、类平衡配对。

采样和解码墙钟时间保存在原始batch中，不当作经过专门预热控制的速度基准。
