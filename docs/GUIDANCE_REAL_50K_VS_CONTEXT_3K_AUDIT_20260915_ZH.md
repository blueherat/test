**本轮标准FM loss 50K与旧Context MLP 3K不是只改变训练步数的对照。训练数据规模与clean latent采样方式同时变了。**

同一SiT 1K噪声/标签bank、相同IG系数0.8与采样窗口下，旧Context MLP的FID为64.907565，本轮real为67.660848，相差2.753283。两者都使用深度4的冻结SiT特征与304,528参数的Context MLP，标准目标都是`z=t*clean+(1-t)*noise`、预测`clean-noise`的MSE。学习率3e-4、batch 32、AdamW weight decay 1e-4与EMA 0.995相同。

|训练设置|旧Context MLP 3K|本轮real 50K|
|---|---:|---:|
|实际训练真实图像数|100类×256=25,600|100类×18=1,800|
|clean latent|每次draw重新采VAE posterior|每张图预采一次后固定|
|训练步数|3,000|50,000|
|总监督样本呈现数|96,000|1,600,000|
|每个图像/固定latent平均呈现次数|3.75|888.89|
|验证图像数|100类×32=3,200|100类×2=200|

旧数据来自`experiments/guidance_distribution_20260912/local_head.py`的RealData与`experiments/sit_measure_guidance_20260912/data.py`的Pool。实际train moments形状为`(100,256,8,32,32)`，validation为`(100,32,8,32,32)`；每次根据均值和标准差加入新的VAE posterior噪声。native分支只返回抽到的第一个parent，其余辅助抽样不计入训练样本数。

新数据来自`experiments/guidance_loss_50k_20260914/components.py`的SourceBank。`real_clean.npy`实际形状为`(100,20,4,32,32)`，每类前18个训练、后2个验证。该bank由`experiments/sit_sampler_reference_20260912/data.py`从旧train moments每类前20个图像各采一个posterior latent，再经strong-reference数据准备流程复制。本轮diffusion时间和噪声仍然每次更新，但clean latent不会重新采样。

本轮头是重新初始化训练，只有输入归一化buffer和位置编码沿用旧头；不是从旧3K参数续训。初始化和训练随机种子也不同。两轮验证数据不同，不能直接比较各自报告的验证MSE；本轮real自身的验证目标从5K时0.88268下降到50K时0.87310，不能仅凭最终FID变差断言已经出现验证loss意义上的过拟合。

因此，目前可以确认的是训练协议存在重大混杂。数据多样性下降是首要怀疑，尚未被控制实验单独证明；“训到50K导致guidance变差”没有被这组比较证明。本轮各新loss与本轮real之间仍是同一小bank协议下的比较，但不能据此把所有差异归因于loss本身或用旧3K建立纯训练时长结论。需要归因时，应恢复原25,600图像与动态posterior流，固定初始化，并比较中间3K和最终50K；此处只记录诊断，不另启任务或插队当前扫描。

1K质量指标原始记录位于`/home/zhoushunyu/data/eqvae/experiments/guidance_loss_50k_20260914/sit_small/screen1000/{context,real}/metrics.json`，两者bank SHA256同为`dbf1a22c09f9586241714c88c5543d1cdbfa50b2c2ee26d5ba5e1d6557d913f0`。
