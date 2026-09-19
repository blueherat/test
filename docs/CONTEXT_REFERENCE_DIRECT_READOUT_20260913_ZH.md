# 直接替换IG弱读出：等价实现与成本

这一改动只优化已经冻结的Context参考的执行方式，不更改指导场、采样时间网格、强度、窗口或权重。原试验为了验证捕获特征，先计算原IG强弱输出，再用新读出替换弱输出；因此测得约2%—3%的附加耗时。实际部署可以直接替换原弱头。

在SiT-S/2的同一次主干遍历里，到depth4时使用新MLP输出弱预测，继续到完整深度输出强预测。只执行一个弱读出，没有独立弱模型、重复prefix或特征捕获hook。

原弱头参数301,840，新读出304,528，替换后净增加2,688。若隐藏宽度为d、输出patch宽度为r，原AdaLN读出参数数目为2d²+(2+r)d+r，新MLP含固定8维位置项的参数数目为2d²+(9+r)d+r；差额为7d。该参数计算说明两者规模接近，不证明二者学习能力或实际计算相等。

同GPU、batch8、完整64步Heun采样和解码，轮换三次取中位数：

|实现|秒/batch|主干查询/图|额外prefix/图|
|---|---:|---:|---:|
|原IG|0.764076|128|0|
|旧捕获方式的Context|0.782719|128|0|
|直接替换后的Context|0.766292|128|0|
|原IG，66步|0.779510|132|0|
|ADG|0.800421|128|0|

直接替换版相对原IG中位数增加0.29%。只有三次计时，这个细小差别不足以主张精确的长期吞吐率；合理表述是“同主干查询数，实测耗时接近原IG”，而不是严格零开销。

固定24条完整轨迹中，直接替换版与原捕获实现的latent及解码像素逐项相同。随后导出的安装接口又逐项复现冻结5K中首批8张的latent及图像，128次full和0次prefix核对通过。额外等价检查不纳入质量样本数，不作为新的独立质量证据。

可复用入口是[reference.py](../experiments/context_reference_5k_20260912/reference.py)：

```python
from experiments.guidance_pasted_20260912 import common
from experiments.context_reference_5k_20260912 import reference

runtime = common.runtime('sit_small')
provenance = reference.install(runtime)  # 校验保留权重，替换原弱头
latents, counts = reference.sample(runtime, noise, labels)
images = runtime.decode(latents)
```

只支持当前已验证的SiT-S/2与depth4 velocity读出。该入口验证权重、训练请求和实际依赖，不会训练模型或加载第二个弱网络。冻结的5K原始采样实现仍保留以便审计。

检查脚本：[direct_readout.py](../experiments/context_reference_5k_20260912/direct_readout.py)。原始逐次时间、24条轨迹检查和接口检查分别保存在`/home/zhoushunyu/data/eqvae/experiments/context_reference_5k_20260912/sit_small/direct_readout_verification.json`与同目录`replacement_api_verification.json`。质量结论以[5K报告](CONTEXT_REFERENCE_5K_RESULTS_20260912_ZH.md)为准。
