# Self-Guidance 在 JiT 与 RAEv2 上的迁移测试

**已全部完成（2026-09-14）：JiT 9组、RAEv2 8组，每组1000张，共17000张配对生成和评价。本轮所有已测SG/SG-prev配置的FID均差于各自基线；按预先规则没有追加新种子复核或参数搜索。代码、全量样本及轻量证据均保留。**

用户要求在小SiT实验后继续测试JiT和RAEv2。本轮沿用已核对的[MAPLE官方实现](https://github.com/maple-research-lab/Self-Guidance)，版本 `843bda799bb531ccf8d98164d8ccf1a3b8ce3f6d`，论文为[Self-Guidance: Boosting Flow and Diffusion Generation on Their Own](https://arxiv.org/abs/2412.05827)。这是新模型上的本地迁移测试；SiT结果及其代码保持原样。

## 结果产生前固定的方法与对照

不训练、不更换权重。JiT-B/16使用本地官方 `model_ema1`、ImageNet1K、256像素；RAEv2使用原DINOv3-L K7、stage2 EMA、既有decoder与归一化统计。两个模型都输出clean预测，因此先在**各自查询时刻**转换为velocity，再应用上一轮核对的发布版SG规则：`v_out=v_baseline+omega*(v_cond-v_reference)`。SG参考保持同一个当前状态和类别，只改时刻；SG-prev缓存上一个接受步骤的原始条件velocity，既不缓存CFG/IG后的输出，也不重复转换旧预测。

|项目|JiT|RAEv2|
|---|---|---|
|桥接时间|`z=t*x+(1-t)*epsilon`|`z=(1-t)*x+t*epsilon`|
|velocity|`(clean-z)/max(1-t,.05)`|`(z-clean)/max(t,t_eps)`，本地t_eps=.05|
|SG更噪查询|`max(0,t-.01)`|`min(1,t+.01)`|
|SG-prev激活|`t>.5`|`t<.5`|
|主求解器|均匀100步Euler|原shift=8的100步Euler|
|已有guidance|CFG=3，`.1<t<1`|IG=1.78，`.1<=t<=1`|

时间偏移和半程门控均按物理噪声时间定义，不按循环索引定义。尤其RAEv2的shift=8网格，SG-prev只在最后约11步生效。历史每条采样链重新清空，在激活前也持续保存原始预测。SG首步更噪时刻裁剪后等于当前时刻，复用预测并省一次重复前向。

JiT CFG保持官方 `v_uncond+scale*(v_cond-v_uncond)` 的运算顺序及窗口外null查询。RAEv2保持项目既有PFR/runtime所用的velocity域IG：`v_weak+1.78*(v_full-v_weak)`；这不是另行交换BF16 clean混合与FP32 velocity转换的次序。RAEv2强/弱头在一次共享模型前向内产生。

这里采用发布版**velocity外推**，不声称直接velocity差严格实现论文跨噪声密度比的精确score；也没有在x-prediction空间另做一套方法并与此混记。

每个模型首先固定8组：strong基线、strong+SG(omega=1)、strong+SG-prev(1)、原guided基线、guided+SG(1/3)、guided+SG-prev(1/3)。JiT额外固定官方Heun50、末步Euler、CFG3对照，共9组。总计17组，每组1000张，首轮17000张。

JiT Euler100 CFG每图200次full，SG299次、SG-prev200次；Heun50每图198次。RAEv2原IG每图100次full，SG199次、SG-prev100次，均无独立prefix查询。真实计数由模型forward hook记录并在每批与预期核对。

所有模型权重和采样状态为FP32，模型BF16 autocast，TF32开启，与采用的既有runtime一致。原模型不做训练或权重转换。原预测转velocity时保留FP32状态运算。

## 数据、评价与复核规则

每个模型建立自己的新噪声bank，NumPy PCG64 seed=2026091327，1000类各一张、打乱标签；所有方法共享该模型的同一输入和batch边界。批大小在正式冻结前由完整生成及解码显存测试确定，最终值写入每个request。四GPU按配置动态取任务，不在一个GPU上同时运行两个采样任务。

两个模型统一使用本地既有nanogen-evals官方评价器、同一 `imagenet_256_fid_stats.npz`、Inception FID和IS，特征缓存保留。这里的FID与上一轮SiT使用的ImageNet100 ADM参考不同，禁止直接跨模型比较绝对值。1K是小样本配对探索，不能当作论文50K指标。固定每类一张也不足以评估类内多样性。

每组保存全量像素、各batch终点状态、输入/labels/request/源码/模型/评价资产哈希、实际forward计数及采样+解码GPU秒。固定前32图提供预览，汇总对比图使用每组相同的前6个ID，不挑图。指标提取和模型加载时间不混入采样成本。

如果某模型的guided SG或SG-prev比同源基线FID改善，固定该family中FID最低的已测参数（差距<.001时选较小omega），在新seed=2026091328上做独立1000张配对复核，不再调参；SG还加入相近前向预算的纯Euler步数对照（JiT150步=300次，RAEv2 199步=199次）。若无改善，报告已完成负结果，不自动扩大搜索。是否继续做5K不在本轮默认范围。

## 实现与运行

目录：[experiments/self_guidance_cross_model_20260913](../experiments/self_guidance_cross_model_20260913/)。独立CPU检查验证两种时间方向、同状态查询、原始条件历史、半程门控、历史重置及零强度一致性；真实模型检查确认JiT与官方Euler逐元素一致、RAEv2与既有velocity-IG循环逐元素一致，两种模型的SG/SG-prev零强度均逐元素恢复基线。生产batch另测完整SG采样及解码。

```bash
CUDA_VISIBLE_DEVICES=0 /home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.self_guidance_cross_model_20260913.check --model jit
CUDA_VISIBLE_DEVICES=1 /home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.self_guidance_cross_model_20260913.check --model raev2
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.self_guidance_cross_model_20260913.run prepare --phase screen_1k --batch 32
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.self_guidance_cross_model_20260913.run controller --phase screen_1k --gpus 0,1,2,3
```

原始输出：`/home/zhoushunyu/data/eqvae/experiments/self_guidance_cross_model_20260913/`。每个阶段的控制器支持续跑；已完成配置和原子写入的完整batch可复用，最终收集会复核其输入与请求身份。任务失败时保留现场并停止同控制器所属worker，避免无声混用失败结果。

## JiT 完整结果

9组均完成，每组1000张，所有配对与FID重算核查通过。

|配置|FID ↓|ΔFID对同源基线|IS ↑|Full/图|采样+解码GPU秒|
|---|---:|---:|---:|---:|---:|
|`strong`|66.220|+0.000|27.780|100|160.9|
|`strong_sg_w1`|66.716|+0.497|26.777|199|317.9|
|`strong_sg_prev_w1`|67.438|+1.219|27.883|100|163.0|
|`cfg`|38.596|+0.000|59.833|200|318.4|
|`cfg_sg_w1`|39.080|+0.484|60.483|299|473.9|
|`cfg_sg_w3`|42.619|+4.023|55.528|299|484.0|
|`cfg_sg_prev_w1`|38.779|+0.183|59.946|200|318.1|
|`cfg_sg_prev_w3`|39.287|+0.691|59.444|200|318.4|
|`cfg_heun50`|38.670|+0.074|58.272|198|314.3|

无CFG和有CFG下，所有已测SG/SG-prev设置的FID均差于同源基线。SG omega=1叠加CFG增加约49%采样及量化耗时，FID恶化0.484；SG-prev不增加模型前向，但FID仍恶化0.183。更高强度omega=3更差。官方Heun50对照FID=38.670，也优于本轮所有叠加SG的候选。按预先规则，不对JiT负结果追加参数搜索或独立种子复核。

[JiT配对样本图](data/self_guidance_cross_model_20260913/screen_1k/jit/comparison.png) · [完整CSV](data/self_guidance_cross_model_20260913/screen_1k/jit/results.csv) · [请求与资产](data/self_guidance_cross_model_20260913/screen_1k/jit/request.json) · [审计](data/self_guidance_cross_model_20260913/screen_1k/jit/results.json)。所有完整图像与batch终点均保存在数据盘原目录。

## RAEv2 完整结果

全部8组完成，每组1000张；固定EMA、decoder与stats、原shift=8网格、IG=1.78。

|配置|FID ↓|ΔFID对同源基线|IS ↑|Full/图|采样+解码GPU秒|
|---|---:|---:|---:|---:|---:|
|`strong`|39.611|+0.000|53.222|100|737.0|
|`strong_sg_w1`|42.780|+3.170|47.123|199|1453.0|
|`strong_sg_prev_w1`|39.842|+0.231|53.100|100|736.8|
|`ig`|38.704|+0.000|58.779|100|728.4|
|`ig_sg_w1`|41.170|+2.466|52.544|199|1467.0|
|`ig_sg_w3`|46.263|+7.559|45.069|199|1463.1|
|`ig_sg_prev_w1`|38.784|+0.079|58.777|100|728.7|
|`ig_sg_prev_w3`|39.032|+0.327|57.669|100|732.9|

无IG时，SG omega=1使FID上升3.170，SG-prev使FID上升0.231。有IG时，SG omega=1/3分别恶化2.466/7.559；SG-prev omega=1/3分别恶化0.079/0.327，没有候选优于原IG基线。SG omega=1使采样加解码时间增加约101.4%，SG-prev则基本保持成本。更高强度没有带来改善。

[RAEv2配对样本图](data/self_guidance_cross_model_20260913/screen_1k/raev2/comparison.png) · [完整CSV](data/self_guidance_cross_model_20260913/screen_1k/raev2/results.csv) · [请求与资产](data/self_guidance_cross_model_20260913/screen_1k/raev2/request.json) · [审计](data/self_guidance_cross_model_20260913/screen_1k/raev2/results.json)。

## 结论、核查与范围

在本次冻结checkpoint、默认偏移.01、默认噪声半程SG-prev窗口及已测强度下，JiT与RAEv2均未观察到FID改善。SG部分样本会改变纹理、构图或物体细节，不能由个别图像观感推断整体质量提升。SG-prev的较小FID差异也未经过多种子显著性检验，因此这里报告的是**本轮配对筛选未见收益**，不宣称已证明所有参数或所有模型上无效。

17组完整输出均保留，每类每配置一张；所有batch覆盖、标签、原始noise身份、源码/模型/评价资产SHA、实际前向计数及计时核对通过。两个真实模型的零强度及主基线逐元素检查通过。17组FID都从缓存特征以FP64重新计算并通过误差阈值0.002；这是独立计算核查，仍使用同一特征提取器，不当作另一套独立评价模型。四个worker退出码全部0，任务已结束，GPU已释放。

本轮使用统一ImageNet1K的nanogen评价参考，不能与此前SiT的ImageNet100 ADM绝对FID直接比较；也不是论文原SD3/FLUX的50K复现。小样本FID存在偏差和波动，完整证据支持保留原采样配置，尚不支持将本轮SG设置作为质量优化。

[机器可读完成摘要](data/self_guidance_cross_model_20260913/summary.json)。生成归档和全部方法的固定样本图：

```bash
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.self_guidance_cross_model_20260913.report --models jit,raev2
```
