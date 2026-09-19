# IG 基础上叠加原版 SG 与 log-score SG

已完成首轮15组与JiT独立复核3组，每组1,000张，共比较18,000张，其中15,000张为本次新生成、3,000张明确复用先前结果。所有样本与 FID 审计通过。**JiT＋原版 SG 在两个种子均改善 FID（0.21和1.45），代价接近两倍前向；SiT 无收益，RAEv2 没有超过更省计算的纯 IG 对照。log-score 版本未显示值得采用的叠加收益。**

本轮直接回答“已有 IG 后，再加 SG 是否还有收益”。三个模型都保留既有 IG 弱头、系数和时间窗口，分别加入论文发布版 SG 和上一轮的 log-score 外推。每个模型还跑两组只增加采样步数的 IG 对照，判断质量变化是否超过额外计算本身。

模型为 SiT-S/2 ImageNet100 EMA800K、官方 JiT-B/16 EMA1 加既有 depth4 读出头，以及 RAEv2 DINOv3-Lk7。SiT 与 JiT 的弱读出头均为之前训练完成的 EMA50K，本轮不训练。

方法定义为

\[
v_{\mathrm{out}}=v_{\mathrm{IG}}+\omega\,(v_s-v_{\mathrm{ref}}).
\]

这里的 SG 对比项来自原始强模型。原版 SG 在同一个状态、同一个类别下，把参考查询时间朝更噪方向移 .01；log-score SG 保持查询时间不变，用一对对称扰动：

\[
v_{\mathrm{ref}}=\tfrac12\left[v_s(z+\xi,t)+v_s(z-\xi,t)\right],
\quad \xi=0.2\,\sigma_{\mathrm{path}}(t)\epsilon,
\quad \omega=1.
\]

这延续了[原版 SG 的发布实现](https://github.com/maple-research-lab/Self-Guidance)与[上一轮 log-score 推导及实验](WEAK_REFERENCE_GUIDANCE_RESEARCH_20260914_ZH.md)。跨时刻版本按发布代码做 velocity 外推，不声称它严格等于跨时刻的精确 score 差。log-score 版本平滑的是 score／log-density，不能把它直接称为密度高斯卷积后的精确 score。本轮不另测试对整个 IG 混合场构造 SG 的递归组合。

| 模型 | 固定 IG | 主求解器 | IG / 原版 SG / log-score SG 前向数 | 纯 IG 成本对照 |
|---|---|---|---|---|
| SiT-S/2 | alpha=.8×6/7（t<.25），.8（.25≤t<.5），之后0 | Heun64；两级沿用左端 IG 系数 | 128 / 255 / 382 | Heun128=256；Heun191=382 |
| JiT-B/16 | alpha=.3（t<.5），之后0 | Euler100 | 100 / 199 / 300 | Euler199；Euler300 |
| RAEv2 | weak+1.78×(strong−weak)，噪声时间 .1≤t≤1 | Euler100，shift=8 | 100 / 199 / 300 | Euler199；Euler300 |

SG 系数、偏移和噪声尺度在生成前固定，没有根据本轮指标调参。原版 SG 首个参考时间与当前时间相同，复用前向；SiT 的 Heun 末级 log-score 扰动尺度为零，也复用前向。因此 SiT 原版 SG 的对照多一次前向，其余成本对照完全一致。弱头复用主干，前向计数通过第一个 transformer block 的实际调用核对。

所有配置共享各模型的同一噪声和标签，seed=2026091407。SiT 为100类每类10张；JiT／RAEv2 为1,000类每类1张。SiT 保留 ADM ImageNet100 5K验证参考；另两个模型使用既有 Nanogen ImageNet256 参考与 Inception 权重。不同评价协议的绝对 FID 不直接横向比较。

RAEv2 的 IG、IG＋log-score、Euler300 三组来自此前 `weak_reference_20260914/cross_screen_1k`；输入、权重与源码核对一致，新旧实现还通过了逐位结果检查。它们明确标记为复用结果，不算本次新生成或独立重复。总计比较15,000张，本次新增12,000张。此前另一个种子的 RAEv2 原版 SG 已是负结果，本轮增加同噪声、同计算量对照，不能把重用的微小 log-score 优势算作第二次证实。

每组保存全量像素、每批终点、输入与请求哈希、实际前向数和采样加解码耗时。源码、强／弱头权重、decoder与评价资产均由请求冻结。生成后逐批核对覆盖、类别、输入、输出与合并样本，并以缓存特征 FP64 重算 FID；该重算检验数值计算，不是独立特征提取器验证。固定前6个样本用于配对展示，不按观感挑图。

[冻结实验协议](../experiments/ig_sg_20260914/PROTOCOL.md) · [采样实现](../experiments/ig_sg_20260914/core.py) · [可续跑队列](../experiments/ig_sg_20260914/run.py) · [结果审计脚本](../experiments/ig_sg_20260914/report.py)

```bash
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.ig_sg_20260914.run prepare
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.ig_sg_20260914.run controller
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.ig_sg_20260914.report
```

原始结果保存在 `/home/zhoushunyu/data/eqvae/experiments/ig_sg_20260914/`。

## 首轮配对结果

FID 越低越好。括号内的成本对照保留原 IG，只增加采样步数。

| 模型 | IG | IG＋原版 SG | 原版 SG 成本对照 | IG＋log-score SG | log-score 成本对照 |
|---|---:|---:|---:|---:|---:|
| SiT-S/2 | 65.494 | 66.119 | 65.578 | 67.342 | 65.584 |
| JiT-B/16 | 55.509 | 55.299 | 56.896 | 65.044 | 57.144 |
| RAEv2 | 39.432 | 41.896 | 39.182 | 39.253 | 39.304 |

SiT：原版 SG 比 IG 恶化0.625，log-score 恶化1.848，也都差于各自的成本对照。

JiT：原版 SG 比 IG 改善0.210、比199步 IG改善1.597；但增加步数的 IG 本身比100步基线差1.387，所以“优于成本对照”不能替代“优于原基线”。log-score 比 IG 恶化9.535。原版 SG 的独立复核见下文。

RAEv2：原版 SG 比 IG 恶化2.464。log-score 比 IG 改善0.179，比300步 IG改善0.051；但199步 IG的39.182还低于log-score的39.253，且只需199次而非300次前向。本轮不足以支持为这点小差异采用log-score。该log-score测量复用此前结果，不能当作独立重复。

| 模型 | IG 耗时/秒 | 原版 SG 耗时/秒 | log-score 耗时/秒 | 原版 SG / IG | log-score / IG |
|---|---:|---:|---:|---:|---:|
| SiT-S/2 | 67.6 | 119.4 | 170.7 | 1.77× | 2.53× |
| JiT-B/16 | 159.2 | 319.0 | 498.4 | 2.00× | 3.13× |
| RAEv2 | 740.6 | 1446.3 | 2179.4 | 1.95× | 2.94× |

耗时为每组1K的采样与解码墙钟时间，排除模型载入、预热、写盘与指标评价；每个任务独占一张4090。RAEv2的部分时间来自复用的先前运行，不能视为严格同步的速度基准。

固定前6个ID配对图：[SiT](data/ig_sg_20260914/sit_small/comparison.png)、[JiT](data/ig_sg_20260914/jit/comparison.png)、[RAEv2](data/ig_sg_20260914/raev2/comparison.png)。图中可见局部纹理、构图变化，但六张图不能估计总体质量或多样性。JiT的log-score组有部分细节变平滑的观感，与FID退化一致；不据此单独做机制归因。

完整CSV含FID、IS、SiT sFID、前向数、耗时和原始输出来源：[SiT](data/ig_sg_20260914/sit_small/results.csv)、[JiT](data/ig_sg_20260914/jit/results.csv)、[RAEv2](data/ig_sg_20260914/raev2/results.csv)。[合并机器结果](data/ig_sg_20260914/results.json)与[首轮审计](data/ig_sg_20260914/audit.json)均已保存。

本轮测试的是固定系数的直接残差叠加。它不能排除其他SG强度、时间窗或IG头下存在收益；也不能把1K FID的细小差异解释成统计显著改善。IG与SG两个方向是否互补，需要额外的匹配引导强度／残差方向分析，本轮的质量测试尚未证明这一机制。

## JiT 独立新种子复核

这是看到首轮结果后选定的跟进实验。保持IG alpha=.3、SG omega=1、时间偏移.01和100步Euler不变，选用全新的PCG64 seed=2026091417，对IG、IG＋原版SG、199步IG各生成1,000张；没有在新噪声bank上调整参数。未追加log-score，因为首轮已明显退化。

| 配置 | FID ↓ | 相对IG的ΔFID | IS ↑ | Full/图 | 采样＋解码/秒 |
|---|---:|---:|---:|---:|---:|
| ig | 56.340 | +0.000 | 31.306 | 100 | 170.0 |
| ig_sg_w1 | 54.895 | -1.445 | 32.878 | 199 | 317.8 |
| ig_sg_cost | 57.141 | +0.800 | 30.822 | 199 | 314.5 |

新种子下，原版SG比标准IG改善1.445，比同计算量IG改善2.246。首轮对应改善0.210和1.597。两次均为正向，但改善幅度不一致；这是值得保留的跨种子信号，尚不是50K评价或统计显著性的证明。两个种子均按相同类别逐样本配对，第二个种子完全独立于首轮选择。

**本轮建议：保留JiT的IG＋原版SG作为后续候选，默认参数为IG alpha=.3（前半程）、SG omega=1、shift=.01、Euler100。** 它把前向数从100增加到199；若目标是固定时延，还需另做步数分配比较。本轮的“同计算量对照”是给纯IG增加步数，不表示叠加SG与原始100步IG具有相同成本。SiT与RAEv2保留现有IG更有依据；本轮未支持采用log-score叠加。

这里验证的是已训练的JiT depth4读出头下，固定SG残差的增量收益，未证明IG与SG的数学协同机制，也不推广到其他JiT尺寸、读出头或参数。

[新种子配对图](data/ig_sg_20260914/confirm_1k/jit/comparison.png) · [完整CSV](data/ig_sg_20260914/confirm_1k/jit/results.csv) · [冻结请求与选择来源](data/ig_sg_20260914/confirm_1k/jit/request.json) · [复核审计](data/ig_sg_20260914/confirm_1k/audit.json)。

复现独立复核（先保留首轮结果用于核验选择来源）：

```bash
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.ig_sg_20260914.confirm
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.ig_sg_20260914.run controller --phase confirm_1k --models jit --gpus 0,3
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.ig_sg_20260914.report --phase confirm_1k --models jit
```

两阶段控制器均正常退出，已完成所有18组的输入身份、覆盖、样本汇合、计数、模型／源码／评价资产和FID重算核验。全部结果保留，不继续用本轮数据调参。
