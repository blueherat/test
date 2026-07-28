# Imagenette-64 噪声阶段责任曲线：正式结果

## 1. 最终判断

本轮严格回答了两个不同问题，结论不能混在一起：

1. **噪声阶段责任曲线是真实、稳定、可区分的现象。** 这一点得到五个 seed、
   三档 bottleneck 的强支持。小 latent 的相对责任更集中在高噪声端；latent
   容量增大后，decoder 在中低噪声继续使用更多样本级信息。
2. **责任曲线没有显示出超越简单重构/验证误差的独立生成质量预测力。** 预注册
   的 leave-one-seed-out 门槛失败，而且独立复算和多种敏感性分析都没有改变
   这个结论。

因此本轮不是“完全失败”，也不是“可以开始端到端”：

> 我们确认了 decoder 会自然形成容量相关的信息使用时间表，但没有证据表明
> 这个时间表比普通拟合质量更能决定生成质量。

按照预注册硬门槛，**不训练 latent prior，不进入端到端联合训练**。这次停止的
是“把责任曲线当作下一阶段选择标准”的链条，不是否定两阶段生成本身。

## 2. 实验规模

- 数据：Imagenette 官方 train `9,469` / val `3,925`，确定性 val 子集 `1,024`。
- 图像：`64x64` RGB，训练随机裁剪/翻转，验证中心裁剪。
- latent：`16/64/256` 维，从头训练的小卷积 encoder。
- decoder：所有时刻都能读取 latent 的条件 velocity U-Net，没有时间门控。
- null 支持：每个训练样本独立 `10%` 零条件 dropout。
- 正式 seed：`0/1/2`；因 seed 2 的 FID 容量排序反向，按预注册补 `3/4`。
- 每个配置：`20,000` step、batch `64`、fp32、EMA、50-step Euler。
- 总计：15 个模型、300,000 optimizer steps、19.2M 次训练样本抽取。
- 每个 run：1,024 张图的 total/frequency paired 责任曲线和条件 rollout。

生成模型训练完全不使用类别标签。标签只用于同类别 shuffle 和最终评估；官方
train/val split 严格分离。预训练 ResNet18 在固定 val 子集上的十类受限准确率为
`92.77%`，说明评估特征对 Imagenette 有足够辨别力。

## 3. 实现审计

所有 15 个 run 同时通过：

- identity 重复 forward 最大绝对 RMS：`0.0`；
- 实测 condition dropout：`9.9584%` 到 `10.0472%`；
- random shuffle 和 within-class shuffle 固定点均为 `0`；
- within-class shuffle 标签不匹配数为 `0`；
- condition embedding RMS 在所有容量和 seed 上均为 `1.0`；
- 同一 seed 的三容量数据流 SHA256 完全相同；
- 同一 seed 的 encoder 共享主干初始化 SHA256 完全相同；
- 同一 seed 的 decoder 共享主干初始化 SHA256 完全相同；
- 所有 loss、latent、condition embedding、FID 均有限。

开发 smoke 曾发现 null condition 的 `sqrt(0)` 反向奇异问题；正式训练前已改为
带 epsilon 的 `rsqrt` 并增加 null backward test。第一批正式 run 又因容量改变
随机数消耗、导致 decoder 初值不完全共享而主动中止；修复并加入共享初始化哈希
后从头重跑。两个废弃批次均未进入结果。

对出现 FID 反序的 `64d seed2` 做独立 checkpoint 重载复算：

- FID、像素 MSE、类别匹配与保存结果逐位一致；
- total profile 最大绝对差 `9.76e-17`；
- frequency profile 最大绝对差 `8.88e-16`；
- identity control 差为 `0`；
- 本项目 Fréchet 实现与 SciPy 标准矩阵平方根公式差 `7.11e-15`。

所以后面的负结论不是缓存、随机采样、汇总或 FID 公式错误。

## 4. 主结果

下表为五个 seed 的均值，括号内为 seed 间样本标准差：

| latent | low fraction | mid fraction | high fraction | feature FID | source pixel MSE | val velocity MSE | class match | effective rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | .0111 (.0008) | .2348 (.0079) | .7541 (.0083) | 120.97 (1.04) | .1913 (.0020) | .13468 (.00023) | 28.40% | 9.51 |
| 64 | .0150 (.0008) | .3383 (.0068) | .6467 (.0065) | 118.51 (1.54) | .1237 (.0022) | .12391 (.00056) | 36.11% | 15.26 |
| 256 | .0171 (.0010) | .3563 (.0045) | .6265 (.0052) | 117.63 (.63) | .1116 (.0014) | .12149 (.00035) | 37.71% | 18.78 |

这里的 fraction 是每个 run 内将正 `Delta_shuffle` 按低/中/高噪声区域归一化
后的曲线形状，不受条件投影绝对尺度影响。

### 4.1 P1：样本责任存在，通过

三个容量在所有 seed 的中高噪声区域都满足：

- `Delta_shuffle` 95% CI 下界大于 0；
- paired positive rate 大于 `0.6`，实测主要区域接近 `1.0`；
- null、random shuffle、within-class shuffle 的方向一致。

这说明 decoder 使用的是与当前图像配对的信息，不只是 latent 的边缘统计。

### 4.2 P2：容量改变曲线形状，通过

五个 seed 的 `16d - 256d` 高噪声责任占比差分别为：

```text
0.1256, 0.1273, 0.1374, 0.1122, 0.1353
```

全部超过预注册的 `0.10`。容量 slope 为 `-0.0850`，分层容量置换检验单侧
`p=0.000257`。

同时，绝对 `Delta_shuffle` 不是随容量下降，而是上升：

| latent | low-noise delta | mid-noise delta | high-noise delta |
|---:|---:|---:|---:|
| 16 | .00577 | .12188 | .39107 |
| 64 | .01154 | .26065 | .49813 |
| 256 | .01404 | .29168 | .51272 |

所以正确解释是：

> 小容量 latent 提供的信息较少，而且相对集中在生成早期；大容量 latent 提供
> 更多信息，并在中低噪声阶段继续帮助 decoder，而不是“高容量不需要早期信息”。

### 4.3 P3：不只是类别信息，通过

同类别 shuffle 的平均 `Delta_within_class` 在所有容量、seed 和三个噪声区域
均为正。高噪声区域的五 seed 均值为：

```text
16d: 0.3463
64d: 0.4519
256d: 0.4668
```

因此曲线不只是十类标签的替代测量，还包含物体实例、布局、颜色和局部外观。

### 4.4 P4：尺度混杂被排除，通过

encoder 输出 per-dimension RMS 约为 1，condition adapter 输出 RMS 固定为 1；
三个容量的 condition RMS 相对 spread 为 `0.0`。共享 decoder 主干和数据流也
逐哈希相同。容量曲线差不能用“256d 注入向量更大”解释。

### 4.5 频带结果

低频在 velocity error 中占绝对主导，但容量效应在三个频带都存在。例如高频
`Delta_shuffle`：

| latent | low-noise | mid-noise | high-noise |
|---:|---:|---:|---:|
| 16 | .000605 | .001189 | .001806 |
| 64 | .000834 | .002305 | .002454 |
| 256 | .000942 | .002492 | .002561 |

这支持“更丰富的 latent 在后期仍提供细节”，但高频绝对贡献很小，不能把该图
解读为 decoder 的大部分质量都由高频 latent 责任决定。

## 5. P5：质量预测门槛失败

预注册 leave-one-seed-out FID RMSE：

| predictor | RMSE |
|---|---:|
| responsibility curve | 1.23399 |
| source pixel MSE | 1.23347 |
| validation velocity MSE | **1.20551** |

责任曲线与 FID 确实相关：整体 Spearman 中，高噪声占比与 FID 为
`rho=0.693, p=0.0042`，中噪声占比为 `rho=-0.707, p=0.0032`。但相关不等于
增量预测价值；它没有超过更简单的 validation MSE 基线。

预注册之外的敏感性分析也没有救回结论：

| post-hoc predictor | LOSO RMSE |
|---|---:|
| only high-noise fraction | 1.25470 |
| only mid-noise fraction | 1.24968 |
| only low-noise fraction | 1.38740 |
| only frequency fraction | 1.27801 |
| latent dimension only | 1.37320 |
| effective rank | **1.15039** |

其中 effective rank 是探索性结果，不能当作预注册发现；但它提示一个更合理的
后续假设：生成质量可能更依赖 latent 实际占用多少稳定方向，而不是 decoder 在
哪个时刻读取它。

## 6. 机制解释

责任曲线反映的是当前像素状态和 latent 之间的信息冗余：

- 高噪声时，`x_t` 几乎没有图像信息，配对 latent 的边际价值最大；
- 噪声下降后，像素状态自己携带越来越多内容，latent 的边际价值自然下降；
- 低容量 latent 主要保存早期最值得传递的粗信息；
- 高容量 latent 同时保存更多实例和细节，因此 decoder 在中低噪声仍会读取它。

这解释了自然责任曲线，却不能保证生成质量单调改善。FID 同时受 encoder 表征、
decoder 容量、latent 有效维度、采样误差和输出分布覆盖影响。一个 latent 被使用
得更久，只说明它含有可利用信息，不说明这些信息更容易由未来 prior 生成，也不
说明它对分布质量最有价值。

## 7. 停止决定

预注册要求 P1--P5 全部通过才训练 latent prior。最终：

```text
P1 sample responsibility: PASS
P2 capacity-dependent curve: PASS
P3 within-class information: PASS
P4 scale/control audit: PASS
P5 quality prediction: FAIL
```

所以本轮在 decoder 责任曲线阶段停止：

- 不训练 latent prior；
- 不训练联合 encoder/prior/decoder；
- 不改 FID、容量、频带、时间点或 predictor 来追求通过；
- 保留“自然责任曲线真实，但不是独立质量指标”的正负混合结论。

若未来重新立项，必须把新假设明确改为 effective rank、显式 rate constraint 或
prior 难度，而不能继续把“主要在早期使用”当成“更 generative-friendly”的
同义词。

## 8. 可复核材料

- 预注册：`docs/IMAGENETTE_NOISE_RESPONSIBILITY_PREREG_ZH.md`
- 主实验：`experiments/imagenette_noise_responsibility.py`
- 四卡启动器：`experiments/run_imagenette_responsibility_sweep.py`
- 汇总与硬门槛：`experiments/summarize_imagenette_noise_responsibility.py`
- 单元测试：`tests/test_imagenette_noise_responsibility.py`、
  `tests/test_summarize_imagenette_noise_responsibility.py`
- 外部结果：`~/data/eqvae/imagenette_noise_responsibility_formal/`
- 最终表图：`~/data/eqvae/imagenette_noise_responsibility_formal/comparison/`

仓库内没有写入 `outputs/` 或 `artifacts/`。
