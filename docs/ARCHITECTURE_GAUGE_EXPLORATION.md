# Architecture-aware Gauge Exploration

## 客观评价

这条方向最有价值的地方，不是再提出一个 latent smoothness 指标，而是构造一个
严格的因果对照族：固定 encoder、decoder 和逐样本重建，只改变正交 latent
坐标 `y = A z`，再观察有限生成架构与有限训练预算是否打破理论坐标不变性。

必须明确区分：

- `H1`：存在人为制造的坏坐标。这只能证明架构对坐标敏感。
- `H2`：存在稳定优于真实 codec identity 的坐标。这才构成质量方法的入口。

附件中最强的机制假设是 higher-order locality mismatch。标量 Fourier all-pass
可以保持每个 channel 的功率谱、二阶 cross-spectrum、欧氏距离和重建，却改变
相位与高阶空间组织；若局部 probe 对它敏感，而扩大感受野或使用 global probe
后差距缩小，才有理由继续做 LocalGauge。

需要收紧的部分：

1. local/global probe 的差值只是 locality tax 的 operational proxy，不是条件期望
   定义本身。必须加入同架构感受野 sweep、参数量和训练预算记录。
2. 搜索 `A` 必须使用 train 更新、validation 选择和 test 报告。联合训练同一批
   probe 与 `A` 容易得到 probe-specific overfitting。
3. decoder 分支必须使用真实 probe residual；各向同性随机噪声在正交变换下分布
   不变，不能证明 directional amplification。
4. 不同 `A` 的训练必须配对样本顺序、噪声、时间、初始化与训练步数。否则早期
   loss 差异可能只是随机性。
5. FID 改善 10% 和 2x compute 是项目门槛，不是理论结论。tiny probe 只能筛选，
   正式质量结论仍需完整生成评估。
6. 当前 novelty 结论未视为已验证。正式写论文前仍需系统检索正交卷积、latent
   preconditioning、decoder-aware loss 和 architecture-conditioned tokenizer。

现有文献只支持相邻边界：RAE 使用 wide shallow DDT head处理高维 latent；
[DiT locality](https://arxiv.org/abs/2410.21273) 说明 attention locality 与归纳偏置
有关；[LPL](https://arxiv.org/abs/2411.04873) 已覆盖 decoder feature loss；
[Diffusing in the Right Space](https://arxiv.org/abs/2606.03578) 做跨 tokenizer、
跨 backbone 的 diffusability 研究。这些都没有替代本文所需的固定 codec、严格
等价坐标因果实验，但也不能据此直接断言 novelty。

## 交付结构

- `notebooks/architecture_gauge_playground.ipynb`：只改一个参数单元即可切换 codec、
  数据、`A`、强度和半径；暴露 `A(z)`、`A_inv(y)` 与可选 tiny probe。
- `notebooks/gauge_mechanism_routing.ipynb`：实现 Step 2 机制分流，并直接画出
  locality、finite horizon、二阶统计、time-bin 和 decoder residual 证据。
- `experiments/architecture_gauge.py`：notebook 背后的严格正交算子、paired probe、
  图表和路由逻辑。

## 分阶段计划

### Step 0：exact gate

每个 `A` 必须同时通过：

- `A_inv(A(z)) ~= z`；
- 范数和样本间欧氏距离不变；
- `A(alpha z + sigma eps) ~= alpha A(z) + sigma A(eps)`；
- `D(A_inv(A(E(x)))) ~= D(E(x))`。

任一误差超过 fp32 合理范围，就停止解释机制。

### Step 1：identity headroom

先扫固定、低容量的 `A`，再做 identity 邻域的 `+delta/-delta` 配对试验。只有
held-out probe risk 在至少 3 个 paired seed 上稳定优于 identity，才进入 learned
gauge。人工 scramble 变差不算 H2。

### Step 2：机制分流

| 证据 | 分流结论 | 下一步 |
|---|---|---|
| all-pass 二阶统计不变；local penalty 大；RF/global 后缩小 | locality mismatch | static LocalGauge + cross-architecture crossover |
| 差异只存在于早期，长训趋同 | finite-horizon optimization | 只主张训练效率 |
| latent residual 接近，decoded error 分离 | decoder amplification | 独立 decoder-aware 项目 |
| time-bin 中最优方向反转 | time dependence | 才考虑 moving gauge |
| 正交族无收益，whitening 有收益 | covariance/SNR | 非正交 preconditioning，不归入 AGF 主线 |
| identity 在真实 codec 上稳定最优 | no headroom | 停止质量方法 |

### Step 3：正式质量验证

冻结 `A*`、丢弃 probe，在 identity 与 `A*` 上从头训练 matched 模型。至少 3 个
seed，报告 FID/sFID/precision/recall、GPU-hours 曲线、time-binned loss，并交换
`A_U*` 与 `A_T*`。只有正式生成指标支持 H2，才能形成方法论文结论。

## 当前成功标准

当前两个 notebook 的成功标准是验证实验逻辑和机制路由可运行，不是证明论文
假设。研究级 go/no-go 仍要求：

- exact gate 全通过；
- 真实 codec 上非 identity 优于 identity；
- 至少 3 个 paired seed；
- probe 排序能迁移到正式模型；
- locality 结论同时通过 RF sweep 和 cross-architecture crossover；
- 最终生成指标在 matched compute 下显著改善。

## 真实 smoke 验证

两个 notebook 已使用本机 ImageNet-1k 与 `RAE-DINOv2` 从头到尾执行。执行副本
位于 `~/data/eqvae/artifacts/gauge_notebook_smoke/`；仓库中的 notebook 已清空输出。

- 可调 `A` notebook：train 16 / validation 8，all-pass 的 inverse、norm、distance、
  paired-noise 与 PSD 相对误差均为 `1e-7` 量级；`D(A_inv(A(z)))` 相对 `D(z)`
  的平均绝对像素误差约 `2.6e-7`。
- 路由 notebook：train 24 / validation 12、12 steps、seed 0。exact gate 最大误差
  `2.42e-7`；最佳非 identity 仅为 `channel_0.5`，held-out loss ratio 约 `0.998`。
  all-pass 与 identity 基本持平，当前没有观察到稳健 H2 headroom、locality 或
  time-bin 机制；decoder amplification 的变异系数约 `0.027`。

这些数值只证明实现、配对和路由逻辑可运行。样本小、训练短且只有一个 seed，
因此不能用于支持或否定论文假设。
