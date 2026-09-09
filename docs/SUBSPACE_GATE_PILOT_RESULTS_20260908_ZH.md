# 子空间分别控制的首轮结果

日期：2026-09-08。状态：三 seed toy 正结果，尚非 ICLR 级贡献或真实图像突破。
协议：[研究起点](SUBSPACE_GATE_RESEARCH_PROTOCOL_20260908_ZH.md)。

## 核心发现

同一组冻结 x/epsilon 预测头，单个混合系数要同时处理数据子空间的形状误差和
法向噪声。分别控制两者，在本次 D512/H128 连续螺旋实验中改善了完整分布指标。
关键对照是：随机子空间没有同等收益，直接投影清除法向噪声也不能替代分别控制。

此前关于“跨时间抵消”的候选解释未得到九个条件的无偏交叉项支持，已记录在
[原协议](TERMINAL_DEFECT_RESEARCH_PROTOCOL_ZH.md)。本轮不把该负结果改写成成功。

## Oracle 因果对照

三个已有训练 seed 为 20260831/20260901/20260902。每条件 4096 samples，
200-step Heun，固定 256 SWD 方向；同 seed 内共享初始噪声和独立真实 reference。
下表为三个 seed 均值，越低越好。

| 控制 | ambient SWD | intrinsic SWD | off-subspace RMS |
|---|---:|---:|---:|
| 原全维标量 Bayes oracle | 0.091829 | 0.075808 | 0.015406 |
| 原 safe gate | 0.080250 | 0.066297 | 0.067572 |
| 只优化 intrinsic 的标量 | 2.821312 | 0.039733 | 4.111801 |
| 只优化 normal 的标量 | 0.115135 | 0.095150 | 0.003261 |
| intrinsic/normal 分别 oracle | **0.048930** | 0.040173 | 0.003204 |
| 随机二维子空间分别 oracle | 0.091768 | 0.075757 | 0.015401 |

子空间分别 oracle 相对原标量 oracle 的全维 SWD 降幅为 53.0%、46.5%、39.9%。
这扩大了控制函数类，因此点态最小 MSE 下降本来就是代数上可预期的；不能说它
“击败了真实 Bayes 速度”。它改善的是受限双头组合。真实子空间也不等同于
螺旋曲线每一点的一维切空间，不使用“已识别非线性流形切空间”的说法。

## 去掉 oracle 的可学习 pilot

只用 1024 个独立训练样本估计 rank-2 PCA。用 clean/noise 配对 velocity target
训练，不调用 Bayes 速度，不读取真实 embedding basis。三个控制均为 86,530 参数，
相同初始化、minibatches、Adam 2e-4，固定 5000 updates、batch1024；不按评估
选择 checkpoint。模型和 rank 的设置预先写入协议。每 seed 三模型训练约 82–83 秒，
不包含最终评估；这不是严格推理成本比较。

每个 seed 使用与 oracle 实验不同的新 bank：202609191/192/193。

| 控制 | ambient SWD | intrinsic SWD | off-subspace RMS |
|---|---:|---:|---:|
| 原全维标量 Bayes oracle | 0.103098 | 0.085351 | 0.015406 |
| 原 safe gate | 0.091183 | 0.075685 | 0.067546 |
| 同规模新标量 gate | 0.098277 | 0.081425 | 0.028298 |
| 新随机子空间双 gate | 0.098045 | 0.081204 | 0.028650 |
| 新 PCA 子空间双 gate | **0.074337** | **0.060892** | **0.004372** |

PCA 相对新标量的三 seed 全维 SWD 分别降低 33.8%、25.7%、9.8%，三者均胜随机
控制。均值之比改善 24.36%。对原 safe gate 的均值改善约 18.48%，但第三 seed
仅从 0.069884 降至 0.069826，实质接近持平，不能宣称三 seed 均有显著优势。
不同 bank 的绝对 SWD 明显波动，禁止跨上、下表直接比较方法。

## 投影是否已经足够

在可学习 pilot 的同一 bank 和冻结 checkpoint 上补做如下对照，未再训练：

| 控制 | ambient SWD | intrinsic SWD | off-subspace RMS |
|---|---:|---:|---:|
| 原 D0 纯 x 头 | 0.154376 | 0.127952 | 0.003301 |
| 每步将 x 预测投影到 PCA 平面 | 0.125281 | 0.103797 | 0.002499 |
| 将新标量 gate 的 implied clean 预测投影 | 0.098619 | 0.081553 | 0.002498 |
| PCA 子空间双 gate | **0.074337** | **0.060892** | 0.004372 |

所以当前收益不能仅由“更贴近数据平面”解释。两个投影基线更贴近平面，却没有
达到双 gate 的分布结果。所有方法沿用原端点 override；投影不是在最后一步强制
零法向值，必须避免把它描述为精确终点流形投影。
重复评估的 D4/scalar/PCA 指标与原 pilot 完全一致。

## 证据边界与下一关

### 追加：独立 8K bank 复核

冻结原 gate checkpoint，用全新 bank 202609211/212/213，每 seed 8192 samples。
没有再次训练或调参。全维 SWD 均值：safe 0.087898、scalar 0.095108、
PCA 双 gate 0.068413、projected x 0.124346、projected scalar 0.095657。
PCA 对这四项比较均为三 seed 同向改善。这是独立 sample bank 复核，不是
新增训练 seed。数据见 `narrow_confirmation.csv`。

### 追加：真实 latent 几何的初查

ImageNet-100 SD-VAE 缓存，仅用训练集、互不重叠的 4096 fit / 4096 holdout。
对 posterior mean 展平后的 4096 维向量做 centered randomized PCA，q256、
5 次迭代。前 8/32/128/256 方向在 holdout 中解释约 29.9/40.4/48.7/52.2%
能量；随机方向约 .22/.82/3.19/6.37%。因此真实数据存在结构性各向异性，
但绝非 toy 的精确低秩支撑；补空间仍有大量信号，不能把它叫作纯法向噪声。
该结果只检查可迁移假设，未证明分区控制会改善 FID。
数据见 `sdvae_energy.csv`；估计的 basis 存在外部数据目录，未接触 validation。

### 追加：解除硬秩瓶颈后的宽模型

原三个 seed，D512/H512、depth4，重新训练 D0/D2/D4 各 15000 updates，
沿用原数据、优化器、batch2048。每 GPU 约 356–357 秒。随后原样训练三种
gate 5000 updates。首次宽模型 bank 的 PCA 相对 scalar 降幅为
2.6%、35.2%、56.8%；这是匹配训练步数的宽度对照，不是匹配 FLOPs。

再使用独立 8192-sample bank 202609251/252/253 完成纯 x / 投影基线复核：

| 控制 | seed20260831 | seed20260901 | seed20260902 |
|---|---:|---:|---:|
| 纯 x | .079060 | .148589 | .303324 |
| 原 safe gate | .056420 | .098409 | .151923 |
| 同规模新 scalar | .056741 | .097891 | .146537 |
| PCA 双 gate | **.053450** | **.066599** | **.064395** |
| projected x | .063322 | .125335 | .238895 |
| projected scalar | .056834 | .097909 | .146234 |

此处均为 ambient SWD。PCA 相对 scalar 改善约 5.8%、32.0%、56.1%，
同时法向残留下降。故收益不只依赖最后线性层的硬 rank<512 限制。
但不同 seed 的纯 x 质量差异极大，训练结束时 loss 仍在下降；不能把 hidden512
当作“已充分优化的强模型”，也不能据此声称优化瓶颈已全部消除。
下一步仍需训练成熟度对照和非线性支撑检验。

数据：`wide_pilot.csv`、`wide_confirmation.csv`；外部数据路径
`~/data/eqvae/experiments/subspace_gate_wide_20260908/seed*/`，含模型/优化器
checkpoint、源码、配置身份和完整历史。确认实验的 base 与 gate 均冻结。

### 追加：真实 SiT 的局部机制检查未显示额外收益

使用真实 450K EMA dual-output SiT、固定 rank32 PCA，另取未用于 PCA 的
2048 个训练样本，fit1024 / hold1024；七个固定时间点分别拟合常数 scalar
以及 top32 / complement 两个系数。普通 SD-VAE posterior 采样，原 BF16
网络、FP32 state，官方源码身份与 checkpoint metadata 一致性检查通过。

split 相对 scalar 的 holdout velocity MSE 绝对变化不超过 6.2e-6，正负混合，
也未稳定优于原空间 gate。系数本身有差异，但头差值的能量及误差关联使得
这版分区提供的额外局部收益很小。该检查不是 FID 实验，不证明质量改善或
所有分区控制无效；它不支持直接宣称 toy 方法已迁移到真实图像模型。
数据：`sit_local_risk.csv`；全部统计量及 fit/hold 索引保存在
`~/data/eqvae/experiments/subspace_gate_sdvae_20260908/local_risk/`。

### 当前仍缺失的证据

1. 目前仍仅三个 seed 的固定线性低秩 toy。硬输出秩瓶颈已用 H512 对照，
   但成熟度、弯曲支撑及不同有效秩尚未验证。
2. PCA rank=2 来自已知 toy 设置。真实 RAE latent 没有精确二维平面，法向低方差
   方向可能含重要细节；旧静态谱方法的失败仍然成立。
3. 已有独立 sample bank 复核和同头纯 x / 投影控制，尚缺 bootstrap 区间、
   严格推理开销和充分训练的强预测基线；“三个 seed 都更好”不等于普遍结论。
4. 尚未形成新颖性结论。高维法向噪声与 x-prediction 的联系已有
   [On the Diffusibility of High-Dimensional Latents](https://cfeng16.github.io/on_the_diffusibility/)
   讨论；预测目标与 guidance 的流形保持已有
   [Not All Prediction Targets Keep Training-Free Diffusion Guidance on the Manifold](https://arxiv.org/html/2607.00647v1)
   讨论。子空间分解本身不认领原创。还需完整比对 TCFG、预测目标混合与几何 guidance。
5. 当前候选问题应表述为“共享标量控制何时产生可解除的子空间竞争，以及如何无需
   oracle 地解除”，而不是泛泛重述“局部误差不代表质量”。
6. 真实图像生成、跨模型迁移、论文主张与完整稿尚未完成。研究目标保持未完成。

## 复现资产

- 代码：`experiments/terminal_defect_spiral.py`、`oracle_subspace_audit.py`、
  `train_subspace_gate_pilot.py`、`evaluate_subspace_gate_controls.py`。
- 小结果：`experiments/results/terminal_defect_20260908/` 下四张 CSV。
- 原始数据、源码快照、输入或随机种子、checkpoint 身份：
  `~/data/eqvae/experiments/terminal_defect_20260907/`；不覆盖既有研究资产。
- 六项解析/实现测试通过，覆盖有限更新 telescoping、U-stat 枚举、投影初始化
  一致性和正交双 oracle 的点态风险性质；这些测试不代替生成实验证据。
