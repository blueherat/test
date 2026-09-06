# RAEv2 完整轨迹分布目标下的共享 gate guidance

日期：2026-09-06

**状态：未通过当前研究目标的机制门槛，暂停。尚未训练，无本方法 FID 结果。**
终点 energy distance 的定义与可微性没有推出具体的 affine token gate，也没有解释
其受限方向能修复何种已识别误差。因此，本方案目前是通用终点损失下的受限控制器，
不能作为“机制导出设计”的候选进入训练或 FID 筛查。

已完成模块实现和 10 项 CPU 测试；8 样本完整轨迹 GPU 梯度审计仅有脚本，尚未运行，
现暂停启动。后文保留原方案和数值检查设计，均不代表执行许可或有效性证据。
即使数值检查通过，也不能补齐机制缺口或自动启动训练。

研究目标仍是：由明确机制导出轨迹内 guidance，在公平计算成本下使 RAEv2 FID
至少降低约 5%，通过配对 1K 筛查及独立 seed 确认。辅助模型和训练是允许的；
不能仅因某方案不免训练或训练较重而淘汰。全部训练、数值审计、验证和推理成本都需记录。

## 1. 这个 proposal 回应的两个断点

仓库已有实验表明，改善 paired latent/decoder 预测目标，不一定改善最终生成分布；
IG 的 raw latent 偏离也可能在解码后对应更好的图像统计。同时，冻结的官方 IG 场
不能直接视为它自身终点分布的 Gaussian bridge Bayes denoiser。

因此，本方案不从某个假设的 posterior mean 或 density ratio 推导新增场，
也不要求把实际状态拉回真实样本的人工加噪路径。它直接定义并优化：

\[
\text{初始 Gaussian noise}
\longrightarrow \text{实际官方 100 步采样器加 gate}
\longrightarrow \text{冻结 decoder}
\longrightarrow \text{最终图像特征分布}.
\]

这使优化对象包含递归状态变化、实际离散网格和 decoder 的有限幅度非线性。
它没有预先证明现有双头方向足以修正图像分布误差；这一点需要实验回答。

相关已有证据见：
[decoder pushforward 审计](RAEV2_IG_DECODER_PUSHFORWARD_MECHANISM_ZH.md)、
[decoder reversal 审计](RAEV2_IG_DECODER_REVERSAL_AUDIT_ZH.md)。
这些文档的观察结果不是本 proposal 有效的证明。

## 2. 唯一固定的控制结构

令当前状态上 full/base clean 输出为
\(F,B\in\mathbb R^{N\times1024\times16\times16}\)，并记 \(D=F-B\)。
先按既有官方实现和运算顺序计算 \(G_{\mathrm{official}}\)：保留其原始 IG
组合规则和原有启用区间，窗外使用原来的 full 输出。

新增 gate 使用 **F/B 原始输出值，不增加归一化、温度或 whitening**：

\[
a_\theta[n,h,w]
=b+\sum_{c=1}^{1024}u_c F[n,c,h,w]
  +\sum_{c=1}^{1024}v_c B[n,c,h,w],
\]

\[
G_\theta
=G_{\mathrm{official}}
+a_\theta[:,\mathrm{None},:,:]\odot(F-B).
\]

- 可训练参数只有 \(u,v\in\mathbb R^{1024}\) 和一个标量 \(b\)，共 **2049** 个。
- 全部 256 个空间 token 和全部 100 个时间步共享同一组参数。
- 不引入每步独立参数，不手工设计新增时间窗口或 guidance schedule。
- 新增项在全部 100 步使用同一公式；不会另行乘上手选的时间 mask。
- 初始化 \(u=v=b=0\)。此时每一步都复现 \(G_{\mathrm{official}}\)，包括最后原本关闭 IG 的一步。
- 不预设 gate 正负号，不通过 clipping 或事后放大系数修补结果。

这仍是一个 guidance 结构：backbone 和 decoder 权重冻结，新增动作在每个 token
上只能缩放当前的 \(F-B\) 方向，不能输出任意独立的 denoising field。
采用这个受限结构是可检验的建模选择，不是它必然包含最优控制的定理。

实现不能用 `if all(theta == 0): return G_official` 跳过 gate，因为这会切断
初始化时对 gate 参数的梯度。零初始化数值一致性由 runner 单独验证。

## 3. 完整 rollout 上的条件 energy distance

将真实有限步采样器的最终状态记为 \(Z^\theta_{\mathrm{end}}(\epsilon,y)\)，
其中 \(\epsilon\sim\mathcal N(0,I)\)，类别 \(y\) 按目标类别先验均匀采样。
定义

\[
U_\theta=\phi\bigl(\mathcal D(Z^\theta_{\mathrm{end}})\bigr),
\qquad Y=\phi(X),\quad X\sim p_{\mathrm{real}}(\cdot\mid y).
\]

\(\phi\) 是实验前冻结的图像特征提取器。数值 pilot 使用已有的 DINOv3-L
final CLS、unit-L2 特征链；真实和生成图像必须经过相同的连续前处理。
这一阶段不搜索不同 feature、kernel 或加权组合。

目标为每个类别的完整 energy distance，再对类别均匀平均：

\[
\mathcal L(\theta)
=\mathbb E_y\left[
2\mathbb E\|U_\theta-Y\|_2
-\mathbb E\|U_\theta-U'_\theta\|_2
-\mathbb E\|Y-Y'\|_2
\right].
\]

同一个类别内，\(U_\theta,U'_\theta\) 使用独立噪声；真实样本与生成噪声独立。
使用普通欧氏距离，指数固定为 1，无 kernel bandwidth。
吸引项与生成样本之间的排斥项的相对系数由 energy distance 定义给出，不能单独调权。
条件分布目标也避免了只匹配 pooled 图像分布时无法发现类别互换的问题。

### 小 batch 的准确计数

对于同一类别的 \(n\ge2\) 个生成特征和 \(m\ge1\) 个真实特征，训练项为

\[
\widehat{\mathcal L}_y
=\frac{2}{nm}\sum_{i,j}\|U_i-Y_j\|_2
-\frac{1}{n(n-1)}\sum_{i\ne i'}\|U_i-U_{i'}\|_2.
\]

生成—生成项排除对角线，分母为 \(n(n-1)\)。两个生成分支都保留梯度。
每类只有一个生成样本时必须报错，不能把排斥项设为零，退化为单样本 reward。

真实—真实项不依赖 gate 参数。\(m=1\) 时可以省略这一常量，获得定义目标的无偏
梯度估计；此时训练项的数值不能直接标记为完整 energy distance。
若要报告完整无偏 energy-distance 估计，必须有 \(m\ge2\)，并再减去
\(\sum_{j\ne j'}\|Y_j-Y_{j'}\|_2/[m(m-1)]\)。
完整 U-statistic 在有限样本下可以为负，不能据此认定实现错误。

正式训练对类别均匀采样，并平均各类损失。不能把每类一个样本的跨类别排斥项
误写成上述 conditional energy distance。

## 4. 可证明的对象与有限模型边界

有限一阶矩条件下，欧氏 energy distance 非负，为零当且仅当对应的 **特征分布**
相同；unit-L2 特征满足有限矩条件。这里比较的是实际 sampler、decoder 和固定
\(\phi\) 共同产生的条件分布，不是 teacher-forced 输入上的配对预测误差。
参见 [energy distance 与 MMD 的关系](https://arxiv.org/abs/1207.6076)。

在允许交换微分与期望、采样器及图像特征链可微或几乎处处可微的条件下，
完整递归反传计算的确实是上述实际终点目标的参数梯度。它不需要 F/B 对应某个
合法密度，也不需要官方场与其自身 endpoint bridge 一致。

但有限参数模型可能无法表达所需控制，优化也可能停在正的 energy distance。
普通梯度下降的目标下降性质不是本研究的新定理；它不提供普适的全局收敛或
FID 改善保证。尤其是：

- \(\phi\) 非单射，特征分布相同不意味着完整图像分布相同。
- Energy distance 下降不意味着 Inception FID 必然下降。
- 有限 batch 的非零梯度不证明 population 梯度有用，可能包含采样噪声。
- 没有 bandwidth 不意味着没有特征几何偏好，也不保证高维检验功效。
- 训练和数值检查通过后，仍需独立样本上的目标验证以及真实配对 FID。

本方案的研究问题是：受限双头控制在真实完整轨迹中的作用，是否足以修正可重复的
图像侧分布误差。不能仅以“优化器降低了训练损失”宣称机制成立或达到质量目标。

## 5. 第一阶段只做数值可行性审计

固定 **4 个预先指定的训练类别，每类 2 个独立初始噪声，共 8 个生成样本**。
这组固定类别只用于数值审计，不能当作 1000 类总体质量的代表。

只检查：

1. 零 gate 时不改变冻结基线。现有未运行脚本只比较同一 F/B 上的本地
   `full + .78*(full-base)` 算式；官方 wrapper 使用 `base + 1.78*(full-base)`，
   数学等价不保证 FP32 bitwise 相同。因此当前脚本不能证明与官方 wrapper
   或独立完整基线轨迹的 bitwise 一致。
2. 完整 rollout、decoder、连续图像前处理和 \(\phi\) 的递归梯度可计算。
3. gate 参数方向导数与完整 rollout 的中心有限差分一致。
4. gate 梯度是否非零、有限；这只验证实现与局部可达性，不作质量判断。
5. 记录峰值显存、耗时、前向次数、反向次数及检查点重算开销。

**冻结权重不等于停止状态梯度。** F/B、Full field 以及 gate 输入对当前状态的
导数都必须保留；不能沿用将 frozen backbone 包在 `no_grad` 中的 common-adapter
训练 wrapper。也不能 detach prefix/suffix 后仍称为完整轨迹目标的梯度。

decoder 和特征提取器参数冻结，但对输入保留梯度。训练链使用连续图像、clamp、
resize 和 normalization，不通过 uint8/PNG 量化链反传；实际图像评测按既有协议独立执行。

此阶段不搜索 gate 结构、guidance 强度或时间窗口，不启动大训练，不报告本方法 FID。
只有补齐机制依据并通过研究准入检查后，才重新考虑训练；届时优化器和预算应事前冻结。
这是研究证据要求，不是要求用户另行授权训练。常规训练设置也要公开记录，不能宣称完全无超参数。

## 6. 与主要已有工作的边界

- [MMD Guidance](https://arxiv.org/html/2601.08379v1) 已把分布吸引和排斥梯度
  加入逐步采样，并给出 latent-space 实现。本方案不把 MMD、吸引/排斥或分布 guidance
  当作新原理；区别在于训练受限 gate，目标通过完整实际 rollout 和 decoder 计算。
- [Diffusion Controller](https://arxiv.org/html/2603.06981v1) 已研究冻结 backbone
  加控制模块，并从随机控制、转移核重加权和正则化终端 reward 导出方法。
  本方案不声称首次提出冻结模型控制，也不将其随机核定理直接套到官方确定性 ODE。
- [Variational Control / DTM](https://arxiv.org/html/2502.03686v1) 已有终端成本和
  轨迹控制框架，其实际算法采用逐步贪心优化和当前 clean prediction 的终端近似。
  本方案明确支付完整 rollout 反传成本，并保留生成—生成项，而非单样本终端 reward。
- [Learn to Guide Your Diffusion Model](https://arxiv.org/html/2510.00815v1)
  已用 energy-kernel MMD 学习 guidance。本方案不能仅靠“用 energy distance 训练 gate”
  声称新颖性；需要实际证明终点图像空间、完整递归监督和受限 IG 控制的作用。

## 7. 公平成本与最终判断

推理仍查询原双头模型 100 次，并增加共享 affine gate 运算；模型 NFE 相同不代表
墙钟时间完全相同，需要实测。推理不需要成批样本相互筛选、拒绝或多候选择优。

训练需计算完整 100 步的前向和状态反向、decoder/feature 前反向，以及必要的
检查点重算。训练数据处理、数值审计、训练、验证和独立 FID 采样都计入总成本；
不能只报 2049 个可训练参数而隐去冻结大模型的反传费用。

当前连数值可行性 GPU 审计也暂不启动，因为它不能解决本方案缺失的机制依据。
是否值得完整训练、是否有独立生成收益、能否达到至少 5% FID 改善，以及相对已有
learned-guidance 方法是否有贡献，均未得到证明。
