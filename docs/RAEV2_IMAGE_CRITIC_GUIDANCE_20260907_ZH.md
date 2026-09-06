# 从图像密度信号到轨迹内的 Gaussian posterior tilt

2026-09-07，固定设计，两个正式1K已经完成，结果见文末。本方向不再只从 Full/Base 的共享错误猜测质量方向，而使用已有独立验证的图像判别器。旧判别器是最终 DINOv3 CLS 的单位 L2 特征上的线性 logistic 回归：1000 真图、1000 官方生成图训练；独立真图及独立生成 seed 上 AUC≈.827。它没有用 Inception/FID 训练。冻结权重 SHA `d0c1e7d9968f4ea4186ee8edd42c08e7407afb352450981fefa5ae781a11967b`。旧 acceptance 实验不恢复，所有生成图均计入新评估。

相关一手工作：[Universal Guidance](https://arxiv.org/abs/2302.07121) 使用 clean 预测承载任意可微约束；[Discriminator Guidance](https://proceedings.mlr.press/v202/kim23i.html) 利用真实/生成判别器。下述 Gaussian 近似与固定协方差设计是本次推导，不宣称复现这些论文的完整算法，也不使用旧 acceptance certificate 证明轨迹改动改善 FID。

记 r(x)=wᵀ normalize(CLS(DINO(decode(x))))+b。若判别器在特征空间是 Bayes 最优、训练先验均衡，它等于该特征的 log density ratio；实际冻结 r 是一个有限的质量代理。其在 raw latent 或任意噪声时刻都不是已证明的真实密度比。

给定当前原生 clean 预测 G，设一个 Gaussian clean surrogate Q=N(G,C_t)。将 r 在 G 处一阶展开，用 exp(r) 对 Q 倾斜，完成平方可得：

    Q_tilt ≈ N(G + C_t ∇r(G), C_t).

因此 clean 更新固定为 G_new=G+C_t∇r(G)，随后走原 Euler。没有额外 strength。该结果对线性 r 和 Gaussian Q 精确，对实际非线性 decoder/DINO 及有限 IG 仅为局部近似。它解释方向、参数化与幅值来源，不证明 native G 为真实 Bayes denoiser，也不承诺终点精确为 q exp(r)。

预定两个协方差模型，仅检验统计结构差异，不扫描幅度：

1. isotropic：C_t=MSE_t I。MSE_t 已由本轮之前保存的 1000 真实前向配对、全部 100 时刻直接估计，与固定均值 Gaussian 负对数似然的最优球形尺度一致。MSE 含 G 偏差，不冒充真实 posterior variance。
2. exchangeable：256 空间 token 的常量模态采用 Σ₀=256 Cov(image mean)，其余正交模态采用 Σᵣ=256/255[Cov(all tokens)−Cov(image mean)]。在空间交换性 Gaussian 家族中，这是训练样本矩决定的协方差，保留全通道相关性。原始 5000 ImageNet TRAIN latent 估计，无 FID 选择。在线性 Gaussian channel 下每个特征值 λ 的 posterior 值为 λ t²/[t²+(1−t)²λ]。最后统一乘 MSE_t/(trace(C_t)/D)，与第一臂保持相同平均尺度，只改变相关结构。未条件化的全局协方差、空间交换性和真实 posterior 的偏差明确保留。

结构的动机有数据依据：constant-mode eigenvalue 最大约 3709，而 residual-mode 最大约 11.77；空间平均方向承载强相关。把所有 latent 元素当独立、只看平均 MSE，可能严重错估图像级语义梯度的自然步长。此观察不等价于证明质量受益。

先在旧 8 图/10 时刻状态缓存上检查梯度有限、中心差分和实际代理函数变化；只作数值诊断，不按时刻选择窗口。然后新 8 图全轨迹、固定 paired 1K，所有 100 时刻使用同一公式。判别器使用可微 FP32 decoder/DINO、FP64 CLS 归一化，量化输出仍由原生 BF16 decoder 生成；这是明确的连续像素 surrogate 差异。完整 encoder/decoder 前后向成本计入，成功候选补充独立 seed 与成本匹配官方采样。验证失败不通过修改几十个时刻系数修补。

## 固定 1K 结果

原生official38.486774，historical interval38.335024；critic_isotropic38.535461（−0.1265%），critic_exchangeable38.434218（+0.1366%）。后者仍弱于interval，实测推理与解码总GPU秒3516.9/3521.1，对照632.3秒。全部1000图计入，目标未达成，不调强度/窗口或宣称判别器AUC已保证FID。完整身份和成本在 [本轮ledger](../experiments/results/raev2_guidance_20260907/screen_ledger.json)。

另有 [固定CPU Jacobian检查](RAEV2_CRITIC_JACOBIAN_AUDIT_20260907_ZH.md)，明确保留球形近似的方向误差与纯噪声端点失配，没有在读取本次FID后偷偷修改这两个候选。
