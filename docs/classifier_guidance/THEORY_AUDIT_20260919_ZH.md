# 分类器弱头训练：理论兼容性与进一步优化审计

2026-09-19；承接 [`b9ce6ba` 的首轮优化](PERFORMANCE_20260919_ZH.md)。

**结论：推荐路径保留原来的端点对抗目标，以及实际离散采样器的一阶完整梯度。**
它没有把完整反传改成截断梯度、连续伴随近似、少步采样或固定判别器。
这个结论不等于浮点结果逐位相同，也不等于已经证明长期训练或 FID 不变。
本轮没有启动新一轮长训、系数扫描或 5K 采样。

## 实际优化的目标

冻结强模型，弱头参数为 θ。令有效额外系数为 aᵢ，实际速度场为

\[
 f_\theta(x,t,y)=v_S(x,t,y)+a_i\{v_S(x,t,y)-v_{W,\theta}(x,t,y)\}.
\]

`a` 是额外系数；RAEv2 的总系数 `w=1.35` 对应 `a=0.35`。
SiT 保留原来的前半程窗口及前四分之一的 6/7 因子。
JiT 保留原 clean-to-velocity 转换与分母下限；RAEv2 仍先分别转换两个 clean prediction，再外推。
SiT 为 64 步 Heun，JiT 为 100 步 Euler，RAEv2 为原 shifted-grid 100 步 Euler。

最终图像为 `Iθ = decode(Solver(fθ, z, y))`，固定特征为 `hθ = Φ(Iθ)`。
真实特征来自训练集 RGB：`h_real = Φ(I_real)`。判别器是条件二分类 logit `Dψ(h,y)`。

\[
 L_D=\mathbb E\,\operatorname{softplus}(-D_\psi(h_{real},y))
     +\mathbb E\,\operatorname{softplus}(D_\psi(\operatorname{stopgrad}(h_\theta),y))
     +\frac\gamma2\mathbb E\|\nabla_{h_{real}}D_\psi(h_{real},y)\|^2.
\]

先用此损失更新 ψ，再对**更新后的判别器**最小化

\[
 L_W=\mathbb E\,\operatorname{softplus}(-D_{\psi^{new}}(h_\theta,y)).
\]

生成器梯度经过判别器的输入、Inception、解码器和全部采样步，回到弱头。
不对 D 的优化器更新再求导；这是原有的一阶交替 GAN 更新，未变成展开优化器的双层目标。
谱归一化在 D 步的 train 模式更新缓冲，在 W 步的 eval 模式冻结缓冲，沿用旧实现。
全局梯度仍先平均，再裁剪，再执行 Adam，最后更新 EMA。

这里要区分几件事：

- 判别器在固定 Inception 特征空间判别真假；R1 也对特征求导，不是像素 R1。
- 训练用连续解码并裁剪的 RGB，评测导出还会量化为 uint8。这是原有可微替代，提速没有引入这个差别。
- 弱头是学习到的向量场控制量；当前损失没有强迫它成为某个显式弱密度的 score，也没有强迫它比强模型更差。因此支持的是“学习端点生成分布”的对抗训练解释，不足以证明原先“强密度与弱密度对数外推”的所有理论前提。
- GAN 损失下降不保证 FID 单调下降，也不自动证明收敛。训练曲线与生成质量仍需独立实验。

## 为什么完整梯度没有改变

冻结 S 的参数不代表可以对 S 的输出 `detach`。优化版保留

\[
 J_x f=(1+a_i)J_xv_S-a_iJ_xv_W,
 \qquad J_\theta f=-a_iJ_\theta v_W.
\]

弱头使用强模型中间特征，`J_x v_W` 也包含这些特征对 x 的导数。
引导关闭后的强模型后缀虽然 `Jθf=0`，仍通过 `Jxf` 把端点梯度传回来。

Euler 步 `x' = x + h f(x,t)` 的反传为

\[
 \lambda_x=\lambda+hJ_x f^T\lambda,
 \quad \Delta g_\theta=hJ_\theta f^T\lambda.
\]

Heun 步为 `p=x+h f₁(x)`、`x'=x+h/2(f₁(x)+f₂(p))`。令

\[
 q=J_xf_2(p)^T(h\lambda/2),\qquad b=h\lambda/2+hq.
\]

则

\[
 \lambda_x=\lambda+q+J_xf_1(x)^Tb,\qquad
 \Delta g_\theta=J_\theta f_2(p)^T(h\lambda/2)+J_\theta f_1(x)^Tb.
\]

[`sampler.py`](../../classifier_guidance/sampler.py) 正是这个递推，只把两个向量场的反传先后执行，避免同时保存两份大模型激活。
保存原前向的状态和预测点，反向重新计算局部向量场；没有反向积分重建轨迹的近似误差。
这是**离散求解器**的完整梯度，不是声称有限步结果等于精确连续 ODE。

## 每项提速的适用条件

| 优化 | 保留的操作与必要条件 |
|---|---|
| CUDA 图重放 | 每次更新状态、时间、标签、系数；head 参数原地更新。复制输出后再复用工作区，禁止并发重放同一实例 |
| 共享图内存池 | 图之间没有需要保留的内部激活依赖，输出已复制；符合 PyTorch 的独立图共享池条件 |
| 激活 checkpoint | 冻结、eval、确定性模块；重算相同输入和整批形状。没有用训练态 BatchNorm 或随机 dropout |
| 冻结权重预存 BF16 | 仅 JiT/RAEv2 已在 BF16 autocast 下计算的 Linear/Conv 权重；不转换 SiT FP32 算子、归一化、embedding 或可训练头 |
| 合并梯度通信 | 相同全局 batch、loss 缩放、平均后裁剪顺序；浮点归约顺序可能不同 |
| 更大的单卡 batch | 需同步减少设备数或合理分配累积，使全局 batch 和 D/W 更新频率不变；不能直接改变全局 batch 后宣称同一训练设置 |

共享内存池规则已核对 [PyTorch 2.11 CUDA 语义](https://docs.pytorch.org/docs/2.11/notes/cuda.html#sharing-memory-across-captures)；
重算机制见 [非重入 checkpoint](https://docs.pytorch.org/docs/2.11/checkpoint.html)。
CPU hook 的计数副作用不会随 CUDA 图重放；这些计数不参与本训练的损失、调度或优化器。

本轮补上的保护包括：检查输入 shape/dtype/device、限制实例使用同一 CUDA stream、检查 head 参数对象及存储地址，
前向后替换 `.data` 存储时也拒绝反传；检查冻结网络和头的 eval 模式；显式拒绝对时间网格或外推系数求导的误用。
训练 helper 的默认反馈也改成整批路径，避免省略参数时意外走逐张微批次。
图的时间/系数静态缓冲保留输入 dtype，额外用非均匀 float64 时间网格检验，避免通用接口静默降成 float32。

这套自定义反传只支持当前目标需要的一阶梯度。未来若训练系数、时间网格，或对 θ 做高阶元学习，须另行实现和验证，不能直接套用。

## 验证和数值边界

- float64 下 Euler/Heun 与完整展开 autograd 对照，覆盖非均匀时间步、冻结强模型后缀和共享输入特征。
- 对噪声与全部头参数做有限差分 `gradcheck`，独立核验解析反传。
- 完整 GAN 更新与直接展开版本核对 loss、feature R1、D/W 参数及缓冲，容差 `1e-12`；使用更新后的 D 计算弱头损失。
- GPU 测试覆盖新标签、头更新、多条未完成反传的轨迹交错、参数版本和存储变化。
- 实际模型另外做更换噪声、标签、原地更新弱头后的参考采样器对照；原始结果见本目录的 refinement JSON。

首轮完整训练对照中，端点与 D/G 损失相同；裁剪后的头梯度相对误差约为 SiT 0.0405%、JiT 0.2293%、RAEv2 0.7557%。
这些是已测输入上的数值检查，不是对所有输入或未来训练步的误差上界。
当前精度包含原有 BF16/TF32，重算、求和顺序及非确定性底层反传均可能带来差异。
“理论兼容”据此指目标、求解器、链式法则和更新规则相同，不把它表述成长期训练逐位一致。

进一步优化的逐项测量、淘汰原因、批量吞吐和重复性对照见 [本轮性能结果](REFINEMENT_20260919_ZH.md)。
