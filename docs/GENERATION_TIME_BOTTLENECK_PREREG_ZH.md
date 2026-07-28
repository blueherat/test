# 生成时间瓶颈小型受控实验：预注册

## 1. 研究问题

已有工作已经覆盖“低维 encoder + 随机 diffusion decoder”这一宽泛框架。
本实验只检验一个更窄、可证伪的问题：

> 若一个紧凑 latent 只允许在高噪声阶段介入像素 flow，能否先确定类别和
> 全局形状，再由不再读取 latent 的后半段 flow 完成细节？

训练时使用一个小 encoder 从图像推断 latent；它只用于学习条件变量和后续
缓存训练 latent prior。完整生成时 encoder 不在采样图中。

## 2. 对照

数据固定为 MNIST 官方 train/test split。第一轮使用 train 10,000、test
2,000，不使用 test 选择训练超参数。

四个模型使用相同初始化、数据顺序、噪声和优化预算：

- `all_time`：所有噪声时刻都读取 latent。
- `high_noise`：latent 作用从 `t=0.55` 到 `t=0.75` 平滑开启，高噪声端为 1。
- `low_noise`：使用互补门，主要在低噪声阶段读取 latent。
- `none`：完全不读取 latent 的无条件像素 flow。

路径和 PiD 探针保持同一约定：

```text
x_t = (1 - t) x_0 + t epsilon
v_target = epsilon - x_0
```

encoder 与像素 flow 联合训练，但不使用重构 decoder、类别标签或外部
representation teacher。类别标签只用于训练一个独立的评估分类器和线性
probe。评估分类器固定训练 20 epochs；在生成实验开始前的资格检查中，
seed 0/1 的 test accuracy 分别为 98.05%/98.55%。

## 3. 预注册预测

P1：`high_noise` 在 held-out teacher path 的高噪声端应满足
`Delta_shuffle > 0`；低噪声端因门为零，identity/null/shuffle 必须数值一致。

P2：条件 rollout 中，`high_noise` 的源类别保持率应显著高于 shuffled-code
对照，且达到 `all_time` 的至少 90%。这才说明早期注入的信息能通过像素状态
传到后半程，而不只是 teacher-forced 损失变好。

P3：`low_noise` 的高噪声 responsibility 应接近零，源类别保持率应弱于
`high_noise`。若相反，说明 MNIST 上全局结构并不需要在生成早期确定。

P4：`none` 不应显示任何 real/shuffle 差异；所有重复 forward identity
控制必须在浮点精度内一致。

P5：`high_noise` 的 train/test latent 线性分类准确率可以高，但这只叫
生成任务诱导的语义压缩，不称为解耦、互信息最优或因果表征。

## 4. 进入 latent prior 的门槛

只有同时满足以下条件，才缓存 train latent 并训练 prior：

- P1、P2、P4 全部通过；
- 至少两个 seed 的方向一致；
- `high_noise` rollout 的源类别保持率比 shuffled 对照高至少 20 个百分点；
- `high_noise` 达到 `all_time` 源类别保持率的 90%，或质量差距可由明显更高的
  同码随机多样性解释；
- 分类器 test accuracy 至少 98%，避免评估器成为瓶颈；
- 结果不是 latent 尺度塌缩、固定点 shuffle 或 test 参与训练造成。

prior 阶段比较 `all_time` 与 `high_noise` 两种 latent，固定 prior 参数量、
训练 step、batch 和采样 NFE。第一轮只用 MNIST 机制指标，不声称自然图像质量。

## 5. 异常与停止规则

若任何预测不符，先检查路径方向、velocity 参数化、门函数、shuffle、
teacher/rollout 状态和分类器。若实现正确且两个 seed 重复，则停止后续 prior，
记录为真实负结果。不得通过改阈值、增加 latent 维度或只报告 teacher path
挽救。

## 6. 文献边界

- [Diffusion Autoencoders](https://arxiv.org/abs/2111.15640) 已用语义 encoder
  配合随机 diffusion decoder。
- [InfoDiffusion](https://arxiv.org/abs/2306.08757) 已用互信息项防止强 decoder
  忽略低维 latent。
- [SWYCC](https://arxiv.org/abs/2409.02529) 已联合学习连续 encoder 与随机
  diffusion decoder，并在 latent 上训练生成模型。
- [RCG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e304d374c85e385eb217ed4a025b6b63-Abstract-Conference.html)
  已覆盖 representation prior + conditional pixel generator。

因此本实验的潜在价值只来自“用生成时间定义信息分工”是否带来可复现的
生成优势，不能来自两阶段框架本身。
