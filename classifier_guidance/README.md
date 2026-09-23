# 分类器引导弱头：当前主线

支持训练弱头、全程引导系数，或联合训练两者；强生成模型、图像解码器和 Inception 特征网络冻结。
真假分类器区分真实训练图像与**强弱外推后的最终图像**；可训练部分通过完整采样轨迹反传来欺骗分类器。
冻结强模型的参数不意味着可以截断它对输入的梯度。

## 代码入口

新的统一实现已经支持 SiT、JiT、RAEv2：

| 职责 | 文件 |
|---|---|
| GPU 资源、源码/资产记录、分布式启动 | [`launch.py`](launch.py) |
| 训练、EMA、保存/恢复、停止标记 | [`train.py`](train.py) |
| D/W 交替更新与合并梯度通信 | [`training.py`](training.py) |
| CUDA 图重放、逐个向量场的精确离散反传 | [`sampler.py`](sampler.py) |
| 冻结反馈网络激活重算、可选微批次 | [`features.py`](features.py) |
| 模型/解码/精度适配 | [`adapters.py`](adapters.py) |
| 保持初始函数的 MLP 加深、权重和 Adam 状态迁移 | [`heads.py`](heads.py) |
| 两档 MLP 的普通 FM 容量对照、训练、5K 评估及队列 | [`capacity_heads.py`](capacity_heads.py)、[`train_capacity.py`](train_capacity.py)、[`evaluate_capacity.py`](evaluate_capacity.py)、[`capacity_pipeline.py`](capacity_pipeline.py) |
| raw / EMA 弱头收敛对照 | [`audit_capacity_convergence.py`](audit_capacity_convergence.py) |
| 全局 batch 不变的微批次梯度累积 | [`training_accumulation.py`](training_accumulation.py) |
| 冻结原始 SiT IG，学习全程有符号时间系数 | [`schedules.py`](schedules.py)、[`train_schedule.py`](train_schedule.py) |
| SiT 有符号系数的真实模型梯度检查 | [`audit_schedule.py`](audit_schedule.py) |
| SiT联合学习1-block弱头与64个有符号系数，独立probe软尺度锚点 | [`sit_joint.py`](sit_joint.py)、[`train_sit_joint.py`](train_sit_joint.py) |
| 联合训练实际梯度检查、单卡预检/30K队列及实时PNG | [`audit_sit_joint.py`](audit_sit_joint.py)、[`sit_joint_pipeline.py`](sit_joint_pipeline.py)、[`monitor_sit_joint.py`](monitor_sit_joint.py) |
| SiT联合训练单双卡配对检查、限时双卡后自动单卡续训 | [`audit_sit_joint_distributed.py`](audit_sit_joint_distributed.py)、[`sit_joint_timed_handoff.py`](sit_joint_timed_handoff.py) |
| 初始化/学后时间曲线的同输入 5K 评测 | [`evaluate_schedule.py`](evaluate_schedule.py) |
| 训练结束后每 3K 保存点的同输入 5K 队列 | [`evaluate_schedule_series.py`](evaluate_schedule_series.py) |
| JiT 系数 / SiT 联合头的 raw、EMA 配对 5K checkpoint 评测 | [`evaluate_learned_guidance.py`](evaluate_learned_guidance.py)、[`checkpoint_fid_pipeline.py`](checkpoint_fid_pipeline.py) |
| 到存档点安全暂停、评测全部通过后恢复完整训练状态 | [`pause_for_checkpoint_eval.py`](pause_for_checkpoint_eval.py)、[`resume_after_checkpoint_fid.py`](resume_after_checkpoint_fid.py) |
| 系数曲线和 GAN 梯度、参数更新的后台只读监测 | [`monitor_schedule.py`](monitor_schedule.py) |
| 真实训练 RGB 数据 | [`data.py`](data.py) |
| 完整训练迭代性能及数值对照 | [`benchmark_train.py`](benchmark_train.py) |
| 动态输入、参数更新后的真实模型梯度审计 | [`audit_replay.py`](audit_replay.py) |
| SiT三个MLP/原生头的同输入误差及W独立生成 | [`audit_head_capacity.py`](audit_head_capacity.py) |
| SiT原生0/1/2-block头与fresh浅MLP的50K训练及两种时间窗5K搜索 | [`sit_transformer_heads.py`](sit_transformer_heads.py)、[`evaluate_sit_transformer.py`](evaluate_sit_transformer.py)、[`sit_transformer_pipeline.py`](sit_transformer_pipeline.py) |
| SiT 原生头复制、输出通道与跨 patch 作用检查 | [`audit_sit_transformer.py`](audit_sit_transformer.py) |
| JiT按SSG结构做真实数据50K容量对照及5K搜索 | [`jit_ssg.py`](jit_ssg.py)、[`evaluate_jit_ssg.py`](evaluate_jit_ssg.py)、[`jit_ssg_pipeline.py`](jit_ssg_pipeline.py) |
| JiT冻结1-block，用GAN学习全程50个引导系数 | [`jit_schedule.py`](jit_schedule.py)、[`train_jit_schedule.py`](train_jit_schedule.py)、[`probe_jit_schedule.py`](probe_jit_schedule.py) |
| JiT 性能预检及短程试训队列 | [`jit_schedule_probe_queue.py`](jit_schedule_probe_queue.py) |
| JiT单/双卡配对检查、实时raw/EMA曲线和checkpoint索引 | [`audit_jit_schedule_distributed.py`](audit_jit_schedule_distributed.py)、[`monitor_jit_schedule.py`](monitor_jit_schedule.py) |

整理前的实现保留在原命名空间，作为对照并兼容旧路径和哈希：

| 职责 | 文件 |
|---|---|
| RGB 二分类器训练 | [`train_binary.py`](../experiments/adversarial_weak_training_20260915/train_binary.py) |
| 分类器和 logistic loss | [`binary_critic.py`](../experiments/adversarial_weak_training_20260915/binary_critic.py) |
| 真实 RGB 训练数据及类别映射 | [`rgb_data.py`](../experiments/adversarial_weak_training_20260915/rgb_data.py) |
| 图像解码 | [`rendering.py`](../experiments/adversarial_weak_training_20260915/rendering.py) |
| 完整离散采样器梯度 | [`discrete_adjoint.py`](../experiments/adversarial_guidance_endpoint_20260915/discrete_adjoint.py) |
| 强模型及弱头适配 | [`models.py`](../experiments/guidance_dynamic_50k_20260915/models.py) |
| 固定噪声 5K 评测 | [`evaluate.py`](../experiments/adversarial_weak_training_20260915/evaluate.py) |

## 已验证的配置

| 模型 | 配置 | 单卡完整更新提速 | 峰值保留显存下降 |
|---|---|---:|---:|
| SiT，batch 6 | 默认 `--feedback checkpoint` | 1.83× | 45.4% |
| JiT，batch 4，速度优先 | 加 `--precast` | 3.77× | 10.4% |
| JiT，batch 4，省显存 | 加 `--precast --checkpoint-backbone` | 3.05× | 40.0% |
| RAEv2，batch 1 | 加 `--precast` | 3.73× | 50.9% |

详见[性能与正确性报告](../docs/classifier_guidance/PERFORMANCE_20260919_ZH.md)。
增加一层弱 MLP 的配对测速和短程训练，见[SiT 容量试验](../docs/classifier_guidance/DEEPER_SIT_MLP_20260919_ZH.md)。
仅训练原始 IG 的 64 个有符号时间系数与 D，见[系数实验](../docs/classifier_guidance/NATIVE_IG_SIGNED_SCHEDULE_20260919_ZH.md)。
保留显存包含 CUDA 图内存池；表中包含采样、图像反馈、D/W backward 和 Adam，排除首次加载、捕获与保存。
JiT 使用已有 3K MLP 作为工程测试夹具，不是完成了新的 50K 对抗训练。

进一步核对了[理论兼容性](../docs/classifier_guidance/THEORY_AUDIT_20260919_ZH.md)，并完成
[后续优化试验](../docs/classifier_guidance/REFINEMENT_20260919_ZH.md)。当前默认路径保留完整的一阶离散梯度。
RAEv2 可选 `--checkpoint-backbone`：batch1 的保留显存约 3.25 GiB，代价是每步约 3.96 秒。
利用剩余显存，SiT 每卡 batch12、JiT 每卡 batch12、RAEv2 每卡 batch4 的吞吐分别比先前测试档提高约 17%、14%、47%；
部署时须通过设备分配保持全局 batch 和 D/W 更新频率，不能直接增大全局 batch 后宣称训练设置相同。

## 启动与恢复

以下是后续正式训练的命令示例，本次没有启动这些长训练。选择实际空闲卡；启动器遇到占用会退出。

```bash
$HOME/miniconda3/envs/myenv/bin/python -m classifier_guidance.launch \
  --gpus 1,3 \
  --output "$HOME/data/eqvae/projects/classifier_guidance/training/sit_rgb_next" -- \
  --model sit_small --updates 800 --global-batch 24 --coefficient 1.05 \
  --resume "$HOME/data/eqvae/experiments/adversarial_weak_training_20260915/endpoint_binary_gan_rgb_v2/checkpoint_001456.pt"
```

- `--coefficient` 始终是额外系数 **a**：`S+a f(t)(S-W)`。RAEv2 以前总系数 `w=1.35` 对应这里 `a=0.35`；不要传成 1.35。
- RAEv2 用 `--model raev2 --coefficient 0.35 --precast`，默认加载 depth8、50K 的 MLP 头。
- JiT 目前默认夹具只有 3K；工程检查需显式 `--engineering-fixture`。正式训练可通过 `--head-checkpoint ... --head-key ema.mlp` 加载带步数记录的 50K MLP。不会把夹具冒充正式基线。
- 从头训练保留默认 16 个 D 预热步；工程验证用 `--warmup 0` 才能在短测试里检查 W 更新。
- 在当前输出目录创建 `STOP_AFTER_CURRENT`，会在下一次更新前保存。恢复时使用新输出目录和 `--resume`。
- checkpoint 保存 head、EMA、D、两个 Adam、各 rank 数据 RNG；改变 GPU 数会记录 `exact_global_stream=false`。
- `--extra-hidden-layer` 为弱头增加一层同宽非线性残差变换：`h -> h + SiLU(hidden_residual(h))`，随后接原输出投影。新增分支零初始化，保留原头的起始预测；原有参数继续训练且保留 Adam 状态，新增参数单独初始化 Adam 状态。SiT 参数量从 304,528 增至 452,368。
- 深头 checkpoint 的恢复和显式加载会自动识别结构，无需再次指定加层；原 SiT 评测器也支持其 `head`/`ema` 权重。
- 每个进程只暴露一张卡，启动器使用 `torchrun --virtual-local-rank`；不要自行改成多卡均可见的普通启动。
- `--eager` 可关闭 CUDA 图，保留相同离散求导实现用于诊断。
- 采样器绑定输入形状、精度、单一 CUDA stream 和参数存储；替换参数对象或迁移存储需重新创建采样器。固定输入时间网格与 `amounts` 表不参与求导；`schedules.py` 将可训练系数置于 field 内的参数模块，通过完整参数 VJP 求导。当前不支持高阶元梯度。
- `--feedback chunk --feature-chunk 1` 是可选极省显存档。SiT 测试中有约 5% 梯度差异，不是默认配置，也未作生成质量验证。
- 输出的 `head`/`ema` 权重兼容原 SiT 评测器的 `--checkpoint`；支持深头只增加结构识别，噪声、采样器及参考统计保持原协议。

## 结果和数据

- [当前方法及结果](../docs/classifier_guidance/README.md)。
- [JiT / SiT 中间 checkpoint 的 FID-5K 结果](../docs/classifier_guidance/GAN_CHECKPOINT_FID_20260923_ZH.md)及[实时 PNG](../docs/classifier_guidance/figures/gan_checkpoint_fid_20260923.png)。评测后从最新完整 checkpoint 续训，目标为 30K 次系数/联合更新；保存原始权重、EMA、判别器、两个优化器及随机状态。JiT 使用 GPU0+2，SiT 使用 GPU1，每300步存档。
- [完整 64 点 5K 扫参](../docs/ADVERSARIAL_HEADS_5K_NEIGHBORHOOD_20260916_ZH.md)。
- 本地数据导航：`$HOME/data/eqvae/projects/classifier_guidance/`。
- 历史训练原件：`$HOME/data/eqvae/experiments/adversarial_weak_training_20260915/`。
- 环境：`$HOME/miniconda3/envs/myenv/bin/python`。

当前扫参已完成，旧训练的停止标记保持有效。整理目录不会恢复任何暂停实验。

## 工作区与检查

训练代码保留在本目录；历史依赖保留原路径。理论脚本见
[`experiments/theory_self_guidance_20260922/`](../experiments/theory_self_guidance_20260922/README.md)，
紧凑证据保存在 `docs/research/`。持续刷新的本地图表及 Git 固定快照见
[图表说明](../docs/classifier_guidance/figures/README.md)。

以下检查使用 CPU，覆盖离散梯度、头部迁移、有符号系数、梯度累积、评估积分器和进程交接；
四项 CUDA 图测试需要显式分配 GPU 后另行运行。

```bash
CUDA_VISIBLE_DEVICES='' CLASSIFIER_TEST_CUDA=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$HOME/miniconda3/envs/myenv/bin/python" -m pytest -q \
  tests/test_classifier_guidance.py tests/test_classifier_heads.py \
  tests/test_classifier_schedules.py tests/test_classifier_accumulation.py \
  tests/test_learned_guidance_evaluation.py tests/test_sit_handoff.py tests/test_sit_joint.py
```
