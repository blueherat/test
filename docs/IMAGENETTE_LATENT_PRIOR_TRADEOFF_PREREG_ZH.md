# Imagenette-64 Latent Prior 与 Decoder 收益权衡：预注册协议

## 1. 研究问题与解释边界

本实验是一个新研究问题，不追认此前已经失败的“噪声责任曲线预测质量”假设。
唯一主问题是：

> 在固定 decoder 家族与训练预算下，增加 latent 容量虽然能改善使用真实 latent
> 时的条件生成质量，是否也会稳定增加固定预算 latent prior 的建模难度，从而使
> 完整两阶段生成在中间容量达到最优？

阶段一只闭合现有 Imagenette-64 系统。冻结已有 `16/64/256d x 5 seeds` 的
encoder 和 decoder，为每个配置训练同构同预算的无条件 latent rectified-flow
prior。阶段一不更新 encoder、decoder，不使用类别标签训练 prior，也不根据结果
修改 prior 架构、容量、步数、NFE 或质量指标。

本实验中的 FID 是 Imagenette-64 上使用 ImageNet ResNet18 pool features 的
受控 feature FID，不称为 ImageNet gFID，也不外推到大型 RAE。

## 2. 冻结输入与数据边界

- 冻结 checkpoint 根目录：
  `~/data/eqvae/imagenette_noise_responsibility_formal/`。
- 配置固定为 `latent_dim in {16,64,256}`、`seed in {0,1,2,3,4}`，共 15 个。
- 数据固定为 `/data/shared/imagenette2-320` 官方 train/val split。
- prior 只读取官方 train 图像经冻结 EMA encoder 得到的 latent。
- held-out prior loss、Oracle、经验 latent 和 prior rollout 只使用官方 val split 或
  与 val 图像无交集的 train latent 经验样本。
- 图像全部使用原实验的确定性 `64x64` validation transform；不通过重复随机增强
  扩大 prior 数据量。
- 类别标签只用于评估预测类别覆盖，不进入 encoder、prior 或 decoder。

## 3. 公平的统一 256 维 prior 接口

不能直接给三种容量使用三个不同输入宽度的 prior，也不能把低维 latent 补零后
从 256 维满秩高斯压到奇异子空间。正式接口如下：

1. 用固定 seed 生成同一个 `256x256` 正交矩阵 `Q`。
2. `d` 维 code 用 `Q` 的前 `d` 列等距嵌入 256 维：

   ```text
   y = z Q_d^T,   z = y Q_d
   ```

3. prior 的源噪声同样只位于该 `d` 维子空间，不要求可逆流把满秩 256 维高斯
   压到低秩流形。
4. prior 网络始终接收和输出 256 维，网络输出再正交投影回对应子空间。
5. flow loss 在 `d` 个正交坐标中按维平均。网络参数量、初始化、优化步数和 NFE
   对三种容量完全相同。

同一 frozen seed 下，三容量必须具有相同 prior 主干初始化哈希、图像顺序、batch
索引、time 和 256 维基础噪声流哈希。正交嵌入往返误差必须不超过 `1e-6`。

## 4. 固定 prior 与训练预算

- prior：256 维 time-conditioned residual MLP rectified flow。
- width `512`，residual blocks `6`，SiLU，LayerNorm，零初始化输出层。
- path：`y_t = (1-t)y_data + t y_noise`，`t=1` 为高斯端；target velocity 为
  `y_noise - y_data`。
- optimizer：AdamW，learning rate `2e-4`，weight decay `1e-4`。
- batch `512`，steps `20,000`，gradient clip `1.0`，EMA `0.999`。
- fp32，关闭 TF32，不使用 autocast。
- 使用最终 EMA，不早停，不按 validation loss 选择不同 step。
- formal sampler：Euler `100` NFE，从 `t=1` 走到 `t=0`。
- 每个 frozen checkpoint 使用一个与 frozen seed 确定绑定的 prior seed。
- 若主结果落在停止阈值附近或容量顺序不一致，只允许补一个预先定义的第二 prior
  seed；不得改网络、预算或选择有利 checkpoint。

## 5. 四种 rollout 与指标定义

所有配置使用同一批 real val features、相同样本数、相同 decoder pixel-noise seed
和相同 decoder `50` NFE。正式评估使用完整 val `3,925` 张图。

1. **Oracle rollout**：冻结 val encoder code 输入冻结 decoder。
2. **Empirical rollout**：从冻结 train code 经验分布等概率采样，再输入 decoder。
3. **Prior rollout**：从高斯经 latent prior 采样 code，再输入 decoder。
4. **Gaussian control**：未经训练的标准高斯 code 输入 decoder。

主指标：

```text
Oracle FID       = FID(Oracle rollout, real val)
End-to-end FID   = FID(Prior rollout, real val)
Total prior gap  = End-to-end FID - Oracle FID
Modeling gap     = End-to-end FID - Empirical FID
```

`Total prior gap` 保留用户指定定义；`Modeling gap` 用来排除 Oracle 的配对 val code
与 prior 所学习 train-code 分布之间的差异。还报告：

- held-out per-active-dimension flow MSE；
- generated/val latent covariance relative Frobenius error；
- covariance eigenvalue overlap；
- real/generated latent effective rank；
- sliced Wasserstein distance；
- predicted class entropy、最小类别占比和类别分布总变差距离；
- Oracle paired source MSE 与 class match，作为原实验复算控制。

所有 feature arrays 和 latent summary 保存到仓库外
`~/data/eqvae/imagenette_latent_prior_tradeoff/`，不写 repo 内 outputs/artifacts。

## 6. 阶段一确认门槛

所有判断以 5 个 frozen seed 的配对结果为单位。正向 trade-off 必须同时满足：

1. **Decoder benefit**：`OracleFID_256 < OracleFID_16` 至少 `4/5` seed，且五 seed
   平均改善至少 `2.0` FID。
2. **Prior difficulty**：`ModelingGap_256 > ModelingGap_16` 至少 `4/5` seed，平均
   增加至少 `2.0` FID；`TotalPriorGap` 方向一致。
3. **Intermediate optimum**：`EndToEndFID_64` 同时低于 `16` 和 `256` 至少
   `4/5` seed，且五 seed 均值至少比两端中较好者低 `1.0` FID。
4. **Not a failed prior**：三容量 prior rollout 均显著优于 Gaussian control；正式
   held-out flow loss 有限，训练后半段无持续发散。
5. **Mechanism prediction**：leave-one-frozen-seed-out 中，`effective rank` 或
   `held-out prior loss` 对 Modeling gap 的 RMSE，至少比“名义 latent dim”和
   “decoder validation loss”两个单变量基线都低 `5%`。

阶段一只有 1--5 全部通过，才允许进入固定 256 维低秩因果实验。相关性、单 seed
中间最优、只有 Total gap 增大或只有 prior loss 增大，都不算确认。

## 7. 相反证据与停止规则

若满足以下条件，则视为该容量范围和固定预算下的明确相反证据：

- Decoder benefit 通过；且
- `EndToEndFID_256 <= EndToEndFID_64 <= EndToEndFID_16` 至少 `4/5` seed；且
- `ModelingGap_256` 不大于 `ModelingGap_16`，或差值小于预注册的 `2.0` 实用
  效应阈值；且
- prior rollout、latent 指标和 NFE 审计确认不是 prior 未收敛、采样离散误差、
  维数归一化、训练/验证泄露、FID 复算或随机流不一致造成。

出现该结果时，必须执行第 8 节独立审计。审计后仍成立则停止本方向，不扩大容量、
不改 prior、不转而挑选有利 seed，也不进入固定秩或联合训练。

若结果方向混合且置信度不足，只允许使用同一协议补第二 prior seed并增加独立 FID
复算；仍不明确则记为“当前预算下无可检测 trade-off”，不宣称正结果。

## 8. 强制实现与反向结果审计

正式结论前必须通过：

1. checkpoint 载入后 Oracle 指标复现原结果，完整 val 重算另行明确标记；
2. encoder/decoder 训练前后参数哈希不变且无梯度；
3. 正交嵌入往返、投影幂等和子空间噪声协方差 toy test；
4. 解析高斯目标下 flow velocity、Euler 方向和 sample moments toy test；
5. train/val 文件路径集合无交集，cache 包含源路径哈希；
6. 三容量 prior 参数量完全相同，同 seed 初始化哈希相同；
7. 三容量训练 batch index、time、基础 256 维 noise 哈希相同；
8. 全部 loss、权重、latent 和 features 有限；
9. 至少一个异常或反向配置从 checkpoint 独立重采样，指标逐项复算；
10. formal `100` NFE 与 `200` NFE 的结论方向一致；若 FID 容量排序变化，则采样
    未收敛，不能形成科学结论。

## 9. 阶段二预先边界

仅当阶段一全部确认门槛通过，才新建独立预注册：所有模型名义 latent 固定 256
维，使用固定正交瓶颈控制 active rank `8/16/32/64/128/256`，保持 encoder/decoder/
prior 家族、参数、条件 RMS、数据流和预算一致。阶段二必须因果复现 decoder benefit、
prior difficulty 和中间 end-to-end optimum，才允许研究 prior-aware bottleneck 方法。

阶段一的 post-hoc effective-rank 相关性不能替代阶段二因果证据。
