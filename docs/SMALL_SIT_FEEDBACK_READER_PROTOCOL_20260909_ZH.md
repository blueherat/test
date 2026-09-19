# 小 SiT：缓存特征上的低成本反馈读出

用户要求继续试验，先考察低成本 guidance。本轮具体化
[反馈建议](IG_READABLE_FEEDBACK_PROPOSAL_20260909_ZH.md)：先把读取接口放在两个输出头，
一次原生共享主干计算后，缓存每个头在最终线性层之前的条件化特征。新增网络不重新
运行 Transformer 主干，不做第二次完整网络查询，不构造 toy。

这与原建议的“浅层之前注入、重跑后续主干”不同，是为限制成本选定的第一步实现。
此次只做一次读出更新，尚不是已求解的计算固定点或跨时间记忆方法。

## 模型与更新

固定已有 ImageNet-100 SiT-S/2 velocity 800K EMA、depth4 velocity 50K EMA 和 MSE VAE。
记原两头 velocity 为 S、W，状态为 z，时间由噪声向数据 0→1。
clean 信息是 a=z+(1−t)S、d=(1−t)(S−W)。将 2×2 latent patch 转为 16 维 token。

每个头独立添加 `Linear(384+52,64) → SiLU → Linear(64,16)` 残差读出。
输入包含 layer-normalized 原头特征，以及 z、a、d 的 patch 和四个时间特征
`[t,1−t,sin(pi*t),cos(pi*t)]`；最后一层全零初始化，主干和原头均冻结。

S'=S+R_S(h_S,z,a,d,t)，W'=W+R_W(h_W,z,a,d,t)。
活动区间采用 S'+gamma(S'−W')；原非活动区间仍使用原 Full。
这种有训练的读出同时改变活动区间的基准预测及引导差，不应把收益自动归因于分歧。
gamma 全部设零时不调用适配器；零初始化适配器也必须严格复现原生 IG。

两种训练／生成配置相同容量与优化设置：

- strong_only：反馈 d 槽为零，读取 strong 判断。
- strong_gap：读取 strong 判断与实际 strong–weak 分歧。

两种配置都拥有相同原模型和原始浅／深特征；对照考察显式反馈分歧的增量作用，
不是假装移除了模型内部所有 weak 信息。strong_only 的 d 输入连接没有非零训练输入，
其可辨识参数比总声明参数少；两组相同架构总参数量不代表输入信息量相同。

## 训练，先冻结再运行

仅从现有 train moments 中以种子 202609977 随机抽取互斥的8192训练图与1024验证图。
不使用生成评价噪声 bank 或 ADM validation reference 训练、拟合或选择检查点。
每图独立抽 VAE posterior、标准高斯 noise、t∼U(0,.5)，构造原生 linear-flow state；
每图均匀无放回采16个 patch。batch64收集一次真实网络特征，缓存 FP32。

两个读出分别以真实 velocity X−epsilon 为监督，损失为两头 MSE 的平均；不训练两头
互相相等，不拟合普通 IG teacher 输出。每个训练 token 以50%概率同时将 a、d 置零。
两个配置使用相同初始化和相同 minibatch／mask 随机序列。

固定1000步、2048 token/batch、AdamW lr.001、weight decay.0001、梯度范数上限1；
前100步线性 warmup，此后 cosine 到初始 lr 的1%，EMA .99。只用固定末步 EMA，
验证日志不用于早停、挑检查点或选超参数。FP32+TF32；特征采集、两组拟合成本单列。
这只是有限训练预算，失败不能证明更充分训练后也无效。

## 生成与对照

复用已完成、覆盖与像素身份可核验的普通 IG FID65.139320；两种新配置各完整1000张。
噪声、标签、B8、四卡分片、主 Dopri5(.001,1e-6)、时间边界0/.125/.25/.375/.5/1、
额外 gamma=.6/.6/.7/.7/0，与该普通 IG 全部相同。

每个 RHS 最多一次 Full 和一次新增双读出，无额外完整网络或 prefix 查询。
自适应求解器的总 NFE 仍可能变化，必须报告实际 Full、读出次数与耗时。
预检在固定同一 batch-state 上计时原生／新 RHS；总生成成本另实测，不能仅凭小参数
量声称低开销。若质量变好但成本明显增加，补足有用计算的普通 IG 成本对照后才能
宣称同成本质量优势；本轮初始预算不自动承诺该优势。

真实网络预检要求：原生 Full／weak-prefix 一致、捕获的线性层输入能复现原输出、
patch往返精确、零适配器精确、禁用读出后复现普通 IG 首批 latent、零 gamma 退化一致。
所有配置直接进行完整1K，不根据几个样例挑选配置。

同时报告 FID、sFID、IS 和固定首8张配对输出。首先看 strong_gap 是否超过 ordinary IG，
再看它是否超过 strong_only；只有后者支持显式分歧的增量价值。MSE改善、分歧缩小、
固定点形式与实际训练完成都不是质量成功。若两组皆无收益，不立即扩大宽度、迭代数、
步数或 guidance scale 搜索。已有 bank 仍是探索数据，不宣称独立确认。

训练请求冻结模型／数据／索引／采集训练代码与本协议；生成请求在训练结束后另冻结
两个实际读出 checkpoint、噪声标签、原生基线、采样与评估源码及 Inception 图身份。
批次原子保存，保留全部失败记录。原始资产位于
`/home/zhoushunyu/data/eqvae/experiments/small_sit_feedback_reader_20260909`。
