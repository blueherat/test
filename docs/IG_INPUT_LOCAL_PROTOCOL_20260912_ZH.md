# 输入局部内部参考：冻结实验协议

本轮落实用户粘贴方案中尚未真正实现的备用项：参考只依赖当前 noisy latent 的一个 tokenizer patch、时间、类别和位置。训练用真实数据的原始回归目标，冻结强模型，不拟合 teacher、不训练独立主干。生成时从强模型第一次 attention 之前的共享 embedding 取输入，增加一个逐 token 读出；不运行额外弱前缀。

在无限容量和总体 MSE 最优极限下，局部 clean 预测是 $E[X_j\mid Z_j,t,c,j]$，对应各 patch 条件边缘分布的乘积。与联合分布的 score 差强调跨 patch 依赖。有限训练读出不自动等于这个理想 score；RAE 的 patch 是语义 latent token，不是像素局部感受野。SWG 已研究删除远程输入的 guidance，本轮不声称首次提出局部 guidance，只检验这种信息限制能否在一次共享主干的 IG 预算下产生收益。

## 唯一方法与匹配控制

- Local：SiT 在 block 0 前、RAE 在 encoder block 0 前取空间 token；Context：分别在 block 4/8 后取空间 token。二者共享相同的原始、未经历 attention 的时间／类别条件。RAE 取原始条件 token 的均值。
- 两个读出结构相同、随机初值相同：逐通道固定训练统计标准化，线性 token 输入＋线性 condition 输入＋8维固定二维 Fourier 位置输入，经 SiLU 和线性输出。隐层宽度等于原 embedding 宽度，不扫描宽度、patch 大小或位置编码。两个 tokenizer 均保留原 patch。
- 从固定训练数据先用32个批次估计输入均值方差，标准差下限1e-4。不能使用生成数据或验证集估计标准化。位置输入为每轴一、二频率 sin/cos。
- SiT 目标为原线性 bridge 的 velocity，RAE 为原线性 bridge 的 clean latent。时间独立均匀采样[.01,.99]。两个 head 使用完全相同的 clean/noise/time/label 批次。
- 固定3000步 AdamW，lr=3e-4、weight_decay=1e-4、EMA=.995，SiT batch32、RAE batch8。最后一步 EMA 进入生成；不根据生成指标挑 checkpoint。固定训练种子2026121301。
- SiT 采用已有每类256个训练／32个验证真实 VAE moments；RAE 采用已有每类5个训练／1个验证真实 latent bank，沿用来源哈希与无重叠检查。该 RAE 数据量限制须报告。

## 生成与判据

两个模型从起点同时进入。每臂400个独立样本、同一初始噪声与类别，种子2026121307。单路径统计，不产生配对第二分支。SiT 为 Heun64、RAE 为 Euler100 shift8；窗口与原 IG 相同。

固定8臂：Local 原幅度／半幅度、Context 原幅度／半幅度、原 IG 原幅度／半幅度／双幅度，以及无引导 Strong。原幅度沿用 SiT .8（前四分之一乘6/7）、RAE .78。不根据 endpoint 污染比例调整此方法的幅度。

Local 候选须比全部6个非 Local 对照至少降低2个 FID 点，且 IS 不低于原 IG 的90%，才能进入新的1000个噪声确认；确认后才考虑5000。400样本是筛选，不能声称统计显著。RAE 400样本只覆盖400类，对完整 ImageNet 参考存在覆盖限制。失败则保留全部结果，禁止用 Local vs Context 的局部胜出冒充超过原 IG。

## 必须通过的实现检查

同标签时间下只修改一个 patch 之外的输入，Local 的该 patch 预测必须保持完全相同；Context 可受影响。共享前向捕获特征与直接 embedding/prefix 计算一致；安装 hook 不改变原 full/weak 输出；零强度完全等于原 Strong。训练后重做信息限制检查，确认所有强模型参数仍 frozen。每个输出的 full/prefix 调用计数和采样解码时间须记录，并另测相同批量、同卡的原 IG 与候选时间，不能把多一个 head 称为零成本。
