# 条件写入后撤去外部条件：完整续生成协议

2026-09-08。用户纠正了研究重点：FSG校准后的高噪状态本身应携带条件，使无条件生成也能趋向有条件结局。本轮据此改变主检验，不再把局部token缓存准确率或真实加噪MSE作为条件接管的充分判断。此前数据保留，但不把它们当作对FSG信息搬运直觉的否定。

## 冻结设置

RAEv2官方EMA，64个类别 `floor(arange(64)*999/63)`，每类一个噪声，seed202609442。FP32、关闭TF32、B4、100步shift8 Euler。全部从同一高斯噪声bank开始；在t=1进行写入，避免类别在前缀轨迹里已有积累的混淆。

写入取固定H=.125、K=2、gamma=2，真实条件/null=1000 Full预测转velocity。每轮为 `q=z−H(2v_c(z,1)−v_u(z,1))`，随后 `z=q+H v_u(q,.875)`。两轮都返回噪声时间1。不是PFR或Full/Base组合。

六组：

- conditional：原始噪声，100步始终给原类别，Full条件场，不加额外CFG。
- unconditional：原始噪声，100步始终给null。
- calibrate_drop：用原类别写入，随后100步全部null。
- calibrate_keep：与calibrate_drop相同写入状态，随后继续原类别。
- unconditional_roundtrip：两腿均null的同H/K往返，随后null，用来检查数值往返作用。
- wrong_class_drop：写入 `(c+1)%1000`，随后null，用来检查输出是否跟随写入类别。相邻ImageNet标签可能语义接近，因此它不是严格异语义负对照。

模型调用分别为每图100/100/106/106/104/106 Full；不是同成本质量比较。两校准续生成组的写入状态要求逐位一致；原始噪声组也要求一致。记录每一后续模型调用的null标签数量，撤条件组必须全部null。底层IG模型虽返回Base，此处从不使用Base输出。

这不是完整FSG论文配置复现：它专门撤掉校准后的所有外部条件，包括原算法常规CFG++校准，检查一次写入阶段之后的持续承载。既往RAE FSG质量负结果不替代这个实验，也不因此被撤销。

## 内部记忆承载对照（在结果出现前补入）

同一64噪声及标签、同一100步采样，在t=1只写一次第14层类别位置差 `m=h_c,class−h_null,class`。之后每步仅输入null，在当前null前缀的类别位置加固定m，再运行后半网络。不在后续查询原类别、不查询未来、不调增益，记忆大小8×1440。该组检验“撤条件后能否靠内部载体维持类别”，此前相对完整条件输出差的误差较大不足以直接淘汰它。

额外8图zero-memory组须与直接null完整生成的前8图逐位一致，核对整条手动拆分rollout。实现 `experiments/probe_raev2_memory_handoff.py`，输出目录 `fsg_memory_handoff_20260908`。这只是最简单的承载对照，不宣称固定缓存有新颖性。若呈现语义保留，仍需错误记忆等因果对照和质量实验，不能用classifier单项充当研究成功。

## 观察量与边界

冻结torchvision ConvNeXt-Tiny ImageNet1K V1，使用官方预处理，在解码uint8图像上测原类别top1/top5、概率和donor类别指标；保存全部概率。分类器未参与写入或采样。64图是机制pilot，不足以宣称FID改善或总体显著性。类别指标不能代替图像质量，保存固定首8样本的六组对照图供查看。

另外比较原始噪声条件/无条件续生成终点差，及同一个写入状态条件/无条件续生成终点差。这两个差只量化latent结局一致性，不是图像语义或分布距离。要分别报告“撤条件后保持类别”和“条件/无条件结局更接近”，不能将两者混为一项成功。

实现 `experiments/probe_raev2_condition_handoff.py`、独立分析 `experiments/analyze_raev2_condition_handoff.py`。所有latent写入状态、终点、图像及源身份记录在 `/home/zhoushunyu/data/eqvae/experiments/fsg_condition_handoff_20260908/`。本协议写于结果完成前；此次没有新方法、质量改善或论文声明。
