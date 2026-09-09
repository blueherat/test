# 同轨迹类别状态保留与输出缓存对照

2026-09-08。继续独立 fixed point 路线，PFR 和论文写作暂停。

上一轮跨轨迹搬运失败不能推断同轨迹失败。本轮固定相同8条缓存轨迹和第14层；当前步取1/47/73/87，donor只取同一轨迹上一步0/46/72/86，不搜索步距、层数或强度。FP32关闭TF32，每个新旧状态的类别/null原生Full与拆分Full输出均逐位一致，总128次。输入状态逐个核对原缓存SHA。

比较在当前null前缀上写入：当前类别token、上一时刻原始类别token、当前null token加上一时刻类别减null token。并比较上一时刻Full clean条件输出差，以及上一时刻velocity条件输出差。后者换算到当前clean差单位，乘以 `max(t_now,.05)/max(t_previous,.05)`；两状态不同，但条件差内部的共同z项抵消，换算没有状态漂移项。所有输出比较均为同一目标空间的Full clean差，未采用PFR修正。

|当前步|旧token差对新token作用余弦|旧token差对新token作用相对平方误差|旧token差对完整条件差相对平方误差|缓存velocity差对完整条件差相对平方误差|
|---|---:|---:|---:|---:|
|1|.999703|.000600|.325200|.001998|
|47|.999216|.002832|.721115|.022853|
|73|.999553|.000946|.476820|.003871|
|87|.998452|.003384|.432209|.007323|

表中都是拼接8图的向量比较，没有统计显著性或未见数据泛化主张。列2/3和列4/5的目标不同，不能拿列3比列5宣称激活记忆胜过输出缓存。当前类别token本身对完整条件差相对平方误差为.326/.724/.478/.429，缓存忠实保留了一个本来不完整的条件作用。减去null状态后，旧token的复用误差在非初始时刻显著小于直接复用原始token（后者对新token作用误差.03584/.03387/.18044）；这仅是机制诊断。

实现 `experiments/probe_raev2_class_token_persistence.py`，分析 `experiments/analyze_raev2_class_token_persistence.py`。128 Full、128前14层、224后14层及decoder，全部B1。GPU执行46154退出0，计算及向量保存22.619321秒（约.00628GPU小时），加载、checkpoint哈希、CPU分析额外。CPU审计15068退出0，重新计算32组保存向量Gram并核对源码SHA。便携结果 `experiments/results/terminal_defect_20260908/raev2_class_token_persistence.json`，原始向量在 `/home/zhoushunyu/data/eqvae/experiments/fsg_class_token_persistence_20260908/`。没有生成图片、FID、训练、加速测量或多步反馈rollout，当前各比较依赖同一现有轨迹。

## 新颖性核对及取舍

本轮实际阅读 [MetaState](https://arxiv.org/html/2603.01331v1) 摘要、Introduction、Related Works：持久状态读入、递归更新、写回冻结模型已是明确设计。另查阅 [RIN 的原始论文摘要](https://proceedings.mlr.press/v202/jabri23a.html)：图像扩散中以先前latent计算条件化下一次latent计算的self-conditioning已有直接先例。未完成两篇全文/实现复现，不将其结果宣称为已在RAE复现。

所以“内部有可复用信息”是可观察现象，“跨步加一组记忆token”本身不是新idea。当前数据不支持靠缓存这8个token完整替换条件计算，也不支持为此直接训练通用读写记忆模块。候选需要先明确比输出缓存多保留或更正确地更新的条件量，以及该差别为什么改善终点分布；不能靠增加层、归一化或延长缓存窗口绕过这个问题。

这轮完成了同轨迹诊断，未找到方法突破，研究目标仍未完成。
