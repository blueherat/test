# 未来原型第二版：扣除弱模型自身往返

用户要求方法优先、只生成新方法、四卡协同且复用已有基线。第一版完整配对1K FID38.687292，对照原生IG38.264239，恶化1.1056%，不扩5K。本轮只改为数值往返相减版本，不重跑基线、不增加机制检测。

固定z，令A=R_B(Phi_IG(z))、N=R_B(Phi_B(z))，更新z←z+(A−N)。理想准确逆下N=z，恢复原来有限区间操作；近似逆下扣除弱模型自身往返造成的偏移。没有质量改善定理，直接按完整1K评价。此相减思路此前已用于初始写入后撤guidance的失败实验；本轮保留全程IG并使用多步未来原型，不将相减本身宣称为新颖性。

其余与第一版完全相同：seed202609413、B4、100步shift8、写回索引20/40/60/80，每次未来4网格步、双向Heun；四卡按全局batch索引mod4协同生成1000张。每图132次完整forward、96次Base前缀，所有批次原子保存可恢复。沿用已有balanced版本8图smoke，前两个全局batch须逐像素复现；已有完整检查结果直接复用。

第一版四卡输出实际在`ig_future_prototype_20260908/rank*`，原迁移协议曾误写为另一个目录；评价端已按实际路径修正，没有重采样。第一版结果见该目录`result.json`以及`experiments/results/terminal_defect_20260908/ig_future_prototype_fourcard_quality.json`。原四臂并行试验按用户纠正停止，无完整质量结论。

第二版唯一新目录`/home/zhoushunyu/data/eqvae/experiments/ig_future_prototype_balanced_fourcard_20260908/`。第一版GPU分片最长285.0秒，合计有效batch GPU秒1115.6；旧取消任务的浪费另保留progress，不算作方法推理成本。论文不写，研究尚未成功。
