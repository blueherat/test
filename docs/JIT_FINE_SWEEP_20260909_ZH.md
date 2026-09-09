# JiT 夜间细扫：depth4 IG → PFR → lifting

用户允许更长时间细扫，并澄清要把JiT上的lifting也做完，不是增加depth8实验。
保持已有depth4/8联合训练到50K，仅depth4参与本轮生成。当前训练不中断。
本协议替代单depth粗略坐标搜索，恢复JiT lifting；小SiT/RAE lifting仍搁置。

IG前半data-time[0,.5)、后半[.5,1]分别设置gamma。
先后段0扫前段0:.15:1.5，再补齐前段上述11点×后段0:.1:.6的完整联合网格。
如果最优落在前段上界，追加1.65/1.8；若随后最优落在后段上界，追加.7/.8。
最后围绕最优，前段±.1以.05间隔，后段±.05以.05间隔细化，限制在已扩展边界内。
每个参数组合只跑一次。选出已评估网格中最好，不宣称全局最优。

按 JIT_ORDERED_METHOD_STUDY_20260909_ZH.md 的模型、B4、seed202609831、
100步Euler、FP32状态/BF16/TF32、原生clean到velocity转换、官方像素量化和
ImageNet256参考执行。每组四卡共同分片1K，组间顺序。复用Full/官方CFG基线。
新共享接口和官方Full、弱prefix逐元素检查，并检查4张旧Full像素。

IG选定后，canonical projected PFR沿用其gamma，仅前半程h=min(1/32,.5-t)，
保持原弱锚点W并修订未来参考。若前半gamma=0则此PFR等同IG，直接复用。
然后lifting使用所选前后gamma，同时乘1/.8/1.2，分别1/2次写入；两次各alpha/2。
辅助Heun和主Strong Euler；block最多4细步，不跨.5。零gamma段只用Strong。
不把lifting扫强度后的最好称作无需调参的收益。

若PFR或最好lifting在1K相对选定IG改善至少1%，冻结该方法配置，自动追加
seed202609932、B4、labels循环0..999的独立5K；同一独立bank IG只生成一次。
两方法均满足阈值则分别与此IG配对。该门槛只是资源分配规则，不是显著性检验。
保存请求、源码/输入/输出哈希、rank覆盖、FID/IS、调用和时间；失败写状态并停止。
不写论文，不把JiT/SiT/RAEv2差异自动归因于latent平均。
