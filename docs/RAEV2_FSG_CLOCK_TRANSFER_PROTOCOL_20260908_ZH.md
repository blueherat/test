# RAEv2 时钟干预：固定直接迁移

在新RAEv2像素/FID之前固定。SiT的同bank低预算优势已确认，但原PFR DOPRI5
仍有更好的最高质量；前向状态校准必要性未获支持。该试验不保证直接迁移成功。

使用官方100080 EMA、原DINOv3L-k7 latent/decoder、shift8 Euler，IG scale1.78及
原区间[.1,1]，BF16/TF32。沿生成时间tau=1-t定义G为原IG生成方向速度，
S为full生成方向速度，R=2S−G，对应SiT的effective reference分解。
这里额外IG系数是.78，不是1.78；模型仍预测clean，通过原clean_to_velocity转速度。

五组固定1K：ordinary100、ordinary110、short100、asynchronous100、time_only100。
后三组在原100步网格首次达到tau>=0,.125,.375时分别校准2、2、1次。
每次h=.025，short查询tau+h，asynchronous/time_only查询tau+.125；
async先z1=z+hG(z,t)，再z←z1−hR(z1,t−.125)；short使用t−h；
time_only用z←z+h[G(z,t)−R(z,t−.125)]。随后执行原生guided Euler步。
三个候选各110次共享full/base模型前向，与ordinary110匹配。没有单独廉价prefix查询。
事件取相同线性桥系数，不声称匹配SiT与RAEv2在所有表示方向上的SNR。

seed202609413，CUDA continuous RNG，B4，global_ids0..999每类一个；全组noise/label
hash必须一致。原native sample_condition与新ordinary100先在相同8张上逐像素比较。
其余组先8张finite/实际调用检查，再原样1K；不据图像调整h/H、事件、IG或种子。
评价保持RAEv2原官方fd_evaluator及imagenet_256_fid_stats，不把其绝对FID与SiT
ImageNet100 ADM分数跨数据集比较。记录全部结果、采样时间、sources及pixel hash。
无需训练。任何1K正面结果均需独立5K确认，不能称为迁移突破。

## 实现检查与运行状态

ordinary100的相同8张与原native sample_condition逐像素一致。
ordinary110、short、asynchronous的8张finite/输入hash/实际调用检查通过。
三个候选的目标都是每样本110次完整共享full/base前向，无单独prefix调用。
实际100步事件索引为0、54、83，对应noise time1、.872037888、.621004581。
四卡运行1K；time_only在ordinary100之后顺序执行，尚未产出其smoke或质量结果。
原生8张采样6.397秒，其他8张约7.1秒；全部5K图像预计约1–1.3 GPU小时，
不包含模型加载与FID。这个估计不是最终计费或硬件测速。

## 时间解释的现有工作边界

本轮继续核对原始文献，避免将阶段调度本身更名为新方法：

- [Stage-wise Dynamics](https://arxiv.org/html/2509.22007v1)：本轮读§4.1–4.2，
  其阶段分析和时变guidance已覆盖“早中晚作用不同”的一般主张。
- [Information-Theoretic CFG](https://arxiv.org/html/2606.24025v1)：本轮读摘要、
  Introduction/§2及§4优化方法。它用实际轨迹目标提出schedule更新，再重新采样验收；
  固定轨迹导数并不冒称完整目标梯度。未在本轮读完其所有证明。
- [General Class Speciation](https://arxiv.org/abs/2602.04404)：本轮仅读摘要，
  已有工作将类别确定阶段推广到非均值可分的类别结构。
- [Semantic Routing](https://arxiv.org/abs/2602.03510)：本轮仅读摘要，
  已指出CFG下数值时间与有效SNR失配可能损害时间相关的语义注入。

本仓库条件撤除实验给出了两模型语义留存阶段差异，但没有证明该差异造成PFR迁移
失败。直接时间迁移试验尚在运行；此时不按未检验的阶段解释改参数，也不将
“语义时钟”“阶段对齐”这些宽泛概念作为已成立的新颖性。

## 完成结果与判定

| arm | 1K FID | IS | 采样秒数 | 每样本完整前向 |
|---|---:|---:|---:|---:|
| ordinary100 | 38.264239 | 59.077 | 802.983 | 100 |
| ordinary110 | 38.392635 | 59.191 | 875.383 | 110 |
| short | 38.939462 | 61.486 | 869.423 | 110 |
| asynchronous | 48.972310 | 70.864 | 880.769 | 110 |
| time_only | 49.931931 | 69.266 | 883.500 | 110 |

五组同noise/label/checkpoint/source hash；像素与评价样本hash、实际调用数全部
通过核对。独立feature-Gram FID重算最大差约2.7e-5。
汇总为`experiments/results/terminal_defect_20260908/raev2_fsg_clock_transfer_all.csv`；
审计代码为`experiments/analyze_raev2_fsg_clock_transfer.py`。
五组采样累计4312.058秒，即1.198 GPU小时；不包含模型加载、smoke及FID。

本轮直接迁移失败。去掉forward状态移动没有解决退化；IS上升不能替代FID。
三个候选均不进入自动5K扩展。历史RAEv2 5K基线/PFR为7.034546/7.224213，
仍没有新5K迁移成功案例，不能与此处1K的数值直接比较。
随后固定32样本原生轨迹分解，见RAEV2_CLOCK_DECOMPOSITION_PROTOCOL_20260908_ZH.md，
该诊断提供尺度和抵消关系证据，不将候选失败强行解释成新方法成功。
遵守用户最新约束：只推进方法和机制，暂停论文撰写。
