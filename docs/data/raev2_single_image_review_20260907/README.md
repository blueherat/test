# 单图质量思考的小型证据归档（2026-09-07）

**最终用户指令：回归 guidance，只等待既定 5K 完成；其他新方向全部停止。** 本目录仅作历史档案，patch 相位假设也不再是继续研究的推荐。

本目录保存本轮非 guidance 思考的小型证据，供离开原数据盘后核对。**没有找到已成立的新单图质量方法，没有 FID 改善结果；不继续训练、参数扫描或质量采样。** 正在完成的四个旧 guidance 候选 5K 补测属于独立冻结协议，不包含在这里。

| 类别 | 状态与内容 | 目录 |
|---|---|---|
| 采样加速、多速率及代理场 | 用户已明确拒绝；停止，不换名推荐。保存十篇原文身份 manifest 与 CPU 理论反例。 | [rejected_sampling](rejected_sampling/) |
| 初始噪声及多视图耦合 | 不构成当前单图分布质量改进；保持单样本边缘的耦合不能改善总体 FID。保存原文身份、失败获取后的正式来源恢复记录和 CPU 代数检查。 | [out_of_scope_coupling](out_of_scope_coupling/) |
| 极小 head、norm、表示对齐、EMA 后训练 | 未识别足够证据的实际瓶颈，不推荐训练。保存四篇原文身份与阅读范围。 | [unestablished_small_training](unestablished_small_training/) |
| patch 相位 | 本轮曾唯一保留的结构假设；最终用户回归 guidance 后一并停止，未成立为方法。保存 request、summary、独立审核及两组逐图相位统计。 | [patch_phase](patch_phase/) |

最终判断见[一个保留假设与收束](../../RAEV2_SINGLE_IMAGE_FINAL_IDEA_20260907_ZH.md)。其他评审见[采样加速](../../RAEV2_ADJACENT_SAMPLING_IDEAS_20260907_ZH.md)、[精确 speculative 初读](../../RAEV2_ADJACENT_EXACT_SAMPLING_REVIEW_20260907_ZH.md)、[代理场反方检查](../../RAEV2_ADJACENT_SURROGATE_REVIEW_20260907_ZH.md)、[耦合](../../RAEV2_ADJACENT_COUPLING_IDEAS_20260907_ZH.md)、[相邻 decoder](../../RAEV2_ADJACENT_DECODER_IDEAS_20260907_ZH.md)、[实际生成域 decoder 适配](../../RAEV2_SINGLE_IMAGE_DECODER_ADAPTATION_REVIEW_20260907_ZH.md)、[极小后训练](../../RAEV2_SINGLE_IMAGE_SMALL_TRAINING_REVIEW_20260907_ZH.md)。

## patch 相位数据如何解释

旧 scale-response seed20260801/20260802 各选每类最小 global sample ID，共 1000 张；原图、其重构、相同类别槽位的生成图为三分支。生成图不是原图的逐图目标配对。NPZ 中 `sample_ids/labels/source_rows/test_mask` 保存身份，`source_real/reconstruction/generated_ig1p78` 各为 `(1000,4,16)` FP64 数组；第二维依次为 `dx,dxx,dy,dyy`，最后一维为 pixel phase0–15。

两个 seed 的重构与生成均有稳定周期特征，生成的首差边界对比低于重构，而完整相位曲线 RMS 偏差略高。不能因此称为生成域独有的额外接缝。旧链包括 encoder、FP16 latent、decoder、clamp 与 FP16 像素存储，尚不能单独归因于 decoder。统计没有证明压平相位或更换读出会改善图像或 FID。独立审核有 26 项检查；48 个原像素 spot 与全部汇总独立复算通过。只读取旧图，没有 GPU、模型、训练、FID 或新增 5K 图片调用。

## 复制与未保存材料的边界

[portable_manifest.json](portable_manifest.json) 将每份原始绝对路径、原 SHA256、字节数映射到本目录相对路径。复制文件保持原始字节；其中旧 manifest/summary 的绝对路径只作来源记录，读取本归档时使用 `copy_path`，不要求原数据盘在线。[verification.json](verification.json) 由复制完成后的独立读取核对源文件与副本哈希，并检查 JSON/NPZ 可读取。

没有复制论文 PDF、HTML、全文 TXT、图片、模型、训练集或大缓存。阅读 manifest 提供外部原文身份与实际阅读范围，并不表示正文已包含在 Git 中。

相邻 decoder 文档中的早期 Procrustes 数字来自 inline CPU 检查，作者确认未保存独立脚本或 JSON；本次如实登记缺口，未重跑或伪造原始记录。decoder/surrogate/actual-q 评审的其他引用与推导只保存在上述 Markdown；未发现单独下载 manifest。精确 speculative 初读尚未完成全文审核，不能按已精读计数。现有文档包含未执行草案，当前收束状态以上表及最终文档为准，不由旧草案自动授权重启。
