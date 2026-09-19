# 辅助评价实现修正：显式保持 cuDNN 开启

首轮辅助评价在第一个生成 bank 的特征复现检查失败，尚未输出任何 precision/recall、KID 或分类器结果。最大特征差 0.00582457，超过冻结阈值 0.0002。原因已定位为新评价脚本调用 `torch.backends.cudnn.flags(allow_tf32=True)` 时遗漏 `enabled`：安装版本该上下文的 `enabled` 默认是 False，使新的参考特征以不同卷积实现计算。

在同一原始首批 32 张图像上的受控检查中，cuDNN=True、TF32=True 与原缓存特征逐位相同（最大差 0）；cuDNN=True、TF32=False 的最大差 0.00582695；cuDNN=False、TF32=True 的最大差 0.00582457。该结果与原 nanogen 子进程使用 torch 默认 cuDNN 设置一致。

修正版明确设置 `enabled=True, benchmark=False, deterministic=False, allow_tf32=True`，保持 matmul TF32=False。完整重新提取 10K 参考特征，并重新执行所有预定二十个 bank 的特征检查与辅助评价；阈值、图像集合和指标公式不变。分类器仍使用已规定的 TF32=False。

首轮原始请求、失败参考特征和原脚本均保留，标注实现失败，不混入正式结果。新目录为 `/home/zhoushunyu/data/eqvae/experiments/official_sit_xl_multimetrics_fixed_20260909`，新入口 `experiments/evaluate_official_sit_xl_multimetrics_fixed_20260909.py`。修正发生在任何新增辅助质量结果产生前，属于执行一致性修复，不是根据辅助质量分数修改协议。
