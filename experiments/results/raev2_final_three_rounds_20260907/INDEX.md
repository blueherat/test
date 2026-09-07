# 最后三轮研究证据：已收束，3%目标未达到

三轮已完成，停止新增实验。第1轮未过机制门槛；第2轮FID6.92572559（+0.34595%）；第3轮FID6.97019678（−0.29394%）；官方基线6.94976848。完整理论与裁决见[最终报告](../../../docs/RAEV2_FINAL_THREE_ROUNDS_CLOSEOUT_20260907_ZH.md)，执行前协议与逐轮结果见[三轮记录](../../../docs/RAEV2_FINAL_THREE_ROUNDS_20260907_ZH.md)。

[可比5K表CSV](paired5k.csv) · [Markdown表](paired5k.md) · [PNG图](paired5k.png) · [PDF图](paired5k.pdf) · [机器结论](closeout.json) · [最终状态](state.json)

| 证据目录 | 内容 | 索引文件数 | 原始文件字节数 |
|---|---|---:|---:|
| [prelude_native_retention](prelude_native_retention/archive_manifest.json) | 三轮限制前的128样本原生有限写入/读取诊断；无FID | 92 | 1,613,155,210 |
| [round1_finite_read_budget](round1_finite_read_budget/archive_manifest.json) | 第1轮，15/128样本改变，4/8时刻组改善；机制门槛阴性，无FID | 51 | 402,786,759 |
| [round2_causal_reference_smoke](round2_causal_reference_smoke/archive_manifest.json) | 第2轮8图原生/零响应实现控制；逐像素一致 | 37 | 36,208,609 |
| [round2_causal_reference_5k](round2_causal_reference_5k/archive_manifest.json) | 第2轮完整配对5K、指标、成本及独立审计 | 52 | 2,063,258,498 |
| [round3_full_read_after_write](round3_full_read_after_write/archive_manifest.json) | 第3轮完整8图控制与配对5K、审计、测试、协议和退出记录 | 94 | 2,099,517,563 |

上述清单实际索引326个文件，共6,214,926,639字节；其中3,828,138字节轻量证据复制进Git。它们是本次前置诊断与最后三轮的清单，不代替全仓库历史清单。每项原始路径、大小、实际SHA256与复制状态可由`archive_manifest.json`追溯；大数组、图像和特征留在数据盘。

[模型与decoder/stat身份](model_identities.json)、[运行环境](environment.json)、[核心论文来源与SHA](paper_reference.json)、[第2轮启动/完成上下文](round2_run_context/manifest.json)、[最终归档核验](archive_verification.json)一并保存。每个采样run的冻结源码与原始request保留在对应证据目录。

本轮全部采样进程已经退出。发现seed202609072已反复探索，独立确认seed202609073本次未用；不把小幅获胜点或归档完成当作质量成功，也不宣称50K/SOTA。此前最佳空间协方差6.90613843仅改善0.62779%。原生有限留存结果、搜索失败和两项5K结果均保留，没有第四轮。
