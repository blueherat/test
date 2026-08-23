# AdvFD clean-room reproduction

本目录保存 AdvFD 的论文级自主复现、官方实现核对、缩放实验和机制审计代码。代码存在
不表示对应假设已经通过；当前结论以
[`ADVFD_OFFICIAL_PMF_B_10K_RESULTS_ZH.md`](../../docs/ADVFD_OFFICIAL_PMF_B_10K_RESULTS_ZH.md)
为准。

## 当前结论

- static FD 后训练稳定改善当前 pMF-B 基线。
- 10K 短预算下，adaptive branch 相对 static 的增量很小，并依赖评估表示。
- 严格配对 5K 评估中，ADM-FID 和 IS 改善，但 JiT-FID 略差，ConvNeXt、DINOv2、
  CLIP 组成的 held-out FDr3 也更差。
- adaptive critic 的 real/fake feature 一起发生尺度漂移和低秩化；当前 checkpoint 不支持
  selective fake-artifact amplification 作为实际训练机制。

## 代码分组

### 论文描述的独立实现

- `core.py`：Fréchet moments、whitening 和 adaptive objective 核心公式。
- `feature_extractors.py`、`generators.py`：冻结表示与生成器适配。
- `run_pmf_pilot.py`：不依赖 AdvFD 官方训练代码的 pMF pilot。
- `run_critic_generalization_toy.py`、`run_selective_amplification_analytic.py`：critic
  generalization 与 selective-amplification 反例。

### 官方实现核对与缩放复现

- `run_official_advfd_packed.py`：对官方入口做最小适配，并记录协议 manifest。
- `run_pmf_b_official_code.sh`、`run_pmf_b_official_static_code.sh`：AdvFD/static
  匹配训练入口。
- `diagnose_distributed_gradient_scaling.py`：验证多卡 loss/gradient averaging 语义。
- `audit_official_timm_equivalence.py`：验证表示模型与预处理等价性。
- `audit_official_advfd_features.py`：checkpoint critic 的 fresh held-out feature 审计。

### 正式评估

- `run_official_pmf_eval.py`：官方 pMF 生成/评估适配器。
- `eval_pmf_b_official_10k_fdr3.sh`：严格配对生成 static/AdvFD 各 5K 张图，并在
  ConvNeXt、DINOv2、CLIP 和两套 Inception reference 上统一评估。
- `summarize_official_fdr3.py`：校验论文 FDr3 三个组成项及固定 normalizer，输出机器可读
  汇总。
- `diagnose_official_advfd_checkpoints.py`、`plot_official_advfd_diagnostics.py`：训练与
  critic 诊断。

### 历史入口

`run_pmf_b_official_25k_pipeline.sh`、`run_pmf_b_official_62p5k_continuation.sh` 和相关
watch 脚本保留用于恢复实验协议，但当前扩训已经停止，不能把未完成 checkpoint 当作
质量结论。

## 验证

```bash
PYTHONPATH=. pytest -q tests/test_advfd_*.py
bash -n experiments/advfd_cleanroom/*.sh
```

大型数据、官方仓库、checkpoint、生成图片、模型缓存和原始 feature 均位于 `/data`
或用户数据目录，不进入 Git。仓库只保存实现、测试、报告和轻量 CSV/JSON 汇总。
