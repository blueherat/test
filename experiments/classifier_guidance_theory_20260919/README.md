# 分类器与控制自由度的解析检查

| 脚本 | 对应报告 |
|---|---|
| [`analytic_checks.py`](analytic_checks.py) | [判别空间、外推系数与弱头自洽性](../../docs/classifier_guidance/DISCRIMINATOR_SCALE_THEORY_20260919_ZH.md) |
| [`control_freedom_checks.py`](control_freedom_checks.py) | [可学习修正场](../../docs/classifier_guidance/GUIDANCE_FREEDOM_20260919_ZH.md) |

两个脚本均通过 `--output` 显式选择 JSON 输出位置，例如：

```bash
"$HOME/miniconda3/envs/myenv/bin/python" -m experiments.classifier_guidance_theory_20260919.analytic_checks \
  --output /tmp/classifier_guidance_analytic_checks.json
"$HOME/miniconda3/envs/myenv/bin/python" -m experiments.classifier_guidance_theory_20260919.control_freedom_checks \
  --output /tmp/classifier_guidance_control_freedom_checks.json
```

后续 self-guidance 与离散反传检查见[理论脚本目录](../theory_self_guidance_20260922/README.md)。
