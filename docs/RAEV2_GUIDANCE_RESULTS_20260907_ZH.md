# 2026-09-07 guidance 研究结果（持续更新）

新 3% 目标仍未达成。固定官方 EMA + native BF16 IG1.78，100-step shift8 Euler，1K 个均衡类别样本，seed202609071，B8，每 batch 初始噪声和类别逐一配对。官方 nanogen Inception / imagenet_256_fid_stats；不能和旧非官方 evaluator 的绝对值混用。改善为 100(1−FID/FID官方)，负值表示变差。

| 方法 | FID | 相对改善 | 采样与解码 GPU 秒总和 |
|---|---:|---:|---:|
| official | 38.486774 | 0 | 632.3 |
| 历史 t≥.5 interval | 38.335024 | +0.3943% | 636.3 |
| Gaussian Markov ancestral η=1 | 38.894423 | −1.0592% | 637.6 |
| partial η=.5 | 38.506604 | −0.0515% | 637.8 |
| 固定 guided reverse 方差 | 38.541200 | −0.1414% | 630.3 |
| 全图速度投影 | 38.478746 | +0.0209% | 636.2 |
| 全图物理噪声投影 | 38.483936 | +0.0074% | 640.3 |
| 随机单块弱参考，固定 .25 | 38.270118 | +0.5629% | 1088.9 |
| 均值门控弱参考，固定 .25 | 38.566323 | −0.2067% | 1118.9 |

以上为同一 seed 的探索检验，没有独立 seed 或 5K 确认。达到当前官方 1K 基线的 3% 门槛需要 FID≤37.332171。本轮没有通过增加成本但变差的控制线制造改善；未改善的方法不再追加成本匹配采样。

诊断：祖先采样在真 clean 条件下精确，但使用预测均值会丢失后验方差。基于 1000 个真实前向配对样本直接估计全时刻 MSE、补齐固定均值下的最优球形 reverse 方差之后，FID 仍未改善，因此不能把第一次失败全部归结为这一项。无新增系数的两种全图最小二乘投影几乎无影响，暂不继续调投影权重。

实现与验证：四个 ancestral 解析测试、两个方差公式测试、三个投影参数化测试通过；原生 IG wrapper 与本地 native_clean 逐位一致。每项运行保存模型、配置、decoder、normalization stats、采样代码 SHA；实际图像 SHA 和跨实验配对身份由汇总脚本复核。数据保存在 `/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907`，紧凑可版本化结果见 [screen_ledger.json](../experiments/results/raev2_guidance_20260907/screen_ledger.json)。失败的 smoke 目录及日志保留，不当作成功样本。

弱参考结论：随机删块比原生官方改善 0.5629%，比历史 interval 改善约 0.1693%；均值门控略变差。这只是方向信号，仍未达标，不能断言随机性本身就是原因（非线性平均也不同）。选择已经冻结的随机删块 .25 进入独立 seed202609072 的 5K，比较原官方及历史 interval。没有基于 FID 修改层、窗口、强度或 mask 分布。实测成本约 1.72 倍；若达到质量门槛，补充实测成本匹配的官方控制。首次 FID 评测因 Hugging Face HEAD 请求 SSL EOF 中断，保留失败日志；恢复时使用原已缓存 reference，HF_HUB_OFFLINE=1，没有重新生成图像。

图像判别器方向：两种 Gaussian posterior covariance 的解析检查通过，真实 decoder/DINO 上 10 个固定时刻梯度有限、中心差分相符，固定修正提高诊断图像的代理 logit。t=1 的一阶预测明显高估实际增益（isotropic 预测约11.04、实际约.348），说明高噪声处局部线性近似很差；不删去该时刻或隐去诊断。正在检查全轨迹，尚无 FID。详见 [独立判别器 guidance 设计](RAEV2_IMAGE_CRITIC_GUIDANCE_20260907_ZH.md)。

进一步验证：critic_isotropic 和 critic_exchangeable 的完整 8 图轨迹均有限，单 B8 轨迹耗时约27.98/28.04秒，峰值显存约12.35GB；保留 encoder 的 official8 输出与先前原始 official8 逐像素一致。约5.5倍推理成本，尚无正式 FID，不能拿代理 logit 或小图预览当目标成功。

FID 独立审计：使用缓存 Inception 特征，以 N×N Gram 特征值重新计算协方差交叉根迹，独立于原 D×D scipy sqrtm 路径。9个完整1K分支最大偏差2.57e−5。固定 reference SHA为 `925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac`。随机删块的 FID 差可以精确拆成：均值项+0.03556、保持原协方差形状时 trace 改变−0.17686、在新 trace 下归一化形状改变−0.07536，总和−0.21666。这是描述性代数分解，不是“某机制导致FID变化”的因果比例；它进一步支持必须做较大样本的独立确认。
