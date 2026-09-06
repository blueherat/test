# Guidance 阅读：上界如何约束设计，评价如何识别新作用

日期：2026-09-06。新增两篇原文；本文没有启动新调度、训练或 FID 实验。全文归档在实验库 `reading_discrepancy_evaluation_v1/`，来源与 SHA 见 `source_manifest.json`。

## C²FG：可取的是可计算的差异尺度，尚未导出最优调度

读了 CVPR 2026 [正文 §3–4](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_C2FG_Control_Classifier-Free_Guidance_via_Score_Discrepancy_Analysis_CVPR_2026_paper.pdf)和[附录 §2.1–2.2、3.2、5](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Gao_C2FG_Control_Classifier-Free_CVPR_2026_supplemental.pdf)。未取得作者实现，不声称核验了运行代码。

对同一高斯加噪核、初始支持半径 R 内的两分布，Tweedie 恒等式给出 `||score_p−score_q|| ≤ 2R α/σ²`；VE 中 α=1。该上界约束真实 score 差异的尺度。它既未刻画有限网络误差，也未证明实际差异逐点单调，更没有确定最优 guidance 强度。正文的 `w(t)=w₀ exp[λ(1−t/T)]` 是有理论启发的设计，两个系数仍按实验设置选择。表 1 的 SiT REPA SDE 为 1.80→1.51；加入已有 interval 的较强对照为 1.42→1.41。不同模型/采样器使用不同 λ，不能直接称为当前无需手调的 RAE 方法。

**独立反例，非论文结论。** 取 VE，正类原分布为 `Uniform[.9,1.1]`，无条件分布为它与 `Uniform[−1.1,−.9]` 的等权混合。在 x=1，正类 score 恒为零。令 `q₊,q₋` 为加高斯噪声后的两个密度，则差异为 `|q₋′(1)|/(q₊(1)+q₋(1))`。σ 为 `.2,.5,1,2,4` 时，差异依次为 `1.31586e−19,.00293866,.23900941,.18866200,.05858743`；均满足 `2.2/σ²`，却先增后减。CPU 双精度与独立 90 位计算一致。初始分布有紧支撑，σ>0 后密度光滑且正；它没有违反上述上界假设。

因此不能从“包络随噪声减小而增大”直接选择同形的 gain。对 RAEv2，F/B 同类且不是已证明属于同一精确加噪核的 score；更应先检验它们的具体误差及实际轨迹响应。这个判断保留论文有效的上界与实证收益，也保留理论到系数之间尚未完成的一步。

## Guidance Matters：改变语义强度本身也会改变评价分数

读了 ICLR 2026 论文的 [arXiv v1 正文 §3–5、附录 B/D 与表 8–9](https://arxiv.org/pdf/2602.22570v1)。作者发现若干偏好指标偏向较强 CFG，并提出按有效 CFG 强度对照。对 Δ=conditional−unconditional，同一状态的正确带符号投影系数是 `⟨new−unconditional,Δ⟩/||Δ||²`。这能诊断方法是否主要沿旧 CFG 方向加强。

但局部投影不是终点等价定理。Eq.9 的范数比会丢掉负系数的符号；沿时间平均系数也不保持非线性轨迹。论文没有证明所有新 guidance 无效：表 1 的 Z-Sampling 在校准后仍有偏好优势。与此同时，表 8 的 COCO FID 为 CFG 13.56、Z-Sampling 17.32、其有效 CFG 对照 14.21，说明奖励优势不自动等于分布质量优势。这些特定 SD/DiT 设置不能代替 RAEv2 实测或完整成本比较。

**对当前实验的具体影响。** 终点类中心对齐量 ψ 是可检验的误差线索；沿其梯度提高 ψ 仍可能只是强化语义。8 图完整伴随实验即使有正响应，也不能据此把新标量网络直接送入训练。还需验证该方向对独立真实分布误差的作用，以及是否仅重现已有 IG 的作用。若后来需要有效强度对照，保留投影符号并清楚报告同状态诊断与实际重放的区别；不能用手设调度替代机制设计。

## 来源锁定

| 原文 | SHA256 |
|---|---|
| C²FG CVPR 正文 | `fb459e6ad772feda2d2c69f86defce62c1cb153ed29e0813b7b05f5de0532ab7` |
| C²FG 官方附录 | `642058386ac519ef6ba1b4cc4ac68decc724bd2d6ae0a11e9c992243ef1de09b` |
| Guidance Matters arXiv v1 PDF | `6acdfb5c66f7d1f8725a761e87b338b721189ef9858f6617394fa9e90318994b` |

另保留下载失败的来源说明：Guidance Matters OpenReview PDF 返回 403，随后成功取得明确版本的 arXiv PDF；C²FG HTML 返回 403，但官方正文和附录 PDF 均成功。没有依据第三方摘要记录成绩。
