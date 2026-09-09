# 强弱预测解码后的差异：视觉与光度对照

2026-09-08。继离散空间匹配至多去掉.24% gap能量后，本轮直接查看冻结Full/Base预测，而非继续假设其差都是位置错位或全局对比度。未部署新算法，未生成质量评测样本。

## 完成的工作

解码前一轮保存的全部8轨迹×4固定索引×3种clean预测：Base、Full、Base+1.78(Full−Base)，共96张。FP32关闭TF32，官方stage1 decoder和stats，图像clamp到[0,1]后转uint8。它们是同一个中间状态的读出，不是各自完成续生成的样本。执行1923退出0，32次B3 decoder调用及保存5.792623秒，加载和后续分析额外；没有新增stage2调用、训练或FID。

已实际查看全部8张contact sheet，没有只挑最好样本。索引0均为低对比度纹理，不能据此判断最终类别。后续索引中，人物/工具边缘重影、猫的纹理、鸟与水面边缘、打印机面板等存在变化；主体位置总体一致。该描述只覆盖这8组，不能证明感知质量总体提升，也不把清晰度等同真实结构正确性。

例如 [id0000配对图](/home/zhoushunyu/data/eqvae/experiments/ig_head_correspondence_20260908/decoded/id0000_sheet.png)、[id0713配对图](/home/zhoushunyu/data/eqvae/experiments/ig_head_correspondence_20260908/decoded/id0713_sheet.png)。每张图列顺序Base/Full/IG clean，行顺序0/47/73/87。

## 检验“只是整体颜色/对比度变化”

对每一对图拟合RGB各自的增益和偏置，共6参数；使用交替8×8图块拟合，余下图块评估。预测不额外裁剪。报告 `1−残余平方误差/原两图平方差`，按同索引全部8图的能量汇总。这里的留出像素不是独立图像样本，参数也不是用于部署的校准器。

|索引|Base→Full差异被解释比例|Full→IG clean差异被解释比例|
|---|---:|---:|
|0|86.27%|87.54%|
|47|4.57%|43.10%|
|73|6.82%|13.99%|
|87|9.23%|7.97%|

CPU执行47832退出0，读取并核对全部96单图及8张sheet哈希。结果表明，在本组非零索引中，Base/Full解码差异的大部分不能由每通道全局仿射变换解释。纯噪声行只表明低对比纹理有较强光度成分，不外推到后续生成。Full→IG在索引47的光度成分较多，也不能用单个比例解释全部guidance收益。

这既不证明剩余部分都是质量修正，也不验证频谱、模糊或局部变形机制。此前全局方差、频谱和读出校准的实际质量负结果继续有效，不因为本次观图重新启动这些旧配方。下一项候选需要具体说明怎样区分有用结构修正与错误结构，并由完整生成裁决。

实现 `experiments/decode_raev2_head_correspondence.py`、`experiments/analyze_raev2_head_photometric.py`；机器结果 `experiments/results/terminal_defect_20260908/raev2_head_photometric.json`；图像及原始clipping比例在 `/home/zhoushunyu/data/eqvae/experiments/ig_head_correspondence_20260908/decoded/`。decoder/stat SHA另在渲染后核对，见该目录 `decoder_identity_audit.json`，不冒称加载前绑定。研究目标尚未完成，论文暂停。
