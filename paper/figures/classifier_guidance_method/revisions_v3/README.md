# v3：三种可编辑论文概念图

三版呈现同一套 SiT 弱头与有符号系数联合训练方法。所有文字、公式、希腊字母和上下标统一使用 **Liberation Sans** 常规／粗体。雪花表示冻结参数，火焰表示可训练参数，橙色虚线表示梯度流。Strong、Weak 和判别器用简化的小网络表示。

[三页可编辑 PPT](concept_variants_v3.pptx) · [三版对比预览](comparison.png) · [参考论文](REFERENCES.md) · [自审记录](REVIEW.md)

| 方案 | 排布与用途 | 图稿 |
|---|---|---|
| A：双栏解释 | 保留接近 AdvFD 的双栏结构及底部分布示意，便于与旧稿比较 | [PNG](A_classic.png) · [SVG](A_classic.svg) · [PDF](A_classic.pdf) |
| B：共享主流程 | 采样、解码、特征提取和判别器只画一次，下方列两类更新目标，适合方法总览 | [PNG](B_shared.png) · [SVG](B_shared.svg) · [PDF](B_shared.pdf) |
| C：交替更新 | 上行更新判别器，下行更新引导；按实际更新顺序展示冻结状态和梯度路径，推荐作为论文主图继续修改 | [PNG](C_lanes.png) · [SVG](C_lanes.svg) · [PDF](C_lanes.pdf) |

B 的主流程图标表示整个训练过程中会学习的参数，每一步的冻结状态由下方两个目标框明确列出。A 和 C 的每个分图直接显示该步的冻结状态。

## 编辑

PPT 中的文字、公式上下标、模块、箭头、小网络、雪花和火焰均为原生可编辑对象。A 仅有两张噪声纹理图片；B、C 全部由文字和矢量对象构成。SVG 内嵌字体并保留文字元素，也可在矢量编辑器中修改。PDF 用于插入论文，PNG 用于预览。

PPT 嵌入常规／粗体字体，主题默认字体也设为 Liberation Sans。若编辑软件不支持嵌入字体，可安装 `fonts/` 中的两份 TTF；许可见 [LICENSE.txt](fonts/LICENSE.txt)。箭头是可编辑路径，移动模块后需同步调整箭头。

小网络的块数和高度仅用于表示模块类型，不对应真实网络层数。Weak 是利用 Strong 上下文的原生弱头，不表示一个独立的完整骨干。A 的散点是概念示意，不是实验测量结果。详细方法定义沿用 [CAPTION.md](../CAPTION.md) 中 SiT 联合训练部分；JiT 仅学习系数的图仍保留在 v2 中。

## 重建

在仓库根目录运行：

```bash
python -m pip install -r paper/figures/requirements.txt
python paper/figures/classifier_guidance_method/revisions_v3/build_variants.py
```

`build_variants.py` 维护三版布局，并复用上一级的 `drawing.py`、`embedded_fonts.py` 和 `build_figure.py`。复制源码时需要同时保留这些依赖；直接编辑导出的 PPT 不需要 Python。重建会覆盖本目录的图稿、场景 JSON 和自动检查记录，手工改动的 PPT 请另存。

所有概念图文件集中在 `paper/figures/`，在 `docs/concept-figure` 分支维护。参考论文下载与临时办公软件渲染不放入仓库。
