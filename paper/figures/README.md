# 论文概念图工作区

概念图相关的图稿、绘图源码、字体、参考原图和审查记录集中维护在本目录。
专用分支：`docs/concept-figure`。

| 目录 | 内容 | 入口 |
|---|---|---|
| `classifier_guidance_method/revisions_v3/` | 当前三版方案：统一无衬线字体，保留冻结／训练图标，简化小网络 | [三页可编辑 PPT](classifier_guidance_method/revisions_v3/concept_variants_v3.pptx)、[三版对比](classifier_guidance_method/revisions_v3/comparison.png)、[使用说明](classifier_guidance_method/revisions_v3/README.md) |
| `classifier_guidance_method/` | v2 历史图稿及共用绘图代码；包含 SiT 联合训练和 JiT 仅学系数两种方法 | [v2 PPT](classifier_guidance_method/classifier_guidance_method_v2.pptx)、[v2 说明](classifier_guidance_method/README.md) |
| `advfd_editable/` | AdvFD Figure 3 原图、早期可编辑重绘及其来源记录 | [参考说明](advfd_editable/README.md) |
| `archive/` | 字体与网络图标修改前的 v1 文件包，仅作历史留存 | [归档说明](archive/README.md) |

## 当前图稿

- v3 A：保留双栏与分布示意。[SVG](classifier_guidance_method/revisions_v3/A_classic.svg) · [PDF](classifier_guidance_method/revisions_v3/A_classic.pdf) · [PNG](classifier_guidance_method/revisions_v3/A_classic.png)。
- v3 B：一条共享主流程与两类目标。[SVG](classifier_guidance_method/revisions_v3/B_shared.svg) · [PDF](classifier_guidance_method/revisions_v3/B_shared.pdf) · [PNG](classifier_guidance_method/revisions_v3/B_shared.png)。
- v3 C：按判别器→引导的更新顺序分两行，推荐作为论文主图继续修改。[SVG](classifier_guidance_method/revisions_v3/C_lanes.svg) · [PDF](classifier_guidance_method/revisions_v3/C_lanes.pdf) · [PNG](classifier_guidance_method/revisions_v3/C_lanes.png)。
- [参考论文与借鉴方式](classifier_guidance_method/revisions_v3/REFERENCES.md) · [v3 自审记录](classifier_guidance_method/revisions_v3/REVIEW.md)。

v3 三版都是同一套 SiT 联合训练内容，全部使用 **Liberation Sans** 常规／粗体，字母、希腊符号和公式上下标保持同一字体。雪花表示冻结参数，火焰表示可训练参数。v2 保留 Comic Sans MS 历史稿；AdvFD 参考底稿保留 Comic Neue / DejaVu Serif。各版本的字体与许可放在对应 `fonts/` 中。

JiT 仅学习系数的历史版本：[SVG](classifier_guidance_method/method_schedule_v2.svg) · [PDF](classifier_guidance_method/method_schedule_v2.pdf) · [PNG](classifier_guidance_method/method_schedule_v2.png)。方法的详细定义见[图注与符号](classifier_guidance_method/CAPTION.md)。

## 修改与重建

可直接修改 PPT 中的文字和图形。v3 布局入口为 `classifier_guidance_method/revisions_v3/build_variants.py`；共用 `drawing.py` 负责绘制，`embedded_fonts.py` 负责 PPT 字体嵌入。运行生成脚本会覆盖导出文件，手工修改的 PPT 请另存后再重建。

在仓库根目录执行：

```bash
python -m pip install -r paper/figures/requirements.txt
python paper/figures/classifier_guidance_method/revisions_v3/build_variants.py
```

若需要重建 AdvFD 参考底稿：

```bash
python paper/figures/advfd_editable/build_figure.py
```

如需重建 v2，运行 `python paper/figures/classifier_guidance_method/build_figure.py`。

Git 跟踪绘图源码、v3 三版方案、v2 历史图稿、对象场景、审查记录、必要参考资源及 v1 历史归档。重复副本、可重新打包的分发 ZIP、字体的本机路径配置和 Python 缓存不进入版本管理。外部论文的新参考图仅记录来源，下载文件和办公软件临时渲染留在 `/tmp`。

`review_checks.json` 保留交付时的检查及源文件哈希；重新绘图会更新对象检查及源文件哈希。完整的交付检查经过独立办公软件渲染和人工目视核对，详见自审记录。
